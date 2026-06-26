use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyArrayMethods};
use pyo3::prelude::*;

pub fn process_pipeline(person_image: &image::RgbImage, necklace_image: &image::RgbaImage, scale_multiplier: f64, style: &str, y_offset: Option<f64>) -> Result<Option<image::RgbImage>, PyErr> {
    let (p_width, p_height) = person_image.dimensions();
    let p_flat: Vec<u8> = person_image.clone().into_raw();
    println!("pflat len: {}", p_flat.len());
    let p_ndarray: Array3<u8> =
        Array3::from_shape_vec((p_height as usize, p_width as usize, 3), p_flat).unwrap();

    let (n_width, n_height) = necklace_image.dimensions();
    let n_flat: Vec<u8> = necklace_image.clone().into_raw();
    let n_ndarray: Array3<u8> =
        Array3::from_shape_vec((n_height as usize, n_width as usize, 4), n_flat).unwrap();

    Python::with_gil(|py| {
        let p_np = p_ndarray.into_pyarray_bound(py);
        let n_np = n_ndarray.into_pyarray_bound(py);

        let os = py.import_bound("os")?;
        let path_str: String = std::env::var("PATH").unwrap_or_default();
        for p in path_str.split(';') {
            if !p.is_empty() {
                let _ = os.call_method1("add_dll_directory", (p,));
            }
        }

        let locals = pyo3::types::PyDict::new_bound(py);
        py.run_bound(r#"
import mediapipe as mp
import cv2
import numpy as np

pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

def largest_component(mask):
    try:
        mask_u8 = (mask.astype(np.uint8) * 255)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
        if num <= 1:
            return mask
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest
    except Exception:
        return mask

def border_background_mask(rgb):
    try:
        h0, w0 = rgb.shape[:2]
        border = max(3, int(min(h0, w0) * 0.03))
        samples = np.concatenate([
            rgb[:border, :, :3].reshape(-1, 3),
            rgb[-border:, :, :3].reshape(-1, 3),
            rgb[:, :border, :3].reshape(-1, 3),
            rgb[:, -border:, :3].reshape(-1, 3),
        ], axis=0).astype(np.float32)
        bg = np.median(samples, axis=0)
        dist = np.linalg.norm(rgb[:, :, :3].astype(np.float32) - bg.reshape(1, 1, 3), axis=2)
        border_dist = np.linalg.norm(samples - bg.reshape(1, 3), axis=1)
        threshold = max(18.0, float(np.percentile(border_dist, 95)) + 10.0)
        mask = dist > threshold

        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = (mask.astype(np.uint8) * 255)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = mask_u8 > 0
        mask = largest_component(mask)
        return mask
    except Exception:
        return np.ones(rgb.shape[:2], dtype=bool)

def trim_to_visible_pixels(rgb, alpha, force_border_mask=False):
    try:
        if force_border_mask:
            mask = border_background_mask(rgb)
            alpha = np.where(mask, alpha, 0.0).astype(np.float32)
        else:
            mask = alpha > 0.05

        visible_fraction = float(np.mean(mask)) if mask.size else 1.0
        if visible_fraction > 0.80:
            mask = border_background_mask(rgb)
            alpha = np.where(mask, alpha, 0.0).astype(np.float32)

        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return rgb, alpha, False
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        pad_x = max(2, int((x2 - x1) * 0.03))
        pad_y = max(2, int((y2 - y1) * 0.03))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(rgb.shape[1], x2 + pad_x)
        y2 = min(rgb.shape[0], y2 + pad_y)
        return rgb[y1:y2, x1:x2], alpha[y1:y2, x1:x2], True
    except Exception:
        return rgb, alpha, False

def process_pipeline(person_np, necklace_np, scale_multiplier, style_str, y_offset_override):
    try:
        # 1. MediaPipe Pose, Face Mesh & Selfie Segmentation
        pose_res = pose.process(person_np)
        face_res = face_mesh.process(person_np)
        seg_res = segmenter.process(person_np)
        
        print(f"[Debug] person_np shape: {person_np.shape}, dtype: {person_np.dtype}")
        print(f"[Debug] pose_res.pose_landmarks: {pose_res.pose_landmarks is not None}")
        print(f"[Debug] face_res.multi_face_landmarks: {face_res.multi_face_landmarks is not None}")
        print(f"[Debug] seg_res.segmentation_mask: {seg_res.segmentation_mask is not None}")
        import sys; sys.stdout.flush()
    
        if not pose_res.pose_landmarks or not face_res.multi_face_landmarks or seg_res.segmentation_mask is None:
            return None
            
        pose_lms = pose_res.pose_landmarks.landmark
        face_lms = face_res.multi_face_landmarks[0].landmark
        
        annotated = np.copy(person_np)
        h, w, _ = annotated.shape
        
        # Upscale segmentation mask back to original image size
        mask_resized = cv2.resize(seg_res.segmentation_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        binary_mask = mask_resized > 0.5
        
        # Draw Pose landmarks (shoulders, chest, arms)
        for idx, lm in enumerate(pose_lms):
            if idx in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(annotated, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(annotated, str(idx), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
        # 2. Geometry & Landmark Acquisition
        l_sh = pose_lms[11]
        r_sh = pose_lms[12]
        nose = pose_lms[0]
        
        # Image space coordinates
        l_x, l_y = l_sh.x * w, l_sh.y * h
        r_x, r_y = r_sh.x * w, r_sh.y * h
        nose_x, nose_y = nose.x * w, nose.y * h
        
        # Distance and angle
        dx = l_x - r_x
        dy = l_y - r_y
        shoulder_width = np.hypot(dx, dy)
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Midpoints
        sh_mid_x = (l_x + r_x) / 2.0
        sh_mid_y = (l_y + r_y) / 2.0
        
        # Face Mesh landmarks
        chin = face_lms[152]
        jaw_l = face_lms[172]
        jaw_r = face_lms[397]
        
        chin_x, chin_y = chin.x * w, chin.y * h
        jaw_l_x, jaw_l_y = jaw_l.x * w, jaw_l.y * h
        jaw_r_x, jaw_r_y = jaw_r.x * w, jaw_r.y * h
        
        # Draw Face Mesh landmarks on annotated image
        for lm_idx, lm_pt in [(152, (chin_x, chin_y)), (172, (jaw_l_x, jaw_l_y)), (397, (jaw_r_x, jaw_r_y))]:
            cv2.circle(annotated, (int(lm_pt[0]), int(lm_pt[1])), 8, (0, 255, 0), -1)
            cv2.putText(annotated, str(lm_idx), (int(lm_pt[0]) + 10, int(lm_pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
        # Jaw-based measurements
        jaw_width = np.hypot(jaw_l_x - jaw_r_x, jaw_l_y - jaw_r_y)
        est_neck_w_kp = jaw_width * 0.85
        face_center_x = int(chin_x)
        
        # Style-based placement & height Y
        style = style_str.lower()
        if style == "choker":
            default_scale_factor = 1.55
            default_y_offset = 0.0
            # Middle-upper neck (so body sits at pit of throat)
            target_y = chin.y + (sh_mid_y / h - chin.y) * 0.40
        elif style == "collar":
            default_scale_factor = 1.20
            default_y_offset = 20.0
            # Upper/mid neck (tight fit)
            target_y = (face_lms[176].y + face_lms[400].y) / 2.0
        elif style == "pendant":
            default_scale_factor = 1.80
            default_y_offset = 10.0
            # Below collarbone/shoulders
            target_y = sh_mid_y / h + (sh_mid_y / h - chin.y) * 0.15
        else:
            # Fallback to choker
            style = "choker"
            default_scale_factor = 1.55
            default_y_offset = 0.0
            target_y = chin.y + (sh_mid_y / h - chin.y) * 0.40
            
        offset_val = y_offset_override if y_offset_override is not None else default_y_offset
        target_y_px = int(target_y * h) + int(offset_val)
        
        # Hybrid Width Scanning utilizing Segmentation Mask and constraints
        search_min_x = int(face_center_x - jaw_width * 0.65)
        search_max_x = int(face_center_x + jaw_width * 0.65)
        
        # Ensure bounds are within image dimensions
        search_min_x = max(0, search_min_x)
        search_max_x = min(w - 1, search_max_x)
        target_y_px = max(0, min(h - 1, target_y_px))
        
        mask_row = binary_mask[target_y_px, :]
        true_indices = np.where(mask_row == True)[0]
        clipped_indices = [idx for idx in true_indices if search_min_x <= idx <= search_max_x]
        
        mask_width = None
        mask_center_x = None
        if len(clipped_indices) > 0:
            segments = np.split(clipped_indices, np.where(np.diff(clipped_indices) > 1)[0] + 1)
            neck_segment = None
            min_dist = float('inf')
            for seg in segments:
                if len(seg) == 0:
                    continue
                seg_left = seg[0]
                seg_right = seg[-1]
                if seg_left <= face_center_x <= seg_right:
                    neck_segment = seg
                    break
                else:
                    dist = min(abs(seg_left - face_center_x), abs(seg_right - face_center_x))
                    if dist < min_dist:
                        min_dist = dist
                        neck_segment = seg
            if neck_segment is not None:
                mask_width = neck_segment[-1] - neck_segment[0]
                mask_center_x = (neck_segment[0] + neck_segment[-1]) // 2

        # Sizing and centering blending/fallback logic
        sizing_source = "KP Fallback"
        final_neck_width = est_neck_w_kp
        neck_seat_x = face_center_x
        
        if mask_width is not None and mask_center_x is not None:
            # Check if mask-based width is physically realistic (60% to 130% of keypoint estimate)
            if 0.6 * est_neck_w_kp <= mask_width <= 1.3 * est_neck_w_kp:
                # Blend: 70% mask width (dynamic adaptation) + 30% keypoint (anatomical constraint)
                final_neck_width = mask_width * 0.7 + est_neck_w_kp * 0.3
                neck_seat_x = mask_center_x
                sizing_source = "Hybrid (Mask)"
                
        neck_seat_y = target_y_px
        
        # Yaw correction (Perspective squish)
        yaw_offset = (nose_x - sh_mid_x) / (shoulder_width + 1e-5)
        yaw_squish = max(0.6, 1.0 - abs(yaw_offset) * 1.5)
        
        # Sizing compatibility layer for old scales
        if scale_multiplier < 0.8:
            final_scale = (shoulder_width * 0.6 * scale_multiplier) / (jaw_width + 1e-5)
            print(f"[Compatibility] Provided scale ({scale_multiplier}) is < 0.8. Converted to jaw-relative scale: {final_scale:.3f}")
        else:
            final_scale = default_scale_factor * scale_multiplier
            
        # 3. Necklace Overlay Renderer
        person = annotated
        person_rgb = person[:, :, :3]
        
        used_border_mask = False
        if necklace_np.shape[-1] == 4:
            neck_rgb = necklace_np[:, :, :3]
            alpha = necklace_np[:, :, 3].astype(np.float32) / 255.0
        else:
            neck_rgb = necklace_np[:, :, :3]
            mask = border_background_mask(neck_rgb)
            alpha = mask.astype(np.float32)
            used_border_mask = True
            try:
                alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
            except Exception:
                pass
    
        if np.nanmax(alpha) <= 0.02:
            return person_rgb
    
        neck_rgb, alpha, trimmed = trim_to_visible_pixels(neck_rgb, alpha, force_border_mask=used_border_mask)
        if not trimmed:
            return person_rgb
            
        nh, nw = neck_rgb.shape[:2]
        true_aspect = nh / max(1.0, float(nw))
        
        # Calculate target dimensions relative to final_neck_width
        target_w = int(final_neck_width * (final_scale / 0.85) * yaw_squish)
        target_h = int(target_w * true_aspect)
    
        # Anchor point fraction based on aspect ratio
        if true_aspect < 0.35:
            anchor_y_frac = 0.05
        elif true_aspect > 0.8:
            anchor_y_frac = 0.02
        else:
            t = (true_aspect - 0.35) / (0.8 - 0.35)
            anchor_y_frac = 0.05 + t * (0.02 - 0.05)
            
        if target_w < 10 or target_h < 10:
            return person_rgb
            
        try:
            neck_rgb_resized = cv2.resize(neck_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            alpha_resized = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)
        except Exception:
            return person_rgb
        
        # 4. Affine Transform Matrix Setup
        anchor_x = target_w / 2.0
        
        # Dynamically find the Y coordinate of the top edge of the main band in the center of the necklace
        ah, aw = alpha_resized.shape[:2]
        c_left = int(aw * 0.45)
        c_right = int(aw * 0.55)
        center_alpha = alpha_resized[:, c_left:c_right]
        non_zero_ys = []
        for col_idx in range(center_alpha.shape[1]):
            col = center_alpha[:, col_idx]
            ys_col = np.where(col > 0.05)[0]
            if len(ys_col) > 0:
                non_zero_ys.append(ys_col.min())
        
        if len(non_zero_ys) > 0:
            anchor_y = float(np.median(non_zero_ys))
        else:
            anchor_y = target_h * anchor_y_frac
            
        # Dynamic Height Compensation for Choker/Collar styles
        orig_seat_y = neck_seat_y
        bottom_dist = target_h - anchor_y
        
        # Define neck limits
        neck_top = int(chin_y)
        neck_bottom = int(sh_mid_y)
        height_shift = 0
        
        if style == "choker":
            # We want the bottom of the choker to sit above the collarbones (neck_bottom - 15)
            ideal_bottom_y = neck_bottom - 15
            if neck_seat_y + bottom_dist > ideal_bottom_y:
                neck_seat_y = ideal_bottom_y - int(bottom_dist)
            # Apply chin guard (at least 15px below the chin)
            neck_seat_y = max(neck_top + 15 + int(anchor_y), neck_seat_y)
            height_shift = neck_seat_y - orig_seat_y
            
        elif style == "collar":
            # Collars are worn higher, bottom of collar should sit above neck_bottom - 30
            ideal_bottom_y = neck_bottom - 30
            if neck_seat_y + bottom_dist > ideal_bottom_y:
                neck_seat_y = ideal_bottom_y - int(bottom_dist)
            # Apply chin guard (at least 10px below the chin)
            neck_seat_y = max(neck_top + 10 + int(anchor_y), neck_seat_y)
            height_shift = neck_seat_y - orig_seat_y
            
        # Print measurement report
        print(f"--- Sizing Report ---")
        print(f"Jaw Width: {jaw_width:.1f} px")
        print(f"Estimated Neck Width (anatomical kp): {est_neck_w_kp:.1f} px")
        if mask_width is not None:
            print(f"Raw Mask Neck Width: {mask_width:.1f} px")
        print(f"Final Neck Width (used): {final_neck_width:.1f} px")
        print(f"Sizing Source: {sizing_source}")
        print(f"Target Necklace Width (scaled + yaw squished): {target_w:.1f} px")
        print(f"Target Necklace Height: {target_h:.1f} px")
        print(f"Necklace is {target_w / (final_neck_width + 1e-5) * 100:.1f}% of the neck width")
        print(f"Style: {style.upper()} (Offset: {offset_val:.1f} px)")
        if height_shift != 0:
            print(f"Height Compensation Shift: {height_shift} px")
        print(f"---------------------")
        
        print(f"[Debug] Calculated dynamic anchor_y: {anchor_y:.1f} px (default bounding-box top was {target_h * anchor_y_frac:.1f} px)")
        if height_shift != 0:
            print(f"[Debug] Height Compensation shifted neck_seat_y from {orig_seat_y} px to {neck_seat_y} px (necklace bottom dist: {bottom_dist:.1f} px)")
        
        M = cv2.getRotationMatrix2D((anchor_x, anchor_y), angle, 1.0)
        M[0, 2] += (neck_seat_x - anchor_x)
        M[1, 2] += (neck_seat_y - anchor_y)
        
        warped_rgb = cv2.warpAffine(neck_rgb_resized, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        warped_a = cv2.warpAffine(alpha_resized, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # 5. Advanced Alpha Compositing
        warped_a = cv2.GaussianBlur(warped_a, (3, 3), 0)
        warped_a = np.expand_dims(np.clip(warped_a, 0.0, 1.0), axis=2)
        
        blended = warped_rgb * warped_a + person_rgb.astype(np.float32) * (1.0 - warped_a)
        result = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Draw cyan measurement bar for final neck width with black outline
        cv2.line(result, 
                 (int(neck_seat_x - final_neck_width/2), int(neck_seat_y)), 
                 (int(neck_seat_x + final_neck_width/2), int(neck_seat_y)), 
                 (0, 0, 0), 5)
        cv2.line(result, 
                 (int(neck_seat_x - final_neck_width/2), int(neck_seat_y)), 
                 (int(neck_seat_x + final_neck_width/2), int(neck_seat_y)), 
                 (0, 255, 255), 3) # Cyan in RGB is (0, 255, 255)
                 
        # Draw orange measurement bar for necklace width with black outline
        cv2.line(result, 
                 (int(neck_seat_x - target_w/2), int(neck_seat_y + 40)), 
                 (int(neck_seat_x + target_w/2), int(neck_seat_y + 40)), 
                 (0, 0, 0), 5)
        cv2.line(result, 
                 (int(neck_seat_x - target_w/2), int(neck_seat_y + 40)), 
                 (int(neck_seat_x + target_w/2), int(neck_seat_y + 40)), 
                 (255, 165, 0), 3) # Orange in RGB is (255, 165, 0)
                 
        # Sizing Info Dashboard (Top-Left)
        card_x1, card_y1 = 40, 40
        card_x2, card_y2 = 480, 290
        overlay = result.copy()
        cv2.rectangle(overlay, (card_x1, card_y1), (card_x2, card_y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, result, 0.25, 0, result)
        
        # Card Border
        cv2.rectangle(result, (card_x1, card_y1), (card_x2, card_y2), (180, 180, 180), 2)
        
        # Title
        cv2.putText(result, "SIZING ANALYSIS", (card_x1 + 20, card_y1 + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Divider line
        cv2.line(result, (card_x1 + 20, card_y1 + 45), (card_x2 - 20, card_y1 + 45), (100, 100, 100), 1)
        
        # Metrics
        cv2.putText(result, f"Style / Fit:      {style.upper()}", (card_x1 + 20, card_y1 + 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    
        cv2.putText(result, f"Final Neck Width: {final_neck_width:.1f} px", (card_x1 + 20, card_y1 + 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA) # Cyan
                    
        cv2.putText(result, f"Necklace Width:   {target_w:.1f} px", (card_x1 + 20, card_y1 + 165), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1, cv2.LINE_AA) # Orange
                    
        cv2.putText(result, f"Sizing Source:    {sizing_source}", (card_x1 + 20, card_y1 + 205), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA) # Green
                    
        cv2.putText(result, f"Vertical Offset:  {offset_val:.1f} px", (card_x1 + 20, card_y1 + 245), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA) # Grey
                    
        import sys
        sys.stdout.flush()
        
        return result
    except Exception as e:
        import traceback
        print("[Python Error] Exception in process_pipeline:")
        traceback.print_exc()
        import sys; sys.stdout.flush()
        return None
"#, Some(&locals), None)?;

        let process_pipeline = locals.get_item("process_pipeline")?.unwrap();
        let ret = process_pipeline.call1((p_np, n_np, scale_multiplier, style, y_offset))?;
        
        if ret.is_none() {
            return Ok(None);
        }

        let annotated_array = ret.downcast::<PyArray3<u8>>()?;
        let annotated_ndarray = annotated_array.to_owned_array();
        let mut annotated_image = image::RgbImage::new(p_width, p_height);
        for y in 0..p_height {
            for x in 0..p_width {
                let r = annotated_ndarray[[y as usize, x as usize, 0]];
                let g = annotated_ndarray[[y as usize, x as usize, 1]];
                let b = annotated_ndarray[[y as usize, x as usize, 2]];
                annotated_image.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }

        Ok(Some(annotated_image))
    })
}
