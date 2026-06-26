use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyArrayMethods};
use pyo3::prelude::*;

pub fn process_pipeline(person_image: &image::RgbImage, necklace_image: &image::RgbaImage, scale_multiplier: f64) -> Result<Option<image::RgbImage>, PyErr> {
    let (p_width, p_height) = person_image.dimensions();
    let p_flat: Vec<u8> = person_image.clone().into_raw();
    println!("pflat {}",p_flat);
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

def process_pipeline(person_np, necklace_np, scale_multiplier):
    # 1. MediaPipe Pose
    results = pose.process(person_np)
    if not results.pose_landmarks:
        return None
        
    annotated = np.copy(person_np)
    h, w, _ = annotated.shape
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(annotated, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(annotated, str(idx), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 2. Geometry & Landmark Acquisition
    l_sh = results.pose_landmarks.landmark[11]
    r_sh = results.pose_landmarks.landmark[12]
    nose = results.pose_landmarks.landmark[0]
    
    # Image space coordinates (x is 0 at left, w at right)
    l_x, l_y = l_sh.x * w, l_sh.y * h
    r_x, r_y = r_sh.x * w, r_sh.y * h
    nose_x, nose_y = nose.x * w, nose.y * h
    
    # Distance and angle
    # Vector from right shoulder to left shoulder (standard angle geometry)
    dx = l_x - r_x
    dy = l_y - r_y
    shoulder_width = np.hypot(dx, dy)
    
    # Calculate body tilt angle (in degrees)
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Midpoints
    mid_x = (l_x + r_x) / 2.0
    mid_y = (l_y + r_y) / 2.0
    
    # Neck seat calculation: push upward by ~18% of shoulder width
    neck_seat_y = mid_y - (shoulder_width * 0.18)
    neck_seat_x = mid_x
    
    # Yaw correction (Perspective squish)
    yaw_offset = (nose_x - mid_x) / (shoulder_width + 1e-5)
    # Squish X if face is significantly turned
    yaw_squish = max(0.6, 1.0 - abs(yaw_offset) * 1.5)
    
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
    
    nh, nw = neck_rgb.shape[:2]
    true_aspect = nh / max(1.0, float(nw))
    
    # Roboflow Sizing Logic (Ported exactly as requested)
    # MediaPipe shoulders are much wider (outer arms) than YOLO's (inner shoulders).
    # We multiply by 0.6 to approximate the inner YOLO shoulder span.
    yolo_shoulder_width = shoulder_width * 0.6
    
    # target_width_pixels = yolo_shoulder_width * roboflow_multiplier (e.g. 0.48)
    target_w = int(yolo_shoulder_width * scale_multiplier * yaw_squish)
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
    # The anchor point on the resized necklace image
    anchor_x = target_w / 2.0
    anchor_y = target_h * anchor_y_frac
    
    # cv2.getRotationMatrix2D builds a 2x3 matrix:
    # [ cos(a) -sin(a)  (1-cos)*cx + sin*cy ]
    # [ sin(a)  cos(a)  -sin*cx + (1-cos)*cy ]
    # This rotates around (anchor_x, anchor_y) but keeps it fixed in the same coordinate location.
    M = cv2.getRotationMatrix2D((anchor_x, anchor_y), angle, 1.0)
    
    # Now add translation to map the anchor point exactly to the neck seat on the main canvas
    M[0, 2] += (neck_seat_x - anchor_x)
    M[1, 2] += (neck_seat_y - anchor_y)
    
    # Warp directly onto a canvas the size of the person image
    warped_rgb = cv2.warpAffine(neck_rgb_resized, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    warped_a = cv2.warpAffine(alpha_resized, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 5. Advanced Alpha Compositing
    # Soften the edges of the mask
    warped_a = cv2.GaussianBlur(warped_a, (3, 3), 0)
    warped_a = np.expand_dims(np.clip(warped_a, 0.0, 1.0), axis=2)
    
    blended = warped_rgb * warped_a + person_rgb.astype(np.float32) * (1.0 - warped_a)
    result = np.clip(blended, 0, 255).astype(np.uint8)
    
    return result


"#, Some(&locals), None)?;

        let process_pipeline = locals.get_item("process_pipeline")?.unwrap();
        let ret = process_pipeline.call1((p_np, n_np, scale_multiplier))?;
        
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
