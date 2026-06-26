mod pose;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    person: String,

    #[arg(short, long)]
    necklace: String,

    #[arg(short, long)]
    output: String,

    #[arg(short, long, default_value_t = 1.0)]
    scale: f64,

    #[arg(short, long, default_value = "choker")]
    style: String,

    #[arg(short, long)]
    y_offset: Option<f64>,
}

fn main() {
    // Force the embedded Python interpreter to load its standard library from Python 3.11 (Windows only)
    #[cfg(target_os = "windows")]
    {
        std::env::set_var("PYTHONHOME", "C:\\Users\\Thethinker\\AppData\\Local\\Programs\\Python\\Python311");
        std::env::set_var("PATH", "C:\\Users\\Thethinker\\AppData\\Local\\Programs\\Python\\Python311;C:\\Users\\Thethinker\\AppData\\Local\\Programs\\Python\\Python311\\Scripts");
    }

    let args = Args::parse();

    println!("Loading person image: {}", args.person);
    let person_img_dynamic = image::open(&args.person).expect("Failed to open person image");
    let person_rgba = person_img_dynamic.to_rgba8();
    
    // Convert to RGB for MediaPipe processing
    let person_rgb = image::DynamicImage::ImageRgba8(person_rgba.clone()).into_rgb8();

    println!("Loading necklace image: {}", args.necklace);
    let necklace_img = image::open(&args.necklace).expect("Failed to open necklace image");
    let necklace_rgba = necklace_img.to_rgba8();

    println!("Extracting pose landmarks and rendering necklace via Python/OpenCV pipeline...");
    let pipeline_result = pose::process_pipeline(&person_rgb, &necklace_rgba, args.scale, &args.style, args.y_offset);
    
    match pipeline_result {
        Ok(Some(rendered_img)) => {
            println!("Pipeline successful! Saving output to: {}", args.output);
            let final_rgba = image::DynamicImage::ImageRgb8(rendered_img).into_rgba8();
            final_rgba.save(&args.output).expect("Failed to save output image");
            println!("Done.");
        }
        Ok(None) => {
            eprintln!("No pose detected in the image.");
        }
        Err(e) => {
            eprintln!("Python error during pose extraction: {:?}", e);
        }
    }
}
