mod pose;

use clap::Parser;
use axum::{
    routing::{get, post},
    extract::Multipart,
    response::{Html, IntoResponse, Response},
    http::{StatusCode, header},
    Router,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    person: Option<String>,

    #[arg(short, long)]
    necklace: Option<String>,

    #[arg(short, long)]
    output: Option<String>,

    #[arg(short, long, default_value = "choker")]
    style: String,

    #[arg(short, long, default_value_t = false)]
    server: bool,

    #[arg(short, long, default_value_t = 3000)]
    port: u16,
}

const INDEX_HTML: &str = include_str!("index.html");

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn handle_render(mut multipart: Multipart) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut person_bytes = None;
    let mut necklace_bytes = None;
    let mut style = "choker".to_string();
    let mut scale = 1.0;
    let mut y_offset = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| (StatusCode::BAD_REQUEST, format!("Multipart error: {}", e)))? {
        let name = field.name().unwrap_or("").to_string();
        if name == "person" {
            person_bytes = Some(field.bytes().await.map_err(|e| (StatusCode::BAD_REQUEST, format!("Error reading person field: {}", e)))?);
        } else if name == "necklace" {
            necklace_bytes = Some(field.bytes().await.map_err(|e| (StatusCode::BAD_REQUEST, format!("Error reading necklace field: {}", e)))?);
        } else if name == "style" {
            let text = field.text().await.unwrap_or_default();
            if !text.is_empty() {
                style = text;
            }
        } else if name == "scale" {
            let text = field.text().await.unwrap_or_default();
            if let Ok(s) = text.parse::<f64>() {
                scale = s;
            }
        } else if name == "y_offset" {
            let text = field.text().await.unwrap_or_default();
            if let Ok(y) = text.parse::<f64>() {
                y_offset = Some(y);
            }
        }
    }

    let person_bytes = person_bytes.ok_or_else(|| (StatusCode::BAD_REQUEST, "Missing person image".to_string()))?;
    let necklace_bytes = necklace_bytes.ok_or_else(|| (StatusCode::BAD_REQUEST, "Missing necklace image".to_string()))?;

    // Load images
    let person_img = image::load_from_memory(&person_bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid person image format: {}", e)))?
        .to_rgb8();

    let necklace_img = image::load_from_memory(&necklace_bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid necklace image format: {}", e)))?
        .to_rgba8();

    // Run pipeline in blocking task to prevent locking the async runtime
    let result = tokio::task::spawn_blocking(move || {
        pose::process_pipeline(&person_img, &necklace_img, scale, &style, y_offset)
    })
    .await
    .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Worker thread panicked".to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Pipeline logic error: {:?}", e)))?;

    match result {
        Some(rendered) => {
            // Encode back to PNG in memory
            let mut buffer = std::io::Cursor::new(Vec::new());
            let dyn_img = image::DynamicImage::ImageRgb8(rendered);
            dyn_img.write_to(&mut buffer, image::ImageOutputFormat::Png)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to encode output image: {}", e)))?;
            
            let response_bytes = buffer.into_inner();
            Ok(Response::builder()
                .header(header::CONTENT_TYPE, "image/png")
                .body(axum::body::Body::from(response_bytes))
                .unwrap())
        }
        None => Err((StatusCode::BAD_REQUEST, "No pose detected in the person image".to_string())),
    }
}

async fn run_web_server(port: u16) {
    let app = Router::new()
        .route("/", get(index))
        .route("/render", post(handle_render))
        .layer(axum::extract::DefaultBodyLimit::max(50 * 1024 * 1024))
        .layer(tower_http::cors::CorsLayer::permissive());

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    println!("Starting local Try-On server at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[tokio::main]
async fn main() {
    #[cfg(target_os = "windows")]
    {
        std::env::set_var("PYTHONHOME", "C:\\Users\\Thethinker\\AppData\\Local\\Programs\\Python\\Python311");
        std::env::set_var("PATH", "C:\\Users\\Thethinker\\AppData\\Local\\Programs\\Python\\Python311;C:\\Users\\Thethinker\\AppData\\Local\\Programs\\Python\\Python311\\Scripts");
    }

    let args = Args::parse();

    if args.server {
        let port = std::env::var("PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(args.port);
        run_web_server(port).await;
        return;
    }

    // CLI Mode
    let person_path = match args.person {
        Some(p) => p,
        None => {
            eprintln!("Error: missing required argument --person (or run with --server)");
            std::process::exit(1);
        }
    };
    let necklace_path = match args.necklace {
        Some(n) => n,
        None => {
            eprintln!("Error: missing required argument --necklace (or run with --server)");
            std::process::exit(1);
        }
    };
    let output_path = match args.output {
        Some(o) => o,
        None => {
            eprintln!("Error: missing required argument --output (or run with --server)");
            std::process::exit(1);
        }
    };

    println!("Loading person image: {}", person_path);
    let person_img_dynamic = image::open(&person_path).expect("Failed to open person image");
    let person_rgba = person_img_dynamic.to_rgba8();
    let person_rgb = image::DynamicImage::ImageRgba8(person_rgba.clone()).into_rgb8();

    println!("Loading necklace image: {}", necklace_path);
    let necklace_img = image::open(&necklace_path).expect("Failed to open necklace image");
    let necklace_rgba = necklace_img.to_rgba8();

    println!("Extracting pose landmarks and rendering necklace via Python/OpenCV pipeline...");
    let pipeline_result = pose::process_pipeline(&person_rgb, &necklace_rgba, 1.0, &args.style, None);
    
    match pipeline_result {
        Ok(Some(rendered_img)) => {
            println!("Pipeline successful! Saving output to: {}", output_path);
            let final_rgba = image::DynamicImage::ImageRgb8(rendered_img).into_rgba8();
            final_rgba.save(&output_path).expect("Failed to save output image");
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
