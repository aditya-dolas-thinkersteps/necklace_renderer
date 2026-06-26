# ==========================================
# Stage 1: Build the Rust Application
# ==========================================
FROM rust:1.75-slim-bookworm AS builder

# Install Python development files and compiler tools required by PyO3/numpy-rust
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the Cargo manifest and locks first to cache dependencies
COPY Cargo.toml Cargo.lock ./

# Create dummy source folder to build and cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Copy actual source files
COPY src ./src

# Rebuild the application in release mode (updates main.rs and pose.rs)
RUN touch src/main.rs && cargo build --release

# ==========================================
# Stage 2: Runtime Image with Python & MediaPipe
# ==========================================
FROM python:3.11-slim-bookworm

# Install graphics and system libraries required by OpenCV (cv2) and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir \
    opencv-python \
    mediapipe \
    numpy

# Copy compiled Rust executable from the builder stage
COPY --from=builder /app/target/release/necklace_renderer /usr/local/bin/necklace_renderer

# Set run directory
WORKDIR /data

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/necklace_renderer"]
