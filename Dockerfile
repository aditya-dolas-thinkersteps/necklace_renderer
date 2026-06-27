# Stage 1: Build the Rust binary
FROM python:3.11-bookworm AS builder

# Install Rust nightly
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
ENV PATH="/root/.cargo/bin:${PATH}"

# Install pkg-config for compilation
RUN apt-get update && apt-get install -y \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy cargo files
COPY Cargo.toml Cargo.lock ./
# Pre-build dependencies to cache layers
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release

# Copy actual source code
COPY src ./src
# Touch main.rs to ensure rebuild with actual source code
RUN touch src/main.rs && cargo build --release

# Stage 2: Final runner image
FROM python:3.11-slim-bookworm

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python packages needed by pipeline
RUN pip3 install --no-cache-dir \
    mediapipe==0.10.14 \
    opencv-python-headless \
    numpy

WORKDIR /app

# Copy compiled binary from builder
COPY --from=builder /app/target/release/necklace_renderer /app/necklace_renderer

# Expose port
EXPOSE 3000

# Run in server mode by default
ENTRYPOINT ["/app/necklace_renderer", "--server", "--port", "3000"]
