# Real-Time Object Detection & Tracking Pipeline

This repository contains an end-to-end, production-ready MLOps pipeline for real-time object detection and multi-object tracking.

The system trains a custom YOLOv8 model, exports it to ONNX for optimized inference, and serves it using NVIDIA's Triton Inference Server. A FastAPI backend handles client requests (images, asynchronous video processing, and real-time WebSockets), passing the data to Triton via gRPC and applying DeepSORT logic for object tracking.

## Tech Stack

* **Model:** YOLOv8 (Detection), DeepSORT (Tracking)
* **Serving:** NVIDIA Triton Inference Server (gRPC)
* **Backend:** FastAPI, Python 3.10, OpenCV
* **Format & Optimization:** ONNX Runtime
* **Infrastructure:** Docker & Docker Compose

## Repository Structure

```text
├── api/                     # FastAPI application and Dockerfile
│   └── main.py              # API endpoints, post-processing, and Triton gRPC client
├── data/                    # Dataset configs (e.g., VisDrone/COCO)
├── models/                  # Triton model repository
│   └── yolov8_onnx/
│       ├── 1/               # Version directory (holds model.onnx)
│       └── config.pbtxt     # Triton configuration (shapes, batching)
├── scripts/                 
│   ├── train.py             # YOLOv8 training script
│   ├── export.py            # PyTorch to ONNX conversion
│   └── evaluate.py          # Generates metrics (mAP, MOTA, FPS, latency)
├── output/                  # Evaluation JSONs and annotated video results
├── weights/                 # PyTorch weight backups (.pt files)
├── docker-compose.yml       # Orchestrates Triton and FastAPI
└── .env.example             # Environment variable templates

```

## ⚠️ Note on Testing vs. Production

By default, this repository is configured for **quick testing and validation**:

* `scripts/train.py` is set to run for **1 epoch**.
* The API confidence thresholds in `api/main.py` are set extremely low (`0.01`) so the 1-epoch model can actually return bounding boxes.

**For real-world deployment:** 1. Increase the training epochs in `scripts/train.py` (e.g., `epochs=100`).
2. Restore the confidence and NMS thresholds in `api/main.py` back to standard production levels (e.g., `0.4`).

## Setup & Installation

### 1. Environment Variables

Create a `.env` file in the root directory using the provided template:

```bash
cp .env.example .env

```

### 2. Train and Export the Model

Before spinning up the server, you need to generate the model weights and convert them to ONNX format so Triton can read them.

```bash
# 1. Train the YOLOv8 model (outputs to weights/best.pt)
python scripts/train.py

# 2. Export the trained model to the Triton repository
python scripts/export.py

```

### 3. Spin Up the Infrastructure

Start the Triton Inference Server and the FastAPI application using Docker Compose. Make sure you have the NVIDIA Container Toolkit installed if you are running this on a GPU.

```bash
docker-compose up --build

```

Both services will run health checks. Once healthy, your API will be live and accessible at: **`http://localhost:8000`**

---

## API Endpoints & Interactive Docs

FastAPI automatically generates a fully interactive UI to test the endpoints. Once your Docker containers are running, you can click the link below to upload images, send videos, and test the API directly from your browser:

👉 **[Interactive Swagger UI Documentation](http://localhost:8000/docs)**

Alternatively, you can test the API locally using `curl`, Postman, or Python's `requests` library:

### `GET /health`

Verifies that the FastAPI service is running and ready to accept traffic.

* **Test it:**
```bash
curl http://localhost:8000/health

```



### `POST /detect/image`

Accepts a multipart image upload, formats it, sends it to Triton for gRPC inference, applies Non-Maximum Suppression (NMS), and returns a JSON array of bounding boxes and class labels.

* **Test it:**
```bash
curl -X POST -F "file=@sample_image.jpg" http://localhost:8000/detect/image

```



### `POST /track/video`

Accepts a video upload and processes the frames asynchronously using FastAPI's `BackgroundTasks`. The annotated video is saved directly to the `/output` directory.

* **Test it:**
```bash
curl -X POST -F "file=@sample_video.mp4" http://localhost:8000/track/video

```



### `WS /track/ws`

A WebSocket endpoint built for real-time video streaming. It accepts binary frame data, updates the DeepSORT tracker, and streams the tracked coordinates and IDs back to the client.

* **Endpoint URL:** `ws://localhost:8000/track/ws`

---

## Evaluation

To generate performance and accuracy metrics (mAP, precision, recall, MOTA, FPS, and latency), run the evaluation script while the Docker containers are active:

```bash
python scripts/evaluate.py

```

This will populate the `output/` folder with `detection_metrics.json`, `tracking_metrics.json`, and `performance.json`.

## Configuration Details: Triton Inference Server

The `config.pbtxt` is configured to handle dynamic batching (up to 16). Note that the ONNX model's output axis is set to dynamic (`[14, -1]`) to prevent shape mismatch errors when Triton automatically prepends the batch dimension.

```

```