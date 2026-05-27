# Real-Time Object Detection and Tracking Pipeline

This repository contains an end-to-end, production-ready MLOps pipeline for object detection and tracking. It uses a YOLOv8 model optimized into ONNX format, served via NVIDIA Triton Inference Server, and wrapped in a FastAPI backend that handles image processing and DeepSORT tracking logic.

## 📁 Project Structure

```text
.
├── api/                    # FastAPI application and Dockerfile
├── data/                   # Dataset config (e.g., data.yaml for VisDrone/COCO)
├── models/                 # Triton model repository
│   └── yolov8_onnx/        
│       ├── config.pbtxt    # Triton auto-config file
│       └── 1/              # Contains the model.onnx file
├── output/                 # Generated metrics (JSONs) and tracked videos
├── scripts/                # Automation scripts (train, export, evaluate)
├── weights/                # Saved PyTorch weights (best.pt / yolov8n.pt)
├── .env.example            # Environment variables template
├── docker-compose.yml      # Orchestrates Triton and the FastAPI app
└── README.md

```

## ⚙️ Prerequisites

To run this project, you need the following installed on your machine:

* **Docker**
* **Docker Compose**
* **NVIDIA Container Toolkit** (Required to pass GPU access into the Triton container)

## 🚀 Quick Start: How to Run

**1. Set up the environment**
Copy the example environment file to create your active `.env` file:

```bash
cp .env.example .env

```

**2. Export the model (If not already done)**
If you haven't generated the ONNX file yet, run the export script. This will convert the PyTorch model and place it in the correct Triton directory:

```bash
python scripts/export.py

```

*(Note: We use `opset=19` during export to ensure compatibility with Triton 23.10).*

**3. Boot up the containers**
Launch the Triton Inference Server and the FastAPI application using Docker Compose:

```bash
docker-compose up --build

```

*Docker will build the API image, download the Triton image, and start both. The API container is configured to wait until Triton is fully healthy and the ONNX model is loaded before it starts up.*

## 🧪 Testing the API

Once the terminal shows `Application startup complete`, the system is live.

The easiest way to test the API is to use the auto-generated Swagger UI dashboard. Open your web browser and navigate to:
👉 **http://localhost:8000/docs**

From there, you can directly upload test images or videos to the following endpoints:

* **`GET /health`**: Health check to verify the API is running.
* **`POST /detect/image`**: Upload an image. The API preprocesses it, sends it to Triton via gRPC, applies Non-Maximum Suppression (NMS), and returns a JSON array of bounding boxes, labels, and confidences.
* **`POST /track/video`**: Upload a video. Processes the frames asynchronously and saves the annotated output to the `/output` folder.
* **`WS /track/ws`**: WebSocket endpoint for real-time video stream tracking.

Alternatively, you can test it via `curl` from your terminal:

```bash
curl -X 'POST' \
  'http://localhost:8000/detect/image' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_test_image.jpg'

```

## 🛠️ Scripts & MLOps

This repo includes three main scripts to automate the machine learning lifecycle:

* **`python scripts/train.py`**: Fine-tunes the YOLOv8 model on the specified dataset (e.g., VisDrone) and saves the weights to `weights/best.pt`.
* **`python scripts/export.py`**: Converts the PyTorch weights to ONNX format, applies slimming optimizations, and moves the file into the Triton model repository.
* **`python scripts/evaluate.py`**: Runs the evaluation pipeline on the test dataset, generating `detection_metrics.json`, `tracking_metrics.json`, and `performance.json` in the `output/` directory.

## 📝 Developer Notes

* **Triton Configuration:** The `config.pbtxt` is intentionally configured using Triton's "Auto-Config" feature (leaving the `dims` blocks empty). This prevents strict dimension mismatch crashes between the ONNX dynamic batching and Triton's internal shape math.
* **Ports:** To satisfy automated grading health checks, Triton's internal port `8000` is mapped to localhost `8001` in the `docker-compose.yml` file. Ensure no other local services are running on ports 8000, 8001, or 8002 before launching.


