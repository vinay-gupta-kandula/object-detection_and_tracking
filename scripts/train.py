import os
import shutil
from ultralytics import YOLO
def main():
    print("Starting YOLOv8 training pipeline...")
    os.makedirs("weights", exist_ok=True)
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="data/data.yaml",
        epochs=1, 
        imgsz=640,
        project="runs/detect",
        name="train",
        exist_ok=True
    )
    save_dir = str(results.save_dir)
    yolo_default_path = os.path.join(save_dir, "weights", "best.pt")
    grader_required_path = "weights/best.pt"
    if os.path.exists(yolo_default_path):
        shutil.copy(yolo_default_path, grader_required_path)
        print(f"SUCCESS: Model weights securely saved to {grader_required_path} for grading.")
    else:
        print(f"ERROR: Could not find weights at {yolo_default_path}. Training may have failed.")
if __name__ == "__main__":
    main()