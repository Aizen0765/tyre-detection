"""
Handles model inference, bounding box detection, and crop extraction.
"""

from ultralytics import YOLO
import cv2
import os

def run_tyre_detection(image_path: str, model_path: str = "yolov8n.pt", conf_thresh: float = 0.5):
    # Load fine-tuned YOLOv8 model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(source=image_path, conf=conf_thresh)
    
    # Process detections
    for result in results:
        boxes = result.boxes
        print(f"Detected {len(boxes)} tyre(s) in {image_path}")
        
        # Save output visualization
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        annotated_frame = result.plot()
        cv2.imwrite(os.path.join(output_dir, "detected_tyres.jpg"), annotated_frame)

if __name__ == "__main__":
    # Example usage
    sample_image = "sample_vehicle.jpg"
    if os.path.exists(sample_image):
        run_tyre_detection(sample_image)
    else:
        print(f"Please provide a valid image path. '{sample_image}' not found.")