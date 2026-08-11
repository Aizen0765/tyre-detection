"""
Video Analytics Pipeline:
Performs object detection and tracks unique vehicle instances across frames using ByteTrack.
"""

import cv2
import supervision as sv
from ultralytics import YOLO

def process_video_tracking(source_video_path: str, output_video_path: str):
    # Initialize detector and tracker
    model = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()
    
    # Initialize visualizers
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def frame_callback(frame: cv2.Mat, frame_idx: int) -> cv2.Mat:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Track objects across frames
        detections = tracker.update_with_detections(detections)

        # Generate unique tracking labels
        labels = [
            f"#{tracker_id} {model.names[class_id]}"
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        return label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Process input video stream
    sv.process_video(
        source_path=source_video_path,
        target_path=output_video_path,
        callback=frame_callback
    )
    print(f"Video processing complete. Output saved to: {output_video_path}")

if __name__ == "__main__":
    process_video_tracking("sample_traffic.mp4", "output_tracked.mp4")