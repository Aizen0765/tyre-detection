"""
Exports PyTorch YOLOv8 and CLIP models to ONNX runtime format
for low-latency production deployment.
"""

from ultralytics import YOLO
import torch
import os

def export_yolo_to_onnx(model_path: str = "yolov8n.pt"):
    model = YOLO(model_path)
    # Export with dynamic axes for batch size & image resolution
    exported_path = model.export(format="onnx", dynamic=True, simplify=True)
    print(f"Successfully exported YOLOv8 model to: {exported_path}")

def export_clip_to_onnx():
    import torch.onnx
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    dummy_input = torch.randn(1, 3, 224, 224)

    os.makedirs("onnx_models", exist_ok=True)
    onnx_path = "onnx_models/clip_vision_encoder.onnx"

    torch.onnx.export(
        model.vision_model,
        dummy_input,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch_size"}},
        opset_version=14
    )
    print(f"Successfully exported CLIP Vision Encoder to: {onnx_path}")

if __name__ == "__main__":
    export_yolo_to_onnx()
    export_clip_to_onnx()