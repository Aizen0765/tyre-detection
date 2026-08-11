"""
CLIP (ViT-B/32) Zero-Shot Vehicle Attribute Classifier
Classifies 8 target vehicle attributes from input images.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 8 Target Vehicle Attributes
ATTRIBUTE_LABELS = [
    "sedan car",
    "SUV car",
    "hatchback car",
    "pickup truck",
    "commercial van",
    "clean vehicle exterior",
    "dirty vehicle exterior",
    "damaged tyre"
]

def classify_vehicle_attributes(image_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load OpenAI CLIP Model & Processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path).convert("RGB")

    # Format zero-shot prompts
    text_prompts = [f"a photo of a {label}" for label in ATTRIBUTE_LABELS]

    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    print("\n--- Attribute Classification Results ---")
    for label, prob in zip(ATTRIBUTE_LABELS, probs):
        print(f"{label:25s}: {prob * 100:.2f}%")

if __name__ == "__main__":
    sample_image = "sample_vehicle.jpg"
    if os.path.exists(sample_image):
        classify_vehicle_attributes(sample_image)
    else:
        print(f"Please provide a valid image path. '{sample_image}' not found.")