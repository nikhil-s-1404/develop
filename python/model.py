from PIL import Image
import random

def detect_tumor(image_path):
    # Placeholder logic
    # In real case, run a trained PyTorch model on the image
    image = Image.open(image_path)
    width, height = image.size

    # Simulated bounding box
    box = {
        "x": random.randint(0, width // 2),
        "y": random.randint(0, height // 2),
        "width": width // 4,
        "height": height // 4,
        "confidence": round(random.uniform(0.7, 0.99), 2)
    }

    return {
    "filename": image_path.split("/")[-1],
    "annotations": [
        {
            "x": 100,
            "y": 150,
            "width": 80,
            "height": 80,
            "label": "Tumor",
            "confidence": 0.95
        }
    ]
}

