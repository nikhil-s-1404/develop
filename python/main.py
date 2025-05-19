from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from model import SegmentationModel
import io
import cv2
import base64
from io import BytesIO
import json
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Allow CORS from Angular frontend
origins = ["http://localhost:4200"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load segmentation model
def load_segmentation_model():
    model = SegmentationModel().model
    model.load_weights('cancer_weights.h5')
    return model

# Initialize LLM pipeline
def initialize_llm_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

segmentation_model = load_segmentation_model()
generator = initialize_llm_pipeline()
set_seed(42)

# Async node: Read and preprocess image
async def read_image(file: UploadFile):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).resize((256, 256))
    return image

# Node: Run segmentation model
def run_segmentation(image):
    image_array = np.array(image)
    image_tensor = tf.expand_dims(image_array, axis=0)
    yhat = segmentation_model.predict(image_tensor)
    return np.squeeze(np.where(yhat > 0.3, 1.0, 0.0))

# Node: Annotate image and extract regions
def draw_segmentation_results(image, mask):
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    detected_regions = []

    if len(mask.shape) == 3:
        for i in range(mask.shape[2]):
            layer = mask[:, :, i]
            contours, _ = cv2.findContours(layer.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                    detected_regions.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})

    return annotated_image, detected_regions

# Node: Construct summary
def construct_summary(detected_regions):
    return f"Detected {len(detected_regions)} suspicious regions. Regions: " + \
           ", ".join([f"({r['x']}, {r['y']}, {r['width']}x{r['height']})" for r in detected_regions])

# Node: Summarize results and call LLM
def call_llm(summary_text: str, patient_info: dict = None) -> dict:
    prompt = f"Findings: {summary_text}."
    if patient_info:
        prompt += f" Patient: {json.dumps(patient_info)}"

    try:
        result = generator(
            prompt,
            max_new_tokens=150,
            pad_token_id=50256
        )

        if isinstance(result, list) and 'generated_text' in result[0]:
            output = result[0]['generated_text']
        elif isinstance(result, dict) and 'generated_text' in result:
            output = result['generated_text']
        else:
            raise ValueError("Unexpected LLM response format")

        # Simple version for layperson
        patient_friendly = f"Hello {patient_info.get('name', 'Patient')}, we reviewed your image and found {len(summary_text.split('Regions: ')[-1].split(','))} area(s) that need attention. Please consult your doctor for more details."

        return {
            "radiologist_summary": output.strip(),
            "patient_summary": patient_friendly
        }
    except Exception as e:
        print(f"HuggingFace LLM Error: {e}")
        return {"radiologist_summary": "Unable to generate report.", "patient_summary": "N/A"}

# Node: Convert image to base64
def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# LangGraph-compatible endpoint using modular nodes
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...), patient_name: str = "", age: int = 0, gender: str = ""):
    image = await read_image(file)
    segmentation_mask = run_segmentation(image)
    annotated_image, detected_regions = draw_segmentation_results(image, segmentation_mask)

    summary_text = construct_summary(detected_regions)

    patient_info = {"name": patient_name, "age": age, "gender": gender}
    ai_report = call_llm(summary_text, patient_info)

    return json.dumps({
        "annotated_image": convert_image_to_base64(annotated_image),
        "detected_regions": detected_regions,
        "summary_text": summary_text,
        "radiologist_summary": ai_report['radiologist_summary'],
        "patient_summary": ai_report['patient_summary']
    })
