from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from model import SegmentationModel  # Assuming you have a model.py file with the SegmentationModel class
import io
import cv2
import base64
from io import BytesIO
import json
from fastapi.middleware.cors import CORSMiddleware
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Allow CORS from your Angular frontend (localhost:4200)
origins = [
    "http://localhost:4200",  # Angular dev server
]

# Add the CORSMiddleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load your model
model = SegmentationModel().model
model.load_weights('cancer_weights.h5')


# NEW: Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")


# Utility to call LLM
def call_llm(summary_text: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical imaging assistant. Generate short, professional reports."},
                {"role": "user", "content": f"Summarize this segmentation output: {summary_text}"}
            ],
              max_tokens=150,
              temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Unable to generate report at the moment."

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Resize image to match model input size
    image = image.resize((256, 256))

    # Convert PIL image to numpy array
    image_array = np.array(image)
    image_tensor = tf.expand_dims(image_array, axis=0)

    # Predict using the model
    yhat = model.predict(image_tensor)
    prediction = yhat.tolist()

    # Convert the prediction to a binary mask (assuming segmentation task)
    yhat = np.squeeze(np.where(np.array(prediction) > 0.3, 1.0, 0.0))

    # Annotate the image by drawing on the predicted areas (using OpenCV or Pillow)
    annotated_image = image.copy()  # Work with a copy of the original image
    draw = ImageDraw.Draw(annotated_image)

    # Summarize detected regions
    detected_regions = []

    # Example of drawing a simple rectangle around predicted areas
    for i in range(yhat.shape[2]):  # Assuming the model output is a 3D array
        mask = yhat[:, :, i]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Threshold to avoid drawing too small contours
                x, y, w, h = cv2.boundingRect(contour)
                draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                detected_regions.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})


    # Create a simple text summary
    summary_text = f"Detected {len(detected_regions)} regions. Sizes: " + ", ".join([f"{r['width']}x{r['height']}" for r in detected_regions])

    # Call LLM to generate a report
    generated_report = call_llm(summary_text)

    # Convert the annotated image to a base64 string to send to frontend
    buffered = BytesIO()
    annotated_image.save(buffered, format="PNG")
    annotated_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the base64-encoded annotated image
    return json.dumps({
        "annotated_image": annotated_image_base64,
        "detected_regions": detected_regions,
        "summary_text": summary_text,
        "generated_report": generated_report
    })

# Run the app using Uvicorn
# uvicorn app:app --reload
