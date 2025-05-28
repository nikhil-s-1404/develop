from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,InteractiveBrowserCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
import requests
import json
import base64
from PIL import Image
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


token = ''
# Replace with your actual endpoint
scoring_uri = ""

frontal = "./uploads/cxr_frontal.jpg"
lateral = "./uploads/cxr_lateral.jpg"

indication = ""
technique = ""
comparison = "None"
subscription_id = ""
resource_group = ""
workspace = ""

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




def read_image(image_file):
    return base64.encodebytes(image_file.read()).decode("utf-8")


@app.post("/predict/")
async def predict(
    frontal_image: UploadFile = File(None),
    lateral_image: UploadFile = File(None),
    indication: str = Form(""),
    technique: str = Form(""),
    comparison: str = Form("")
):
    try:
        input_data = {}

        if frontal_image:
            input_data["frontal_image"] = base64.encodebytes(await frontal_image.read()).decode("utf-8")
        if lateral_image:
            input_data["lateral_image"] = base64.encodebytes(await lateral_image.read()).decode("utf-8")
        if indication:
            input_data["indication"] = indication
        if technique:
            input_data["technique"] = technique
        if comparison:
            input_data["comparison"] = comparison

        if not input_data:
            return JSONResponse(content={"error": "No valid inputs provided"}, status_code=400)

        columns = list(input_data.keys())
        values = list(input_data.values())

        data = {
            "input_data": {
                "columns": columns,
                "index": [0],
                "data": [values]
            },
            "params": {}
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        response = requests.post(scoring_uri, json=data, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            return JSONResponse(content={"error": response.text}, status_code=response.status_code)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# score_image(frontal, lateral, indication, technique, comparison)

