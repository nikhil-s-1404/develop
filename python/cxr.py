from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential,InteractiveBrowserCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
import requests
import json
import base64
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image

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



def read_image(image_path):
    """Reads an image from a file path and returns the image as a byte array."""
    with open(image_path, "rb") as f:
        return f.read()


def score_image(frontal_path, lateral_path, indication, technique, comparison):
    """Scores frontal and lateral images using the deployed model."""
    print('score')  
    input_data = {
        "frontal_image": base64.encodebytes(read_image(frontal_path)).decode("utf-8"),
        "lateral_image": base64.encodebytes(read_image(lateral_path)).decode("utf-8"),
        "indication": indication,
        "technique": technique,
        "comparison": comparison,
    }

    data = {
        "input_data": {
           'columns': ['frontal_image', 'lateral_image', 'indication', 'technique', 'comparison'],
           'index': [0],
            "data": [
                list(input_data.values()),
            ],
        },
        "params": {},
    }

    # Create request json
    request_file_name = "sample_request_data.json"
    with open(request_file_name, "w") as request_file:
        json.dump(data, request_file)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    print('Input')
    response = requests.post(scoring_uri, json=data, headers=headers)

    # Handle response
    if response.status_code == 200:
      print("Prediction:", response.json())
    else:
      print("Error:", response.status_code, response.text)





score_image(frontal, lateral, indication, technique, comparison)

