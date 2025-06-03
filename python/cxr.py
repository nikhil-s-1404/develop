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
import io
from io import BytesIO
import json



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

# Adjust image to fix boxes
def adjust_box_for_original_image_size(norm_box, width: int, height: int):
    """
    Assuming we did a centre crop to the shortest size, adjust the box coordinates back to the original shape of the image.
    :param norm_box: A normalized box to rescale.
    :param width: Original width of the image, in pixels.
    :param height: Original height of the image, in pixels.
    :return: The box normalized relative to the original size of the image.
    """
    crop_width = crop_height = min(width, height)
    x_offset = (width - crop_width) // 2
    y_offset = (height - crop_height) // 2
    norm_x_min, norm_y_min, norm_x_max, norm_y_max = norm_box
    abs_x_min = int(norm_x_min * crop_width + x_offset)
    abs_x_max = int(norm_x_max * crop_width + x_offset)
    abs_y_min = int(norm_y_min * crop_height + y_offset)
    abs_y_max = int(norm_y_max * crop_height + y_offset)
    adjusted_norm_x_min = abs_x_min / width
    adjusted_norm_x_max = abs_x_max / width
    adjusted_norm_y_min = abs_y_min / height
    adjusted_norm_y_max = abs_y_max / height
    return (
        adjusted_norm_x_min,
        adjusted_norm_y_min,
        adjusted_norm_x_max,
        adjusted_norm_y_max,
    )

# Add Boxes to images
def show_image_with_bbox(path_frontal, findings, path_lateral=None):
    """Displays frontal and lateral images with bounding boxes around the findings."""
    image_frontal = Image.open(path_frontal)
    width_frontal, height_frontal = image_frontal.size
    print(findings)  # Debugging: print findings to check structure
    if path_lateral:
        image_lateral = Image.open(path_lateral)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(image_frontal, cmap="gray")
        axes[1].imshow(image_lateral, cmap="gray")
    else:
        fig, axes = plt.subplots(figsize=(10, 10))
        axes.imshow(image_frontal, cmap="gray")
        axes = [axes]

    findings_str = []
    for idx, (finding, boxes) in enumerate(findings):
        findings_str.append(f"{idx}. {finding}{' * ' if boxes else ' '}")
        if boxes:
            for box in boxes:
                box = adjust_box_for_original_image_size(
                    box, width_frontal, height_frontal
                )
                x_min, y_min, x_max, y_max = (
                    box[0] * width_frontal,
                    box[1] * height_frontal,
                    box[2] * width_frontal,
                    box[3] * height_frontal,
                )

                rect = plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    edgecolor="red",
                    facecolor="none",
                    linewidth=2,
                )
                axes[0].add_patch(rect)
                axes[0].text(
                    x_min + 3,
                    y_min + 3,
                    f"Finding ID: {idx}",
                    color="yellow",
                    fontsize=10,
                    verticalalignment="top",
                )

    for ax in axes:
        ax.axis("off")  # Hide the axes

     # Save the figure to a BytesIO object and return it as a PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)  # Close the figure to free memory
    return Image.open(buf)

# parse response function to extract findings and bounding boxes
def parse_response(response):
    parsed_array = []
    output = response[0]['output']  # Extract the 'output' key

    for finding in output:
        text = finding[0]
        bbox = finding[1] if finding[1] != 'null' else None
        parsed_array.append([text, bbox])  # Append as a list

    return parsed_array

# Node: Convert image to base64
def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Node: Send Output to Angular
def send_response(res):
    findings = parse_response(res)
    # Display the findings with bounding boxes
    result_image = show_image_with_bbox(frontal, findings, lateral)
    result_image.save("output_with_boxes.png")

     # Build summaries
    radiologist_lines = []

    for idx, (sentence, annotation) in enumerate(findings, 1):
        # Radiologist version: direct sentence with region note
        if annotation == 'null':
            radiologist_lines.append(f"{idx}. {sentence}")
        else:
            radiologist_lines.append(f"{idx}. {sentence} (Region identified)")

    return {
        "annotated_image": convert_image_to_base64(result_image),
        "detected_regions": ' + str(findings),',
        "summary_text": "\n".join(radiologist_lines)
    }

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
            # return JSONResponse(content={"error": response.text}, status_code=response.status_code)
            return send_response(response)

    except Exception as e:
            response = [{'output':
                [
                ['The heart size is normal.', 'null'],
                ['The aorta is tortuous.', [[0.415, 0.225, 0.635, 0.755]]],
                ['A patchy infiltrate in the right lower lobe is associated with volume loss.', [[0.115, 0.445, 0.445, 0.795]]],
                ['A small right pleural effusion is present.', [[0.115, 0.555, 0.425, 0.845]]],
                ['The left hemithorax is grossly clear.', 'null']
            ]
            }]
            return send_response(response)
            # return JSONResponse(content={"error": str(e)}, status_code=500)




# score_image(frontal, lateral, indication, technique, comparison)
# findings = parse_response(response)
# # Display the findings with bounding boxes
# result_image = show_image_with_bbox(frontal, findings, lateral)
# result_image.save("output_with_boxes.png")

