from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import torch

app = FastAPI()

is_cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {is_cuda_available}")

# Load the YOLO model (ensure it's the correct classification or detection model)
model = YOLO('https://drive.google.com/uc?export=download&id=10MCPKxEnlsS2ZWLV1a8FF-TMs8OHm-Xt')  # or to your local model
# Optionally, send the model to the GPU if CUDA is available
if is_cuda_available:
    model = model.to('cuda')

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="File format not supported. Please upload a JPG or PNG image.")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')

        # Run the YOLO model on the image
        results = model.predict(image)
        print(results[0].boxes)

        names = results[0].names
        # Check if it's a classification result
        if results[0].probs is not None:
            # Classification model response
            top_class = results[0].probs.top1.item()  # Convert tensor to a scalar
            confidence = results[0].probs.top1conf.item() * 100  # Convert tensor to a scalar

            response = {
                "tire_condition": names[top_class],  # Predicted class name
                "confidence": f"{confidence:.2f}%",
                "speed": f"{results[0].speed['inference'] / 1000:.2f} seconds",
            }

        # Handle object detection results (if using detection model)
        elif results[0].boxes is not None:
            detections = []
            for box in results[0].boxes:
                detection = {
                    "class": names[int(box.cls.item())],
                    "confidence": f"{float(box.conf.item()):.2f}%",
                    "x_top_left": f"{float(box.xywh[0][0].item() - box.xywh[0][2].item() / 2):.2f}",
                    "y_top_left": f"{float(box.xywh[0][1].item() - box.xywh[0][3].item() / 2):.2f}",
                    "width": f"{float(box.xywh[0][2].item()):.2f}",
                    "height": f"{float(box.xywh[0][3].item()):.2f}",
                }
                detections.append(detection)
            response = {
                "detections": detections,
                "speed": f"{results[0].speed['inference'] / 1000:.2f} seconds",
            }

        # No results found
        else:
            response = {"message": "No results found, try uploading another image."}

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {e}")


@app.get("/new-endpoint/")
async def new_endpoint():
    return {"message": "This is a new endpoint."}
