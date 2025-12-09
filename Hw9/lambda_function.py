import json
import base64
import io
from PIL import Image
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms

# Preprocessing from HW8
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ONNX model (included in image)
session = ort.InferenceSession("hair_classifier_empty.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def predict_pil(img: Image.Image):
    x = transform(img)
    x = x.unsqueeze(0).numpy()

    logits = session.run([output_name], {input_name: x})[0][0]
    return float(logits)


def lambda_handler(event, context):
    b64 = event["image"]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    score = predict_pil(img)

    return {"score": score}
