import io
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image
from torchvision import transforms as T
import cv2

from .model import get_model


app = FastAPI()

# ============== DEVICE ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== LOAD MODEL ==================
root_dir = Path(__file__).resolve().parents[1]
model_path = root_dir / "models" / "isic_fast_model.pt"

model = get_model(pretrained=False).to(device)

if model_path.exists():
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(f"✔ Loaded trained model from {model_path}")
else:
    print(f"⚠ WARNING: Model file {model_path} not found; predictions will be random!")

model.eval()

# ============== IMAGE TRANSFORM =============
tfm = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3)
])


# ================= GRAD-CAM FUNCTION ===================
def generate_gradcam(pil_img: Image.Image):
    """Generate Grad-CAM overlay + malignant probability."""
    input_tensor = tfm(pil_img).unsqueeze(0).to(device)

    # Last feature layer of MobileNetV3 Small
    target_layer = model.features[-1]

    gradients, activations = [], []

    # Hooks
    def save_activation(module, inp, out):
        activations.append(out)

    def save_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer.register_forward_hook(save_activation)
    target_layer.register_full_backward_hook(save_gradient)

    # Forward
    logits = model(input_tensor)
    score = logits.squeeze()

    # Backward
    model.zero_grad()
    score.backward(retain_graph=True)

    # Compute CAM
    grad = gradients[-1].detach().cpu().numpy()
    act = activations[-1].detach().cpu().numpy()

    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, pil_img.size)
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    cam_uint8 = np.uint8(cam * 255)

    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.6 * np.array(pil_img) + 0.4 * heatmap).astype(np.uint8)

    prob = torch.sigmoid(score).item()
    return overlay, prob


# ================== WEB UI ==========================
@app.get("/", response_class=HTMLResponse)
async def index():
    """Simple HTML upload interface."""
    html = """\
    <!DOCTYPE html>
    <html>
    <head>
    <title>Skin Lesion Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; text-align:center; margin:40px; }
        img { max-width:320px; border:2px solid #333; margin-top:20px; }
        button { margin-top:15px; padding:10px 20px; }
    </style>
    </head>
    <body>
        <h2>Melanoma Detection with Grad-CAM</h2>
        <p>Upload a dermoscopic skin lesion image.</p>
        <input type="file" id="imgInput" accept="image/*"/>
        <br/>
        <button onclick="upload()">Analyze</button>
        <p id="prediction"></p>
        <img id="resultImg"/>

        <script>
        function upload() {
            let file = document.getElementById('imgInput').files[0];
            if (!file) { alert("Please choose an image first!"); return; }
            let form = new FormData();
            form.append("file", file);

            fetch("/predict-json", { method: "POST", body: form })
            .then(r => r.json())
            .then(data => {
                document.getElementById("prediction").innerText =
                    data.prediction + " (p=" + data.probability.toFixed(3) + ")";
            });

            fetch("/predict-image", { method: "POST", body: form })
            .then(r => r.blob())
            .then(img => {
                document.getElementById("resultImg").src = URL.createObjectURL(img);
            });
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


# ================== API ENDPOINTS ====================
@app.post("/predict-json")
async def predict_json(file: UploadFile = File(...)):
    """Return JSON with predicted label + probability."""
    pil_img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    _, prob = generate_gradcam(pil_img)
    label = "Malignant" if prob > 0.5 else "Benign"
    return JSONResponse({"prediction": label, "probability": prob})


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """Return Grad-CAM overlay image."""
    pil_img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    overlay, prob = generate_gradcam(pil_img)
    _, out_png = cv2.imencode(".png", overlay)
    return StreamingResponse(io.BytesIO(out_png.tobytes()), media_type="image/png")


# ================= RUN LOCAL =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
