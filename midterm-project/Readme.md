# 🩺 Skin Lesion Classification with Grad-CAM Explainability (ISIC-2020)

This project predicts whether a dermoscopic image of a skin lesion is **benign or malignant** using deep learning.  
It also produces **Grad-CAM heatmaps** to visualize which areas of the lesion influence the prediction.

## 🚀 Tech Stack
- PyTorch
- FastAPI
- HTML/JS Web UI
- Docker
- Grad-CAM Explainability

---

## 📊 Dataset

🔗 **ISIC 2020 Challenge Dataset**  
https://challenge2020.isic-archive.com/

📥 You must download:
- `ISIC_2020_Training_JPEG.zip`
- `ISIC_2020_Training_GroundTruth.csv`
- `ISIC_2020_Training_Duplicates.csv`

Place them inside:

skin-cancer-detection/data/


---

## 🧪 Training

Run:

python src/train.py
This:
cleans dataset

trains MobileNetV3 Small (fast mode)

saves model to models/isic_fast_model.pt

🤖 Start API + Web UI

python src/predict.py
Then open in browser:
http://localhost:8000
Upload an image → view prediction + Grad-CAM.

🐳 Docker Deployment
Build image:

docker build -t skin-lesion-app .
Run container:

docker run -p 8000:8000 skin-lesion-app
Open browser:

http://localhost:8000
