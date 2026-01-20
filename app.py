import os
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# =========================================================
# 1) Page config
# =========================================================
st.set_page_config(page_title="Plant Disease Predictor", page_icon="🌿", layout="centered")
DEVICE = torch.device("cpu")

# =========================================================
# 2) Dark UI + fix widgets/button visibility
# =========================================================
st.markdown("""
<style>
.stApp{
  background: linear-gradient(135deg, #0b1220 0%, #0f172a 45%, #071826 100%) !important;
}
main .block-container{ max-width: 920px; padding-top: 1.2rem; padding-bottom: 2rem; }
html, body, [class*="css"]{ color:#e5e7eb !important; }
h1,h2,h3,h4,h5,h6,p,span,label,li,div{ color:#e5e7eb !important; }

.card{
  background: rgba(17,24,39,0.75) !important;
  border: 1px solid rgba(148,163,184,0.18) !important;
  border-radius: 16px !important;
  padding: 16px 18px !important;
  margin: 12px 0 !important;
  box-shadow: 0 10px 25px rgba(0,0,0,0.25) !important;
}
.muted{ color:#a8b3c7 !important; font-size:0.92rem !important; }

.stButton > button{
  background-color: #2563eb !important;
  color: #ffffff !important;
  border-radius: 12px !important;
  border: 1px solid #2563eb !important;
  padding: 0.65rem 1.2rem !important;
  font-size: 16px !important;
  font-weight: 700 !important;
}
.stButton > button *{ color:#ffffff !important; }
.stButton > button:hover{ background-color:#1d4ed8 !important; border-color:#1d4ed8 !important; }

div[data-testid="stSelectbox"] div[data-baseweb="select"]{
  background: rgba(15,23,42,0.85) !important;
  border: 1px solid rgba(148,163,184,0.25) !important;
  border-radius: 12px !important;
}
div[data-baseweb="select"] *{ color:#e5e7eb !important; }
div[data-baseweb="select"] input{
  color:#e5e7eb !important;
  -webkit-text-fill-color:#e5e7eb !important;
}
ul[role="listbox"]{
  background:#0b1220 !important;
  border:1px solid rgba(148,163,184,0.25) !important;
  border-radius:12px !important;
}
li[role="option"]{ color:#e5e7eb !important; }
li[role="option"]:hover{ background: rgba(148,163,184,0.12) !important; }

div[data-testid="stFileUploader"]{
  background: rgba(15,23,42,0.75) !important;
  border: 1px solid rgba(148,163,184,0.25) !important;
  border-radius: 12px !important;
  padding: 10px !important;
}
div[data-testid="stFileUploader"] *{ color:#e5e7eb !important; }
section[data-testid="stFileUploaderDropzone"]{
  background: rgba(15,23,42,0.55) !important;
  border: 1px dashed rgba(148,163,184,0.35) !important;
  border-radius: 12px !important;
}
section[data-testid="stFileUploaderDropzone"] button{
  background: #334155 !important;
  color: #ffffff !important;
  border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3) HuggingFace auto-download (robust)
# =========================================================
HF_REPO = "ijghb/plant-disease-resnet18"

# Use hf.co domain (often more reliable on Streamlit Cloud)
MODEL_URLS = {
    "Best (Color model)": f"https://hf.co/{HF_REPO}/resolve/main/color_optuna_resnet18.pth?download=true",
    "Grayscale model": f"https://hf.co/{HF_REPO}/resolve/main/grayscale_optuna_resnet18.pth?download=true",
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def ensure_model(url: str) -> str:
    """
    Downloads model file if missing and returns local path.
    Prints detailed error info (status code + URL) when it fails.
    Supports optional HF_TOKEN in Streamlit secrets.
    """
    filename = os.path.basename(url.split("?")[0])
    local_path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(local_path):
        return local_path

    headers = {"User-Agent": "Mozilla/5.0"}

    # Optional: if repo is PRIVATE, add token in Streamlit Secrets:
    # HF_TOKEN = "..."
    token = None
    try:
        token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        token = None

    if token:
        headers["Authorization"] = f"Bearer {token}"

    with st.spinner("Downloading model (first time only)..."):
        r = requests.get(url, stream=True, timeout=240, headers=headers, allow_redirects=True)

        if r.status_code != 200:
            # Show exact reason
            st.error(f"Download failed ❌  |  HTTP {r.status_code}")
            st.code(f"URL: {url}")
            st.code(f"Hint: 401/403 = private/gated repo or token missing | 404 = wrong path/branch/filename")
            st.stop()

        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return local_path

# =========================================================
# 4) Model build/load
# =========================================================
def build_model(num_classes: int):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

@st.cache_resource
def load_checkpoint(local_path: str):
    ckpt = torch.load(local_path, map_location=DEVICE)
    if "classes" not in ckpt or "state_dict" not in ckpt:
        raise ValueError("Checkpoint format invalid. Expected keys: 'classes', 'state_dict'.")
    classes = ckpt["classes"]
    model = build_model(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes

# =========================================================
# 5) Preprocessing
# =========================================================
def get_transform(model_choice: str):
    t = [transforms.Resize((224, 224))]
    if "Grayscale" in model_choice:
        t.append(transforms.Grayscale(num_output_channels=3))
    t += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(t)

def pretty_label(label: str) -> str:
    return label.replace("___", " / ").replace("_", " ")

def topk(probs: torch.Tensor, classes, k=5):
    k = min(k, len(classes))
    vals, idxs = torch.topk(probs, k)
    return [(classes[int(i)], float(v)) for v, i in zip(vals, idxs)]

# =========================================================
# 6) Simple care tips
# =========================================================
def care_tips(label: str):
    s = label.lower()
    if "healthy" in s:
        return [
            "Leaf looks healthy. No treatment needed.",
            "Keep normal watering and balanced nutrition.",
            "Remove old/dry leaves and keep the area clean."
        ]
    if "powdery_mildew" in s or "powdery mildew" in s:
        return [
            "Remove heavily infected leaves.",
            "Increase airflow (spacing) and avoid overwatering.",
            "Avoid excess nitrogen fertilizer.",
            "If it spreads, follow local agriculture guidance."
        ]
    if "rust" in s:
        return [
            "Remove infected leaves and nearby weeds.",
            "Avoid splashing water on leaves.",
            "Improve sunlight and airflow.",
            "Follow local guidance if it increases."
        ]
    if "late_blight" in s or "late blight" in s:
        return [
            "Remove infected leaves quickly (do not compost).",
            "Water near soil; keep leaves dry.",
            "Increase spacing for airflow.",
            "Consult local experts for safe treatment."
        ]
    if "early_blight" in s or "early blight" in s:
        return [
            "Remove infected leaves and fallen debris.",
            "Avoid overhead watering; keep leaves dry.",
            "Rotate crops to reduce recurrence.",
            "Consult local guidance if it spreads."
        ]
    if "virus" in s or "mosaic" in s:
        return [
            "Isolate/remove infected plant to prevent spread.",
            "Disinfect tools after use.",
            "Control insects safely (aphids/whiteflies).",
            "Use healthy seedlings/resistant varieties if available."
        ]
    return [
        "Try a clearer image (close-up, good light, plain background).",
        "Remove infected leaves and keep the area clean.",
        "Avoid overhead watering; keep leaves dry.",
        "For exact treatment, consult local agriculture experts."
    ]

# =========================================================
# 7) UI
# =========================================================
st.title("🌿 Plant Disease Predictor")
st.markdown('<div class="muted">Upload a leaf image to get a simple diagnosis and guidance.</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
model_choice = st.selectbox("Choose a model", list(MODEL_URLS.keys()), index=0)
uploaded = st.file_uploader("Upload leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if not uploaded:
    st.markdown('<div class="card"><div class="muted">⬆️ Upload an image to start.</div></div>', unsafe_allow_html=True)
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.image(img, width=560)
st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Optional: Show more details"):
    show_details = st.checkbox("Show top predictions and chart", value=False)
    top_k = st.slider("How many alternative predictions to show:", 1, 5, 3)

if st.button("Predict"):
    url = MODEL_URLS[model_choice]

    # Download only when user clicks Predict
    local_model_path = ensure_model(url)

    model, classes = load_checkpoint(local_model_path)

    x = get_transform(model_choice)(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0)

    preds = topk(probs, classes, k=5)
    top_label, top_prob = preds[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✅ Result")
    st.write(f"**Prediction:** {pretty_label(top_label)}")
    st.write(f"**Confidence:** {top_prob*100:.2f}%")
    st.markdown('<div class="muted">Note: This tool provides guidance only. For exact pesticide/dosage, follow local agriculture recommendations.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌱 What you can do (simple tips)")
    for tip in care_tips(top_label):
        st.write(f"- {tip}")
    st.markdown('</div>', unsafe_allow_html=True)

    if show_details:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Top predictions")
        for label, p in preds[:top_k]:
            st.write(f"**{pretty_label(label)}** — {p*100:.2f}%")
            st.progress(int(p * 100))
        st.markdown('</div>', unsafe_allow_html=True)

        df = pd.DataFrame({
            "Class": [pretty_label(x[0]) for x in preds[:top_k]],
            "Probability": [x[1] for x in preds[:top_k]],
        }).set_index("Class")

        st.subheader("📊 Confidence chart")
        st.bar_chart(df)
