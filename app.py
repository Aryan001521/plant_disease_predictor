import os
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# =========================================================
# 1) App Config
# =========================================================
st.set_page_config(page_title="Plant Disease Predictor", layout="centered")
DEVICE = torch.device("cpu")

# =========================================================
# 2) CSS (Background + Fix black widgets + Fix button)
# =========================================================
st.markdown("""
<style>

/* ---------- APP BACKGROUND ---------- */
.stApp{
  background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #ecfeff 100%) !important;
}

/* ---------- TEXT READABILITY ---------- */
html, body, [class*="css"]{
  color: #111827 !important;
}
h1,h2,h3,h4,h5,h6,p,span,label,li,div{
  color: #111827 !important;
}

/* ---------- LAYOUT ---------- */
main .block-container{
  max-width: 900px;
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}

/* ---------- CARDS ---------- */
.card{
  background: rgba(255,255,255,0.96) !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 16px !important;
  padding: 16px 18px !important;
  margin: 12px 0 !important;
  box-shadow: 0 10px 25px rgba(15,23,42,0.10) !important;
}
.small{
  color: #4b5563 !important;
  font-size: 0.92rem !important;
}

/* ---------- BUTTON (FIX INVISIBLE TEXT) ---------- */
.stButton > button{
  background-color: #111827 !important;
  color: #ffffff !important;
  border-radius: 12px !important;
  border: 1px solid #111827 !important;
  padding: 0.65rem 1.2rem !important;
  font-size: 16px !important;
  font-weight: 700 !important;
}
.stButton > button *{ color: #ffffff !important; }
.stButton > button:hover{
  background-color: #1f2937 !important;
  border-color: #1f2937 !important;
}

/* ---------- SELECTBOX FIX ---------- */
div[data-testid="stSelectbox"] div[data-baseweb="select"]{
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 12px !important;
}
div[data-baseweb="select"] > div{ background: #ffffff !important; }
div[data-baseweb="select"] *{ color: #111827 !important; }
div[data-baseweb="select"] input{
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}
div[data-baseweb="select"] svg{ fill: #111827 !important; }

/* Dropdown list */
ul[role="listbox"]{
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 12px !important;
}
li[role="option"]{
  background: #ffffff !important;
  color: #111827 !important;
}
li[role="option"]:hover{ background: #f3f4f6 !important; }

/* ---------- FILE UPLOADER FIX ---------- */
div[data-testid="stFileUploader"]{
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 12px !important;
  padding: 10px !important;
}
div[data-testid="stFileUploader"] *{ color: #111827 !important; }

section[data-testid="stFileUploaderDropzone"]{
  background: #ffffff !important;
  border: 1px dashed #c7d2fe !important;
  border-radius: 12px !important;
}
section[data-testid="stFileUploaderDropzone"] button{
  background: #2563eb !important;
  color: #ffffff !important;
  border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 3) HuggingFace model auto-download
# =========================================================
MODEL_URLS = {
    "color": "https://huggingface.co/jighb/plant-disease-resnet18/resolve/main/color_optuna_resnet18.pth",
    "grayscale": "https://huggingface.co/jighb/plant-disease-resnet18/resolve/main/grayscale_optuna_resnet18.pth",
}

os.makedirs("models", exist_ok=True)

def ensure_model(name: str) -> str:
    """Download model file once (if missing) and return local path."""
    local_path = f"models/{name}_optuna_resnet18.pth"
    if os.path.exists(local_path):
        return local_path

    url = MODEL_URLS[name]
    with st.spinner(f"Downloading {name} model (first time only)..."):
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return local_path

# Model choices shown to user
MODEL_PATHS = {
    "Best (Color model)": ensure_model("color"),
    "Grayscale model": ensure_model("grayscale"),
}

# =========================================================
# 4) Preprocessing
# =========================================================
def get_transform(model_name: str):
    t = [transforms.Resize((224, 224))]
    if "Grayscale" in model_name:
        t.append(transforms.Grayscale(num_output_channels=3))
    t += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(t)

# =========================================================
# 5) Build / Load model
# =========================================================
def build_model(num_classes: int):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

@st.cache_resource
def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location=DEVICE)

    if "classes" not in ckpt or "state_dict" not in ckpt:
        raise ValueError("Checkpoint format invalid. Expect keys: 'classes', 'state_dict'.")

    classes = ckpt["classes"]
    state_dict = ckpt["state_dict"]

    model = build_model(len(classes))
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes

# =========================================================
# 6) Prediction helpers
# =========================================================
def predict_probs(image: Image.Image, model, model_name: str):
    x = get_transform(model_name)(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0)
    return probs

def topk_from_probs(probs, classes, k=5):
    k = min(k, len(classes))
    top_probs, top_idx = torch.topk(probs, k=k)
    return [(classes[int(i)], float(p)) for p, i in zip(top_probs, top_idx)]

def pretty_label(label: str) -> str:
    return label.replace("___", " / ").replace("_", " ")

def confidence_badge(p: float) -> str:
    # simple user-friendly confidence label
    if p >= 0.85:
        return "High"
    if p >= 0.60:
        return "Medium"
    return "Low"

# =========================================================
# 7) Care Tips (simple, non-technical)
# =========================================================
def cure_tips(label: str):
    low = label.lower()

    if "healthy" in low:
        return [
            "Looks healthy. No treatment needed.",
            "Keep normal watering and balanced fertilizer.",
            "Remove old/dry leaves and keep the area clean."
        ]

    if "late_blight" in low or "late blight" in low:
        return [
            "Remove infected leaves quickly (do not compost).",
            "Water near soil, avoid wet leaves.",
            "Give more spacing for airflow.",
            "Follow local agriculture guidance for safe treatment."
        ]

    if "early_blight" in low or "early blight" in low:
        return [
            "Remove infected leaves and fallen debris.",
            "Keep leaves dry (avoid overhead watering).",
            "Rotate crops to reduce recurrence.",
            "Consult local guidance if it spreads."
        ]

    if "powdery_mildew" in low or "powdery mildew" in low:
        return [
            "Remove heavily infected leaves.",
            "Improve ventilation and spacing.",
            "Avoid excess nitrogen fertilizer.",
            "Use recommended treatment as per local guidance."
        ]

    if "rust" in low:
        return [
            "Remove infected leaves and nearby weeds.",
            "Avoid water splashing on leaves.",
            "Improve airflow and sunlight exposure.",
            "Consult local guidance if it increases."
        ]

    if "mosaic" in low or "virus" in low:
        return [
            "Remove/isolate infected plants to prevent spread.",
            "Disinfect tools after use.",
            "Control insects (aphids/whiteflies) safely.",
            "Use healthy seedlings/resistant varieties if available."
        ]

    return [
        "Try a clearer photo (close-up, good light, plain background).",
        "Remove infected leaves and keep the area clean.",
        "Avoid overhead watering; keep leaves dry.",
        "For exact treatment, consult local agriculture experts."
    ]

# =========================================================
# 8) UI (Simple + Professional)
# =========================================================
st.title("🌿 Plant Disease Predictor")
st.write("Upload a leaf image to get a diagnosis with simple care tips.")

model_choice = st.selectbox("Choose a model", list(MODEL_PATHS.keys()), index=0)

uploaded = st.file_uploader("Upload leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.markdown('<div class="card"><span class="small">⬆️ Upload an image to start.</span></div>', unsafe_allow_html=True)
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.image(image, width=520)

if st.button("Predict"):
    try:
        model_path = MODEL_PATHS[model_choice]
        model, classes = load_checkpoint(model_path)

        probs = predict_probs(image, model, model_choice)
        preds = topk_from_probs(probs, classes, k=5)

        top_label, top_prob = preds[0]
        nice_label = pretty_label(top_label)
        conf_text = confidence_badge(top_prob)

        # Result card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✅ Result")
        st.write(f"**Prediction:** {nice_label}")
        st.write(f"**Confidence:** {top_prob*100:.2f}%  ({conf_text})")
        st.markdown('<div class="small">Note: This is a guidance tool. For exact pesticide/dosage, follow local agriculture recommendations.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Tips card
        tips = cure_tips(top_label)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🌱 What you can do")
        for t in tips:
            st.write(f"- {t}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Chart
        df = pd.DataFrame({
            "Class": [pretty_label(x[0]) for x in preds],
            "Probability": [x[1] for x in preds]
        }).set_index("Class")

        st.subheader("📊 Confidence Chart")
        st.bar_chart(df)

        # Optional: show details
        with st.expander("Show Top Predictions"):
            for label, p in preds:
                st.write(f"**{pretty_label(label)}** — {p*100:.2f}%")
                st.progress(int(p * 100))

    except Exception as e:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.error(f"Something went wrong: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="card"><span class="small">Click <b>Predict</b> to see the result.</span></div>', unsafe_allow_html=True)
