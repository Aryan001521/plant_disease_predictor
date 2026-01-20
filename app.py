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
# 2) Clean UI + Fix black widgets + Fix button text
# =========================================================
st.markdown("""
<style>
.stApp{
  background: linear-gradient(135deg, #0b1220 0%, #0f172a 45%, #071826 100%) !important;
}

/* Main container width */
main .block-container{
  max-width: 920px;
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}

/* Text */
html, body, [class*="css"] { color: #e5e7eb !important; }
h1,h2,h3,h4,h5,h6,p,span,label,li,div { color: #e5e7eb !important; }

/* Card */
.card{
  background: rgba(17,24,39,0.75) !important;
  border: 1px solid rgba(148,163,184,0.18) !important;
  border-radius: 16px !important;
  padding: 16px 18px !important;
  margin: 12px 0 !important;
  box-shadow: 0 10px 25px rgba(0,0,0,0.25) !important;
}
.muted{ color:#a8b3c7 !important; font-size:0.92rem !important; }

/* Button fix (Predict invisible text issue) */
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
.stButton > button:hover{
  background-color: #1d4ed8 !important;
  border-color: #1d4ed8 !important;
}

/* Selectbox fix (dark theme) */
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
  background: #0b1220 !important;
  border: 1px solid rgba(148,163,184,0.25) !important;
  border-radius: 12px !important;
}
li[role="option"]{ color:#e5e7eb !important; }
li[role="option"]:hover{ background: rgba(148,163,184,0.12) !important; }

/* File uploader fix */
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

/* Small badges */
.badge{
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  background: rgba(34,197,94,0.15);
  border: 1px solid rgba(34,197,94,0.35);
  color:#bbf7d0 !important;
  font-weight:600;
  font-size: 0.85rem;
}
.badge-low{
  background: rgba(239,68,68,0.12);
  border: 1px solid rgba(239,68,68,0.30);
  color:#fecaca !important;
}
.badge-mid{
  background: rgba(245,158,11,0.12);
  border: 1px solid rgba(245,158,11,0.30);
  color:#fde68a !important;
}

/* Progress bars look better in dark */
div[data-testid="stProgress"] > div > div{
  background-color: #60a5fa !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3) HuggingFace auto-download (IMPORTANT)
# =========================================================
HF_REPO = "ijghb/plant-disease-resnet18"

MODEL_URLS = {
    "Best (Color model)": f"https://huggingface.co/{HF_REPO}/resolve/main/color_optuna_resnet18.pth",
    "Grayscale model": f"https://huggingface.co/{HF_REPO}/resolve/main/grayscale_optuna_resnet18.pth",
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def ensure_model(url: str) -> str:
    """
    Downloads the model file if missing and returns local path.
    Works on Streamlit Cloud.
    """
    filename = os.path.basename(url)
    local_path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(local_path):
        return local_path

    with st.spinner("Downloading model (first time only)..."):
        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()
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

def confidence_level(p: float) -> str:
    if p >= 0.85:
        return "High"
    if p >= 0.60:
        return "Medium"
    return "Low"

def confidence_badge_html(level: str) -> str:
    if level == "High":
        return '<span class="badge">High confidence</span>'
    if level == "Medium":
        return '<span class="badge badge-mid">Medium confidence</span>'
    return '<span class="badge badge-low">Low confidence</span>'

def topk(probs: torch.Tensor, classes, k=5):
    k = min(k, len(classes))
    vals, idxs = torch.topk(probs, k)
    out = []
    for v, i in zip(vals, idxs):
        out.append((classes[int(i)], float(v)))
    return out

# =========================================================
# 6) Simple care tips (non-medical, guidance only)
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
# 7) UI (Simple + professional)
# =========================================================
st.title("🌿 Plant Disease Predictor")
st.markdown('<div class="muted">Upload a leaf image to get a simple diagnosis and care guidance.</div>', unsafe_allow_html=True)

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

# User-friendly optional details
with st.expander("Optional: Show more details"):
    show_details = st.checkbox("Show top predictions and chart", value=False)
    confidence_threshold = st.slider("Show result only if confidence is at least:", 0.30, 0.95, 0.60)
    top_k = st.slider("How many alternative predictions to show:", 1, 5, 3)

if st.button("Predict"):
    try:
        # Download model only when user clicks Predict (fast + stable)
        url = MODEL_URLS[model_choice]
        local_model_path = ensure_model(url)

        model, classes = load_checkpoint(local_model_path)

        x = get_transform(model_choice)(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).squeeze(0)

        preds = topk(probs, classes, k=5)
        top_label, top_prob = preds[0]

        # If confidence too low, show warning
        if top_prob < confidence_threshold:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.warning("Confidence is low for this image. Try a clearer photo (close-up, good light, plain background).")
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()

        nice = pretty_label(top_label)
        level = confidence_level(top_prob)

        # Result (simple, top-1)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✅ Result")
        st.markdown(confidence_badge_html(level), unsafe_allow_html=True)
        st.write(f"**Prediction:** {nice}")
        st.write(f"**Confidence:** {top_prob*100:.2f}%")
        st.markdown('<div class="muted">Note: This tool provides guidance only. For exact pesticide/dosage, follow local agriculture recommendations.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Care tips
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🌱 What you can do (simple tips)")
        for tip in care_tips(top_label):
            st.write(f"- {tip}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Optional details
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

    except requests.exceptions.HTTPError:
        st.error(
            "Model download failed. Please make sure your Hugging Face repo is PUBLIC and the file names match.\n\n"
            f"Repo: {HF_REPO}"
        )
    except Exception as e:
        st.error(f"Something went wrong: {e}")
