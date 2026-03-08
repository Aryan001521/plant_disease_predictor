import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================================================
# 1) App Config
# =========================================================
st.set_page_config(page_title="Plant Disease Predictor", layout="centered")
DEVICE = torch.device("cpu")

# =========================================================
# 2) CSS
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
  max-width: 950px;
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

/* ---------- BUTTON ---------- */
.stButton > button{
  background-color: #111827 !important;
  color: #ffffff !important;
  border-radius: 12px !important;
  border: 1px solid #111827 !important;
  padding: 0.65rem 1.2rem !important;
  font-size: 16px !important;
  font-weight: 700 !important;
}
.stButton > button *{
  color: #ffffff !important;
}
.stButton > button:hover{
  background-color: #1f2937 !important;
  border-color: #1f2937 !important;
}

/* ---------- SELECTBOX ---------- */
div[data-testid="stSelectbox"] div[data-baseweb="select"]{
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 12px !important;
}
div[data-baseweb="select"] > div{
  background: #ffffff !important;
}
div[data-baseweb="select"] *{
  color: #111827 !important;
}
div[data-baseweb="select"] input{
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}
div[data-baseweb="select"] svg{
  fill: #111827 !important;
}

/* ---------- DROPDOWN LIST ---------- */
ul[role="listbox"]{
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 12px !important;
}
li[role="option"]{
  background: #ffffff !important;
  color: #111827 !important;
}
li[role="option"]:hover{
  background: #f3f4f6 !important;
}

/* ---------- FILE UPLOADER ---------- */
div[data-testid="stFileUploader"]{
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 12px !important;
  padding: 10px !important;
}
div[data-testid="stFileUploader"] *{
  color: #111827 !important;
}
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
# 3) Model paths
# Put your .pth files inside ./models/
# =========================================================
MODEL_PATHS = {
    "Best (Color model)": "models/color_optuna_resnet18.pth",
    "Grayscale model": "models/grayscale_optuna_resnet18.pth",
    "Segmented model": "models/segmented_optuna_resnet18.pth",
}

def get_available_models():
    return {k: v for k, v in MODEL_PATHS.items() if os.path.exists(v)}

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
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

@st.cache_resource
def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location=DEVICE)
    classes = ckpt["classes"]
    state_dict = ckpt["state_dict"]

    model = build_model(len(classes))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, classes

# =========================================================
# 6) Prediction helpers
# =========================================================
def preprocess_image(image: Image.Image, model_name: str):
    x = get_transform(model_name)(image).unsqueeze(0).to(DEVICE)
    return x

def predict_probs(image: Image.Image, model, model_name: str):
    x = preprocess_image(image, model_name)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    return probs

def topk_from_probs(probs, classes, k=5):
    k = min(k, len(classes))
    top_probs, top_idx = torch.topk(probs, k=k)
    return [(classes[int(i)], float(p)) for p, i in zip(top_probs, top_idx)]

def pretty_label(label: str) -> str:
    return label.replace("___", " / ").replace("_", " ")

# =========================================================
# 7) Disease info + cure tips
# =========================================================
def disease_info(label: str):
    low = label.lower()

    if "healthy" in low:
        return {
            "cause": "No disease detected.",
            "severity": "None",
            "spread": "None",
            "note": "The leaf appears healthy."
        }

    if "late_blight" in low or "late blight" in low:
        return {
            "cause": "Usually linked to fungal-like pathogen infection in wet conditions.",
            "severity": "High",
            "spread": "Fast in humid conditions",
            "note": "Can spread quickly and damage nearby plants."
        }

    if "early_blight" in low or "early blight" in low:
        return {
            "cause": "Usually fungal infection, often seen on older leaves first.",
            "severity": "Medium to High",
            "spread": "Moderate",
            "note": "Common in warm and humid environments."
        }

    if "powdery_mildew" in low or "powdery mildew" in low:
        return {
            "cause": "Fungal disease that appears like white powder on leaves.",
            "severity": "Medium",
            "spread": "Moderate",
            "note": "Can reduce plant growth if ignored."
        }

    if "rust" in low:
        return {
            "cause": "Fungal infection causing rust-colored spots.",
            "severity": "Medium",
            "spread": "Moderate",
            "note": "Often spreads through moisture and wind."
        }

    if "mosaic" in low or "virus" in low:
        return {
            "cause": "Likely viral infection.",
            "severity": "High",
            "spread": "Can spread through insects/tools",
            "note": "Isolation is important to reduce spread."
        }

    return {
        "cause": "Disease detected, but exact biological cause may vary.",
        "severity": "Unknown",
        "spread": "Possible",
        "note": "Use a clearer image or consult local experts for exact confirmation."
    }

def cure_tips(label: str):
    low = label.lower()

    if "healthy" in low:
        return [
            "No treatment needed.",
            "Keep regular watering and balanced fertilizer.",
            "Remove old leaves and keep the area clean."
        ]

    if "late_blight" in low or "late blight" in low:
        return [
            "Remove infected leaves quickly (do not compost).",
            "Avoid watering on leaves; water near soil.",
            "Improve airflow with proper spacing/pruning.",
            "Follow local agriculture guidance for fungicide use."
        ]

    if "early_blight" in low or "early blight" in low:
        return [
            "Remove infected leaves and fallen debris.",
            "Water at soil level and keep leaves dry.",
            "Rotate crops to reduce recurrence.",
            "Consult local agriculture guidance if it spreads."
        ]

    if "powdery_mildew" in low or "powdery mildew" in low:
        return [
            "Remove heavily infected leaves.",
            "Avoid too much nitrogen fertilizer.",
            "Improve ventilation and spacing.",
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
            "Isolate or remove infected plants to prevent spread.",
            "Disinfect tools after use.",
            "Control insects like aphids/whiteflies safely.",
            "Use healthy seedlings/resistant varieties if available."
        ]

    return [
        "Take a clearer photo with good light and close focus.",
        "Remove infected leaves and keep the area clean.",
        "Avoid overhead watering and keep leaves dry.",
        "For exact treatment, consult local agriculture experts."
    ]

# =========================================================
# 8) Grad-CAM
# =========================================================
def generate_gradcam(image: Image.Image, model, model_name: str, class_idx=None):
    """
    Returns:
        heatmap_rgb_pil: PIL Image heatmap
        overlay_pil: PIL Image overlay on original resized image
    """
    model.eval()

    # Target layer for ResNet18
    target_layer = model.layer4[-1]

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    # full backward hook is preferred
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        x = preprocess_image(image, model_name)
        x.requires_grad_(True)

        output = model(x)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        score = output[:, class_idx]
        model.zero_grad()
        score.backward()

        acts = activations[0]      # shape: [1, C, H, W]
        grads = gradients[0]       # shape: [1, C, H, W]

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True) # [1, 1, H, W]
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = Image.fromarray(np.uint8(cam * 255)).resize((224, 224))
        cam_np = np.array(cam).astype(np.float32) / 255.0

        # Original resized image
        orig = image.resize((224, 224)).convert("RGB")
        orig_np = np.array(orig).astype(np.float32)

        # Create simple red heatmap
        heatmap = np.zeros((224, 224, 3), dtype=np.float32)
        heatmap[..., 0] = cam_np * 255.0  # red channel

        # Overlay
        overlay = 0.6 * orig_np + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        heatmap_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        heatmap_rgb[..., 0] = (cam_np * 255).astype(np.uint8)

        heatmap_rgb_pil = Image.fromarray(heatmap_rgb)
        overlay_pil = Image.fromarray(overlay)

        return heatmap_rgb_pil, overlay_pil

    finally:
        fh.remove()
        bh.remove()

# =========================================================
# 9) Download report
# =========================================================
def create_text_report(prediction, confidence, preds, info, tips):
    lines = []
    lines.append("Plant Disease Prediction Report")
    lines.append("=" * 35)
    lines.append(f"Prediction: {prediction}")
    lines.append(f"Confidence: {confidence:.2f}%")
    lines.append("")
    lines.append("Disease Info")
    lines.append("-" * 20)
    lines.append(f"Cause: {info['cause']}")
    lines.append(f"Severity: {info['severity']}")
    lines.append(f"Spread Risk: {info['spread']}")
    lines.append(f"Note: {info['note']}")
    lines.append("")
    lines.append("Top Predictions")
    lines.append("-" * 20)
    for label, p in preds:
        lines.append(f"{pretty_label(label)}: {p * 100:.2f}%")
    lines.append("")
    lines.append("Recommended Care Tips")
    lines.append("-" * 20)
    for t in tips:
        lines.append(f"- {t}")

    return "\n".join(lines)

# =========================================================
# 10) UI
# =========================================================
st.title("🌿 Plant Disease Predictor")
st.write("Upload a leaf image to get a disease diagnosis, confidence score, Grad-CAM explanation, and simple care tips.")

available = get_available_models()
if not available:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.error("No model files found in the `models/` folder. Please add your trained `.pth` files.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📌 Image Tips")
st.write("- Use a clear leaf photo")
st.write("- Keep good lighting")
st.write("- Avoid blur")
st.write("- Try to capture one main leaf")
st.markdown('</div>', unsafe_allow_html=True)

model_choice = st.selectbox("Choose a model", list(available.keys()), index=0)
uploaded = st.file_uploader("Upload leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.markdown('<div class="card"><span class="small">⬆️ Upload an image to start.</span></div>', unsafe_allow_html=True)
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Uploaded Leaf Image", width=520)

if st.button("Predict"):
    model, classes = load_checkpoint(available[model_choice])
    probs = predict_probs(image, model, model_choice)
    preds = topk_from_probs(probs, classes, k=5)

    top_label, top_prob = preds[0]
    nice_label = pretty_label(top_label)

    info = disease_info(top_label)
    tips = cure_tips(top_label)

    # ---------------- Result ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✅ Result")
    st.write(f"**Prediction:** {nice_label}")
    st.write(f"**Confidence:** {top_prob * 100:.2f}%")

    if top_prob < 0.60:
        st.warning("Low confidence prediction. Please upload a clearer image with better lighting and focus.")
    elif top_prob < 0.80:
        st.info("Moderate confidence prediction. The result looks reasonable, but a clearer image may improve reliability.")
    else:
        st.success("High confidence prediction.")

    st.markdown('<div class="small">Note: For exact pesticide or dosage, always follow local agriculture recommendations.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Disease info ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🦠 Disease Information")
    st.write(f"**Possible Cause:** {info['cause']}")
    st.write(f"**Severity:** {info['severity']}")
    st.write(f"**Spread Risk:** {info['spread']}")
    st.write(f"**Note:** {info['note']}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Grad-CAM ----------------
    try:
        heatmap_img, overlay_img = generate_gradcam(
            image=image,
            model=model,
            model_name=model_choice,
            class_idx=classes.index(top_label)
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔥 Grad-CAM Explanation")
        st.write("This highlights the image regions the model focused on for its prediction.")
        col1, col2 = st.columns(2)
        with col1:
            st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
        with col2:
            st.image(overlay_img, caption="Heatmap Overlay on Leaf", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.warning(f"Grad-CAM could not be generated for this prediction. Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Tips ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌱 What you can do")
    for t in tips:
        st.write(f"- {t}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Chart ----------------
    df = pd.DataFrame({
        "Class": [pretty_label(x[0]) for x in preds],
        "Probability": [x[1] for x in preds]
    }).set_index("Class")

    st.subheader("📊 Confidence Chart")
    st.bar_chart(df)

    with st.expander("Show Top Predictions"):
        for label, p in preds:
            st.write(f"**{pretty_label(label)}** — {p * 100:.2f}%")
            st.progress(int(p * 100))

    # ---------------- Download report ----------------
    report_text = create_text_report(
        prediction=nice_label,
        confidence=top_prob * 100,
        preds=preds,
        info=info,
        tips=tips
    )

    st.download_button(
        label="📥 Download Prediction Report",
        data=report_text,
        file_name="plant_disease_report.txt",
        mime="text/plain"
    )

else:
    st.markdown('<div class="card"><span class="small">Click <b>Predict</b> to see the result.</span></div>', unsafe_allow_html=True)
