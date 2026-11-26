import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib
import os
import json
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="AI-Based Multi-Crop Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f4f7f5;
        color: #1e293b;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #165b33 0%, #1f7a43 100%);
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    .banner-box {
        background: #ffffff;
        border: 1px solid #d6e4da;
        border-left: 6px solid #1f7a43;
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 16px;
    }
    .card-box {
        background: #ffffff;
        border: 1px solid #dce5df;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #14532d;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #e5ece7;
    }
    .sev-card {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .grad-scale {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #1d4ed8, #06b6d4, #22c55e, #eab308, #ef4444);
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# All 3 models included
GATEKEEPER_PATH = "model_gatekeeper.h5"
DISEASE_MODEL_PATH = "model_disease_v7.h5"
POTATO_MODEL_PATH = "model_potato.h5"
IMG_SIZE = (224, 224)
HISTORY_DIR = "history"

GATE_CLASSES = ["non_leaf", "other_plant_leaf", "supported_leaf"]

CLASS_NAMES = [
    "Pepper Bacterial Spot",
    "Pepper Healthy",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight"
]

POTATO_CLASSES = [
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight"
]

RECOMMENDATIONS = {
    "Pepper Bacterial Spot": {
        "Very Mild": "Apply copper-based bactericide as a preventive measure.",
        "Mild": "Apply copper-based bactericide. Avoid overhead watering.",
        "Moderate": "Apply bactericide within 2-3 days. Remove affected leaves.",
        "Severe": "Remove severely affected plants. Apply bactericide to remaining plants immediately."
    },
    "Pepper Healthy": {
        "Very Mild": "No action needed. Continue regular monitoring.",
        "Mild": "No action needed. Continue regular monitoring.",
        "Moderate": "No action needed. Continue regular monitoring.",
        "Severe": "No action needed. Continue regular monitoring."
    },
    "Potato Early Blight": {
        "Very Mild": "Monitor the plant. Remove any visibly spotted lower leaves.",
        "Mild": "Apply a preventive fungicide and improve airflow between plants.",
        "Moderate": "Apply fungicide within 2-3 days. Remove and destroy affected leaves.",
        "Severe": "Apply fungicide immediately. Consider removing severely affected plants."
    },
    "Potato Healthy": {
        "Very Mild": "No action needed. Continue regular monitoring.",
        "Mild": "No action needed. Continue regular monitoring.",
        "Moderate": "No action needed. Continue regular monitoring.",
        "Severe": "No action needed. Continue regular monitoring."
    },
    "Potato Late Blight": {
        "Very Mild": "Inspect daily - late blight spreads fast. Apply preventive fungicide now.",
        "Mild": "Apply a systemic fungicide immediately. Avoid overhead irrigation.",
        "Moderate": "Urgent fungicide treatment required within 24-48 hours.",
        "Severe": "Remove and destroy affected plants immediately to prevent spread."
    },
    "Tomato Early Blight": {
        "Very Mild": "Monitor the plant. Remove any visibly spotted lower leaves.",
        "Mild": "Apply a preventive fungicide and improve airflow between plants.",
        "Moderate": "Apply fungicide within 2-3 days. Remove and destroy affected leaves.",
        "Severe": "Apply fungicide immediately. Consider removing severely affected plants."
    },
    "Tomato Healthy": {
        "Very Mild": "No action needed. Continue regular monitoring.",
        "Mild": "No action needed. Continue regular monitoring.",
        "Moderate": "No action needed. Continue regular monitoring.",
        "Severe": "No action needed. Continue regular monitoring."
    },
    "Tomato Late Blight": {
        "Very Mild": "Inspect daily - late blight spreads fast. Apply preventive fungicide now.",
        "Mild": "Apply a systemic fungicide immediately. Avoid overhead irrigation.",
        "Moderate": "Urgent fungicide treatment required within 24-48 hours.",
        "Severe": "Remove and destroy affected plants immediately to prevent spread."
    }
}

os.makedirs(HISTORY_DIR, exist_ok=True)

@st.cache_resource
def load_models():
    for path in [GATEKEEPER_PATH, DISEASE_MODEL_PATH, POTATO_MODEL_PATH]:
        if not os.path.exists(path):
            st.error(f"Model file '{path}' not found in repository.")
            st.stop()
    gate_model = tf.keras.models.load_model(GATEKEEPER_PATH)
    disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    potato_model = tf.keras.models.load_model(POTATO_MODEL_PATH)
    return gate_model, disease_model, potato_model

def prepare_mobilenet_batch(image_np):
    resized = cv2.resize(image_np, IMG_SIZE).astype(np.float32)
    preprocessed = preprocess_input(resized)
    return np.expand_dims(preprocessed, axis=0)

def validate_input_image(image_np, gate_model):
    if image_np is None or image_np.size == 0:
        return False, "Not a Leaf", "The uploaded image could not be read. Please upload a valid leaf photo."

    h, w = image_np.shape[:2]
    if h < 40 or w < 40:
        return False, "Not a Leaf", "The image resolution is too small. Please upload a clear leaf photo."

    batch = prepare_mobilenet_batch(image_np)
    probs = gate_model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = GATE_CLASSES[pred_idx]
    supported_prob = float(probs[2])

    if pred_label == "non_leaf" and float(probs[0]) > 0.80:
        return False, "Not a Leaf", "Not a leaf image — please upload a plant leaf photo."
    if pred_label == "other_plant_leaf" and float(probs[1]) > 0.80 and supported_prob < 0.25:
        return False, "Not a Leaf (Unsupported Plant)", "Not a supported leaf — only Potato, Tomato, and Bell Pepper leaves are supported."

    return True, "", ""

def split_crop_and_disease(class_label):
    parts = class_label.split(" ", 1)
    return parts[0], (parts[1] if len(parts) > 1 else class_label)

def segment_leaf_and_disease(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    leaf_mask = cv2.inRange(hsv, np.array([10, 30, 20]), np.array([90, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    disease_mask_raw = cv2.inRange(hsv, np.array([5, 40, 20]), np.array([30, 255, 200]))
    disease_mask = cv2.bitwise_and(disease_mask_raw, leaf_mask)
    leaf_area = np.sum(leaf_mask > 0)
    disease_area = np.sum(disease_mask > 0)
    severity_percent = (disease_area / leaf_area * 100) if leaf_area > 0 else 0.0
    return leaf_mask, disease_mask, severity_percent

def severity_label(p):
    if p < 10:
        return "Very Mild"
    elif p < 25:
        return "Mild"
    elif p < 50:
        return "Moderate"
    else:
        return "Severe"

def make_severity_overlay(image_np, disease_mask):
    overlay = image_np.copy()
    overlay[disease_mask > 0] = [255, 0, 0]
    return cv2.addWeighted(image_np, 0.6, overlay, 0.4, 0)

def build_gradcam_model(model, input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    first_layer = model.layers[0]
    if hasattr(first_layer, "layers"):
        base_output = first_layer(inputs)
        y = base_output
        for layer in model.layers[1:]:
            y = layer(y)
        return tf.keras.Model(inputs=inputs, outputs=[base_output, y])
    else:
        x = inputs
        conv_output = None
        for layer in model.layers:
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_output = x
        return tf.keras.Model(inputs=inputs, outputs=[conv_output, x])

def make_gradcam_overlay(image_np, grad_model, pred_index):
    img_resized = cv2.resize(image_np, IMG_SIZE)
    img_array = prepare_mobilenet_batch(image_np)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap_resized = cv2.resize(heatmap.numpy(), IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet_colors = matplotlib.colormaps["jet"](np.arange(256))[:, :3]
    jet_heatmap = (jet_colors[heatmap_uint8] * 255).astype(np.uint8)
    return cv2.addWeighted(img_resized, 0.6, jet_heatmap, 0.4, 0)

def save_to_history(original_img, severity_img, gradcam_img, result):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    Image.fromarray(original_img).save(f"{HISTORY_DIR}/{timestamp}_original.jpg")
    Image.fromarray(severity_img).save(f"{HISTORY_DIR}/{timestamp}_severity.jpg")
    Image.fromarray(gradcam_img).save(f"{HISTORY_DIR}/{timestamp}_gradcam.jpg")
    result["timestamp"] = timestamp
    with open(f"{HISTORY_DIR}/{timestamp}_result.json", "w") as f:
        json.dump(result, f)

def load_history():
    entries = []
    for fname in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if fname.endswith("_result.json"):
            with open(f"{HISTORY_DIR}/{fname}") as f:
                entries.append(json.load(f))
    return entries

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

with st.sidebar:
    st.markdown("### 🌿 Multi-Crop Plant Disease Detection")
    st.caption("with Severity Estimation")
    nav_choice = st.radio(
        "Navigation",
        ["Home", "Upload Image", "About Project", "Dataset Classes"],
        index=["Home", "Upload Image", "About Project", "Dataset Classes"].index(st.session_state.nav_page),
        label_visibility="collapsed"
    )
    st.session_state.nav_page = nav_choice
    st.markdown("---")
    st.markdown("**Supported Crops**")
    st.markdown("🥔 **Potato**\n\n🍅 **Tomato**\n\n🫑 **Pepper**")

st.markdown("""
<div class="banner-box">
    <h2 style="margin:0; color:#14532d;">🌿 AI-Based Multi-Crop Plant Disease Detection and Severity Estimation</h2>
    <p style="margin:4px 0 0 0; color:#475569;">Upload a Potato, Tomato, or Bell Pepper leaf image to detect disease, estimate severity, and view Grad-CAM explanation.</p>
</div>
""", unsafe_allow_html=True)

gate_model, disease_model, potato_model = load_models()
grad_disease_model = build_gradcam_model(disease_model)
grad_potato_model = build_gradcam_model(potato_model)

if st.session_state.nav_page == "About Project":
    st.subheader("About Project: Multi-Crop Plant Disease Detection and Severity Estimation")
    st.write(
        "This system uses a MobileNetV2 pipeline with an Input Gatekeeper, an 8-class Multi-Crop Disease Classifier, "
        "and a dedicated Potato Specialist Classifier for **Potato**, **Tomato**, and **Bell Pepper** leaves."
    )
    st.stop()

if st.session_state.nav_page == "Dataset Classes":
    st.subheader("Supported Dataset Classes (8 Classes)")
    for idx, cls in enumerate(CLASS_NAMES, start=1):
        st.write(f"{idx}. **{cls}**")
    st.stop()

if st.session_state.nav_page == "Upload Image":
    st.info("Upload a clear Potato, Tomato, or Bell Pepper leaf image below.")

st.markdown('<div class="card-box"><div class="card-title">1. Upload Leaf Image</div>', unsafe_allow_html=True)
col_up, col_mode = st.columns([2.4, 1])
with col_up:
    uploaded_file = st.file_uploader("Drag and drop an image here or click to browse (JPG, PNG)", type=["jpg", "jpeg", "png"])
with col_mode:
    crop_hint = st.selectbox("Crop Filter (Optional)", ["Auto Detect", "Potato", "Tomato / Pepper"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
    except Exception:
        st.error("Invalid image file. Please upload a JPG or PNG photo.")
        st.stop()

    is_valid, err_title, err_msg = validate_input_image(image_np, gate_model)
    if not is_valid:
        st.error(f"🚫 {err_title}")
        st.warning(err_msg)
        st.stop()

    with st.spinner("Analyzing leaf image..."):
        img_resized = cv2.resize(image_np, IMG_SIZE)
        img_array = prepare_mobilenet_batch(image_np)
        predictions = disease_model.predict(img_array, verbose=0)[0]
        base_idx = int(np.argmax(predictions))
        base_class = CLASS_NAMES[base_idx]

        use_potato_specialist = (crop_hint == "Potato") or (
            crop_hint == "Auto Detect" and "Potato" in base_class
        )

        if use_potato_specialist:
            p_preds = potato_model.predict(img_array, verbose=0)[0]
            p_idx = int(np.argmax(p_preds))
            predicted_class = POTATO_CLASSES[p_idx]
            confidence = float(p_preds[p_idx]) * 100
            gradcam_overlay = make_gradcam_overlay(image_np, grad_potato_model, p_idx)
        else:
            predicted_class = base_class
            confidence = float(predictions[base_idx]) * 100
            gradcam_overlay = make_gradcam_overlay(image_np, grad_disease_model, base_idx)

        crop_name, disease_name = split_crop_and_disease(predicted_class)

        if "Healthy" not in predicted_class:
            _, disease_mask, severity_percent = segment_leaf_and_disease(image_np)
            sev_label = severity_label(severity_percent)
            severity_overlay = make_severity_overlay(img_resized, cv2.resize(disease_mask, IMG_SIZE))
        else:
            severity_percent = 0.0
            sev_label = "Very Mild"
            severity_overlay = img_resized

        recommendation = RECOMMENDATIONS[predicted_class][sev_label]

        save_to_history(img_resized, severity_overlay, gradcam_overlay, {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 1),
            "severity_percent": round(severity_percent, 1),
            "severity_label": sev_label,
            "recommendation": recommendation
        })

    c1, c2, c3 = st.columns([1.1, 1.15, 1.25])

    with c1:
        st.markdown('<div class="card-box"><div class="card-title">1. Upload Leaf Image</div>', unsafe_allow_html=True)
        st.image(image, caption="Original Upload", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card-box"><div class="card-title">2. Prediction Result</div>', unsafe_allow_html=True)
        st.markdown(f"**Crop :** {crop_name}")
        st.markdown(f"**Disease :** {disease_name}")
        st.markdown(f"**Confidence :** **{confidence:.2f}%**")
        st.markdown(f"""
        <div class="sev-card">
            <div style="font-weight:700; color:#92400e;">Severity Estimation (HSV-based)</div>
            <div style="font-size:1.5rem; font-weight:800; color:#b45309;">{severity_percent:.1f}%</div>
            <div style="font-weight:700; color:#78350f;">{sev_label}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("🟢 Very Mild (< 10%)  |  🟡 Mild (10% - 25%)")
        st.caption("🟠 Moderate (25% - 50%)  |  🔴 Severe (≥ 50%)")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card-box"><div class="card-title">3. Grad-CAM Visualization</div>', unsafe_allow_html=True)
        st.image(gradcam_overlay, caption="Grad-CAM Heatmap", use_container_width=True)
        st.markdown('<div class="grad-scale"></div>', unsafe_allow_html=True)
        st.caption("Low Attention ⬅️  ➡️ High Attention")
        st.markdown("</div>", unsafe_allow_html=True)

    if confidence < 60:
        st.warning("⚠️ Low confidence prediction. Try a closer, well-lit photo of a single leaf.")

    if "Healthy" in predicted_class:
        st.success(f"✅ {recommendation}")
    elif sev_label in ["Very Mild", "Mild"]:
        st.warning(f"⚠️ {recommendation}")
    else:
        st.error(f"🚨 {recommendation}")

if st.session_state.nav_page == "Home":
    st.markdown("---")
    st.subheader("📁 Upload History")
    history_entries = load_history()

    if len(history_entries) == 0:
        st.info("No past uploads yet. Upload a leaf image above to start building history.")
    else:
        st.write(f"Showing {len(history_entries)} past upload(s), most recent first.")
        for entry in history_entries:
            ts = entry["timestamp"]
            with st.expander(f"{entry['predicted_class']} - {entry['confidence']}% confidence - {ts}"):
                h_col1, h_col2, h_col3 = st.columns(3)
                with h_col1:
                    st.image(f"{HISTORY_DIR}/{ts}_original.jpg", caption="Original", use_container_width=True)
                with h_col2:
                    st.image(f"{HISTORY_DIR}/{ts}_severity.jpg", caption="Severity", use_container_width=True)
                with h_col3:
                    st.image(f"{HISTORY_DIR}/{ts}_gradcam.jpg", caption="Grad-CAM", use_container_width=True)
                st.write(f"**Disease:** {entry['predicted_class']}")
                st.write(f"**Confidence:** {entry['confidence']}%")
                st.write(f"**Severity:** {entry['severity_percent']}% ({entry['severity_label']})")
                st.write(f"**Recommendation:** {entry['recommendation']}")