import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib
import os
import json
import re
from datetime import datetime
from tensorflow.keras.applications import MobileNetV2 as ImageNetMobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

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

MODEL_PATH = "model_mobilenet_v4.h5"
IMG_SIZE = (224, 224)
HISTORY_DIR = "history"
CLASS_NAMES = [
    "Pepper Bacterial Spot", "Pepper Healthy", "Potato Early Blight",
    "Potato Healthy", "Potato Late Blight", "Tomato Early Blight",
    "Tomato Healthy", "Tomato Late Blight"
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
def load_input_validator():
    return ImageNetMobileNetV2(weights="imagenet", include_top=True)

PERSON_ANIMAL_LABELS = (
    "person", "man", "woman", "boy", "girl", "face", "red panda", "panda",
    "lynx", "cat", "dog", "fox", "wolf", "tiger", "lion", "bear", "monkey",
    "baboon", "gorilla", "chimpanzee", "elephant", "horse", "zebra", "cow",
    "sheep", "goat", "deer", "antelope", "rabbit", "hare", "squirrel",
    "raccoon", "skunk", "otter", "badger", "weasel", "hamster", "mouse",
    "rat", "bird", "eagle", "owl", "parrot", "penguin", "snake", "lizard",
    "turtle", "frog", "toad", "fish", "shark", "whale", "dolphin", "jellyfish",
    "starfish", "snail", "slug", "spider", "scorpion", "bee", "wasp", "ant"
)

OBJECT_LABELS = (
    "car", "bus", "truck", "bicycle", "mountain bike", "motorcycle", "scooter",
    "airplane", "airliner", "boat", "ship", "train", "tractor", "building",
    "computer", "keyboard", "phone", "chair", "sofa", "bottle", "cup", "bag",
    "backpack", "shoe", "clock", "book", "ball", "toy", "helmet", "camera",
    "remote control", "television", "laptop", "microwave", "refrigerator", "suit",
    "gown", "jersey", "sweatshirt", "cardigan", "wig", "mask", "beard", "sunglasses",
    "lab coat", "uniform", "bow tie", "microphone", "tie", "trench coat", "vestment",
    "cloak", "stethoscope", "handbag", "wallet", "umbrella", "syringe", "guitar",
    "drum", "sports car", "jeep", "minibus", "moped", "streetcar", "forklift",
    "lawn mower", "plow", "vacuum", "screen", "monitor", "reel", "wrench", "hammer",
    "scarf", "bandana", "veil", "turban", "headscarf", "shawl", "purse", "suitcase",
    "shopping bag", "plastic bag", "toyshop"
)

OTHER_PLANT_LABELS = (
    "daisy", "sunflower", "rose", "tulip", "orchid", "mushroom", "corn",
    "cabbage", "broccoli", "cauliflower", "cucumber", "zucchini", "pumpkin",
    "squash", "strawberry", "pineapple", "banana", "lemon", "orange",
    "granny smith", "fig", "acorn", "rapeseed", "cardoon", "artichoke",
    "ear of corn", "head cabbage", "bell cote"
)

TARGET_LABELS = ("bell pepper", "pepper", "tomato", "potato", "capsicum")

def _label_matches(label, phrases):
    normalized = label.lower().replace("_", " ").replace(",", " ")
    return any(
        re.search(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", normalized)
        for p in phrases
    )

def _crop_model_confidence(image_np, disease_model):
    if disease_model is None:
        return None
    try:
        resized = cv2.resize(image_np, IMG_SIZE).astype(np.float32) / 255.0
        probs = disease_model.predict(np.expand_dims(resized, axis=0), verbose=0)[0]
        return float(np.max(probs))
    except Exception:
        return None

def validate_leaf_image(image_np, disease_model=None):
    if image_np is None or image_np.size == 0:
        return False, "Not a Leaf Image — the image could not be read. Upload a clear leaf photo."

    h, w = image_np.shape[:2]
    if h < 80 or w < 80:
        return False, "Not a Leaf Image — the image is too small. Upload a clear leaf photo."

    crop_confidence = _crop_model_confidence(image_np, disease_model)

    try:
        gray_original = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        scales = [gray_original]
        max_dim = max(gray_original.shape)
        if max_dim < 1400:
            scale = min(1400.0 / max_dim, 3.0)
            if scale > 1.05:
                scales.append(cv2.resize(
                    gray_original,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_CUBIC
                ))

        cascade_names = (
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_profileface.xml",
        )
        for gray in scales:
            gray = cv2.equalizeHist(gray)
            gray_h, gray_w = gray.shape[:2]
            for cascade_name in cascade_names:
                cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
                if cascade.empty():
                    continue
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.08, minNeighbors=4,
                    minSize=(35, 35), flags=cv2.CASCADE_SCALE_IMAGE
                )
                area = float(gray_h * gray_w)
                for (_, _, fw, fh) in faces:
                    if float(fw * fh) / area >= 0.012:
                        return False, "Not a Leaf Image — a human face was detected. Upload only potato, tomato, or bell pepper leaves."
    except Exception:
        pass

    try:
        validator = load_input_validator()
        resized = cv2.resize(image_np, (224, 224)).astype(np.float32)
        batch = preprocess_input(np.expand_dims(resized, axis=0))
        probabilities = validator.predict(batch, verbose=0)
        decoded = decode_predictions(probabilities, top=5)[0]
        labels = [(label.lower().replace("_", " "), float(score)) for _, label, score in decoded]
        top_label, top_score = labels[0]

        person_animal_score = max(
            (score for label, score in labels if _label_matches(label, PERSON_ANIMAL_LABELS)),
            default=0.0
        )
        object_score = max(
            (score for label, score in labels if _label_matches(label, OBJECT_LABELS)),
            default=0.0
        )
        other_plant_score = max(
            (score for label, score in labels if _label_matches(label, OTHER_PLANT_LABELS)),
            default=0.0
        )
        target_evidence = max(
            (score for label, score in labels if _label_matches(label, TARGET_LABELS)),
            default=0.0
        )

        if person_animal_score >= 0.25:
            return False, "Not a Leaf Image — the image appears to contain a person or animal."
        if object_score >= 0.25 or (top_score >= 0.70 and _label_matches(top_label, OBJECT_LABELS)):
            return False, "Not a Leaf Image — the image appears to contain an object or vehicle."

        strong_crop_evidence = crop_confidence is not None and crop_confidence >= 0.80
        if other_plant_score >= 0.45 and not strong_crop_evidence and target_evidence < 0.10:
            return False, "Not a Supported Leaf — upload only potato, tomato, or bell pepper leaves. Other plant leaves are not supported."

        if crop_confidence is not None and crop_confidence < 0.20 and other_plant_score < 0.20:
            return False, "Not a Supported Leaf — upload a clear potato, tomato, or bell pepper leaf."
    except Exception:
        st.warning("Advanced image screening is unavailable; please upload a clear supported leaf image.")

    return True, ""

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found in this folder.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

def split_crop_and_disease(class_label):
    parts = class_label.split(" ", 1)
    return parts[0], (parts[1] if len(parts) > 1 else class_label)

def crop_to_leaf(image_np, padding=20):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    leaf_mask = cv2.inRange(hsv, np.array([10, 30, 20]), np.array([90, 255, 255]))
    kernel = np.ones((7, 7), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_np, False
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return image_np, False
    x, y, w, h = cv2.boundingRect(largest)
    H, W = image_np.shape[:2]
    cropped = image_np[max(0, y - padding):min(H, y + h + padding), max(0, x - padding):min(W, x + w + padding)]
    if cropped.size == 0:
        return image_np, False
    return cropped, True

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
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
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
    <p style="margin:4px 0 0 0; color:#475569;">Upload a leaf image to detect the disease, estimate severity and visualize Grad-CAM explanation.</p>
</div>
""", unsafe_allow_html=True)

model = load_model()
grad_model = build_gradcam_model(model)

if st.session_state.nav_page == "About Project":
    st.subheader("About Project: Multi-Crop Plant Disease Detection and Severity Estimation")
    st.write(
        "This project classifies leaf photographs of **Potato**, **Tomato**, and **Bell Pepper** into eight disease and healthy categories, "
        "estimates approximate affected leaf area using HSV segmentation, and generates Grad-CAM heatmaps to highlight influential image regions."
    )
    st.markdown("### Key Architecture & Workflow")
    st.markdown("""
    - **Model Progression:** Started with a custom CNN on potato leaves, expanded to eight multi-crop classes, compared ResNet50, and adopted **MobileNetV2** (224×224×3 input, GlobalAveragePooling2D, Dense 128 ReLU, Dropout 0.4, 8-class softmax) with backbone freezing followed by fine-tuning the top 30 layers.
    - **Automatic Leaf Cropping & HSV Severity:** Uses HSV colour thresholding and morphological filtering to crop the leaf region and compute an approximate affected-area percentage (`Very Mild < 10%`, `Mild 10%–25%`, `Moderate 25%–50%`, `Severe ≥ 50%`).
    - **Grad-CAM Explainability:** Visualizes relative spatial attention behind each prediction.
    - **Domain Gap Finding:** Controlled PlantVillage validation accuracy (~95.17% for MobileNetV2 v6) is higher than performance on real-world-style PlantDoc test images (32/66 = 48.5% for v6), highlighting lighting, background, and framing differences between lab-style and field-style photos.
    - **Future Scope:** Adding a dedicated input-validation model for unsupported/non-leaf images, more diverse field images, deterministic unaugmented validation splits, and expert-validated severity checks.
    """)
    st.stop()

if st.session_state.nav_page == "Dataset Classes":
    st.subheader("Supported Dataset Classes (8 Classes)")
    for idx, cls in enumerate(CLASS_NAMES, start=1):
        st.write(f"{idx}. **{cls}**")
    st.stop()

upload_anchor = "upload-section"
if st.session_state.nav_page == "Upload Image":
    st.markdown(f'<div id="{upload_anchor}"></div>', unsafe_allow_html=True)
    st.info("Upload a clear Potato, Tomato, or Bell Pepper leaf image below to run classification, HSV severity estimation, and Grad-CAM.")

st.markdown('<div class="card-box"><div class="card-title">1. Upload Leaf Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop an image here or click to browse (JPG, PNG)", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_np_original = np.array(image)
    except Exception:
        st.error("The uploaded file is not a valid image. Please upload a JPG or PNG leaf photo.")
        st.stop()

    is_leaf, validation_message = validate_leaf_image(image_np_original, model)
    if not is_leaf:
        if validation_message.startswith("Not a Supported Leaf"):
            st.error("🚫 Not a Potato/Tomato/Bell Pepper Leaf")
        else:
            st.error("🚫 Not a Leaf Image")
        st.warning(validation_message)
        st.stop()

    with st.spinner("Analyzing leaf image..."):
        image_np, was_cropped = image_np_original, False
        img_resized = cv2.resize(image_np, IMG_SIZE)
        img_array = np.expand_dims(img_resized / 255.0, axis=0)
        predictions = model.predict(img_array, verbose=0)[0]
        pred_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[pred_index]
        crop_name, disease_name = split_crop_and_disease(predicted_class)
        confidence = float(predictions[pred_index]) * 100

        if "Healthy" not in predicted_class:
            _, disease_mask, severity_percent = segment_leaf_and_disease(image_np)
            sev_label = severity_label(severity_percent)
            severity_overlay = make_severity_overlay(img_resized, cv2.resize(disease_mask, IMG_SIZE))
        else:
            severity_percent = 0.0
            sev_label = "Very Mild"
            severity_overlay = img_resized

        gradcam_overlay = make_gradcam_overlay(image_np, grad_model, pred_index)
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
        st.image(image_np, caption="Cropped Leaf" if was_cropped else "Full Image", use_container_width=True)
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