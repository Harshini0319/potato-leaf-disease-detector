import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib
import matplotlib.cm as cm
import os
import json
from datetime import datetime


# ---------------- INPUT VALIDATION: SUPPORTED CROP LEAVES ----------------
# This is a screening gate, not a perfect botanical classifier. The thresholds
# below are intentionally tuned to reduce false rejection of valid target leaves.
# ImageNet is only a general-purpose screening model, not a crop-specific validator.
from tensorflow.keras.applications import MobileNetV2 as ImageNetMobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import re

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
    """Return the strongest supported-crop class probability as screening evidence."""
    if disease_model is None:
        return None
    try:
        resized = cv2.resize(image_np, IMG_SIZE).astype(np.float32) / 255.0
        probs = disease_model.predict(np.expand_dims(resized, axis=0), verbose=0)[0]
        return float(np.max(probs))
    except Exception:
        return None


def validate_leaf_image(image_np, disease_model=None):
    """Reject obvious non-leaf inputs while avoiding brittle color-only rejection of crop leaves."""
    if image_np is None or image_np.size == 0:
        return False, "Not a Leaf Image — the image could not be read. Upload a clear leaf photo."

    h, w = image_np.shape[:2]
    if h < 80 or w < 80:
        return False, "Not a Leaf Image — the image is too small. Upload a clear leaf photo."

    # Do NOT reject an image only because it has little green color. Target leaves can
    # be yellow, brown, diseased, shadowed, or pale; a color threshold caused false rejects.

    crop_confidence = _crop_model_confidence(image_np, disease_model)

    # Face screening: the previous 12% face-area threshold was much too high for normal
    # photos. Try several cascades on original and enlarged grayscale images; a detected
    # medium/large face should stop the workflow before any displayed prediction.
    face_detections = []
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
                    ratio = float(fw * fh) / area
                    # This threshold is 1.2% (not the former 12%).
                    if ratio >= 0.012:
                        face_detections.append((cascade_name, ratio))
        if face_detections:
            return False, "Not a Leaf Image — a human face was detected. Upload only potato, tomato, or bell pepper leaves."
    except Exception:
        # Continue to the independent image-label screening if face detection is unavailable.
        pass

    # ImageNet is a broad screening model, not a crop-specific classifier. Combine its
    # label with the crop classifier's confidence so a weak object label doesn't reject
    # a valid crop leaf. The crop confidence is used only as screening evidence here.
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

        # Strong person/animal evidence takes priority. A small chance of an animal label
        # on a leaf is not enough; the label must be reasonably confident.
        if person_animal_score >= 0.25:
            return False, "Not a Leaf Image — the image appears to contain a person or animal."

        # Do not let a high disease-model score override a clearly recognized object.
        # The 8-class disease model always chooses one of its crop classes, even for a
        # backpack/person photo, so its confidence is not proof that the upload is a leaf.
        if object_score >= 0.25:
            return False, "Not a Leaf Image — the image appears to contain an object or vehicle."
        if top_score >= 0.70 and _label_matches(top_label, OBJECT_LABELS):
            return False, "Not a Leaf Image — the image appears to contain an object or vehicle."

        # Reject a clearly recognized unsupported plant when ImageNet gives strong evidence.
        # Keep a moderate crop-model signal as a tie-breaker only for ambiguous generic labels.
        strong_crop_evidence = crop_confidence is not None and crop_confidence >= 0.80
        if other_plant_score >= 0.45 and not strong_crop_evidence and target_evidence < 0.10:
            return False, "Not a Supported Leaf — upload only potato, tomato, or bell pepper leaves. Other plant leaves are not supported."

        # If the uploaded image strongly matches neither the supported-crop classifier nor
        # any plant-like class, reject only extremely weak crop predictions. Do not apply a
        # global high-confidence threshold: real photos of valid leaves may be low-confidence.
        if crop_confidence is not None and crop_confidence < 0.20 and other_plant_score < 0.20:
            return False, "Not a Supported Leaf — upload a clear potato, tomato, or bell pepper leaf."

    except Exception:
        # If ImageNet weights are not available, face screening still runs. Do not reject
        # every upload just because a general-purpose screening model could not load.
        st.warning("Advanced image screening is unavailable; please upload a clear supported leaf image.")

    return True, ""

# ---------------- CONFIG ----------------
MODEL_PATH = "model_mobilenet_v6.h5"
IMG_SIZE = (224, 224)
HISTORY_DIR = "history"
CLASS_NAMES = ["Pepper Bacterial Spot", "Pepper Healthy", "Potato Early Blight",
               "Potato Healthy", "Potato Late Blight", "Tomato Early Blight",
               "Tomato Healthy", "Tomato Late Blight"]

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
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found in this folder.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

# ---------------- NEW: AUTO-CROP TO LEAF ----------------
def crop_to_leaf(image_np, padding=20):
    """Finds the largest leaf-colored region and crops to it, removing background clutter."""
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_leaf = np.array([10, 30, 20])
    upper_leaf = np.array([90, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    kernel = np.ones((7, 7), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_np, False  # no leaf detected, use original

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:  # too small, likely noise
        return image_np, False

    x, y, w, h = cv2.boundingRect(largest)
    H, W = image_np.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(W, x + w + padding)
    y1 = min(H, y + h + padding)

    cropped = image_np[y0:y1, x0:x1]
    if cropped.size == 0:
        return image_np, False
    return cropped, True

def segment_leaf_and_disease(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_leaf = np.array([10, 30, 20])
    upper_leaf = np.array([90, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    lower_disease = np.array([5, 40, 20])
    upper_disease = np.array([30, 255, 200])
    disease_mask_raw = cv2.inRange(hsv, lower_disease, upper_disease)
    disease_mask = cv2.bitwise_and(disease_mask_raw, leaf_mask)
    leaf_area = np.sum(leaf_mask > 0)
    disease_area = np.sum(disease_mask > 0)
    severity_percent = (disease_area / leaf_area * 100) if leaf_area > 0 else 0.0
    return leaf_mask, disease_mask, severity_percent

def severity_label(p):
    if p < 10: return "Very Mild"
    elif p < 25: return "Mild"
    elif p < 50: return "Moderate"
    else: return "Severe"

def make_severity_overlay(image_np, disease_mask):
    overlay = image_np.copy()
    overlay[disease_mask > 0] = [255, 0, 0]
    return cv2.addWeighted(image_np, 0.6, overlay, 0.4, 0)

def build_gradcam_model(model, input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    first_layer = model.layers[0]

    if hasattr(first_layer, 'layers'):
        # Transfer-learning model (e.g. MobileNetV2): the base model's own
        # output IS already the last conv feature map (since include_top=False)
        base_output = first_layer(inputs)
        conv_output = base_output
        y = base_output
        for layer in model.layers[1:]:
            y = layer(y)
        return tf.keras.Model(inputs=inputs, outputs=[conv_output, y])
    else:
        # Custom CNN: find the last Conv2D layer manually
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
    heatmap = heatmap.numpy()
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
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

st.set_page_config(page_title="Multi-Crop Leaf Disease Detector", page_icon="🌿", layout="wide")
st.title("🌿 Multi-Crop Leaf Disease Detection")
st.write("Upload a Potato, Tomato, or Pepper leaf photo to get real-time disease classification, severity estimation, and treatment recommendations.")

model = load_model()
grad_model = build_gradcam_model(model)

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_np_original = np.array(image)
    except Exception:
        st.error("The uploaded file is not a valid image. Please upload a JPG or PNG leaf photo.")
        st.stop()

    # Validate the image BEFORE the disease classifier runs.
    is_leaf, validation_message = validate_leaf_image(image_np_original, model)
    if not is_leaf:
        if validation_message.startswith("Not a Supported Leaf"):
            st.error("🚫 Not a Potato/Tomato/Bell Pepper Leaf")
        else:
            st.error("🚫 Not a Leaf Image")
        st.warning(validation_message)
        st.info("Please upload only potato, tomato, or bell pepper leaves. No disease prediction, severity, Grad-CAM, or treatment recommendation was generated for this image.")
        st.stop()

    with st.spinner("Analyzing leaf image..."):
        # Auto-crop only after the input passes validation.
        image_np, was_cropped = crop_to_leaf(image_np_original)

        img_resized = cv2.resize(image_np, IMG_SIZE)
        img_array = np.expand_dims(img_resized / 255.0, axis=0)
        predictions = model.predict(img_array, verbose=0)[0]
        pred_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[pred_index]
        confidence = float(predictions[pred_index]) * 100

        if "Healthy" not in predicted_class:
            leaf_mask, disease_mask, severity_percent = segment_leaf_and_disease(image_np)
            sev_label = severity_label(severity_percent)
            severity_overlay = make_severity_overlay(cv2.resize(image_np, IMG_SIZE), cv2.resize(disease_mask, IMG_SIZE))
        else:
            severity_percent = 0.0
            sev_label = "Very Mild"
            severity_overlay = img_resized

        gradcam_overlay = make_gradcam_overlay(image_np, grad_model, pred_index)
        recommendation = RECOMMENDATIONS[predicted_class][sev_label]

        result = {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 1),
            "severity_percent": round(severity_percent, 1),
            "severity_label": sev_label,
            "recommendation": recommendation
        }
        save_to_history(img_resized, severity_overlay, gradcam_overlay, result)

    st.subheader("Latest Result")

    if was_cropped:
        st.caption("The image below was automatically cropped to focus on the leaf before analysis.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(image, caption="Original Upload", use_container_width=True)
    with col2:
        st.image(image_np, caption="Cropped to Leaf" if was_cropped else "No Crop Applied", use_container_width=True)
    with col3:
        st.image(severity_overlay, caption="Diseased Region", use_container_width=True)
    with col4:
        st.image(gradcam_overlay, caption="Grad-CAM Focus", use_container_width=True)

    st.markdown("---")
    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric("Predicted Disease", predicted_class)
        st.metric("Confidence", f"{confidence:.1f}%")
    with result_col2:
        st.metric("Severity", f"{severity_percent:.1f}%", sev_label)

    if confidence < 60:
        st.warning("⚠️ Low confidence prediction. For better accuracy, try a closer, well-lit photo of a single leaf against a plain background.")

    if "Healthy" in predicted_class:
        st.success(f"✅ {recommendation}")
    elif sev_label in ["Very Mild", "Mild"]:
        st.warning(f"⚠️ {recommendation}")
    else:
        st.error(f"🚨 {recommendation}")

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