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


# ---------------- INPUT VALIDATION: LEAF VS NON-LEAF ----------------
# This extra ImageNet classifier helps reject obvious unrelated objects
# (people, animals, buses, bags, buildings, etc.) before disease prediction.
# It is a general-purpose object classifier, not a dedicated leaf detector;
# test carefully with the leaf photos used in your project.
from tensorflow.keras.applications import MobileNetV2 as ImageNetMobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

@st.cache_resource
def load_input_validator():
    # Downloads ImageNet weights on first run if they are not cached.
    return ImageNetMobileNetV2(weights="imagenet", include_top=True)

# ImageNet labels that are commonly botanical / plant-like. This is deliberately
# broad to reduce false rejection of leaves, but cannot guarantee perfect detection.
PLANT_LABEL_KEYWORDS = (
    "leaf", "plant", "pot", "bell pepper", "cabbage", "cauliflower",
    "broccoli", "artichoke", "cardoon", "corn", "ear", "rapeseed",
    "daisy", "flower", "fungus", "mushroom", "acorn", "buckeye",
    "cucumber", "zucchini", "spaghetti squash", "butternut squash",
    "pumpkin", "strawberry", "fig", "pineapple", "banana", "lemon",
    "orange", "granny smith", "head cabbage", "greenhouse"
)

def validate_leaf_image(image_np):
    """Return (accepted, reason). General ImageNet screening + color sanity check."""
    if image_np is None or image_np.size == 0:
        return False, "The image could not be read. Please upload a JPG or PNG leaf photo."
    h, w = image_np.shape[:2]
    if h < 80 or w < 80:
        return False, "The image is too small. Please upload a clearer leaf photo."

    # First reject obvious human faces.
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(max(35, w // 10), max(35, h // 10))
        )
        if len(faces) > 0:
            return False, "A human face was detected. Please upload a potato, tomato, or pepper leaf image."
    except Exception:
        pass

    # Classify the whole image with ImageNet before passing it to the disease model.
    # If its top predictions are clearly unrelated to plants, reject it.
    try:
        validator = load_input_validator()
        small = cv2.resize(image_np, (224, 224)).astype(np.float32)
        batch = np.expand_dims(small, axis=0)
        scores = validator.predict(preprocess_input(batch), verbose=0)
        top_labels = [label.lower().replace("_", " ") for _, label, _ in decode_predictions(scores, top=5)[0]]
        if not any(any(keyword in label for keyword in PLANT_LABEL_KEYWORDS) for label in top_labels):
            return False, (
                "This image does not appear to contain a plant or leaf. "
                "Please upload a clear potato, tomato, or pepper leaf photo."
            )
    except Exception as exc:
        # Do not silently claim reliable non-leaf detection if the validator cannot run.
        return False, (
            "The image validation model could not run. Please try again after the app "
            "finishes loading, or contact the project administrator."
        )

    # Retain a basic color/region check as a second sanity check.
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(
        hsv, np.array([20, 25, 25], dtype=np.uint8),
        np.array([100, 255, 255], dtype=np.uint8)
    )
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_ratio = float(np.count_nonzero(green_mask)) / float(h * w)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_ratio = (
        float(cv2.contourArea(max(contours, key=cv2.contourArea))) / float(h * w)
        if contours else 0.0
    )
    if green_ratio < 0.025 or largest_ratio < 0.015:
        return False, (
            "A plant-leaf region could not be identified confidently. Please upload a close, "
            "well-lit photo of one potato, tomato, or pepper leaf against a simple background."
        )
    return True, ""

# ---------------- CONFIG ----------------
MODEL_PATH = "model_mobilenet_v6.h5"
IMG_SIZE = (224, 224)
HISTORY_DIR = "history"
CLASS_NAMES = ["Pepper Bacterial Spot", "Pepper Healthy", "Potato Early Blight",
               "Potato Healthy", "Potato Late Blight", "Tomato Early Blight",
               "Tomato Healthy", "Tomato Late Blight"]
# Predictions below this threshold are treated as uncertain, not as a reliable diagnosis.
# This reduces overconfident-looking advice, but does not repair a model trained with
# incorrect class ordering or insufficient data.
CONFIDENCE_THRESHOLD = 60.0

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
    is_leaf, validation_message = validate_leaf_image(image_np_original)
    if not is_leaf:
        st.error("🚫 Not a Leaf Image")
        st.warning(validation_message)
        st.info("Please upload a clear potato, tomato, or pepper leaf image. No disease prediction, severity, or Grad-CAM was generated.")
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
        top_indices = np.argsort(predictions)[::-1][:3]
        top_predictions = [
            (CLASS_NAMES[int(i)], float(predictions[int(i)]) * 100)
            for i in top_indices
        ]
        prediction_is_reliable = confidence >= CONFIDENCE_THRESHOLD

        if "Healthy" not in predicted_class:
            leaf_mask, disease_mask, severity_percent = segment_leaf_and_disease(image_np)
            sev_label = severity_label(severity_percent)
            severity_overlay = make_severity_overlay(cv2.resize(image_np, IMG_SIZE), cv2.resize(disease_mask, IMG_SIZE))
        else:
            severity_percent = 0.0
            sev_label = "Very Mild"
            severity_overlay = img_resized

        gradcam_overlay = make_gradcam_overlay(image_np, grad_model, pred_index)
        if prediction_is_reliable:
            recommendation = RECOMMENDATIONS[predicted_class][sev_label]
        else:
            recommendation = (
                "No treatment recommendation shown because the model confidence is below "
                f"{CONFIDENCE_THRESHOLD:.0f}%. Retake a clear, close-up photo and verify the "
                "crop and disease with a reliable source."
            )

        result = {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 1),
            "top_predictions": [
                {"class": name, "confidence": round(score, 1)}
                for name, score in top_predictions
            ],
            "prediction_status": "tentative" if not prediction_is_reliable else "model_prediction",
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
        st.metric("Model's top prediction", predicted_class)
        st.metric("Confidence", f"{confidence:.1f}%")
    with result_col2:
        if prediction_is_reliable:
            st.metric("Estimated severity", f"{severity_percent:.1f}%", sev_label)
        else:
            st.metric("Estimated severity", "Not reported", "Prediction uncertain")

    st.subheader("Top 3 model predictions")
    for class_name, class_score in top_predictions:
        st.write(f"- **{class_name}** — {class_score:.1f}%")

    if not prediction_is_reliable:
        st.warning(
            f"⚠️ Uncertain result: the top prediction is only {confidence:.1f}% "
            f"(required threshold: {CONFIDENCE_THRESHOLD:.0f}%). Do not rely on this as a "
            "confirmed disease or crop identification. No treatment advice is provided."
        )
    elif "Healthy" in predicted_class:
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