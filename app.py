import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.cm as cm
import os
import json
from datetime import datetime

# ---------------- CONFIG ----------------
MODEL_PATH = "model_multi_crop_cnn_v2.h5"
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
    jet = cm.get_cmap("jet")
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
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    with st.spinner("Analyzing..."):
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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(severity_overlay, caption="Diseased Region Highlighted", use_container_width=True)
    with col3:
        st.image(gradcam_overlay, caption="Grad-CAM (Model Focus Area)", use_container_width=True)

    st.markdown("---")
    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric("Predicted Disease", predicted_class)
        st.metric("Confidence", f"{confidence:.1f}%")
    with result_col2:
        st.metric("Severity", f"{severity_percent:.1f}%", sev_label)

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