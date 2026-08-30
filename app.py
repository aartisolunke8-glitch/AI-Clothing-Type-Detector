import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="AI Clothing Detector",
    page_icon="👕",
    layout="centered"
)

# Title
st.title("👕 AI Clothing Type Detector")
st.write("Upload a clothing image and let AI identify its category.")

# Clothing categories
class_names = [
    "Tshirts",
    "Shirts",
    "Tops",
    "Casual Shoes",
    "Sports Shoes",
    "Sandals",
    "Handbags",
    "Dresses"
]

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("clothing_model.h5")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "📤 Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Clothing Image",
        use_container_width=True
    )

    # Prepare image
    image_resized = image.resize((160, 160))
    image_array = np.array(image_resized, dtype="float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(
        f"Prediction: {class_names[predicted_index]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

st.divider()

st.subheader("✨ About the Project")

st.write(
    "This AI-based project uses a deep learning image-classification "
    "model trained on real fashion product images. It can identify "
    "six different clothing categories from an uploaded image."
)

st.subheader("🎯 Supported Categories")

st.write(
    "👕 Tshirts  |  👔 Shirts  |  👟 Casual Shoes  |  "
    "👟 Sports Shoes  |  👜 Handbags  |  👚 Tops"
)

st.caption("AI Clothing Type Detector • Internship Final Project")
