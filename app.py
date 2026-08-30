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

# Clothing categories
class_names = [
    "T-shirt / Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# Title
st.title("👕 AI Clothing Type Detector")
st.write("Upload a clothing image and let AI identify its category.")

# Load model
model = tf.keras.models.load_model("clothing_model.keras")

# Upload image
uploaded_file = st.file_uploader(
    "📤 Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(
        image,
        caption="Uploaded Clothing Image",
        use_container_width=True
    )

    # Prepare image
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28)

    # Prediction
    prediction = model.predict(image_array, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(
        f"Prediction: {class_names[predicted_class]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

st.divider()

st.subheader("ℹ️ About the Project")
st.write(
    "This project uses an AI image-classification model "
    "trained on the Fashion-MNIST dataset to identify "
    "different types of clothing."
)
