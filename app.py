import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import joblib
import time

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:white;
}

.subtitle{
    color:#B0B0B0;
    font-size:18px;
}

.prediction{
    background:#1c1c1c;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

.metric{
    background:#161616;
    padding:20px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    model = keras.models.load_model("cnn_model.keras")

    return model


@st.cache_resource
def load_classes():

    return joblib.load("class_names.pkl")


model = load_model()

class_names = load_classes()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🧠 CNN Image Classifier")

st.sidebar.markdown("---")

st.sidebar.info(
"""
Upload an image and let the CNN model classify it.
"""
)

st.sidebar.markdown("### Model")

st.sidebar.write("TensorFlow / Keras CNN")

st.sidebar.markdown("### Image Size")

st.sidebar.write("256 x 256")

# -----------------------------
# Header
# -----------------------------
st.markdown('<p class="title">AI Image Classifier</p>',unsafe_allow_html=True)

st.markdown(
'<p class="subtitle">Upload an image and let the CNN model predict its class.</p>',
unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1,col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        if st.button("Predict"):

            start = time.perf_counter()

            img = image.resize((256,256))

            img = np.array(img)

            if img.shape[-1] == 4:
                img = img[:,:,:3]

            img = img.astype("float32") / 255.0

            img = np.expand_dims(img,axis=0)

            prediction = model.predict(img,verbose=0)

            inference_time = (time.perf_counter()-start)*1000

            st.success("Prediction Completed")

            # Binary Classification
            if prediction.shape[1] == 1:

                probability = float(prediction[0][0])

                if probability > 0.5:

                    predicted_class = class_names[1]

                    confidence = probability

                else:

                    predicted_class = class_names[0]

                    confidence = 1-probability

                st.markdown("## Prediction")

                st.markdown(
                    f"<div class='prediction'><h2>{predicted_class}</h2></div>",
                    unsafe_allow_html=True
                )

                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )

                st.progress(float(confidence))

            # Multi Class
            else:

                predicted_index = np.argmax(prediction)

                predicted_class = class_names[predicted_index]

                confidence = prediction[0][predicted_index]

                st.markdown("## Prediction")

                st.markdown(
                    f"<div class='prediction'><h2>{predicted_class}</h2></div>",
                    unsafe_allow_html=True
                )

                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )

                st.progress(float(confidence))

                st.markdown("### Class Probabilities")

                for i,name in enumerate(class_names):

                    st.write(name)

                    st.progress(float(prediction[0][i]))

                    st.write(f"{prediction[0][i]*100:.2f}%")

            st.divider()

            c1,c2,c3 = st.columns(3)

            c1.metric(
                "Image Width",
                image.size[0]
            )

            c2.metric(
                "Image Height",
                image.size[1]
            )

            c3.metric(
                "Inference Time",
                f"{inference_time:.2f} ms"
            )

st.divider()

st.markdown(
"""
<center>

Built with ❤️ using TensorFlow, Keras and Streamlit

</center>
""",
unsafe_allow_html=True
)