# backend/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import os
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

app = FastAPI()

# CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your custom classifier model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vehicle_classifier", "vehicle_classifier_custom_cnn.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Load VGG16 base model (used for feature extraction)
vgg_model = VGG16(include_top=False, input_shape=(224, 224, 3))
vgg_model.trainable = False  # Just to be safe, disable training

class_names = ["Car", "Truck", "Cyclist"]  # Update according to training labels

@app.get("/")
def read_root():
    return {"message": "Tolling API is running"}

from fastapi.responses import JSONResponse

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)

        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return {
            "class": predicted_class,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "class": None, "confidence": None}
        )


# To run:
# uvicorn main:app --reload
