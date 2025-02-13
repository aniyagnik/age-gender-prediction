from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import json
import os
from fastapi.middleware.cors import CORSMiddleware

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU, use CPU only

tf.config.set_visible_devices([], 'GPU')  # Explicitly disable GPU usage

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Load model architecture from JSON
with open("age_gender_model/config.json", "r") as f:
    model_config = json.load(f)

model = tf.keras.models.model_from_json(json.dumps(model_config))

# Load model weights
model.load_weights("age_gender_model/model.weights.h5")

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Function to preprocess image
def preprocess_image(image: Image.Image):
    image = np.array(image)  # Convert to NumPy array
    image = cv2.resize(image, (128, 128))  # Resize to match model expected size
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale if needed
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (for grayscale)
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# API Endpoint for prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = preprocess_image(image)

    # Perform prediction
    prediction = model.predict(image)

    # Ensure model output is a list of two arrays
    if isinstance(prediction, list) and len(prediction) == 2:
        pred_gender = prediction[0][0][0]  # Extract probability
        pred_age = prediction[1][0][0]  # Extract age

        gender = "Male" if pred_gender < 0.5 else "Female"
        age = int(round(pred_age))  # Convert to integer

        return {"age": age, "gender": gender}
    else:
        return {"error": "Unexpected model output format"}

# Run the server (only needed for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
