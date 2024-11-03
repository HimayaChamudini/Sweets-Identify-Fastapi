from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import io

# Load your trained model
model = tf.keras.models.load_model('improved_model.keras')

# Define class names based on the categories in your dataset
class_names = ['asmi', 'helapa', 'mun_kawum', 'undu_walalu']

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this based on your needs; "*" allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint for prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image = Image.open(io.BytesIO(await file.read()))

        # Preprocess the image: resize and normalize
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the sweet category
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]

        # Prepare the response with the expected key 'prediction'
        response = {"prediction": predicted_class}

        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
