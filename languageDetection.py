#LANGUAGE DETECTION API

#INSTALL DEPENDENCIES
# pip install fastapi uvicorn pillow tensorflow requests

# COMMAND TO RUN THE API
# Paste the following line in the terminal: uvicorn languageDetection:app --host 0.0.0.0 --port 5000

# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load THE edge impulse model
model = tf.keras.models.load_model('model.h5')

# Define a route for image classification
@app.post('/classify/')
async def classify(image: UploadFile):
    try:
        # Read image
        img = Image.open(image.file)

        # Convert image to Grayscale
        img = img.convert('L')  

        # Resize the image
        img = img.resize((96, 96))  

        # Convert the image to a numpy array
        image_array = np.array(img)

        # Prediction using the model
        predictions = model.predict(np.expand_dims(image_array, axis=0))

        # Predictions to labels
        labels = ['English', 'Hindi', 'Arabic', 'Chinese']

        # Check if confidence is above 0.5
        confidence = float(predictions.max())
        if confidence >= 0.5:
            prediction_label = labels[np.argmax(predictions)]
        else:
            prediction_label = 'Uncertain'

        # JSON response
        response = {
            "prediction": prediction_label,
            # Uncomment the line below to check print the confidence level
            #"confidence": confidence  
        }

        return response
    except Exception as e:
        return {"error": str(e)}

