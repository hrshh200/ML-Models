# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:55:58 2024

@author: User
"""

from fastapi import FastAPI, File, UploadFile
import pickle
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras




app= FastAPI()

model_image = keras.models.load_model("model_CNN.keras")

def preprocessedimage(image):
    image.reshape(100,100,3)
    return image

@app.post('/image_classification')

async def image_pred(file: UploadFile = File(...)):
    contents= await file.read()
    image=Image.open(io.BytesIO(contents))
    preprocessed_image=preprocessedimage(image)
    prediction = model_image.predict(preprocessed_image)
    y_pred = prediction > 0.5

    if(y_pred==0):
        return{"The image contains dog"}
    else:
        return{"The image contains cat"}
    
    