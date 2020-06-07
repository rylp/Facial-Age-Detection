# -*- coding: utf-8 -*-

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__)

print(os.getcwd())

labels = ['Middle','Old','Young']

# Keras
from keras.models import load_model
from keras.preprocessing import image 
from keras.models import model_from_json
from keras.optimizers import SGD

MODEL_PATH = 'C:/Users/rohan/Desktop/Work/Age_detection_dataset/App/model/model.json'
MODEL_PATH2 = 'C:/Users/rohan/Desktop/Work/Age_detection_dataset/App/model/model.h5' 

# opening and store file in a variable

json_file = open('model/model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights('model/model.h5')
print("Loaded Model from disk")

opt = SGD(lr=0.01)

loaded_model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

loaded_model._make_predict_function()  

def model_predict(img_path,loaded_model):
    images=[]
    img = cv2.imread(img_path)
    img = cv2.resize(img , (64,64))
    
    images.append(img)
    
    images = np.array(images, dtype="float") / 255.0
    
    pred=loaded_model.predict(images[0])
    
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, loaded_model)
        
        i=preds.argmax(axis=1)
        vals=np.amax(preds,axis=1)
        perc_vals = vals*100
        perc_vals_rounded = perc_vals.round(2)
        
        label_img = labels[i]
        
        result = label_img+": "+str(perc_vals_rounded)
        
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
