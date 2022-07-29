import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from PIL import Image
from pickle import load
import cv2
import argparse
import time
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from keras.applications.xception import Xception
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model

#Initialize the flask App
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['GET', 'POST'])
def predict():
    f = request.files['file']
    # time.sleep(5)

    img_path= f.filename
    max_length = 32
    tokenizer = load(open("tokenizer.p","rb"))
    model = load_model('model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")
    #Feature Extraction
    try:
        image = Image.open(img_path)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    photo = xception_model.predict(image)

    #Caption Generation
    description = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([description])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word=None
        for x, index in tokenizer.word_index.items():
            if index==pred:
                word=x
                break
        if word is None:
            break
        description += ' ' + word
        if word == 'end':
            break
    temp = description.split()
    data1 = 'start'
    data2 = 'end'
    for word in temp:
        if ((word == data1) or (word==data2)):
            temp.remove(word)
    result = ' '.join(temp)

    filename = secure_filename(f.filename)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    flash(result)
    
    return render_template('index.html', filename=filename)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run(debug=True)