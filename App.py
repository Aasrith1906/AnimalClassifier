from flask import Flask , render_template , redirect , url_for , session ,flash , request 
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap 


from wtforms import FileField  , SubmitField
from wtforms.validators import DataRequired 

from keras.models import Sequential
import keras

from matplotlib import image

from werkzeug.utils import secure_filename

import Model 

import numpy as np

import os

import cv2

app = Flask(__name__)

bootstrap = Bootstrap(app)

app.config['SECRET_KEY'] = 'ABASCASFD'

class PhotoForm(FlaskForm):
    photo = FileField()
    submit = SubmitField()

global cnn_model


def load_model(jsonfile:str , h5file:str):
    json_file = open(jsonfile , 'r')
    data = json_file.read()
    loaded_model = keras.models.model_from_json(data)
    loaded_model.load_weights(h5file)
    optimizer = keras.optimizers.SGD(lr = 0.1)
    loaded_model.compile(optimizer=optimizer , loss = 'categorical_crossentropy',
                 metrics=['accuracy'])

@app.route('/' , methods = ['GET' , 'POST'])
def index():

    prediction = ""

    form = PhotoForm()

    if form.validate_on_submit():
        
        f = form.photo.data
        filename = secure_filename(f.filename)
        f.save(filename)

        img = image.imread(filename)

        image_resized = cv2.resize(img , (100,100))
        image_final = cv2.cvtColor(image_resized , cv2.COLOR_BGR2RGB)

        image_array = np.array([image_final])

        loaded_model = load_model("model.json" , "model.h5")

        prediction = loaded_model.predict(image_array)

        os.remove(image_path)
        
        return render_template('index.html' , prediction = prediction , form = form)

    return render_template('index.html' , prediction = prediction,form = form)

if __name__ == '__main__':

    app.run(debug=True)