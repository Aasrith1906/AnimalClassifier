from flask import Flask , render_template , redirect , url_for , session ,flash , request 
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap 


from wtforms import FileField
from wtforms.validators import DataRequired 

from keras.models import Sequential
import keras

from werkzeug.utils import secure_filename

import Model 

app = Flask(__name__)

bootstrap = Bootstrap(app)

app.config['SECRET_KEY'] = 'ABASCASFD'

class PhotoForm(FlaskForm):
    photo = FileField()

global cnn_model

def load_model():

    optimizer = keras.optimizers.SGD(lr = 0.1)
    json_file = open("model.json" , 'r')
    data = json_file.read()
    loaded_model = keras.models.model_from_json(data)
    loaded_model.load_weights("model.h5")
    loaded_model.compile(optimizer=optimizer , loss = 'categorical_crossentropy',
                metrics=['accuracy'])

    return loaded_model

    


@app.route('/' , methods = ['GET' , 'POST'])
def index():

    prediction = ""

    form = PhotoForm()

    return render_template('index.html' , prediction = prediction,form = form)

if __name__ == '__main__':

    loaded_model = load_model()
    cnn_model = Model.Model(loaded_model)

    app.run(debug=True)