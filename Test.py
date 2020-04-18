import keras
import numpy as np

def load_model(jsonfile:str , h5file:str):
    json_file = open(jsonfile , 'r')
    data = json_file.read()
    loaded_model = keras.models.model_from_json(data)
    loaded_model.load_weights(h5file)
    optimizer = keras.optimizers.SGD(lr = 0.1)
    loaded_model.compile(optimizer=optimizer , loss = 'categorical_crossentropy',metrics=['accuracy'])

    return loaded_model

def get_list_classes():

        list_classes = []

        for root , dirs , files in os.walk(r'E:\Code\AnimalClassifierCNN\AnimalClassifier\images'):
    
            for folder in dirs:
                
                list_classes.append(folder)
        
        return list_classes


for i in range(2):
    cnn_model = load_model("model.json" , "model.h5")
    cnn_model._make_predict_function()
    cnn_model.predict(np.array([np.zeros((100,100,3))]))