import os

class Model():

    def __init__(self , model):

        self.model = model

        self.get_list_classes()
        
        self.generate_dict()

    def get_list_classes(self):

        self.list_classes = []

        for root , dirs , files in os.walk(r'E:\Code\AnimalClassifierCNN\AnimalClassifier\images'):
    
            for folder in dirs:
                
                self.list_classes.append(folder)

    def generate_dict(self):

        self.class_dict = dict()

        for i in range(len(self.list_classes)):
            
            self.class_dict[self.list_classes[i]] = i 


    def get_class_name(self , class_num):
    
        for c in self.class_dict.keys():
            
            if self.class_dict[c] == class_num:
                
                return c
    

    def Predict(self , image):

        prediction = self.model.predict(image)[0]
    
        max_prob = 0
        max_index = 0
        
        for i in range(len(prediction)):
            
            if prediction[i] > max_prob:
                
                max_prob = prediction[i]
                max_index = i
                
        
        return str(self.get_class_name(max_index))