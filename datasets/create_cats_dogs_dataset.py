'''
The CSV file is being created by using the cats-dogs image dataset taken from kaggle. https://www.kaggle.com/c/dogs-vs-cats/data
'''

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
from keras.models import Model
import csv, os
import pandas as pd
import pdb

def get_vgg16():
    # Includes VGG fc layers
    model = VGG16(include_top=True)

    # Pops the output layer( used for classification ) from the model.
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()

    # Restructure model.
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    return model

def get_features():
    model = get_vgg16()
    features = {}
    input_size = 224
    target_size = (input_size, input_size, 3)

    img_dir = '/home/sidharth/Documents/cs848/datasets/dogs-vs-cats/train'

    dataset_csv = '/home/sidharth/Documents/cs848/datasets/dogs_cats.csv'
    fil = open(dataset_csv, 'a')
    csv_writer = csv.writer(fil, delimiter=',')    
    
    for img_file in os.listdir(img_dir):
        # Loads image
        image = load_img(img_dir + '/' + img_file, target_size=target_size)

        # Converts image to a numpy array
        image = img_to_array(image)

        # Reshape image
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # Preprocesses image input for model.
        image = preprocess_input(image)

        img_features = model.predict(image)
        features[img_file] = img_features
        #pdb.set_trace()
        df = pd.DataFrame(img_features)
        df.insert(0, 'label', img_file.split('.')[0])
        df.to_csv(fil, header=False)
        #csv_writer.writerow(img_features)

get_features()
