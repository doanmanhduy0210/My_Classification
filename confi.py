import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
root_dir = '/'
batch_size = 8
image_size = (112,112)
input_shape = (112,112,3)
epochs = 1
test_size = 0.2

transfroms = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1)

models_path = 'my_weights.h5'
root_dir = '/home/minglee/Documents/aiProjects/dataset/dogs-vs-cats/Traffic_sign_classification/data'
