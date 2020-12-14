import os
import numpy
import math
import random
import pandas as pd

from PIL import Image
from keras.preprocessing import image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import SGD
import keras
from keras import metrics
from keras import optimizers
from keras import regularizers
from keras.preprocessing import image

from tensorflow.keras.models import load_model
from keras.models import Model, load_model, Sequential
from keras import layers
from keras import models
from keras import callbacks
from keras import losses
from keras import regularizers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from keras.layers import Activation, Dense
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.utils import to_categorical
from keras.layers  import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
import confi


def print_of_folder_paths(paths):
    check_dir = os.listdir(paths) # tìm và duyệt tất cả các tệp trong thư mục cuối cùng của đường daanx
    cnt = 0
    index = range(10)
    for i in index:
        print(check_dir[i])

def Convert_csv(folder_paths):

    filenames = os.listdir(folder_paths) # như trên đã nói
    label_list = []
    for filename in filenames:
        label = filename.split('.')[0]  # cắt ra chữ dog or cat ở cuối path
        if label == 'dog':
            label_list.append('dog')
        else:
            label_list.append('cat')

    return pd.DataFrame({'filename': filenames,'label': label_list})

def print_sample(filename):
    sample = random.choice(filename) # chọn ngẫu nhiên một ảnh trong filename
    test_img = image.load_img(os.path.join(paths_train,sample))
    plt.imshow(test_img)

def Model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape= confi.input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def  Optimize(model):
    sgd = optimizers.SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov= True)
    #  chọn đạo hàm để tối ưu , chon learning rate start, chọn decay để  stop, momentum là động lượng vượt qua điểm local minimun

    model.compile(optimizer = sgd,
                    loss = 'binary_crossentropy',   # chọn vì khi generator data chỉ có 1 class , nên ko xài được binary crossentropy
                    metrics=['accuracy']) #  chọn loại để giám sát mô hình
    return model

def Callbacks():
    earlystop = EarlyStopping(patience=10) # pp dừng đạo hàm

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', # adjust learning rate
                                  factor=0.1,       # giảm đạo hàm xuống bn lần
                                  patience = 4,   # số epoch chọn khi model ko thay đổi performance
                                  min_lr=0.00001)  # min của đạo hàm

    return [reduce_lr,earlystop]




if __name__ == '__main__':

#_____ paths of dataset __________
    paths_train = os.path.join(confi.root_dir,'train')
    paths_test  = os.path.join(confi.root_dir,'test')
    csv_labels =  os.path.join(confi.root_dir,'sampleSubmission.csv')

#______load paths to csv file ________
    df_train = Convert_csv(paths_train)
#______ split dataset _________________
    train_df_split, valid_df_split = train_test_split(df_train, test_size= confi.test_size, random_state=42, shuffle = True)
    train_df_split = train_df_split.reset_index(drop=True)
    valid_df_split = valid_df_split.reset_index(drop=True)
#_________ dataloader + augmentation ________________

    train_generator = confi.transfroms.flow_from_dataframe(
                                dataframe = train_df_split,  # day la datafram shape = (20000,2)
                                directory = paths_train,  # day la path_dirictory
                                x_col='filename',   # gép tên file vào path_train
                                y_col='label',   # gép cột label (count label)
                                target_size= confi.image_size,  # mục tiêu reshape ảnh thành
                                class_mode='binary', # vì loss function : categorical crossentropy
                                batch_size= confi.batch_size,
                                shuffle=True)
    valid_generator = confi.transfroms.flow_from_dataframe(
                                dataframe = valid_df_split,  # day la datafram shape = (20000,2)
                                directory = paths_train,  # day la path_dirictory
                                x_col='filename',   # gép tên file vào path_train
                                y_col='label',   # gép cột label (count label)
                                target_size= confi.image_size,  # mục tiêu reshape ảnh thành
                                class_mode='binary', # vì loss function : categorical crossentropy
                                batch_size= confi.batch_size,
                                shuffle=True)

#________ load models _______________
    model = Model()
    print(model.summary())
    model = Optimize(model)
    model_path = os.path.join(confi.root_dir,confi.models_path)
    if os.path.exists(model_path):
        print('load model path:{} -> status: {}'.format(model_path,os.path.exists(model_path)))
        model.save_weights(model_path)
# ____________training models ________________
    history = model.fit_generator(
        generator = train_generator,
        epochs = confi.epochs,
        validation_data = valid_generator,
        validation_steps = len(valid_generator),
        steps_per_epoch = len(train_generator),
        callbacks = Callbacks()
    )

    model.save_weights(os.path.join(confi.root_dir,confi.models_path))
    model = model.load_weights(os.path.join(confi.root_dir,confi.models_path))



# convert the history.history dict to a pandas DataFrame:
    df_history_models = pd.DataFrame(history.history)
# or save to csv:
    hist_csv_file = os.path.join(confi.root_dir,'history.csv')
    with open(hist_csv_file, mode='w') as f:
        df_history_models.to_csv(f)
