import numpy as np 
import os
import pandas as pd 
from scipy.io import wavfile

import librosa
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from tensorflow.keras import regularizers, activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from datetime import datetime 

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split



def loading_data() : 

    us8k_df = pd.read_pickle("./pkl/us8k_augmented_df.pkl")
    df = us8k_df.drop(['fold'],axis=1)
    X = np.stack(df.melspectrogram.to_numpy())

    X_dim = (128,128,1)
    X = X.reshape(X.shape[0], *X_dim)
    Y = np.array(df['label'])
    Y = to_categorical(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,shuffle=True,stratify = Y)
    X_val, X_test, Y_val,Y_test = train_test_split(X_test,Y_test,test_size=0.5,shuffle=True,stratify = Y_test)

    return X_train,Y_train,X_val,Y_val,X_test,Y_test


def init_model():
    model1 = Sequential()
    
    #layer-1
    model1.add(Conv2D(filters=24, kernel_size=5, input_shape=(128, 128, 1),
                      kernel_regularizer=regularizers.l2(1e-3)))
    model1.add(MaxPooling2D(pool_size=(3,3), strides=3))
    model1.add(Activation(activations.relu))
    
    #layer-2
    model1.add(Conv2D(filters=36, kernel_size=4, padding='valid', kernel_regularizer=regularizers.l2(1e-3)))
    model1.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model1.add(Activation(activations.relu))
    
    #layer-3
    model1.add(Conv2D(filters=48, kernel_size=3, padding='valid'))
    model1.add(Activation(activations.relu))
    
    model1.add(GlobalAveragePooling2D())
    
    #layer-4 (1st dense layer)
    model1.add(Dense(60, activation='relu'))
    model1.add(Dropout(0.5))
    
    #layer-5 (2nd dense layer)
    model1.add(Dense(10, activation='softmax'))

    
    # compile
    model1.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    return model1   







if __name__ == "__main__":

    X_train,Y_train,X_val,Y_val,X_test,Y_test = loading_data()
    print(Y_train.shape)
    model = init_model()
    model.summary()

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)
    save_best = tf.keras.callbacks.ModelCheckpoint(filepath = "logs/checkpoints/", save_weights_only = True,
                                                    monitor = "val_accuracy", mode = "max", save_best_only = True)

    initial_epochs = 100
    num_batch_size = 32
    model_fit = model.fit(X_train,Y_train, epochs=initial_epochs,validation_data=(X_val,Y_val),
                            batch_size=num_batch_size, callbacks = [tensorboard_callback, save_best])


    model.save('logs/soundClass_augmented_model.h5')                                          
