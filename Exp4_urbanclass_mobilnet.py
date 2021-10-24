import numpy as np 
import os
import pandas as pd 
from scipy.io import wavfile


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
import cv2







def show_results(modele):


    fig = plt.figure()
    plt.figure(figsize=(15,5))

    plt.subplot(121)
    plt.plot(modele.history['accuracy'])
    plt.plot(modele.history['val_accuracy'])
    plt.grid(linestyle='--')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.subplot(122)
    plt.plot(modele.history['loss'])
    plt.plot(modele.history['val_loss'])
    plt.grid(linestyle='--')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.savefig('result3/resultVal.png')
    
    
    loss, accuracy = model.evaluate(X_test,Y_test)
    loss1, accuracy1 = model.evaluate(X_train,Y_train)
    
    file = open('result3/result.txt','a')
    file.write('\n=========TRANSFERT LEARNING =========\n')
    file.write('\nMax validation accuracy: %.4f %%' % (np.max(modele.history['val_accuracy']) * 100))
    file.write('\nMin validation loss: %.5f' % np.min(modele.history['val_loss']))
    
  
    file.write('\nTest evaluate accuracy: %f ' %accuracy)
    file.write('\nTest evaluate loss: %f ' %loss)
    
    file.write('\nTrain evaluate accuracy: %f ' %accuracy1)
    file.write('\nTrain evaluate loss: %f ' %loss1)




if __name__ == "__main__":
    us8k_df = pd.read_pickle("./pkl/us8k_df.pkl")
    
    
    df = us8k_df.drop(['fold'],axis=1)
    X = np.stack(df.melspectrogram.to_numpy())
    X_dim = (128,128,1)
    X = X.reshape(X.shape[0], *X_dim)
    Y = np.array(df['label'])
    Y = to_categorical(Y)
    
    X_new = np.zeros((8732,128,128,3))
    
    for i in range(len(X)):
        X_new[i]=cv2.cvtColor(X[i], cv2.COLOR_GRAY2RGB)
    
    X=X_new
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,shuffle=True,stratify = Y)
    X_val, X_test, Y_val,Y_test = train_test_split(X_test,Y_test,test_size=0.5,shuffle=True,stratify = Y_test)
    
    
    #def du modèle 
    
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
    
    IMG_SHAPE = (128,128,3)

    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    
    prediction_layer = tf.keras.layers.Dense(10)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    
    
    # model : MobileNet puis du dropout et une couche dense pour la prédiction
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    x = rescale(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    
    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    initial_epochs = 100
    num_batch_size = 32
    
    
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)
    save_best = tf.keras.callbacks.ModelCheckpoint(filepath = "logs/checkpoints/", save_weights_only = True,
                                                   monitor = "val_accuracy", mode = "max", save_best_only = True)
    
    
    model_fit = model.fit(X_train,Y_train, epochs=initial_epochs,
                          validation_data=(X_val,Y_val),
                          batch_size=num_batch_size,callbacks = [tensorboard_callback, save_best])

    
    
    
    model.save('logs/soundClass_model.h5')
    
    show_results(model)
    
    
    
