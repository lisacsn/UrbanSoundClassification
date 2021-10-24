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



def loading_data(file_train,file_test) : 

    us8k_df_train = pd.read_pickle(file_train)
    us8k_df_test = pd.read_pickle(file_test)


    df_train = us8k_df_train.drop(['fold'],axis=1)
    df_test = us8k_df_test.drop(['fold'],axis=1)



    X_train = np.stack(df_train.melspectrogram.to_numpy())
    X_test = np.stack(df_test.melspectrogram.to_numpy())



    X_dim = (128,128,1)
    X_train = X_train.reshape(X_train.shape[0], *X_dim)
    X_test = X_test.reshape(X_test.shape[0], *X_dim)


    Y_train = np.array(df_train['label'])
    Y_train = to_categorical(Y_train)
        
    Y_test = np.array(df_test['label'])
    Y_test = to_categorical(Y_test)



    return X_train,Y_train,X_test,Y_test


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


def show_results(tot_history):
    """Show accuracy and loss graphs for train and test sets."""
    pngName = ['resultaugmentedfold/res1.png','resultaugmentedfold/res2.png','resultaugmentedfold/res3.png','resultaugmentedfold/res4.png','resultaugmentedfold/res5.png','resultaugmentedfold/res6.png','resultaugmentedfold/res7.png','resultaugmentedfold/res8.png','resultaugmentedfold/res9.png','resultaugmentedfold/res10.png']
    for i, history in enumerate(tot_history):
        print('\n({})'.format(i+1))
        fig = plt.figure()
        plt.figure(figsize=(15,5))

        plt.subplot(121)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.grid(linestyle='--')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.grid(linestyle='--')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
            
        plt.savefig(pngName[i])
        
        file = open('resultaugmentedfold/result.txt','a')
        file.write('\n=========FOLD%s=========\n'% (i+1))
        file.write('\nMax validation accuracy: %.4f %%' % (np.max(history.history['val_accuracy']) * 100))
        file.write('\nMin validation loss: %.5f' % np.min(history.history['val_loss']))

        print('\tMax validation accuracy: %.4f %%' % (np.max(history.history['val_accuracy']) * 100))
        print('\tMin validation loss: %.5f' % np.min(history.history['val_loss']))






if __name__ == "__main__":
    history = []
    test_list = [1,2,3,4,5,6,7,8,9]
    for test in test_list:
        filename_train = 'fold_pkl/train_augmented'+str(test)+'.pkl'
        filename_test = 'fold_pkl/test'+str(test)+'.pkl'
        print(filename_test)
        X_train,Y_train,X_test,Y_test = loading_data(filename_train,filename_test)

        model = init_model()
        log_dir = "logs/fit/afold" + datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)
        save_best = tf.keras.callbacks.ModelCheckpoint(filepath = "logs/checkpoints/", save_weights_only = True,
                                                    monitor = "val_accuracy", mode = "max", save_best_only = True)

        initial_epochs = 100
        num_batch_size = 32
        model_fit = model.fit(X_train,Y_train, epochs=initial_epochs,validation_data=(X_test,Y_test),
                                batch_size=num_batch_size, callbacks = [tensorboard_callback, save_best])


        model.save('logs/soundClass_augmented_model'+str(test)+'.h5')   
        
        history.append(model_fit)
    show_results(history)
        
    


