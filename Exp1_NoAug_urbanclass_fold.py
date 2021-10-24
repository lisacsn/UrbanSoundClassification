import numpy as np 

import pandas as pd 



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



def train_test_split(fold_k, data, X_dim=(128, 128, 1)):
    X_train = np.stack(data[data.fold != fold_k].melspectrogram.to_numpy())
    X_test = np.stack(data[data.fold == fold_k].melspectrogram.to_numpy())

    y_train = data[data.fold != fold_k].label.to_numpy()
    y_test = data[data.fold == fold_k].label.to_numpy()

    XX_train = X_train.reshape(X_train.shape[0], *X_dim)
    XX_test = X_test.reshape(X_test.shape[0], *X_dim)
    
    yy_train = to_categorical(y_train)
    yy_test = to_categorical(y_test)
    
    return XX_train, XX_test, yy_train, yy_test



def show_results(tot_history):
    """Show accuracy and loss graphs for train and test sets."""
    pngName = ['Exp1_result/res1.png','Exp1_result/res2.png','Exp1_result/res3.png','Exp1_result/res4.png','Exp1_result/res5.png','Exp1_result/res6.png','Exp1_result/res7.png','Exp1_result/res8.png','Exp1_result/res9.png','Exp1_result/res10.png']
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
        
        file = open('Exp1_result/result.txt','a')
        file.write('\n=========FOLD%s=========\n'% (i+1))
        file.write('\nMax validation accuracy: %.4f %%' % (np.max(history.history['val_accuracy']) * 100))
        file.write('\nMin validation loss: %.5f' % np.min(history.history['val_loss']))

        print('\tMax validation accuracy: %.4f %%' % (np.max(history.history['val_accuracy']) * 100))
        print('\tMin validation loss: %.5f' % np.min(history.history['val_loss']))


def process_fold(fold_k, data, epochs=100, num_batch_size=32):
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(fold_k, data)

    # init model
    model = init_model()

    # pre-training accuracy
    log_dir = "logs/fit/folds/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)

    
    # train the model

    history = model.fit(X_train,y_train, epochs=epochs,validation_data=(X_test,y_test),batch_size=num_batch_size, callbacks=[tensorboard_callback])
    
    return history


if __name__ == "__main__":
    us8k_df = pd.read_pickle("./pkl/us8k_df.pkl")
    history1 = []
    for i in range(10) : 
        FOLD = i+1
        
        history = process_fold(FOLD, us8k_df, epochs=100)
        history1.append(history)


    show_results(history1)
