'''
José E. Maese Álvarez. 
TFG: Uso de redes neuronales para identificación de matrículas.
Funciones de AlexNet
'''
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import funciones as fun
import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# learn = 0.001
# epocas = 25
def AlexNet(learn, epocas):
    BATCH_SIZE = 16
    IMG_SIZE = (227, 227)
    train_dataset = image_dataset_from_directory(directory = 'Imagenes/Numeros', 
                                                 shuffle=True,
                                                 color_mode = 'rgb',
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE,
                                                 validation_split=0.25,
                                                 subset='training',
                                                 seed = 1)
                                                     
    test_dataset = image_dataset_from_directory(directory = 'Imagenes/Numeros',
                                                shuffle=True,
                                                batch_size=BATCH_SIZE,
                                                color_mode = 'rgb',
                                                image_size=IMG_SIZE,
                                                validation_split=0.25,
                                                subset='validation',
                                                seed = 1)
    class_names = train_dataset.class_names
                
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    data_augmentation = fun.data_augmenter()
        
        
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    # =tf.optimizers.SGD(lr=0.001))
    opt = keras.optimizers.SGD(learning_rate=learn)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    
    history = model.fit(train_dataset, epochs=epocas, validation_data=test_dataset)
    model.save("AlexNet_Model.h5")
    
    
    df_loss_acc = pd.DataFrame(history.history)
    
    plt.figure(1)
    df_loss= df_loss_acc[['loss','val_loss']]
    df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
    df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
    plt.close(1)
    
    plt.figure(2)
    df_acc= df_loss_acc[['accuracy','val_accuracy']]
    df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
    df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
    plt.close(2)
    
    AlexNet_pred = model.predict(test_dataset)
    
    resultados_AlexNet = df_loss_acc.to_numpy()
    
    return (resultados_AlexNet[-1,0], resultados_AlexNet[-1,1], resultados_AlexNet[-1,2], resultados_AlexNet[-1,3])
            
    
    
    

