'''
José E. Maese Álvarez. 
TFG: Uso de redes neuronales para identificación de matrículas.
Funciones de LeNet-5
'''
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import funciones as fun
import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# learn=0.001
# epocas=20
def Lenet(learn, epocas):
    BATCH_SIZE = 16
    IMG_SIZE = (32, 32)
    directory = "Imagenes/Numeros/"
    train_dataset = image_dataset_from_directory(directory = 'Imagenes/Numeros', 
                                                 shuffle=True,
                                                 color_mode = 'grayscale',
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE,
                                                 validation_split=0.25,
                                                 subset='training',
                                                 seed = 1)
                                                 
    test_dataset = image_dataset_from_directory(directory = 'Imagenes/Numeros',
                                                shuffle=True,
                                                batch_size=BATCH_SIZE,
                                                color_mode = 'grayscale',
                                                image_size=IMG_SIZE,
                                                validation_split=0.25,
                                                subset='validation',
                                                seed = 1)
                                                 
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    data_augmentation = fun.data_augmenter()
    
        ############################################################################
        ############################################################################
        
    model = Sequential()
    model.add(Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = 'relu', input_shape = (32,32,1)))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Conv2D(filters = 16,  kernel_size = 5, strides = 1, activation = 'relu', input_shape = (14,14,6)))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Flatten())
    model.add(Dense(units = 120, activation = 'relu'))
    model.add(Dense(units = 84, activation = 'relu'))
    model.add(Dense(units = 10, activation = 'softmax'))
    
    
    # conv_model = convolutional_model((32, 32, 1))
    opt = keras.optimizers.Adam(learning_rate=learn)
    model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.summary()
    history = model.fit(train_dataset, epochs=epocas, validation_data=test_dataset)
    model.save("LeNet_Model.h5")
    
    ############################################################################
    ############################################################################
    
    
    df_loss_acc = pd.DataFrame(history.history)
    
    plt.figure(1)
    df_loss= df_loss_acc[['loss','val_loss']]
    df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
    df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
    plt.savefig('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Archivos/Imagenes/Lenet/Loss: ' + str(learn) + ', ' + str(epocas) +'.jpg', facecolor='w', bbox_inches="tight", pad_inches=0.3, transparent=True)
    plt.close(1)
    
    plt.figure(2)
    df_acc= df_loss_acc[['accuracy','val_accuracy']]
    df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
    df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
    plt.savefig('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Archivos/Imagenes/Lenet/Acc: ' + str(learn) + ', ' + str(epocas) +'.jpg', facecolor='w', bbox_inches="tight", pad_inches=0.3, transparent=True)
    plt.close(2)
    
    LeNet_pred = model.predict(test_dataset)
    resultados_Lenet = df_loss_acc.to_numpy()
    
    return (resultados_Lenet[-1,0], resultados_Lenet[-1,1], resultados_Lenet[-1,2], resultados_Lenet[-1,3])
