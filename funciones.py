'''
José Enrique Maese Álvarez
TFG: 
'''
    
import tensorflow as tf    
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation    
import tensorflow.keras.layers as tfl
 

###############################################################################
###############################################################################
###############################################################################
        
def data_augmenter():
    data_augmentation = tf.keras.Sequential()
    # data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.05))
    return data_augmentation    

###############################################################################
###############################################################################
###############################################################################


def procesado_AlexNet(dataset):
    # dataset = tf.image.per_image_standardization(dataset)
    dataset = tf.image.resize(dataset, (227, 227, 3))
    return dataset()

###############################################################################
###############################################################################
###############################################################################

def number_mobileNet_model(image_shape, data_augmentation=data_augmenter()):  
    
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    input_shape = image_shape + (3,)
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                    include_top=False,
                                                    weights="imagenet") # From imageNet
    
    # freeze the base model by making it non trainable
    base_model.trainable = False 

    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape) 
    
    # apply data augmentation to the inputs
    x = data_augmentation(inputs)
    
    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(x) 
    
    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False) 
    
    # add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    x = tfl.GlobalAveragePooling2D()(x) 
    # include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(0.2)(x)
        
    # use a prediction layer with one neuron (as a binary classifier only needs one)
    outputs = tfl.Dense(1, activation = 'linear')(x)
    
    
    model = tf.keras.Model(inputs, outputs)
    
    return model



