'''
José Enrique Maese Álvarez
TFG: 
'''

import Dataset_Funciones as df
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import IPython.display as display
from PIL import Image


img = 0;

print('Recortando matricula ' + str(img))
df.RecorteMatricula_Def(str(img)) 
display.display(Image.open('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Coches/' + str(img) + '.jpg'))

print('Recortando caracteres de matricula  ' + str(img)) 
df.RecorteCaracteres_Def(str(img),  alpha1=0.8, alpha2=0.6, beta=0.05, limite_binarizado=125, tam=32)
display.display(Image.open('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/DEFENSA/matriculas/' + str(img) + '.jpg'))


j = 1
print('Calculando resultado...')


################### LENET-5 ####################

if j==0:       
    # load model
    Matricula = np.zeros((1,4))
    model = load_model('LeNet_Model2.h5')
    # model.summary()
    tam = 32
    for k in range(1, 5):
        i=k-1
        # Lectura del caracter
        caracter = cv2.imread('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/DEFENSA/Caracteres/' + str(img) + '_' + str(i) + '.jpg')
        # Preprocesado de la imagen
        caracterResize = cv2.resize(caracter, (tam, tam))
        caracterGris = cv2.cvtColor(caracterResize, cv2.COLOR_BGR2GRAY) 
        caracterPre = np.reshape(caracterGris, (1, tam, tam, 1))
        # Resultado
        resultado = model.predict(caracterPre)
        indice = resultado.argmax(axis=1)
        # print(indice)
        Matricula[0,i] = indice
    print('Matrícula calculada usando LeNet-5: ' + str(round(Matricula[0,0])) + ' ' + str(round(Matricula[0,1])) + ' ' + str(round(Matricula[0,2])) + ' ' + str(round(Matricula[0,3]))) 


################### ALEXNET ####################

if j==1:        
    Matricula = np.zeros((1,4))
    # load model
    model = load_model('AlexNet_Model.h5')
    # model.summary()
    tam = 227
    for k in range(1, 5):
        i=k-1
        # Lectura del caracter
        caracter = cv2.imread('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/DEFENSA/Caracteres/' + str(img) + '_' + str(i) + '.jpg')
        # Preprocesado de la imagen
        caracterResize = cv2.resize(caracter, (tam, tam))
        np_img = np.array(caracterResize)
        caracterPre = np.reshape(np_img, [-1, tam, tam, 3])
        resultado = model.predict(caracterPre)
        np_resultado = np.array(resultado)
        indice = resultado.argmax(axis=1)
        # print(indice)
        Matricula[0,i] = indice
    print('Matrícula calculada usando AlexNet: ' + str(round(Matricula[0,0])) + ' ' + str(round(Matricula[0,1])) + ' ' + str(round(Matricula[0,2])) + ' ' + str(round(Matricula[0,3]))) 

################### ALEXNET ####################

if j==2:        
    Matricula = np.zeros((1,4))
    # load model
    model = load_model('ResNet_Model.h5')
    # model.summary()
    tam = 180
    for k in range(1, 5):
        i=k-1
        # Lectura del caracter
        caracter = cv2.imread('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/DEFENSA/Caracteres/' + str(img) + '_' + str(i) + '.jpg')
        # Preprocesado de la imagen
        caracterResize = cv2.resize(caracter, (tam, tam))
        np_img = np.array(caracterResize)
        caracterPre = np.reshape(np_img, [-1, tam, tam, 3])
        resultado = model.predict(caracterPre)
        np_resultado = np.array(resultado)
        indice = resultado.argmax(axis=1)
        # print(indice)
        Matricula[0,i] = indice
    print('Matrícula calculada usando ResNet50: ' + str(round(Matricula[0,0])) + ' ' + str(round(Matricula[0,1])) + ' ' + str(round(Matricula[0,2])) + ' ' + str(round(Matricula[0,3]))) 

