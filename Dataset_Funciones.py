# '''
# Jos� E. Maese �lvarez. 
# TFG: Uso de redes neuronales para identificaci�n de matr�culas.
# Funciones de creacion de base de datos 
# '''
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


""" RECORTE DE MATRICULA
1. Cargar imagen
2. Convertir en escala de grises
3. Filtro de reducción de ruido
4. Detección de bordes
5. Localización de contornos y ordenacion por tamaño
6. Escoger el rectangulo correspondiente a la matricula:
    - Proporción ancho x alto = 4.7
    - Área de la matrícula entre dos valores límite
7. Recortar imagen de matrícula
"""

def RecorteMatricula(img):
    #1. Cargar imagen
    imagColor = cv2.imread('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Coches/' + str(img) + '.jpg') 
    imagColor = cv2.resize(imagColor, (864, 864) )
    
    #2. Cargar imagen en escala de grises
    imagGris = cv2.cvtColor(imagColor, cv2.COLOR_BGR2GRAY) 

    #3. Filtro de reduccion de ruido
    imagFiltrada = cv2.GaussianBlur(imagGris, (5, 5), 0)
    cv2.imwrite('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Pruebas/' + str(img) + '_filtro_gauss.jpg', imagFiltrada)
    
    #4. Detección de bordes con Canny. Incluye algoritmo Sobel, supresion de 
    # no-max (pixeles fuera del borde) y umbral por histeresis.
    imagBordes = cv2.Canny(imagFiltrada, 30, 200)
    imagBordes = cv2.dilate(imagBordes, None, iterations=1)
    cv2.imwrite('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Pruebas/' + str(img) + 'bordes.jpg', imagBordes)

    
    #5. Localizacion de contornos y creacion de rectangulos en cada uno
    (contornos, jerarquia) = cv2.findContours(imagBordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Ordena y dibuja solo los 10 contornos más grandes
    contornos = sorted(contornos, key = cv2.contourArea, reverse = True)[:10]    
    #6. Escoger el rectangulo correspondiente a la matricula:
    matrizDetectada = 0
    for c in contornos:
       area = cv2.contourArea(c)
       epsilon = 0.02*cv2.arcLength(c, True)
       approx = cv2.approxPolyDP(c, epsilon, True)
       # vertices = len(approx)
       x,y,w,h = cv2.boundingRect(approx)
       aspect_ratio = float(w)/(h)
       
       if (aspect_ratio > 3.6 and aspect_ratio < 6):
           if area>5000 and area<18000:
               matrizDetectada = 1
               break 
            
    #7. Recortamos la matri�cula
    if matrizDetectada == 1:    
        
        # Masking the part other than the number plate
        mask = np.zeros(imagGris.shape, np.uint8)
        new_image = cv2.drawContours(mask, [approx], 0, 255, -1,)
        new_image = cv2.bitwise_and(imagGris, imagGris, mask=mask)
        

        #Recortamos la matricula
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        imagMatricula = imagGris[topx:bottomx+1, topy:bottomy+1]
        
        cv2.imwrite('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Pruebas/' + str(img) + '.jpg', imagMatricula)       
    else:
        print('Error al detectar la matricula del coche ' +str(img))


    
""" RECORTE DE CARACTERES
1. Cargar imagen en escala de grises
4. Detección de bordes
5. Localización de contornos y ordenacion por tamaño
6. Escoger el rectangulo correspondiente a la matricula:
    - Proporción ancho x alto = 4.7
    - Área de la matrícula entre dos valores límite
7. Recortar imagen de matrícula
"""  
 
def RecorteCaracteres(img, alpha1, alpha2, beta, limite_binarizado, tam):
        imagMat = cv2.imread('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Matriculas/' + str(img) + '.jpg') 
        if imagMat is None:
            print('Error al recortar la matricula del coche ' + str(img))
            return
        height = 64
        width = 256
        
        # Aplicamos filtros a la imagen y binarizamos
        imagColor = cv2.resize(imagMat, (width, height) )
        imagGris = cv2.cvtColor(imagColor, cv2.COLOR_BGR2GRAY)  
        imagFiltrada = cv2.GaussianBlur(imagGris, (5, 5), 0)
        ret, imagBin = cv2.threshold(imagFiltrada, limite_binarizado, 255, cv2.THRESH_BINARY_INV)
        imagBin[:, 0:5]=255
        imagMatriz = np.asarray(imagBin)
        
        plt.figure(1)
        plt.imshow(imagMatriz,cmap = 'gray')
        plt.close(1)
        
        grad = np.sum(imagMatriz, axis=0)/(255*height)
        for i in range(5,256):
            grad[i] = alpha1*grad[i] + (1 - alpha1)*grad[i-2]
                       
        # Creamos el limite inferior que separa cada caracter utilizando su nivel de blanco en cada columna
        x = np.array(range(0, 256))
        y = grad.copy()
        limCaracter = grad.copy()
        limCaracter[:] = 0

        for i in range(30, 226):
            limCaracter[i] = np.amin(y[i-30:i+30]) + beta
        limCaracter[0:30] = np.amin(y[0:30]) + beta
        limCaracter[225:256] = np.amin(y[225:256]) + beta
        
        alpha2 = 0.6
        for i in range(5,256):
            limCaracter[i] = alpha2*limCaracter[i] + (1 - alpha2)*limCaracter[i-2]
        
        plt.figure(2)    
        plt.plot(x, y, color="black")
        plt.plot(x, limCaracter, color="red")
        #plt.title('Matricula '+str(img))
        plt.xlabel('Columnas de la imagen')
        plt.ylabel('Nivel de blanco')
        plt.savefig('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Pruebas/hist.jpg', facecolor='w', bbox_inches="tight", pad_inches=0.3, transparent=True)

        #imagHist = np.asarray(x,y)
        #plt.show(imagHist)  
       # plt.savefig('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Pruebas/imagHist', facecolor='w', bbox_inches="tight", pad_inches=0.3, transparent=True)
        plt.close(2)
        
        #Obtenemos la posicion de cada caracter diferenciando entre letra y numero
        j = 0
        flag = 1     #No esta en caracter
        inicioCaracter = np.zeros(8)
        finCaracter = np.zeros(8)
        margen = 3
        for i in range(256):
            if (grad[i] > limCaracter[i] and flag == 0):
                flag = 1
                inicioCaracter[j] = i - margen
                #print('Inicio caracter en ' + str(inicioCaracter[j]))
            if (grad[i] < limCaracter[i] and flag == 1):
                flag = 0
                finCaracter[j] = i + margen
                #print('Fin caracter en ' + str(finCaracter[j]))
                j = j + 1
                if j == 8:
                    break
        
        #Recorto cada caracter
        inicioCaracterX = np.zeros(7)
        finCaracterX = np.zeros(7)
        inicioCaracterY = np.zeros(7)
        finCaracterY = np.zeros(7)
        caracter = np.zeros(7)
        
        for i in range(7):
            inicioCaracterX[i] = inicioCaracter[i+1]
            finCaracterX[i] = finCaracter[i+1]
            inicioCaracterY[i] = 3      
            finCaracterY[i] = 60      
            # print(str(finCaracterX[i]))
            caracter = imagGris[int(inicioCaracterY[i]):int(finCaracterY[i]), int(inicioCaracterX[i]):int(finCaracterX[i])]
            if finCaracterX[i] == 0:
                break
            caracter = cv2.resize(caracter, (tam, tam))
            cv2.imwrite('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Caracteres/' + str(img) + '_' + str(i) + '.jpg', caracter)


def RecorteMatricula_Def(img):
    #1. Cargar imagen
    imagColor = cv2.imread('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Coches/' + str(img) + '.jpg') 
    imagColor = cv2.resize(imagColor, (864, 864) )
    
    #2. Cargar imagen en escala de grises
    imagGris = cv2.cvtColor(imagColor, cv2.COLOR_BGR2GRAY) 

    #3. Filtro de reduccion de ruido
    imagFiltrada = cv2.GaussianBlur(imagGris, (5, 5), 0)
    # cv2.imwrite('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Pruebas/' + str(img) + '_filtro_gauss.jpg', imagFiltrada)
    
    #4. Detección de bordes con Canny. Incluye algoritmo Sobel, supresion de 
    # no-max (pixeles fuera del borde) y umbral por histeresis.
    imagBordes = cv2.Canny(imagFiltrada, 30, 200)
    imagBordes = cv2.dilate(imagBordes, None, iterations=1)
    # cv2.imwrite('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Pruebas/' + str(img) + 'bordes.jpg', imagBordes)

    
    #5. Localizacion de contornos y creacion de rectangulos en cada uno
    (contornos, jerarquia) = cv2.findContours(imagBordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Ordena y dibuja solo los 10 contornos más grandes
    contornos = sorted(contornos, key = cv2.contourArea, reverse = True)[:10]   
    IMAG_contornos= cv2.drawContours(imagColor, contornos, -1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.imwrite('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/DEFENSA/Matriculas/contornos.jpg', IMAG_contornos)       

    #6. Escoger el rectangulo correspondiente a la matricula:
    matrizDetectada = 0
    for c in contornos:
       area = cv2.contourArea(c)
       epsilon = 0.02*cv2.arcLength(c, True)
       approx = cv2.approxPolyDP(c, epsilon, True)
       # vertices = len(approx)
       x,y,w,h = cv2.boundingRect(approx)
       aspect_ratio = float(w)/(h)
       
       if (aspect_ratio > 3.6 and aspect_ratio < 6):
           if area>5000 and area<18000:
               matrizDetectada = 1
               break 
            
    #7. Recortamos la matri�cula
    if matrizDetectada == 1:    
        
        # Masking the part other than the number plate
        mask = np.zeros(imagGris.shape, np.uint8)
        new_image = cv2.drawContours(mask, [approx], 0, 255, -1,)
        new_image = cv2.bitwise_and(imagGris, imagGris, mask=mask)
        

        #Recortamos la matricula
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        imagMatricula = imagGris[topx:bottomx+1, topy:bottomy+1]
        
        cv2.imwrite('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/DEFENSA/Matriculas/' + str(img) + '.jpg', imagMatricula)       
    else:
        print('Error al detectar la matricula del coche ' +str(img))


    
""" RECORTE DE CARACTERES
1. Cargar imagen en escala de grises
4. Detección de bordes
5. Localización de contornos y ordenacion por tamaño
6. Escoger el rectangulo correspondiente a la matricula:
    - Proporción ancho x alto = 4.7
    - Área de la matrícula entre dos valores límite
7. Recortar imagen de matrícula
"""  
 
def RecorteCaracteres_Def(img, alpha1, alpha2, beta, limite_binarizado, tam):
        imagMat = cv2.imread('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/DEFENSA/Matriculas/' + str(img) + '.jpg') 
        if imagMat is None:
            print('Error al recortar la matricula del coche ' + str(img))
            return
        height = 64
        width = 256
        
        # Aplicamos filtros a la imagen y binarizamos
        imagColor = cv2.resize(imagMat, (width, height) )
        imagGris = cv2.cvtColor(imagColor, cv2.COLOR_BGR2GRAY)  
        imagFiltrada = cv2.GaussianBlur(imagGris, (5, 5), 0)
        ret, imagBin = cv2.threshold(imagFiltrada, limite_binarizado, 255, cv2.THRESH_BINARY_INV)
        imagBin[:, 0:5]=255
        imagMatriz = np.asarray(imagBin)
        
        plt.figure(1)
        plt.imshow(imagMatriz,cmap = 'gray')
        plt.close(1)
        
        grad = np.sum(imagMatriz, axis=0)/(255*height)
        for i in range(5,256):
            grad[i] = alpha1*grad[i] + (1 - alpha1)*grad[i-2]
                       
        # Creamos el limite inferior que separa cada caracter utilizando su nivel de blanco en cada columna
        x = np.array(range(0, 256))
        y = grad.copy()
        limCaracter = grad.copy()
        limCaracter[:] = 0

        for i in range(30, 226):
            limCaracter[i] = np.amin(y[i-30:i+30]) + beta
        limCaracter[0:30] = np.amin(y[0:30]) + beta
        limCaracter[225:256] = np.amin(y[225:256]) + beta
        
        alpha2 = 0.6
        for i in range(5,256):
            limCaracter[i] = alpha2*limCaracter[i] + (1 - alpha2)*limCaracter[i-2]
        
        plt.figure(2)    
        plt.plot(x, y, color="black")
        plt.plot(x, limCaracter, color="red")
        #plt.title('Matricula '+str(img))
        plt.xlabel('Columnas de la imagen')
        plt.ylabel('Nivel de blanco')
        # plt.savefig('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Pruebas/hist.jpg', facecolor='w', bbox_inches="tight", pad_inches=0.3, transparent=True)

        #imagHist = np.asarray(x,y)
        #plt.show(imagHist)  
       # plt.savefig('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/Pruebas/imagHist', facecolor='w', bbox_inches="tight", pad_inches=0.3, transparent=True)
        plt.close(2)
        
        #Obtenemos la posicion de cada caracter diferenciando entre letra y numero
        j = 0
        flag = 1     #No esta en caracter
        inicioCaracter = np.zeros(8)
        finCaracter = np.zeros(8)
        margen = 3
        for i in range(256):
            if (grad[i] > limCaracter[i] and flag == 0):
                flag = 1
                inicioCaracter[j] = i - margen
                #print('Inicio caracter en ' + str(inicioCaracter[j]))
            if (grad[i] < limCaracter[i] and flag == 1):
                flag = 0
                finCaracter[j] = i + margen
                #print('Fin caracter en ' + str(finCaracter[j]))
                j = j + 1
                if j == 8:
                    break
        
        #Recorto cada caracter
        inicioCaracterX = np.zeros(7)
        finCaracterX = np.zeros(7)
        inicioCaracterY = np.zeros(7)
        finCaracterY = np.zeros(7)
        caracter = np.zeros(7)
        
        for i in range(7):
            inicioCaracterX[i] = inicioCaracter[i+1]
            finCaracterX[i] = finCaracter[i+1]
            inicioCaracterY[i] = 3      
            finCaracterY[i] = 60      
            # print(str(finCaracterX[i]))
            caracter = imagGris[int(inicioCaracterY[i]):int(finCaracterY[i]), int(inicioCaracterX[i]):int(finCaracterX[i])]
            if finCaracterX[i] == 0:
                break
            caracter = cv2.resize(caracter, (tam, tam))
            cv2.imwrite('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/DEFENSA/Caracteres/' + str(img) + '_' + str(i) + '.jpg', caracter)
        

    
def Lenet(img):
    RecorteMatricula_Def(str(img))
    RecorteCaracteres_Def(str(img),  alpha1=0.8, alpha2=0.6, beta=0.05, limite_binarizado=125, tam=32)
    Matricula = np.zeros((1,4))
    model = load_model('LeNet_Model2.h5')
    # model.summary()
    tam = 32
    for k in range(1, 5):
        i=k-1
        # Lectura del caracter
        caracter = cv2.imread('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/DEFENSA/Caracteres/' + str(img) + '_' + str(i) + '.jpg')
        if caracter is not None:
            # Preprocesado de la imagen
            caracterResize = cv2.resize(caracter, (tam, tam))
            caracterGris = cv2.cvtColor(caracterResize, cv2.COLOR_BGR2GRAY) 
            caracterPre = np.reshape(caracterGris, (1, tam, tam, 1))
            # Resultado
            resultado = model.predict(caracterPre)
            indice = resultado.argmax(axis=1)
            # print(indice)
            Matricula[0,i] = indice
        else:
            Matricula[0,i] = "E"
    return Matricula
         

def AlexNet(img):
    Matricula = np.zeros((1,4))
    # load model
    model = load_model('AlexNet_Model.h5')
    # model.summary()
    tam = 227
    for k in range(1, 5):
        i=k-1
        # Lectura del caracter
        caracter = cv2.imread('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/DEFENSA/Caracteres/' + str(img) + '_' + str(i) + '.jpg')
        if caracter is not None:
            # Preprocesado de la imagen
            caracterResize = cv2.resize(caracter, (tam, tam))
            np_img = np.array(caracterResize)
            caracterPre = np.reshape(np_img, [-1, tam, tam, 3])
            resultado = model.predict(caracterPre)
            indice = resultado.argmax(axis=1)
            # print(indice)
            Matricula[0,i] = indice
        else:
            Matricula[0,i] = "E"
    return Matricula

def ResNet(img):
    Matricula = np.zeros((1,4))
    # load model
    model = load_model('ResNet_Model.h5')
    # model.summary()
    tam = 180
    for k in range(1, 5):
        i=k-1
        # Lectura del caracter
        caracter = cv2.imread('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes/DEFENSA/Caracteres/' + str(img) + '_' + str(i) + '.jpg')
        if caracter is not None:
            # Preprocesado de la imagen
            caracterResize = cv2.resize(caracter, (tam, tam))
            np_img = np.array(caracterResize)
            caracterPre = np.reshape(np_img, [-1, tam, tam, 3])
            resultado = model.predict(caracterPre)
            indice = resultado.argmax(axis=1)
            # print(indice)
            Matricula[0,i] = indice
        else:
            Matricula[0,i] = "E"
    return Matricula


 