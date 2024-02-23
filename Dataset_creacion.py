'''
José Enrique Maese Álvarez
TFG: 
'''

import Dataset_Funciones as df
import cv2


for i in range(1):
    print('Recortando imagen ' + str(i))
    df.RecorteMatricula(str(i))
cv2.destroyAllWindows()    
    
   #%%
   
#José Enrique Maese Álvarez
#TFG: 

import Dataset_Funciones as df
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


for i in range(126):
    print('Recortando matricula ' + str(i))
    df.RecorteCaracteres(str(i),  alpha1=0.8, alpha2=0.6, beta=0.05, limite_binarizado=125, tam=32)
cv2.destroyAllWindows()

#%%   LENET

import LeNet
import numpy as np

LeNet.Lenet(0.0001, 25)

# ep = [10, 15, 20, 25]
# lr = [0.1, 0.01, 0.001, 0.0001]
# datos_Lenet=np.zeros((16, 4))
# pos = 0

# for i in lr:
#     for j in ep:
#         (loss, acc, val_loss, val_acc) = LeNet.Lenet(i, j)
#         datos_Lenet[pos, 0] = loss
#         datos_Lenet[pos, 1] = acc
#         datos_Lenet[pos, 2] = val_loss
#         datos_Lenet[pos, 3] = val_acc
#         pos = pos + 1
        
        
        
#%%   ALEXNET

import AlexNet
import numpy as np

AlexNet.AlexNet(0.001, 25)

# ep = [15, 20, 25, 30]
# lr = [0.01, 0.001, 0.0001, 0.00001] 
# datos_Alexnet = np.zeros((16, 4))
# pos = 0

# for i in lr:
#     for j in ep:
#         (loss, acc, val_loss, val_acc) = AlexNet.AlexNet(i, j)
#         datos_Alexnet[pos, 0] = loss
#         datos_Alexnet[pos, 1] = acc
#         datos_Alexnet[pos, 2] = val_loss
#         datos_Alexnet[pos, 3] = val_acc
#         pos = pos + 1


#%%   RESNET

import ResNet
import numpy as np

ep = [30, 35, 40, 45]
lr = [0.001, 0.0001, 0.00001, 0.000001]
datos_Resnet = np.zeros((16, 4))
pos = 0

for i in lr:
    for j in ep:
        (loss, acc, val_loss, val_acc) = ResNet.ResNet(i, j)
        datos_Resnet[pos, 0] = loss
        datos_Resnet[pos, 1] = acc
        datos_Resnet[pos, 2] = val_loss
        datos_Resnet[pos, 3] = val_acc
        pos = pos + 1






