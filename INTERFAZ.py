'''
Jose Enrique Maese Alvarez
TFG: 
'''
import PySimpleGUI as sg
import os.path
from PIL import Image
import io
import Dataset_Funciones as df
import numpy as np

#Definimos el tama�o de la ventana
screen_width, screen_height = sg. Window.get_screen_size()

w_width = int(screen_width*0.9)
w_height = int(screen_height*0.7)

# First the window layout in 2 columns

file_list_column = [
    [   sg.Text("Image Folder", font=('HELVETICA', 12)), sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),  sg.FolderBrowse()],
    [   sg.Listbox(values=[], enable_events=True, size=(40, w_height), key="-FILE LIST-")], 
]


# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Escoge una imagen de la lista:", font=('HELVETICA', 12))],
    [sg.Text(size=(int(w_width/20), 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]


Matricula_LeNet = np.zeros((1,4))
Matricula_AlexNet = np.zeros((1,4))
Matricula_ResNet = np.zeros((1,4))

n_network_viewer_column = [
    [sg.Text("Matricula:", font=('HELVETICA', 12))],
    [sg.Image(key="-MATRICULA-")],
    [sg.Text("Resultado:", font=('HELVETICA', 12))],
    [sg.Text(f'LeNet-5: {Matricula_LeNet[0,0]} {Matricula_LeNet[0,1]} {Matricula_LeNet[0,2]} {Matricula_LeNet[0,3]}', key='-MAT_LENET-', font=('HELVETICA', 15))],
    [sg.Text(f'AlexNet: {Matricula_AlexNet[0,0]} {Matricula_AlexNet[0,1]} {Matricula_AlexNet[0,2]} {Matricula_AlexNet[0,3]}', key='-MAT_ALEXNET-', font=('HELVETICA', 15))],
    [sg.Text(f'ResNet50: {Matricula_ResNet[0,0]} {Matricula_ResNet[0,1]} {Matricula_ResNet[0,2]} {Matricula_ResNet[0,3]}', key='-MAT_RESNET-', font=('HELVETICA', 15))],
    [sg.Text(size=(50, 1), key="-TOUT-")],
]



# ----- Full layout -----
layout = [
    [   sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
        sg.VSeperator(),
        sg.Column(n_network_viewer_column)]
]



#  Creamos la ventana
window = sg.Window("Reconocimiento de matr�culas", layout, size = (w_width,w_height) )


#  Loop
while True:
    
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []
            
        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".jpg", ".png"))
        ]
        window["-FILE LIST-"].update(fnames)
    
    
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.abspath(os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0]))
            image = Image.open(filename)
            image.thumbnail((int(w_width/3), int(w_width/3)))
            image=image.rotate(-90)
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window["-IMAGE-"].update(data=bio.getvalue())

            #### Codigo para llamar a las funciones.
            ## Extraemos el valor de la imagen
            im = (os.path.join(values["-FILE LIST-"][0]))
            im=im[:-4]
            # print(im)
            df.RecorteMatricula_Def(im)
            df.RecorteMatricula_Def(im)
                  
            ###DIBUJAMOS LA MATRICULA
            filename_mat = ('C:/Users/josen/OneDrive/Documents/Python Scripts/TFG/Imagenes//DEFENSA/Matriculas/' + str(im) + '.jpg')
            image_mat = Image.open(filename_mat)
            # print('P1')
            image_mat.thumbnail((int(w_width/3), int(w_width/3)))
            image_mat=image_mat.rotate(0)
            bio_mat = io.BytesIO()
            image_mat.save(bio_mat, format="PNG")
            window["-MATRICULA-"].update(data=bio_mat.getvalue())
            # print('P2')
            
            #ANALIZAMOS LA MATRICULA
            Matricula_LeNet = df.Lenet(im)
            Matricula_LeNet=np.round(Matricula_LeNet).astype(int)
            window['-MAT_LENET-'].update(f'LeNet-5:    {Matricula_LeNet[0,0]}    {Matricula_LeNet[0,1]}    {Matricula_LeNet[0,2]}    {Matricula_LeNet[0,3]}')
            # print('P3')
            
            Matricula_AlexNet = df.AlexNet(im)
            Matricula_AlexNet=np.round(Matricula_AlexNet).astype(int)
            window['-MAT_ALEXNET-'].update(f'AlexNet:     {Matricula_AlexNet[0,0]}    {Matricula_AlexNet[0,1]}    {Matricula_AlexNet[0,2]}    {Matricula_AlexNet[0,3]}')
            # print('P4')

            Matricula_ResNet = df.ResNet(im)
            Matricula_ResNet=np.round(Matricula_ResNet).astype(int)
            window['-MAT_RESNET-'].update(f'ResNet50:  {Matricula_ResNet[0,0]}    {Matricula_ResNet[0,1]}    {Matricula_ResNet[0,2]}    {Matricula_ResNet[0,3]}')

            print(Matricula_LeNet)
            print(Matricula_AlexNet)
            print(Matricula_ResNet)
            
        except:
            pass


window.close()
    
    
    
