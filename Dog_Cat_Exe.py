from tensorflow.keras.models import model_from_json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

#Funzione per il calcolo del Recall
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
#FUnzione per il calcolo della Precision
def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

print("Seleziona il modello che vuoi utilizzare:")
print("<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>")
print("1. Modello bilanciato [12K Gatti | 12K Cani]")
print("2. Modello sbilanciato [12K Gatti | 9.5K Cani]")
print("3. Modello sbilanciato [12K Gatti | 5K Cani]")
print("4. Modello sbilanciato [12K Gatti | 1.2K Cani]")

#-------------Menu di scelta------------------------------
while True:
    print("         ")
    scelta = input("-----------> ")
    errore_scelta = 0

    if scelta == '1':
        file_json = 'Desktop/CNN/Model_12_12/model_simple.json'
        file_model = 'Desktop/CNN/Model_12_12/model_simple.h5'
        file_flatten = 'Desktop/CNN/Flatten_Output/12_12/12_12_flatten_out.csv'
        file_weight = 'Desktop/CNN/Weight/12_12/12_12_weight.txt'
        file_matrix = 'Desktop/CNN/Model_12_12/Matrice confusione/matrix_12_12.txt'
        file_value = 'Desktop/CNN/Model_12_12/values.txt'
        errore_scelta = 0
    else:
        errore_scelta = errore_scelta + 1
        if scelta == '2':
            file_json = 'Desktop/CNN/Model_12_9/model_simple.json'
            file_model = 'Desktop/CNN/Model_12_9/model_simple.h5'
            file_flatten = 'Desktop/CNN/Flatten_Output/12_9/12_9_flatten_out.csv'
            file_weight = 'Desktop/CNN/Weight/12_9/12_9_weight.txt'
            file_matrix = 'Desktop/CNN/Model_12_9/Matrice confusione/matrix_12_9.txt'
            file_value = 'Desktop/CNN/Model_12_9/values.txt'
            errore_scelta = 0
        else:
            errore_scelta = errore_scelta + 1
            if scelta == '3':
                file_json = 'Desktop/CNN/Model_12_5/model_simple.json'
                file_model = 'Desktop/CNN/Model_12_5/model_simple.h5'
                file_flatten = 'Desktop/CNN/Flatten_Output/12_5/12_5_flatten_out.csv'
                file_weight = 'Desktop/CNN/Weight/12_5/12_5_weight.txt'
                file_matrix = 'Desktop/CNN/Model_12_5/Matrice confusione/matrix_12_5.txt'
                file_value = 'Desktop/CNN/Model_12_5/values.txt'
                errore_scelta = 0
            else:
                errore_scelta = errore_scelta + 1
                if scelta == '4':
                    file_json = 'Desktop/CNN/Model_12_1/model_simple.json'
                    file_model = 'Desktop/CNN/Model_12_1/model_simple.h5'
                    file_flatten = 'Desktop/CNN/Flatten_Output/12_1/12_1_flatten_out.csv'
                    file_weight = 'Desktop/CNN/Weight/12_1/12_1_weight.txt'
                    file_matrix = 'Desktop/CNN/Model_12_1/Matrice confusione/matrix_12_1.txt'
                    file_value = 'Desktop/CNN/Model_12_1/values.txt'
                    errore_scelta = 0
                else:
                    errore_scelta = errore_scelta + 1
    if(errore_scelta == 0):
        break
  
print("------------------------------------------------")    
print("Json file: ", file_json)
print("Model file: ", file_model)
print("Directory Flatten Output: ", file_flatten)
print("Directory Weight Output: ", file_weight)
print("Directory Matrix Confusion: ", file_matrix)
print("File Values: ", file_value)
print("------------------------------------------------")

json_file = open(file_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights(file_model)
print("Modello[",file_model,"] caricato con successo!")

loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy',precision_m, recall_m])

path_test_cat = "Desktop/CNN/test_set/cat/"
path_test_dog = "Desktop/CNN/test_set/dog/"
count_ok = 0 #Contatore per classificazione corretta
count_no = 0 #Contatore per classificazione errata
count = 0 #Numero di elementi del test set
x_test = [] #Array di immagini di test
y_test = [] #Array di classi dei vari test
y_pred_class = []

print("Caricamento immagini in corso...")

#Caricamento di foto di gatti nel test set
for img in tqdm(os.listdir(path_test_cat)):
    count = count + 1
    path = os.path.join(path_test_cat,img)
    img_cat = cv.imread(path)
    img = cv.resize(img_cat, (50,50))
    img = img.reshape(1, 50, 50, 3)  
    x_test.append(img)
    y_test.append(0)
    predictions = loaded_model.predict(img)
    y_pred_class.append(int(predictions))
    if predictions == 0:
        count_ok = count_ok + 1
    else:
        count_no = count_no + 1
        
    
#Caricamento di foto di cani nel test set
for img in tqdm(os.listdir(path_test_dog)):
    count = count + 1
    path = os.path.join(path_test_dog,img)
    img = cv.imread(path)
    img = cv.resize(img, (50,50))
    img = img.reshape(1, 50, 50, 3)  
    x_test.append(img)
    y_test.append(1)
    predictions = loaded_model.predict(img)
    y_pred_class.append(int(predictions))
    if predictions == 1:
        count_ok = count_ok + 1
    else:
        count_no = count_no + 1
        
print("Caricamento completato!")    

print("Livelli CNN:")
print
print(loaded_model.summary())
        
print("----------------------------------------")
print("Matrice di confusione:")
print("_______________________________________")
print
matrix_conf = confusion_matrix(y_test,y_pred_class)
print(matrix_conf)
print
print("_______________________________________")

print("----------------VALORI------------------------")
values = classification_report(y_test, y_pred_class)
print(values)
print("----------------------------------------------")
        
print("Salvataggio output livello di flatten...")
print("Livello input: ", loaded_model.layers[0].output)
print("Livello output: ", loaded_model.layers[6].output)
print(" ")

#Cancellazione del file obsoleto per poter scrivere il nuovo output del flatten
try:
    os.remove(file_flatten)
    os.remove(file_weight)
    os.remove(file_matrix)
    os.remove(file_value)
    print("----------------------> File rimosso/i con successo!")
except:
    print("----------------------> File inesistente/i...")
    
file_out = open(file_flatten, "a") #Creazione file per la scrittura del flatten
file_out_weight = open(file_weight, "a") #Creazione file per la scrittura dei pesi del flatten
file_out_matrix = open(file_matrix, "a")
file_out_value = open(file_value, "a")
print("-----------------------------------> Creazione file...")

#-----------Creazione file di feature uscente dal flatten-------------------------
get_3rd_layer_output = K.function([loaded_model.layers[0].input],
                                  [loaded_model.layers[6].output])
classe = "Gatto" #Classe di appartenenza della feature

for count in range(0, len(x_test)):
    if y_test[count] == 0:
        classe = "Gatto"
    else:
        classe = "Cane"
    layer_output = get_3rd_layer_output([x_test[count]])[0]
    output = str(layer_output.tolist())
    output = output.replace("[[","")
    output = output.replace("]]","")
    output = output.replace(",",";")
    output = output.replace(".",",")
    file_out.write(output)
    file_out.write("; ")
    file_out.write(classe)
    file_out.write("\n")
#-----------------------------------------------------------------------------------


file_out_matrix.write(str(matrix_conf))
print("---------------------------------------> File di matrice di confusione creato!")
file_out_matrix.close()
print("----------------------------------------------> File di flatten creato/modificato con successo!")
file_out.close()
fourth_layer_weights  = loaded_model.layers[0].get_weights()
file_out_weight.write(str(fourth_layer_weights))
print("-----------------------------------------------------> File di weight creato/modificato con successo!")
file_out_weight.close()
file_out_value.write(values)
print("-----------------------------------------------------------> File di valori creato con successo!")
file_out_value.close()

print("_______________________________________")
print("Classificazione corretta: ", count_ok)
print("Classificazione errata: ", count_no)
print("_______________________________________")
print("Errore di classificazione: ", (count_no/count)*100)
