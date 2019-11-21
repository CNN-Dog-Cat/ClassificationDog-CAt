from keras.models import model_from_json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


json_file = open('Desktop/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("Desktop/model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

path_test = "Desktop/test/199.jpg"
img = cv.imread(path_test)
plt.imshow(img)
plt.show()

img = cv.resize(img, (50,50))
img = img.reshape(1, 50, 50, 3)

# Previsioni di output su campioni di input 'x_test'
predictions = loaded_model.predict(img)
if predictions == 0:
    print("Gatto")
else:
    print("Cane")