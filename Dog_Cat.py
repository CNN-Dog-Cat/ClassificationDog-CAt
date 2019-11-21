from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import matplotlib.pyplot as plt

path_train = "Desktop/train_1/" #Path della cartella contenente il training set

# Initialising the CNN
model = Sequential()

# Convolution
model.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))

# Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
model.add(Flatten())

# Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory(path_train,
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

model.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 20,
                         validation_steps = 2000)

model.save('Cat_Dog_Test') #Salva il modello creato per poterlo utilizzare quando si ha necessità
# Creazione del json e del file rappresentativi del modello per poterlo riprendere senza effettuare ogni volta il fit
model_json = model.to_json()
with open("Desktop/model.json","w") as json_file:
  json_file.write(model_json)

model.save_weights("Desktop/model.h5")
print("Modello Salvato!")
import keras.models as keras
new_model = keras.load_model('Cat_Dog_Test') # Carica il modello salvato col nome 'CNNTest'

path_test = "Desktop/test/156.jpg"
img = cv.imread(path_test)
plt.imshow(img)
plt.show()

img = cv.resize(img, (50,50))
img = img.reshape(1, 50, 50, 3)

# Previsioni di output su campioni di input 'x_test'
predictions = new_model.predict(img)
if predictions == 0:
    print("Gatto")
else:
    print("Cane")