import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "./in/sign-language-img/Train"
test_dir = "./in/sign-language-img/Test"

#Normailzar base de datos
train_datagen = ImageDataGenerator(rescale = 1/255, validation_split= 0.2)
test_datagen = ImageDataGenerator(rescale = 1/255, validation_split= 0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (28, 28),        #Tamaño
    batch_size = 128,              #Nº de lotes por lo que se entrenara
    class_mode = "categorical",    #Multiclase
    color_mode = "grayscale",
    subset = "training"
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (28, 28),
    batch_size = 128,
    class_mode = "categorical",
    color_mode = "grayscale",
    subset = "validation"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (28, 28),
    batch_size = 128,
    class_mode = "categorical",
    color_mode = "grayscale"
)

#Crear clases
classes = [char for char in string.ascii_uppercase if char != "J" if char != "Z"]
print(classes)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize = (10, 10))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img[:,:,0])
        ax.axis("off")
    #plt.tight_layout()
    #plt.show()
    plt.savefig("./out/load_from_json.png", bbox_inches="tight", pad_inches=0)
    plt.close()

sample_training_images, _= next(train_generator)
plotImages(sample_training_images[:5])

model_base = tf.keras.models.Sequential([
              tf.keras.layers.Flatten(input_shape= (28, 28, 1)), #Convierte las imagenes de 28 x 28 en un array
              tf.keras.layers.Dense(256, activation = "relu"),   #Nº de neuronas
              tf.keras.layers.Dense(128, activation = "relu"),   #Otra capa
              tf.keras.layers.Dense(len(classes), activation = "softmax") #Capa de salida Softmax para mas de dos clases
])

model_base.summary() #Ver como esta estructurada nuestra data

#Entrenamiento

model_base.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy']) #Haci se trabaja para multiples clases


history = model_base.fit(
    train_generator,                           
    epochs= 20, 
    validation_data = validation_generator
)

#Testeamos el modelo
results = model_base.evaluate(test_generator)

#Generamos una clase para visualizar y analizar los resultados
def visualizacion_resultados(history):
    epochs = [i for i in range(20)]
    fig, ax = plt.subplots(1,2)
    train_acc = history.history["accuracy"]
    train_loss = history.history["loss"]
    val_acc = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]
    fig.set_size_inches(16, 9)

    ax[0].plot(epochs, train_acc, "go-", label =" Entrenamiento accuracy")
    ax[0].plot(epochs, val_acc, "ro-", label= "Validacion accuracy")
    ax[0].set_title("Entrenamiento & validación accuracy")
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs, train_loss, "go-", label =" Entrenamiento Loss")
    ax[1].plot(epochs, val_loss, "ro-", label= "Validacion Loss")
    ax[1].set_title("Entrenamiento & validación Loss")
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")

    #plt.show()
    plt.savefig("./out/history_model_base.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    

visualizacion_resultados(history)

#Creamos nuestro modelo con regularizadores
from tensorflow.keras import regularizers

model_optimizer = tf.keras.models.Sequential([
              tf.keras.layers.Flatten(input_shape= (28, 28, 1)),
              tf.keras.layers.Dense(256, kernel_regularizer = regularizers.l2(1e-5), activation = "relu"),  #Se aplica regularizador
              tf.keras.layers.Dropout(0.2),                                                                 #Se eliminan neuronas para no generalizar
              tf.keras.layers.Dense(128, kernel_regularizer= regularizers.l2(1e-5),activation = "relu"),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(len(classes), activation = "softmax")
])

model_optimizer.summary()

model_optimizer.compile(optimizer= "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

history_optimizer = model_optimizer.fit(
    train_generator,
    epochs = 20,
    validation_data = validation_generator
)

visualizacion_resultados(history_optimizer)

#Creamos nuestro modelo con convoluciones
model_convolutional = tf.keras.models.Sequential([
              tf.keras.layers.Conv2D(75, (3,3), activation = "relu", input_shape= (28, 28, 1)),  #Convolución  74 filtros de a 3 x 3
              tf.keras.layers.MaxPool2D((2,2)),  #Redicción de la dimensionalidad
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(256, kernel_regularizer = regularizers.l2(1e-5), activation = "relu"),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(128, kernel_regularizer= regularizers.l2(1e-5),activation = "relu"),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(len(classes), activation = "softmax")
])

model_convolutional.summary()

model_convolutional.compile(optimizer= "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

history_convolutional = model_convolutional.fit(
    train_generator,
    epochs = 20,
    validation_data = validation_generator
)

visualizacion_resultados(history_convolutional)

#Calbacks personalizados
model_callback = tf.keras.models.Sequential([
              tf.keras.layers.Conv2D(75, (3,3), activation = "relu", input_shape= (28, 28, 1)),
              tf.keras.layers.MaxPool2D((2,2)),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(256, kernel_regularizer = regularizers.l2(1e-5), activation = "relu"),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(128, kernel_regularizer= regularizers.l2(1e-5),activation = "relu"),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(len(classes), activation = "softmax")
])

model_callback.summary()

model_callback.compile(optimizer= "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

from tensorflow.keras.callbacks import Callback

class TrainingCallback(Callback):
  def on_epoch_end(self, epoch, logs = {}):
    if logs.get("accuracy") > 0.95:
      print("Lo logramos, nuestro modelo llego a 95%, detenemos el entrenamiento")
      self.model.stop_training = True

callback = TrainingCallback()

history_callback = model_callback.fit(
    train_generator,
    epochs = 20,
    callbacks = [callback],
    validation_data = validation_generator
)

def get_model():
  model = tf.keras.models.Sequential([
              tf.keras.layers.Conv2D(75, (3,3), activation = "relu", input_shape= (28, 28, 1)),
              tf.keras.layers.MaxPool2D((2,2)),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(256, kernel_regularizer = regularizers.l2(1e-5), activation = "relu"),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(128, kernel_regularizer= regularizers.l2(1e-5),activation = "relu"),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(len(classes), activation = "softmax")
  ])
  return model

model_early = get_model()
model_early.summary()

model_early.compile(optimizer= "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

#Early stopping
callback_early = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience=3, mode = "auto")

history_early = model_early.fit(
    train_generator, 
    epochs= 20, 
    callbacks = [callback_early],
    validation_data = validation_generator
)

import kerastuner as kt
from tensorflow import keras

def constructor_modelos(hp):
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(75, (3,3), activation= "relu", input_shape = (28, 28, 1)))
  model.add(tf.keras.layers.MaxPool2D((2,2)))
  model.add(tf.keras.layers.Flatten())

  hp_units = hp.Int("units", min_value = 32, max_value = 512, step = 32)
  model.add(tf.keras.layers.Dense(units=hp_units,activation = "relu", kernel_regularizer= regularizers.l2(1e-5)))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(128,activation = "relu", kernel_regularizer= regularizers.l2(1e-5)))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(len(classes), activation = "softmax"))

  hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

  model.compile(optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate),loss = "categorical_crossentropy", metrics = ["accuracy"])

  return model

tuner = kt.Hyperband(
    constructor_modelos,
    objective = "val_accuracy",
    max_epochs = 20,
    factor = 3,
    directory = "models/",
    project_name = "platzi-tunner"
)

tuner.search(train_generator, epochs =20, validation_data = validation_generator)

best_hps = tuner.get_best_hyperparameters(num_trials =1)[0]

print(best_hps.get("units"))
print(best_hps.get("learning_rate"))

hypermodel = tuner.hypermodel.build(best_hps)

history_hypermodel = hypermodel.fit(
    train_generator,
    epochs = 20,
    callbacks = [callback_early],
    validation_data = validation_generator
)

#Guardar únicamente arquitectura del modelo
config_dict = hypermodel.get_config()
print(config_dict)

# Crear un nuevo modelo desde la arquitectura
model_same_config = tf.keras.Sequential.from_config(config_dict)
model_same_config.summary()

#Guardar y cargar unicamente los pesos
#Entrenar el modelo con los checkpoints

from tensorflow.keras.callbacks import ModelCheckpoint

model_weight = get_model()
model_weight.summary()

model_weight.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# Creamos unos checkpoint de nuestros modelos con TensorFlow, verbose: mostrar mas información
checkpoint_path = "model_checpoints/checkpoint"
checkpoint_weighs= ModelCheckpoint(
    filepath = checkpoint_path,
    frecuency = "epoch",
    save_weights_only = True,
    verbose = 1

)
# Entrenamos el modelo, y almacenamos en cada steps los pesos del moedelo.
history_weight = model_weight.fit(
    train_generator,
    epochs = 20,
    callbacks = [checkpoint_weighs],
    validation_data = validation_generator
)

#Guardamos los pesos de forma manual
model_weight.save_weights("model_manul/my_model")
#Creamos la arquitectura y cargamos los pesos
model_weights2 = get_model()
model_weights2.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model_weights2.load_weights(checkpoint_path)  #Cargar los pesos al modelo
model_weights2.evaluate(test_generator)
model_weight.evaluate(test_generator)

## Criterios para almacenar modelos
#### Almacenar infraestructura y peso

checkpoint_path = "model_checpoints_complete"
checkpoint_weighs= ModelCheckpoint(
    filepath = checkpoint_path,
    frecuency = "epoch",
    save_weights_only = False,
    monitor = "val_accuracy",
    save_best_only = True,
    verbose = 1

)

model_complete = get_model()
model_complete.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

history_complete = model_complete.fit(
    train_generator,
    epochs = 20,
    callbacks = [checkpoint_weighs],
    validation_data = validation_generator
)
#### Almacenar infraestructura y peso manual
model_complete.save("saved_model_complete/mymodel")

## Cargamos nuestro modelo completo
model_complete3 = tf.keras.models.load_model("saved_model_complete/mymodel")
model_complete3.evaluate(test_generator)

model_complete.evaluate(test_generator)

#### Usar el formato .h5 para guardar modelos
#Hierarchical Data Format (HDF5 format)

#!pip install pyyaml h5py
model_complete3.save("my_model.h5")
model_complete4 = tf.keras.models.load_model("my_model.h5")

## Creamos nuestra modelo pre-entrenado con modelos de Keras
#### Preprocesamiento de nuestra base de datos
#Debemos revisar cual es el input de nuestro modelos (Tamaño del tensor de entrada)

train_generator_resize = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150, 150),
    batch_size = 128,
    class_mode = "categorical",
    color_mode = "rgb",
    subset = "training"
)

validation_generator_resize = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150, 150),
    batch_size = 128,
    class_mode = "categorical",
    color_mode = "rgb",
    subset = "validation"
)

test_generator_resize = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150, 150),
    batch_size = 128,
    class_mode = "categorical",
    color_mode = "rgb"
)

from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(
    include_top = False,
    input_tensor = tf.keras.layers.Input(shape = (150, 150, 3))
)

for layer in pre_trained_model.layers:
  layer.trainable = False

pre_trained_model.summary()

#### Congelamos el modelo 
# Congelamos hasta la capa mixed 7 para agregar nuestra capas de salida.

last_layers= pre_trained_model.get_layer("mixed7")
last_output = last_layers.output

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(128, activation = "relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(len(classes), activation= "softmax")(x)

model_keras = tf.keras.Model(pre_trained_model.input, x)

model_keras.compile(optimizer= "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model_keras.summary()

history_keras = model_keras.fit(
    train_generator_resize,
    epochs = 5,
    validation_data = validation_generator_resize
)

## Creamos nuestra modelo pre-entrenado con modelos desde TensorFlow Hub
import tensorflow_hub as hub
module_url = "https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4"

model_hub = tf.keras.Sequential([
                                 tf.keras.layers.InputLayer(input_shape=(150, 150, 3)),
                                 hub.KerasLayer(module_url, trainable=False),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(128, activation ="relu"),
                                 tf.keras.layers.Dropout(rate=0.2),
                                 tf.keras.layers.Dense(len(classes), activation = "softmax")
])

model_hub.build((None, 150, 150, 3))
model_hub.summary()

model_hub.compile(optimizer= "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

history_hub = model_hub.fit(
    train_generator_resize,
    epochs= 5,
    validation_data = validation_generator_resize
)
results = model_hub.evaluate(test_generator_resize)

# Conectar con TensorBoard

from tensorflow.python.keras.callbacks import TensorBoard
from time import time

model_tensorboard = get_model()
model_tensorboard.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

callback_tensorboard = TensorBoard(log_dir = "logs/{}".format(time()))

history_tensorboard = model_tensorboard.fit(
    train_generator,
    epochs = 20,
    callbacks = [callback_tensorboard],
    validation_data = validation_generator
)

#### Ejecutamos nuestro tensorboard con los comandos magic Se puede visualizar desde Google Colab o desde el terminal

"""load_ext tensorboard
tensorboard --logdir logs 

## Comportamos los resultados con el mundo entero
Mayor información ingresa sobre Tensorflow.dev [aquí](https://tensorboard.dev/)"""
