from tensorflow.keras import regularizers
from tensorflow import keras
import tensorflow as tf


class Models:

    def get_model(self, classes):
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
    
    def model_base(self, train_generator, validation_generator, test_generator, classes):
        model_base = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape= (28, 28, 1)), #Convierte las imagenes de 28 x 28 en un array
                    tf.keras.layers.Dense(256, activation = "relu"),   #Nº de neuronas
                    tf.keras.layers.Dense(128, activation = "relu"),   #Otra capa
                    tf.keras.layers.Dense(len(classes), activation = "softmax") #Capa de salida Softmax para mas de dos clases
        ])

        model_base.summary()  #Tabla
        model_base.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy']) #Haci se trabaja para multiples clases

        history = model_base.fit(
            train_generator,                           
            epochs= 20, 
            validation_data = validation_generator
        )

        #Testeamos el modelo
        results = model_base.evaluate(test_generator)
        print(f"Results: {results}\nLoss: {results[0]}\nAccuracy: {results[1]}")

        return history

    def model_optimizer(self, train_generator, validation_generator, test_generator, classes):
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

        results = model_optimizer.evaluate(test_generator)
        print(f"Results: {results}\nLoss: {results[0]}\nAccuracy: {results[1]}")

        return history_optimizer

    def model_convolutional(self, train_generator, validation_generator, test_generator, classes):
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

        results = model_convolutional.evaluate(test_generator)
        print(f"Results: {results}\nLoss: {results[0]}\nAccuracy: {results[1]}")

        return history_convolutional

    def model_callback(self, train_generator, validation_generator, test_generator, classes, callback):
        
        model_callback = self.get_model(classes)
        model_callback.summary()
        model_callback.compile(optimizer= "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

        history_callback = model_callback.fit(
            train_generator,
            epochs = 20,
            callbacks = [callback],
            validation_data = validation_generator,
            verbose=1
        )

        results = model_callback.evaluate(test_generator)
        print(
            f"Results: {results}\n"
            f"Loss: {results[0]}\n"
            f"Accuracy: {results[1]}"
        )
        return history_callback

    def constructor_modelos(self, hp, classes):
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
