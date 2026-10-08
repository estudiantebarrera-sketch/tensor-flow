import os
import zipfile
import json
import codecs
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import base64
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Load:
    def load_from_json(self, url):

        data_json = []
        with open(url, 'r', encoding='utf-8') as js:
            for line in js:
                data_json.append(json.loads(line))

        print("{} imagenes encontradas".format(len(data_json)))

        print(data_json[0])

        images = []

        for data in data_json:
            response = requests.get(data['content'])
            img = np.asarray(Image.open(BytesIO(response.content)))
            images.append([img, data["label"]])

        """# Mostrar y guardar
        plt.imshow(images[0][0].reshape(28, 28), cmap="gray")
        plt.axis("off")

        plt.savefig("./out/load_from_json.png", bbox_inches="tight", pad_inches=0)
        plt.close()"""

    def load_from_base64(self, url):

        with open(url) as f:
            data = json.load(f)

        print(data)

        """base64_img_bytes = data['b'].encode('utf-8')
        path_img = "./out/decoded_images_base64.png"
        with open(path_img, "wb") as file_to_save:
            decoded_image_data = base64.decodebytes(base64_img_bytes)
            file_to_save.write(decoded_image_data)

        img = Image.open(path_img)
        img"""

    def load_from_csv(self, path):
        df = pd.read_csv(path)
        print(f"Shape: {df.shape}")
        return df

    def normalizar(self, train_dir, test_dir):
        train_datagen = ImageDataGenerator(rescale = 1/255, validation_split= 0.2)
        test_datagen = ImageDataGenerator(rescale = 1/255, validation_split= 0.2)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size = (28, 28),        # Tamaño 28 x 28 pixeles
            batch_size = 128,              # Número de imágenes procesadas en cada batch
            class_mode = "categorical",    # Multiclase
            color_mode = "grayscale",      # Escala de grises
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

        classes = list(train_generator.class_indices.keys())

        return train_generator, validation_generator, test_generator, classes

