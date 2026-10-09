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

class Utils:

    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        images = X.values
        plt.imshow(images[1].reshape(28, 28))
        plt.axis("off")
        plt.savefig("./out/imagen_prueba.png", bbox_inches="tight", pad_inches=0)
        plt.close()
        return X, y

    def normalizar(self, dataset, label):
        dataset = dataset.drop_duplicates()
        dataset = dataset.loc[(dataset.drop(columns=label) != "0").any(axis=1)]

        columnas = dataset.columns.drop(label)
        dataset[columnas] = dataset[columnas].apply(pd.to_numeric, errors='coerce')
        dataset = dataset.dropna(subset=columnas)

        """plt.figure(figsize = (10, 10))
        sns.set_style("darkgrid")
        sns.countplot(data=dataset, x=label, hue=label, palette="husl", legend=False)
        plt.savefig("./out/clases.png", bbox_inches="tight", pad_inches=0)
        plt.close()"""

        y = dataset[label]
        X = dataset.drop(label, axis=1)

        X = X.astype(int)

        X = X / 255


        #unique_val = np.array(y)
        
        """print("="*64)
        print('Info:')
        print(dataset.info())
        print("="*64)
        print('dtypes:')
        print(dataset.dtypes)
        print("="*64)
        print(f"Labels: {np.unique(unique_val)}")
        print(F"Nulos: {dataset.isnull().values.any()}")
        print("="*64)
        print(dataset[dataset.duplicated()])"""

        return X, y

    def plotImages(self, images_arr):
        fig, axes = plt.subplots(1, 5, figsize = (10, 10))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img[:,:,0])
            ax.axis("off")
        #plt.tight_layout()
        #plt.show()
        plt.savefig("./out/load_from_json.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    def visualizacion_resultados(self, history, name):
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
        plt.savefig(f"./out/history_{name}.png", bbox_inches="tight", pad_inches=0)
        plt.close()