import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
import json

(x_train, y_train), (x_test,y_test) = cifar100.load_data(label_mode = "fine")

print(x_train.shape)
print(y_train.shape)

# Leer labels
with open("./in/databasesLoadData/cifar100_labels.json", "r") as file:
    labels = json.load(file)



num_image = 40

# Mostrar label y nombre
label = y_train[num_image][0]

print("Label:", label)
print("Clase:", labels[label])

plt.imshow(x_train[num_image])
plt.show()

from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist

(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = cifar10.load_data()
(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()