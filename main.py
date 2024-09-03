# Importamos las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

# Leemos el archivo csv
data = pd.read_csv("altura_peso.csv", sep=",")

# Se crean las dos variables con sus respectivos datos
x = data["Altura"].values
y = data["Peso"].values

#### IMPLEMENTACION DEL MODELO KERAS

# Se crea el contenedor o modelo de keras
np.random.seed(2)
modelo = Sequential()

# Definimos el contenido del modelo
# Tamaño de datos de salida
output_dim = 1

# Tamaño de datos de entrada
input_dim = 1

# Modelo definido de regresión lineal
modelo.add(Dense(output_dim, input_dim = input_dim, activation="linear" ))

# Se define el método que usará el entrenamiento
modelo.compile(loss="mse", optimizer=SGD(learning_rate=0.0004))

#### ENTRENAMIENTO DEL MODELO

# Cantidad de iteraciones
cant_epochs = 1000
# Cantidad de datos a utilizar
batch_size = x.shape[0]
historia = modelo.fit(x, y, epochs=cant_epochs, batch_size = batch_size, verbose = 1)

#### VISUALIZACION DE RESULTADOS

capas = modelo.layers[0]
w, b = capas.get_weights()
print("Parámetros: ", w[0][0], ", ", b[0])

loss = historia.history["loss"]

plt.subplot(1,2,1)
plt.plot(loss)
plt.xlabel("epoch")
plt.ylabel("ECM")
plt.title("ECM vs Epochs")

y_reg = modelo.predict(x)
plt.subplot(1,2,2)
plt.scatter(x,y)
plt.plot(x,y_reg, "r")
plt.title("Datos originales y regresion lineal")
plt.show()