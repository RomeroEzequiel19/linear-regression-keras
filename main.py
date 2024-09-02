# Importamos las librerías
import pandas as pd

# Leemos el archivo csv
data = pd.read_csv("altura_peso.csv")

# Lo almacenamos en un dataframe
df = pd.DataFrame(data)

# Se crean las dos variables con sus respectivos datos
x = df["Altura"]
y = df["Peso"]

