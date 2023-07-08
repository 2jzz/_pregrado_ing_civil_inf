import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time

import os

from sklearn.model_selection import train_test_split
from keras.models import load_model

import shap

start_code = time.time()

print("Comenzando carga de valores ANN.")
x = pd.read_csv("F:/Universidad/Tesis/Codigo/Datos/Procesados/x_values.csv", dtype = int)
y_cdm = 0 # pd.read_csv("../../../Datos/Procesados/y_dummies_cdm.csv", dtype = int)
y_grdt = 0 # pd.read_csv("../../../Datos/Procesados/y_dummies_grdtipo.csv", dtype = int)
y_grd = pd.read_csv("F:/Universidad/Tesis/Codigo/Datos/Procesados/y_dummies_grd.csv", dtype = int)
y_sev = 0 # pd.read_csv("../../../Datos/Procesados/y_dummies_sev.csv", dtype = int)
print("Carga de archivos de la ANN finalizada.\n")

master = 1
y = [y_cdm, y_grd, y_grdt, y_sev]
y_str = ['CDM', 'GRD', 'GRDT', 'SEV']
path_ = f"F:/Universidad/Tesis/Codigo/Versiones/Códigos Alex/Versión 4 - Lectura/SHAP Datos/{y_str[master]}/shap_values_{y_str[master]}_datasize_0.0075.csv"


print("Comenzando carga de valores SHAP.")
start_time = time.time()
# Leer el archivo CSV y convertirlo en un DataFrame de pandas
shap_values_read = pd.read_csv(path_)
# Leer las dimensiones originales desde el archivo de texto
with open(f"F:/Universidad/Tesis/Codigo/Versiones/Códigos Alex/Versión 4 - Lectura/SHAP Datos/{y_str[master]}/shap_values_{y_str[master]}_dimensions.txt", "r") as f:
    dimensions = tuple(map(int, f.read().split(",")))
# Convertir el DataFrame de pandas en una matriz numpy y remodelar a su forma original de 3 dimensiones
shap_values = shap_values_read.values.reshape(dimensions)

end_time = time.time()
time_ = end_time - start_time
minutes, seconds = divmod(time_, 60)
print(f"Tiempo total de ejecución: {int(minutes)}:{int(seconds)} minutos.\n")

print("Realizando train split")

x_train, x_test, y_train, y_test = train_test_split(x, y[master], train_size= 0.0075, test_size = 0.0075, random_state = 0, stratify = y[master])

print("Train split finalizado.\n")

shap_values_converted = [shap_values[i] for i in range(shap_values.shape[0])]

type_plot = 'BAR PLOT'

# Establecer la cantidad de gráficos y el número de filas aleatorias
n_plots = 500
n_random_rows = 500

# Establecer la semilla para garantizar resultados consistentes
np.random.seed(42)

for i, class_shap_values in enumerate(shap_values):
    start_time = time.time()
    if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
        continue
    print(f"Obteniendo gráficas de la clase: {y_train.columns[i]}")
    # Generar índices aleatorios
    random_indices = np.random.choice(shap_values_converted[i].shape[0], n_random_rows, replace=False)
    for j in range(n_plots):
        # Seleccionar una fila aleatoria de los valores SHAP
        random_shap_values = class_shap_values[random_indices[j], :]

        # Crear una figura para cada clase
        plt.figure(figsize = [18, 18], dpi = 160)

        # Generar el bar_plot para la clase actual y el valor SHAP seleccionado
        shap.bar_plot(
            random_shap_values,
            feature_names=x_train.columns,
            max_display=25,
            show=False  # Establecer en False para evitar que se muestre la trama en la ventana de salida
        )

        # Guardar cada gráfico en un archivo separado
        output_folder = f"F:/Universidad/Tesis/Codigo/Visualización de datos/SHAP/{y_str[master]}/{type_plot}/{y_train.columns[i]}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(f"{output_folder}/bar_plot_value_{j}.png")

        # Cerrar la figura para liberar memoria
        plt.close()
    end_time = time.time()
    celda_time = end_time - start_time
    minutes, seconds = divmod(celda_time, 60)
    print(f"Graficas obtenidas, Tiempo total de ejecución: {int(minutes)}:{int(seconds)} minutos.")
    print("--------------------------\n")


end_code = time.time()

f_time = end_code - start_code

f_minutes, f_seconds = divmod(f_time, 60)

print(f"Tiempo total de ejecución: {int(f_minutes)}:{int(f_seconds)} minutos.")