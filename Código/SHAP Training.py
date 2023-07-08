import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import datetime
import os

from sklearn.model_selection import train_test_split
from keras.models import load_model

import shap

# Reading de los modelos.

# Ruta donde están los modelos.

start_code = time.time()

# Carga de los modelos.
relative_0 = "F:/Universidad/Tesis/Codigo/Versiones/Códigos Alex/Versión 3 - actual/modelos"

model_CDM = load_model(filepath = f"{relative_0}/datasetCDM_2023_04_17_19_35")
model_GRD = load_model(filepath = f"{relative_0}/datasetGRD_2023_04_17_20_53")
model_GRDT = load_model(filepath = f"{relative_0}/datasetGRDT_2023_04_17_11_29")
model_SEV = load_model(filepath = f"{relative_0}/datasetSEV_2023_04_17_22_09")

# Aca se deben cargar los modelos ya entrenados, se le debe dar la ruta donde se guardó la carpeta, sin ninguna extensión.

######################


# OJO QUE ESTO SE DEBE HACER EN UN .PY, en un jupyternotebook la RAM se colapsa porque a pesar de que le del plt.close(), igual queda almacenado en el cache de vscode, esto no pasa en un .py


#######################


# Carga de los archivos.
relative_1 = "F:/Universidad/Tesis/Codigo/Datos/Procesados"
x = pd.read_csv(f"{relative_1}/x_values.csv", dtype = int)
y_cdm = pd.read_csv(f"{relative_1}/y_dummies_cdm.csv", dtype = int)
y_grdt = pd.read_csv(f"{relative_1}/y_dummies_grdtipo.csv", dtype = int)
y_grd = pd.read_csv(f"{relative_1}/y_dummies_grd.csv", dtype = int)
y_sev = pd.read_csv(f"{relative_1}/y_dummies_sev.csv", dtype = int)
df = pd.read_csv(f'{relative_1}/dataset_procesado_pbi.csv', dtype = str)

# Archivos con los que fueron entrenados los modelos, ojo aca con las dimensiones, si no cuadran la ejecución se cae.



lst_y = [y_cdm, y_grdt, y_grd, y_sev]
lst_y_str = ['CDM', 'GRDT', 'GRD', 'SEV']
lst_model = [model_CDM, model_GRDT, model_GRD, model_SEV]

x_train, x_test, y_train, _ = train_test_split(x, lst_y[0], train_size = 0.0004, test_size = 0.0004, random_state = 0, stratify = lst_y[0])

# Trainsize muy muy pequeño porque SHAP consume bastantes recursos y en muy lento

lista_indices_test = x_test.index.to_list()

df_recortado = df.iloc[lista_indices_test]
df_recortado.to_csv(f'{relative_1}/761_pacientes_shap_correcto.csv')

y_test_cdm = y_cdm.iloc[lista_indices_test]
y_test_grdt = y_grdt.iloc[lista_indices_test]
y_test_grd = y_grd.iloc[lista_indices_test]
y_test_sev = y_sev.iloc[lista_indices_test]

y_test_models = [y_test_cdm, y_test_grdt, y_test_grd, y_test_sev]

# rng = np.random.default_rng(seed = 42)
# Función para predecir las probabilidades

def predict_proba(X):
    return model.predict(X, verbose = False)

def shap_function(i):
    prediccion = lst_model[i].predict(x_test)
    def class_labels(row_index):
        return [f'Clase {y_test_models[i].columns[f]} ({prediccion[row_index, f].round(2):.2f})' for f in range(len(y_test_models[i].columns))]
    # Crear un objeto KernelExplainer de SHAP
    background_data = shap.sample(x_train, 100)  # Usar una muestra del conjunto de entrenamiento como datos de fondo
    explainer = shap.KernelExplainer(predict_proba, background_data)

    # Calcular los valores SHAP para el conjunto de datos de prueba
    shap_values = explainer.shap_values(x_test, nsamples = 100)


    #####
    # Aqui arriba se obtiene el explainer para cada modelo, el explainer es como un objeto de SHAP que tiene bastante info respecto al modelo, algunos gráficos de SHAP necesitan el explainer
    # otros se conforman con los valores SHAP, los valores SHAP son un cubo para el caso de los modelos multiclase, donde cada capa del eje Z es una clase, luego cada eje x/y
    # se comporta como, columnas vs shap_values (valor que aporta cada una de las columnas para el paciente en concreto que se le haya tirado al modelo)

    # De aqui par abajo se obtienen los gráficos
    #####


    type_plot = "BAR_PLOT"
    y_str = ['CDM', 'GRDT', 'GRD', 'SEV']

    print(f"Trabajando modelo de {y_str[i]}")
    for n, class_shap_values in enumerate(shap_values):
        start_time = time.time()
        print(f"Obteniendo gráficas de la clase: {y_test_models[i].columns[n]}")
        
        for j in range(300):
            # Crear una figura para cada clase
            plt.figure(figsize = [18, 15], dpi = 300)
            shap.multioutput_decision_plot(list(explainer.expected_value), list(shap_values),
                                            row_index = j,
                                            feature_names=list(x_test.columns),
                                            highlight=[np.argmax(prediccion[j])],
                                            legend_labels=class_labels(j),
                                            legend_location='lower right',
                                            auto_size_plot = False,
                                            show = False)
            # Guardar cada gráfico en un archivo separado
            output_folder = f"F:/Universidad/Tesis/Codigo/Visualización de datos/SHAP/Multioutput decision plot"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plt.savefig(f"{output_folder}/patient_{x_test['DIAGNOSTICO1'].index[j]}_{y_str[i]}.png")
            # Cerrar la figura para liberar memoria
            plt.close()

            random_shap_values = class_shap_values[j, :]
            # Crear una figura para cada clase
            plt.figure(figsize = [15, 15], dpi = 150)

            # Generar el bar_plot para la clase actual y el valor SHAP seleccionado
            shap.bar_plot(
                random_shap_values,
                feature_names=x_test.columns,
                max_display = 25,
                show=False  # Establecer en False para evitar que se muestre la trama en la ventana de salida
            )

            # Guardar cada gráfico en un archivo separado
            output_folder = f"F:/Universidad/Tesis/Codigo/Visualización de datos/SHAP"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plt.savefig(f"{output_folder}/patient_{x_test['DIAGNOSTICO1'].index[j]}_{y_str[i]}_{type_plot}_class_{y_test_models[i].columns[n]}.png")

            # Cerrar la figura para liberar memoria
            plt.close()

        end_time = time.time()
        celda_time = end_time - start_time
        minutes, seconds = divmod(celda_time, 60)
        print(f"Graficas obtenidas, Tiempo total de ejecución: {int(minutes)}:{int(seconds)} m:s.")
        print("--------------------------\n")

for i in range(3, -1, -1):
    model = lst_model[i]
    shap_function(i)

end_code = time.time()
f_time = end_code - start_code
td = datetime.timedelta(seconds = f_time)
dt = datetime.datetime(2023, 5, 1, 0, 0, 0)
dt_final = dt + td
hora_final = dt_final.strftime("%H:%M:%S")

print(f"Tiempo total de ejecución: {hora_final} Hrs.")