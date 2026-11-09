# ==========================================
# CELDA 1: IMPORTACIÓN DE LIBRERÍAS Y DATOS
# ==========================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Simulación de los datos del caso NutriFresh (Basados en el PDF)
data_pedidos = {
    'ID_Pedido': ['P-2023101', 'P-2023102', 'P-2023103', 'P-2023104'],
    'Ciudad_destino': ['Barranquilla', 'Cartagena', 'Santa Marta', 'Soledad'],
    'Tipo_producto': ['Frutas', 'Lácteos', 'Carnes frías', 'Verduras'],
    'Peso_kg': [120, 350, 420, 250],
    'Fecha': ['2023-10-15', '2023-10-15', '2023-10-15', '2023-10-15'],
    'Estado_entrega': ['A tiempo', 'Retrasado', 'Devuelto', 'A tiempo']
}

data_rutas = {
    'Ciudad_destino': ['Cartagena', 'Santa Marta', 'Sincelejo', 'Barranquilla'],
    'Tiempo_min': [140, 180, 210, 170],
    'Km_recorridos': [110, 155, 170, 155],
    'Consumo_gal': [22, 30, 34, 29]
}

data_externos = {
    'Ciudad_destino': ['Barranquilla', 'Cartagena', 'Santa Marta', 'Soledad'],
    'Clima': ['Soleado', 'Lluvia moderada', 'Lluvia intensa', 'Nublado'],
    'Precipitacion_mm': [0, 15, 45, 5],
    'Nivel_transito': [3, 4, 5, 2]
}

# Convertir a DataFrames
df_pedidos = pd.DataFrame(data_pedidos)
df_rutas = pd.DataFrame(data_rutas)
df_externos = pd.DataFrame(data_externos)

# Unir los datos (Merge) por la ciudad de destino
df = pd.merge(df_pedidos, df_rutas, on='Ciudad_destino', how='left')
df = pd.merge(df, df_externos, on='Ciudad_destino', how='left')

# Transformar la variable objetivo: 1 si es retrasado/devuelto, 0 si es a tiempo
df['Retraso_binario'] = df['Estado_entrega'].apply(lambda x: 1 if x in ['Retrasado', 'Devuelto'] else 0)

print("--- DATASET CONSOLIDADO ---")
display(df)