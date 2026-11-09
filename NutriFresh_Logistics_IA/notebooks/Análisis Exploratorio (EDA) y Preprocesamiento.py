# ==========================================
# CELDA 2: ANÁLISIS EXPLORATORIO Y ENCODING
# ==========================================
print("--- ESTADÍSTICAS DESCRIPTIVAS ---")
display(df.describe())

# 1. Visualización de correlaciones (Heatmap)
plt.figure(figsize=(8,5))
sns.heatmap(df[['Peso_kg', 'Tiempo_min', 'Km_recorridos', 'Precipitacion_mm', 'Nivel_transito', 'Retraso_binario']].corr(), 
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación - Variables Logísticas')
plt.show()

# 2. Transformación de variables categóricas (One-Hot Encoding)
# Justificación: Los algoritmos no leen texto, necesitan 0s y 1s.
df_ml = pd.get_dummies(df, columns=['Ciudad_destino', 'Tipo_producto', 'Clima'], drop_first=True)

# 3. Definir Variables Predictoras (X) y Variable Objetivo (y)
X = df_ml.drop(columns=['ID_Pedido', 'Fecha', 'Estado_entrega', 'Retraso_binario'])
y = df_ml['Retraso_binario']

# 4. Estandarización de variables numéricas (Opcional para Random Forest, pero buena práctica)
scaler = StandardScaler()
X[['Peso_kg', 'Tiempo_min', 'Km_recorridos', 'Precipitacion_mm']] = scaler.fit_transform(X[['Peso_kg', 'Tiempo_min', 'Km_recorridos', 'Precipitacion_mm']])

print("\n--- VARIABLES PREDICTORAS (X) ---")
display(X.head())