# ==========================================
# CELDA 3: ENTRENAMIENTO DEL MODELO BASE
# ==========================================
# Dividir datos en Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Configuración del modelo (Random Forest)
# class_weight='balanced' se usa para manejar el desbalanceo (pocos retrasos vs muchos a tiempo)
modelo_rf = RandomForestClassifier(
    n_estimators=100,       # 100 árboles
    max_depth=5,            # Evitar sobreajuste (overfitting)
    class_weight='balanced',# Penaliza más los errores en la clase minoritaria
    random_state=42
)

# Entrenar el modelo
modelo_rf.fit(X_train, y_train)
print("Modelo Random Forest entrenado exitosamente.")