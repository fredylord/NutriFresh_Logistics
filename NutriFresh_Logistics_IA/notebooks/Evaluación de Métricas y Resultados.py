# ==========================================
# CELDA 4: EVALUACIÓN Y MÉTRICAS
# ==========================================
# Predicciones
y_pred = modelo_rf.predict(X_test)

# 1. Reporte de Clasificación (Accuracy, Precision, Recall, F1-Score)
print("--- REPORTE DE CLASIFICACIÓN ---")
print(classification_report(y_test, y_pred))

# 2. Matriz de Confusión
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción (0=A tiempo, 1=Retraso)')
plt.ylabel('Realidad')
plt.show()

# 3. Importancia de las Variables (Feature Importance)
importancias = modelo_rf.feature_importances_
features = X.columns
df_importancia = pd.DataFrame({'Variable': features, 'Importancia': importancias}).sort_values(by='Importancia', ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x='Importancia', y='Variable', data=df_importancia, palette='viridis')
plt.title('Importancia de las Variables en la Predicción de Retrasos')
plt.show()

print("\n--- TOP VARIABLES QUE CAUSAN RETRASOS ---")
display(df_importancia.head(5))