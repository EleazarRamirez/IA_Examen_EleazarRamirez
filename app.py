import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 3. Carga del dataset
try:
    df = pd.read_csv('titanic.csv') 
    print("Archivo cargado exitosamente")
except FileNotFoundError:
    print("Error: No se encontró el archivo .csv")

# 4. Preprocesamiento
# 4.1 Limpieza de nulos
# Eliminamos columnas con demasiados nulos
cols_to_drop = [c for c in ['deck', 'embark_town', 'alive', 'who', 'adult_male', 'cabin', 'name', 'ticket'] if c in df.columns]
df = df.drop(columns=cols_to_drop)

df['age'] = df['age'].fillna(df['age'].median())
df = df.dropna(subset=['embarked'])

# Convertimos 'sex' y 'embarked' a números
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 5. Separación de datos
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6 & 7. Selección y Entrenamiento
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_train, y_train) 

# 8. Evaluación
y_pred = modelo.predict(X_test)

print("\n--- RESULTADOS ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Matriz de Confusión
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.show()


print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
