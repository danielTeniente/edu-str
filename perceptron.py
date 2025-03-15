import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la app
st.title("Visualización del Perceptrón")
st.write("Modifica los valores de los pesos y el bias para ver cómo cambia la frontera de decisión.")

# Controles deslizantes para modificar los pesos y el bias
w1 = st.slider("Peso w1", -2.0, 2.0, 0.5, 0.1)
w2 = st.slider("Peso w2", -2.0, 2.0, -0.8, 0.1)
b = st.slider("Bias (b)", -2.0, 2.0, 0.2, 0.1)

# Datos de entrada
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
w = np.array([w1, w2])

# Función del perceptrón
def perceptron_output(x, w, b):
    return np.sign(np.dot(x, w) + b)

# Calcular la salida para cada punto de X
outputs = np.array([perceptron_output(x, w, b) for x in X])

# Separar puntos por clase
class_1 = X[outputs == 1]
class_neg_1 = X[outputs == -1]

# Rango de x1 para la frontera de decisión
x1_vals = np.linspace(-2, 2, 100)

# Evitar división por cero si w2 == 0
if w2 != 0:
    x2_vals = (-w1 * x1_vals - b) / w2
else:
    x2_vals = np.full_like(x1_vals, -b / w1)  # Recta vertical si w2 = 0

# Crear la figura
fig, ax = plt.subplots()
ax.scatter(class_1[:, 0], class_1[:, 1], color='blue', label="Clase 1 (y=1)")
ax.scatter(class_neg_1[:, 0], class_neg_1[:, 1], color='red', label="Clase -1 (y=-1)")
ax.plot(x1_vals, x2_vals, 'k--', label="Frontera de decisión")

# Configurar el gráfico
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
ax.set_title("Clasificación del Perceptrón y Frontera de Decisión")
ax.grid()

# Mostrar la gráfica en Streamlit
st.pyplot(fig)
