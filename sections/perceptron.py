import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.i18n import get_text

def show_perceptron():
    st.title(get_text("title", "perceptron"))
    st.write(get_text("description", "perceptron"))

    # Add mathematical equations section
    st.header(get_text("equations_title", "perceptron"))
    
    # Perceptron equation
    st.subheader(get_text("perceptron_eq_title", "perceptron"))
    st.write(get_text("perceptron_eq_desc", "perceptron"))
    st.latex(r"y = \text{sign}(w_1x_1 + w_2x_2 + b)")
    
    # Decision boundary equation
    st.subheader(get_text("boundary_eq_title", "perceptron"))
    st.write(get_text("boundary_eq_desc", "perceptron"))
    st.latex(r"x_2 = -\frac{w_1}{w_2}x_1 - \frac{b}{w_2}")
    
    # Parameters explanation
    st.subheader(get_text("explanation_title", "perceptron"))
    st.write(get_text("explanation_text", "perceptron"))

    # Controles deslizantes para modificar los pesos y el bias
    w1 = st.slider(get_text("weight1", "perceptron"), -2.0, 2.0, 0.5, 0.1)
    w2 = st.slider(get_text("weight2", "perceptron"), -2.0, 2.0, -0.8, 0.1)
    b = st.slider(get_text("bias", "perceptron"), -2.0, 2.0, 0.2, 0.1)

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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(class_1[:, 0], class_1[:, 1], color='blue', label=get_text("class1", "perceptron"))
    ax.scatter(class_neg_1[:, 0], class_neg_1[:, 1], color='red', label=get_text("class_neg1", "perceptron"))
    ax.plot(x1_vals, x2_vals, 'k--', label=get_text("decision_boundary", "perceptron"))

    # Configurar el gráfico
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend()
    ax.set_title(get_text("graph_title", "perceptron"))
    ax.grid(True)

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig) 