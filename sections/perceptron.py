import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.i18n import get_text
import time

# Cache for generated data to avoid recomputing
@st.cache_data
def generate_circular_data(n_points=200):
    # Generate points for the outer circle (class 1)
    theta = np.random.uniform(0, 2*np.pi, n_points)
    r = np.random.uniform(1.5, 2.5, n_points)
    x1 = r * np.cos(theta)
    y1 = r * np.sin(theta)
    
    # Generate points for the inner circle (class -1)
    theta = np.random.uniform(0, 2*np.pi, n_points)
    r = np.random.uniform(0, 0.8, n_points)
    x2 = r * np.cos(theta)
    y2 = r * np.sin(theta)
    
    return x1, y1, x2, y2

@st.cache_data
def generate_exercise_data(n_points=50, noise_scale=0.1):
    # Generate points for class 1 (upper right)
    x1 = np.random.uniform(0, 1.0, n_points)
    y1 = np.random.uniform(0.1, 1.0, n_points)
    
    # Generate points for class -1 (lower left)
    x2 = np.random.uniform(-1.0, 0.5, n_points)
    y2 = np.random.uniform(-1.0, 0, n_points)
    
    # Add noise
    noise = np.random.normal(0, noise_scale, (n_points, 2, 2))
    x1 += noise[:, 0, 0]
    y1 += noise[:, 0, 1]
    x2 += noise[:, 1, 0]
    y2 += noise[:, 1, 1]
    
    return x1, y1, x2, y2

def show_limitations():
    st.header(get_text("limitations_title", "perceptron"))
    st.write(get_text("limitations_desc", "perceptron"))
    
    # Get cached data
    x1, y1, x2, y2 = generate_circular_data()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the points
    ax.scatter(x1, y1, c='blue', label=get_text("class1", "perceptron"), alpha=0.6)
    ax.scatter(x2, y2, c='red', label=get_text("class_neg1", "perceptron"), alpha=0.6)
    
    # Add a sample perceptron line that would fail to separate the classes
    x = np.linspace(-3, 3, 100)
    y = 0.5 * x + 0.2
    ax.plot(x, y, 'k--', label=get_text("decision_boundary", "perceptron"), alpha=0.5)
    
    # Configure the plot
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.grid(True)
    ax.legend()
    ax.set_title(get_text("limitations_plot_title", "perceptron"))
    
    # Show the plot
    st.pyplot(fig)
    
    # Add explanatory text
    st.write(get_text("limitations_text", "perceptron"))

def show_exercise():
    st.header(get_text("exercise_title", "perceptron"))
    st.write(get_text("exercise_desc", "perceptron"))
    
    # Get cached data
    x1, y1, x2, y2 = generate_exercise_data()
    
    # Combine points for classification
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.array([1] * len(x1) + [-1] * len(x2))
    
    # Create two columns for controls and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Sliders for weights and bias
        w1 = st.slider(get_text("weight1", "perceptron"), -2.0, 2.0, -0.1, 0.1)
        w2 = st.slider(get_text("weight2", "perceptron"), -2.0, 2.0, 0.1, 0.1)
        b = st.slider(get_text("bias", "perceptron"), -2.0, 2.0, 0.1, 0.1)
        
        # Show current line equation or warning
        if abs(w2) < 0.001:  # Check if w2 is effectively zero
            st.warning(get_text("division_by_zero", "perceptron"))
        else:
            st.latex(f"x_2 = {-w1/w2:.2f}x_1 + {-b/w2:.2f}")
    
    with col2:
        # Create an empty container for the plot
        plot_container = st.empty()
        
        # Calculate perceptron outputs for checking correctness
        w = np.array([w1, w2])
        outputs = np.sign(np.dot(X, w) + b)  # Vectorized operation
        
        # Check if classification is correct
        is_correct = np.all(outputs == y)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot points with their original colors
        ax.scatter(x1, y1, c='blue', label=get_text("class1", "perceptron"))
        ax.scatter(x2, y2, c='red', label=get_text("class_neg1", "perceptron"))
        
        # Plot decision boundary
        x1_vals = np.linspace(-1.5, 1.5, 100)
        if abs(w2) < 0.001:  # Check if w2 is effectively zero
            x_vertical = -b/w1
            ax.axvline(x=x_vertical, color='k', linestyle='--', label=get_text("decision_boundary", "perceptron"))
        else:
            x2_vals = (-w1 * x1_vals - b) / w2
            ax.plot(x1_vals, x2_vals, 'k--', label=get_text("decision_boundary", "perceptron"))
        
        # Configure the plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.grid(True)
        ax.legend()
        ax.set_title(get_text("exercise_plot_title", "perceptron"))
        
        # Show the plot in the container
        plot_container.pyplot(fig)
        
        # Show success/failure message
        if is_correct:
            st.success(get_text("exercise_success", "perceptron"))
        else:
            st.info(get_text("exercise_continue", "perceptron"))

def show_perceptron():
    st.title(get_text("title", "perceptron"))
    st.write(get_text("description", "perceptron"))

    # Add mathematical equations section
    st.header(get_text("equations_title", "perceptron"))
    
    # Perceptron equation
    st.subheader(get_text("perceptron_eq_title", "perceptron"))
    st.write(get_text("perceptron_eq_desc", "perceptron"))
    st.latex(r"y = \text{sign}(w_1x_1 + w_2x_2 + b)")
    
    # Standard line equation
    st.subheader(get_text("line_eq_title", "perceptron"))
    st.write(get_text("line_eq_desc", "perceptron"))
    st.latex(r"y = mx + b")
    st.write(get_text("line_eq_params", "perceptron"))
    
    # Comparison between standard line and perceptron boundary
    st.subheader(get_text("comparison_title", "perceptron"))
    st.write(get_text("comparison_text", "perceptron"))
    
    # Decision boundary equation
    st.subheader(get_text("boundary_eq_title", "perceptron"))
    st.write(get_text("boundary_eq_desc", "perceptron"))
    st.latex(r"x_2 = -\frac{w_1}{w_2}x_1 - \frac{b}{w_2}")
    
    # Parameters explanation
    st.subheader(get_text("explanation_title", "perceptron"))
    st.write(get_text("explanation_text", "perceptron"))

    # Create two columns for the interactive part
    col1, col2 = st.columns([1, 2])

    with col1:
        # Sliders for weights and bias
        w1 = st.slider(get_text("weight1", "perceptron"), -2.0, 2.0, 0.5, 0.1)
        w2 = st.slider(get_text("weight2", "perceptron"), -2.0, 2.0, -0.8, 0.1)
        b = st.slider(get_text("bias", "perceptron"), -2.0, 2.0, 0.2, 0.1)

        # Show current line equation or warning
        if abs(w2) < 0.001:  # Check if w2 is effectively zero
            st.warning(get_text("division_by_zero", "perceptron"))
        else:
            st.latex(f"x_2 = {-w1/w2:.2f}x_1 + {-b/w2:.2f}")

    with col2:
        # Create an empty container for the plot
        plot_container = st.empty()
        
        # Input data
        X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        w = np.array([w1, w2])

        # Calculate outputs
        outputs = np.sign(np.dot(X, w) + b)  # Vectorized operation

        # Separate points by class
        class_1 = X[outputs == 1]
        class_neg_1 = X[outputs == -1]

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(class_1[:, 0], class_1[:, 1], color='blue', label=get_text("class1", "perceptron"))
        ax.scatter(class_neg_1[:, 0], class_neg_1[:, 1], color='red', label=get_text("class_neg1", "perceptron"))

        # Plot decision boundary
        if abs(w2) < 0.001:  # Check if w2 is effectively zero
            x_vertical = -b/w1
            ax.axvline(x=x_vertical, color='k', linestyle='--', label=get_text("decision_boundary", "perceptron"))
        else:
            x1_vals = np.linspace(-2, 2, 100)
            x2_vals = (-w1 * x1_vals - b) / w2
            ax.plot(x1_vals, x2_vals, 'k--', label=get_text("decision_boundary", "perceptron"))

        # Configure the plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend()
        ax.set_title(get_text("graph_title", "perceptron"))
        ax.grid(True)

        # Show the plot in the container
        plot_container.pyplot(fig)
    
    # Add the line equation section
    st.markdown("---")  # Add a separator
    
    # Add the exercise section
    st.markdown("---")  # Add a separator
    show_exercise()
    
    # Add the limitations section
    st.markdown("---")  # Add a separator
    show_limitations() 