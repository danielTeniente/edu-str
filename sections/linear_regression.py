import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.i18n import get_text

@st.cache_data
def generate_height_data(n_points=40):  # 40 points total, multiple per age
    """Generate synthetic height data from birth to 20 years."""
    np.random.seed(42)
    
    # Generate ages (integers from 0 to 20)
    # We'll have multiple measurements per age
    ages = np.random.randint(1, 21, n_points)  # Random integer ages
    ages.sort()  # Sort ages for better visualization
    
    # Average height at birth (cm) and approximate yearly growth (cm)
    initial_height = 50  # ~50 cm at birth
    growth_rate = 6.5    # ~6.5 cm per year average
    
    # Generate base heights with growth rate that slows down with age
    # Use sigmoid function to simulate growth spurt and then slowdown
    growth_factor = 1 / (1 + np.exp(-(ages - 12)/2))  # Growth spurt around age 12
    heights = initial_height + growth_rate * (ages + 3 * growth_factor)
    
    # Add random variation that increases with age
    # Small variation at birth (0.5-1 cm), larger variation in final height (±10-15 cm)
    base_variation = 0.5 + ages/2  # Increases with age
    variation = np.random.normal(0, base_variation, n_points)
    
    # Add extra random variation to final height
    final_height_variation = np.random.normal(0, 8, n_points) * (ages/20)**2  # More variation in final height
    
    heights = heights + variation + final_height_variation
    
    # Round heights to 1 decimal place for cleaner display
    heights = np.round(heights, 1)
    
    return ages, heights

def calculate_rmse(x, y, m, b):
    """Calculate Root Mean Squared Error for given line parameters."""
    y_pred = m * x + b
    return np.sqrt(np.mean((y - y_pred) ** 2))

def show_line():
    st.header(get_text("title", "line_basic"))
    st.write(get_text("description_p1", "line_basic"))
    st.latex(r"y=mx+b")
    st.write(get_text("description_p2", "line_basic"))
    st.write(get_text("exercise_desc", "line_basic"))
    # Create two columns for controls and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        m = st.slider(get_text("slope", "line_basic"), -5.0, 5.0, 1.0, 0.1)
        b = st.slider(get_text("intercept", "line_basic"), -5.0, 5.0, 0.0, 0.1)
        # Show current equation
        equation = f"y = {m:.2f}x + {b:.2f}" if b >= 0 else f"y = {m:.2f}x - {abs(b):.2f}"
        st.latex(equation)
    
    with col2:
        # Create an empty container for the plot
        plot_container = st.empty()
        
        # Generate points for the line
        x = np.linspace(-5, 5, 200)
        y = m * x + b
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, 'b-', label='y = mx + b')
        
        # Configure the plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.legend()
        ax.set_title(get_text("plot_title", "line_basic"))
        
        # Show the plot in the container
        plot_container.pyplot(fig)

def show_linear_regression():
    st.title(get_text("title", "linear_regression"))
    st.write(get_text("description_p1", "linear_regression"))

    show_line()

    # example section
    st.header(get_text("ex_title", "linear_regression"))
    st.write(get_text("ex_desc_p1", "linear_regression"))
    st.write(get_text("ex_desc_p2", "linear_regression"))
    st.write(get_text("ex_interactive_desc", "linear_regression"))

    # Interactive visualization section
    # Get height data
    ages, heights = generate_height_data(n_points=50)

    # Create two columns for controls and visualization
    col1, col2 = st.columns([1, 2])

    with col1:
        # Sliders for line parameters
        m = st.slider(get_text("slope", "linear_regression"), 4.0, 9.0, 6.5, 0.1)  # Growth rate in cm/year
        b = st.slider(get_text("intercept", "linear_regression"), 45.0, 55.0, 50.0, 0.5)  # Initial height in cm
        
        # Show current line equation
        st.latex(f"y = {m:.1f} \cdot x + {b:.1f}")
        
        # Calculate and show MSE
        rmse = calculate_rmse(ages, heights, m, b)
        st.write(get_text("ex_rmse", "linear_regression"))
        st.write(f"**{rmse:.2f}** cm")
        
        # Option to show error lines
        show_errors = st.checkbox(get_text("show_errors", "linear_regression"), value=True)

    with col2:
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot data points
        ax.scatter(ages, heights, color='blue', alpha=0.6, label='Height Measurements')
        
        # Plot regression line
        x_line = np.linspace(0, 20, 100)
        y_line = m * x_line + b
        ax.plot(x_line, y_line, 'r-', label='Growth Prediction')
        
        # Plot error lines if enabled
        if show_errors:
            y_pred = m * ages + b
            for xi, yi, y_predi in zip(ages, heights, y_pred):
                ax.plot([xi, xi], [yi, y_predi], 'gray', alpha=0.3)
            ax.plot([], [], 'gray', alpha=0.3, label=get_text("error_lines", "linear_regression"))
        
        # Configure the plot
        ax.grid(True)
        ax.legend()
        ax.set_xlabel(get_text("ex_plot_x_label", "linear_regression"))
        ax.set_ylabel(get_text("ex_plot_y_label", "linear_regression"))
        ax.set_xlim(0, 21)
        ax.set_ylim(40, 220)
        
        # Set integer ticks for x-axis
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        
        ax.set_title(get_text("ex_plot_title","linear_regression"))
        
        # Show the plot
        st.pyplot(fig)
        st.info(get_text("ex_hint", "linear_regression")) 
    
    # Error equation
    st.header(get_text("error_eq_title", "linear_regression"))
    st.write(get_text("error_eq_desc_p1", "linear_regression"))
    st.write(get_text("error_eq_desc_p2", "linear_regression"))
    st.latex(r"MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2")
    st.write(get_text("error_eq_params", "linear_regression"))
    st.write(get_text("error_eq_desc_p3", "linear_regression"))
    st.latex(r"(y_i - \hat{y}_i)^2")
    st.write(get_text("error_eq_desc_p4", "linear_regression"))
    st.write(get_text("error_eq_desc_p5", "linear_regression"))
    st.write(get_text("error_eq_desc_p6", "linear_regression"))
    st.latex(r"RMSE = \sqrt{MSE}")


    # Explanation section
    st.header(get_text("explanation_title", "linear_regression"))
    st.write(get_text("explanation_text", "linear_regression"))

    # Limitations section
    st.header(get_text("limitations_title", "linear_regression"))
    st.write(get_text("limitations_text", "linear_regression"))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Example 1: Quadratic pattern
    x1 = np.linspace(-5, 5, 100)
    y1 = x1**2 + np.random.normal(0, 0.5, 100)
    ax1.scatter(x1, y1, alpha=0.5, label='Data points')
    ax1.plot(x1, x1**2, 'r-', label='True pattern')
    ax1.set_title('Quadratic Pattern')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    ax1.legend()
    
    # Example 2: Exponential pattern
    x2 = np.linspace(0, 2, 100)
    y2 = np.exp(x2) + np.random.normal(0, 0.5, 100)
    ax2.scatter(x2, y2, alpha=0.5, label='Data points')
    ax2.plot(x2, np.exp(x2), 'r-', label='True pattern')
    ax2.set_title('Exponential Pattern')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    ax2.legend()
    
    # Example 3: Periodic pattern
    x3 = np.linspace(0, 4*np.pi, 100)
    y3 = np.sin(x3) + np.random.normal(0, 0.2, 100)
    ax3.scatter(x3, y3, alpha=0.5, label='Data points')
    ax3.plot(x3, np.sin(x3), 'r-', label='True pattern')
    ax3.set_title('Periodic Pattern')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.grid(True)
    ax3.legend()
    
    # Adjust layout and display
    plt.tight_layout()
    st.pyplot(fig)
    