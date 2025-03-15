import streamlit as st
from sections.perceptron import show_perceptron
from utils.i18n import setup_language_selector, get_text

# Configure the page
st.set_page_config(
    page_title=get_text("title"),
    page_icon="🧠",
    layout="wide"
)

# Setup language selector in sidebar
setup_language_selector()

# Sidebar menu
st.sidebar.title(get_text("navigation"))
page = st.sidebar.radio("Go to", ["Home", "Perceptron"])

# Main content
if page == "Home":
    st.title(get_text("welcome"))
    st.write(get_text("description"))
    
    st.write(get_text("available_sections"))
    st.write("- **Perceptron**: " + get_text("title", "perceptron"))
elif page == "Perceptron":
    show_perceptron() 