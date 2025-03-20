import streamlit as st
from utils.i18n import get_text, setup_language_selector
from sections.perceptron import show_perceptron
from sections.linear_regression import show_linear_regression
from sections.books import show_books

def show_main_page():
    st.title(get_text("title", "app"))
    st.markdown(get_text("welcome", "app"))
    st.markdown(get_text("description", "app"))
    st.markdown("---")
    st.markdown(f"### {get_text('available_sections', 'app')}")
    st.markdown("""
    - **Linear Regression**: {linear_regression_title}
    - **Perceptron**: {perceptron_title}
    """.format(
        linear_regression_title=get_text("title", "linear_regression"),
        perceptron_title=get_text("title", "perceptron")
    ))

def main():
    setup_language_selector()

    st.sidebar.title(get_text("navigation", "app"))
    pages = {
        "Main": show_main_page,
        "Linear Regression": show_linear_regression,
        "Perceptron": show_perceptron,
        "Books": show_books
    }
    
    page = st.sidebar.radio(
        label="",
        options=list(pages.keys()),
        label_visibility="collapsed"
    )
    
    pages[page]()

if __name__ == "__main__":
    main() 