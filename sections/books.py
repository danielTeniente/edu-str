import streamlit as st
from utils.i18n import get_text

def show_books():
    st.title("Mis libros de tecnología")
    st.write("En esta sección encontrarás los libros que he escrito sobre redes neuronales y programación competitiva.")
    
    # Create two columns for the books
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Atrapado en redes neuronales")
        st.write("Un libro donde hablo de mi experiencia trabajando en el campo de la inteligencia artificial.")
        st.write("Puedes comprarlo en Payhip.")
        st.markdown(f"[Purchase on Payhip](https://payhip.com/b/Vtn4Z)")
    
    with col2:
        st.subheader("Un mundo muy complejo")
        st.write("Este es un libro de divulgación donde explico algunos algoritmos que aprendí en competencias de programación.")
        st.write("Puedes comprarlo en Payhip.")
        st.markdown(f"[Purchase on Payhip](https://payhip.com/b/miKIt)")
    
    st.write("---")
    st.write("Gracias por visitar mi sitio web.")