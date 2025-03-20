import streamlit as st
from utils.i18n import get_text

def show_books():
    st.title(get_text("title", "books"))
    st.write(get_text("description", "books"))
    
    # Create two columns for the books
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(get_text("neural_networks.title", "books"))
        st.write(get_text("neural_networks.description", "books"))
        st.write(f"**{get_text('neural_networks.price', 'books')}**")
        st.markdown(f"[Purchase on Payhip](https://payhip.com/b/Vtn4Z)")
    
    with col2:
        st.subheader(get_text("competitive_programming.title", "books"))
        st.write(get_text("competitive_programming.description", "books"))
        st.write(f"**{get_text('competitive_programming.price', 'books')}**")
        st.markdown(f"[Purchase on Payhip](https://payhip.com/b/miKIt)")
    
    st.write("---")
    st.write(get_text("thank_you", "books"))
    st.info(get_text("note", "books")) 