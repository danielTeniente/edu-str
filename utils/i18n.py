import json
import os
import streamlit as st

LANGUAGES = {
    "en": "English",
    "es": "Español",
    "pt": "Português"
}

def load_translations(lang_code):
    """Load translations for the specified language code."""
    try:
        with open(f"languages/{lang_code}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading translations for {lang_code}: {str(e)}")
        # Fallback to English
        with open("languages/en.json", "r", encoding="utf-8") as f:
            return json.load(f)

def get_text(key, section="app"):
    """Get translated text for a given key and section."""
    if "translations" not in st.session_state:
        st.session_state.translations = load_translations("en")
        st.session_state.language = "en"
    
    try:
        return st.session_state.translations[section][key]
    except KeyError:
        return f"Missing translation: {section}.{key}"

def change_language():
    """Callback function to handle language changes."""
    selected_lang = st.session_state.language_selector
    if selected_lang != st.session_state.language:
        st.session_state.translations = load_translations(selected_lang)
        st.session_state.language = selected_lang
        st.rerun()

def setup_language_selector():
    """Add language selector to sidebar and handle language changes."""
    if "language" not in st.session_state:
        st.session_state.language = "en"
        st.session_state.translations = load_translations("en")
    
    st.sidebar.selectbox(
        "Language/Idioma/Língua",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        key="language_selector",
        index=list(LANGUAGES.keys()).index(st.session_state.language),
        on_change=change_language
    ) 