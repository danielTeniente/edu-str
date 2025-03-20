import json
import os
import streamlit as st

# Initialize session state for language if not exists
if 'language' not in st.session_state:
    st.session_state.language = 'es'

LANGUAGES = {
    "es": "Español",
    "en": "English",
    "pt": "Português"
}

def load_translations(lang_code):
    """Load translations for the specified language code."""
    try:
        file_path = os.path.join('languages', f'{lang_code}.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to English if translation file not found
        file_path = os.path.join('languages', 'en.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def get_text(key, section=None):
    """Get translated text for a key, optionally within a section."""
    translations = load_translations(st.session_state.language)
    
    try:
        if section:
            return translations[section][key]
        return translations[key]
    except KeyError:
        # Return the key itself if translation not found
        return key

def set_language(lang_code):
    """Set the current language."""
    st.session_state.language = lang_code

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
        st.session_state.language = "es"
        st.session_state.translations = load_translations("es")
    
    st.sidebar.selectbox(
        "Language/Idioma/Língua",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        key="language_selector",
        index=list(LANGUAGES.keys()).index(st.session_state.language),
        on_change=change_language
    ) 