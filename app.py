import streamlit as st
from transformers import pipeline, M2M100Tokenizer, M2M100ForConditionalGeneration

# Initialize language detection model
lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

def detect_language(text: str) -> str:
    result = lang_detector(text, top_k=1)
    return result[0]['label']

# Initialize translation model
m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate text from source language to target language.
    """
    m2m_tokenizer.src_lang = src_lang.lower()
    encoded = m2m_tokenizer(text, return_tensors="pt")
    generated_tokens = m2m_model.generate(
        **encoded,
        forced_bos_token_id=m2m_tokenizer.get_lang_id(tgt_lang.lower())
    )
    return m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Supported languages (M2M100 uses ISO language codes)
language_options = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Hindi": "hi",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Portuguese": "pt",
    "Arabic": "ar",
    "Italian": "it"
}

# --- Streamlit UI ---

st.title("🌍 Translation & Detection Tool")
st.write("This tool detects the language of your input text and translates it into the chosen target language.")

# Text input area for the text to be translated.
input_text = st.text_area("Enter text to translate:")

# Language selection dropdown
target_language = st.selectbox("Select target language:", list(language_options.keys()))

if st.button("Translate"):
    if input_text:
        # Step 1: Detect the language of the input text.
        detected_lang = detect_language(input_text)
        st.write(f"**Detected language:** {detected_lang}")

        try:
            translation = translate_text(input_text, src_lang=detected_lang, tgt_lang=language_options[target_language])
            
            # Display translated text
            st.text_area("Translated Text:", translation, height=100)
            
            # Copy text button
            st.button("📋 Copy Translated Text", key="copy")
            st.markdown(
                f"""
                <script>
                    function copyText() {{
                        navigator.clipboard.writeText("{translation}");
                        alert("Text copied to clipboard!");
                    }}
                    document.getElementById("copy").addEventListener("click", copyText);
                </script>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Translation error: {e}")
    else:
        st.warning("Please enter some text to translate.")
