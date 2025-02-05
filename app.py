import streamlit as st
from transformers import pipeline, M2M100Tokenizer, M2M100ForConditionalGeneration

# Initialize session state for translation output
if "translation_output" not in st.session_state:
    st.session_state.translation_output = ""

# Initialize Language Detection and Translation Models
lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

def detect_language(text: str) -> str:
    """Detects the language of the given text."""
    if text.strip() == "":
        return "Unknown"
    
    result = lang_detector(text, top_k=1)
    return result[0]['label']

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translates text from source language to target language."""
    if text.strip() == "":
        return ""
    
    m2m_tokenizer.src_lang = src_lang.lower()
    encoded = m2m_tokenizer(text, return_tensors="pt")
    generated_tokens = m2m_model.generate(
        **encoded,
        forced_bos_token_id=m2m_tokenizer.get_lang_id(tgt_lang.lower())
    )
    return m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Supported Languages (M2M100 100+ Languages)
language_options = {

        "Afrikaans": "af", "Amharic": "am", "Arabic": "ar", "Asturian": "ast", "Azerbaijani": "az",
    "Bashkir": "ba", "Belarusian": "be", "Bulgarian": "bg", "Bengali": "bn", "Breton": "br",
    "Bosnian": "bs", "Catalan": "ca", "Cebuano": "ceb", "Czech": "cs", "Welsh": "cy",
    "Danish": "da", "German": "de", "Greek": "el", "English": "en", "Spanish": "es",
    "Estonian": "et", "Basque": "eu", "Persian": "fa", "Finnish": "fi", "French": "fr",
    "Irish": "ga", "Galician": "gl", "Gujarati": "gu", "Hausa": "ha", "Hebrew": "he",
    "Hindi": "hi", "Croatian": "hr", "Hungarian": "hu", "Armenian": "hy", "Indonesian": "id",
    "Igbo": "ig", "Icelandic": "is", "Italian": "it", "Japanese": "ja", "Javanese": "jv",
    "Georgian": "ka", "Kazakh": "kk", "Khmer": "km", "Kannada": "kn", "Korean": "ko",
    "Kurdish": "ku", "Kyrgyz": "ky", "Latin": "la", "Luxembourgish": "lb", "Ganda": "lg",
    "Lingala": "ln", "Lao": "lo", "Lithuanian": "lt", "Latvian": "lv", "Malagasy": "mg",
    "Maori": "mi", "Macedonian": "mk", "Malayalam": "ml", "Mongolian": "mn", "Marathi": "mr",
    "Malay": "ms", "Maltese": "mt", "Burmese": "my", "Nepali": "ne", "Dutch": "nl",
    "Norwegian": "no", "Occitan": "oc", "Punjabi": "pa", "Polish": "pl", "Pashto": "ps",
    "Portuguese": "pt", "Romanian": "ro", "Russian": "ru", "Sindhi": "sd", "Sinhala": "si",
    "Slovak": "sk", "Slovenian": "sl", "Shona": "sn", "Somali": "so", "Albanian": "sq",
    "Serbian": "sr", "Sundanese": "su", "Swedish": "sv", "Swahili": "sw", "Tamil": "ta",
    "Telugu": "te", "Tajik": "tg", "Thai": "th", "Turkmen": "tk", "Tagalog": "tl",
    "Turkish": "tr", "Tatar": "tt", "Ukrainian": "uk", "Urdu": "ur", "Uzbek": "uz",
    "Vietnamese": "vi", "Xhosa": "xh", "Yoruba": "yo", "Chinese": "zh", "Zulu": "zu"
}

# --- Streamlit UI ---
st.title("🌍 Multilingual Translation & Detection Tool")
st.write("Detect the language of your text and translate it into over **100 languages**.")

# User selects input language
input_language = st.selectbox("Select input language:", list(language_options.keys()))

# Input text area
input_text = st.text_area("Enter text to translate:", height=100)

# User selects target language
target_language = st.selectbox("Select target language:", list(language_options.keys())[1:])  # Remove "Auto Detect" for target

# Translation button
if st.button("Translate"):
    if input_text:
        # Detect language only if "Auto Detect" is selected
        detected_lang = detect_language(input_text) if input_language == "Auto Detect" else language_options[input_language]

        st.write(f"**Detected Language:** {detected_lang}")

        try:
            # Translate text
            translation = translate_text(input_text, src_lang=detected_lang, tgt_lang=language_options[target_language])

            # ✅ Update session state before using it in widgets
            st.session_state.translation_output = translation  

        except Exception as e:
            st.error(f"Translation error: {e}")
    else:
        st.warning("Please enter some text to translate.")

# Display translated text
st.text_area("Translated Text:", value=st.session_state.translation_output, height=100, disabled=True)

# Copy text button
if st.button("📋 Copy Translated Text"):
    st.session_state.clipboard = st.session_state.translation_output
    st.success("Text copied to clipboard! (Copy manually if needed)")
