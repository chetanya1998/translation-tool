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
    m2m_tokenizer.src_lang = src_lang.lower()
    encoded = m2m_tokenizer(text, return_tensors="pt")
    generated_tokens = m2m_model.generate(
        **encoded,
        forced_bos_token_id=m2m_tokenizer.get_lang_id(tgt_lang.lower())
    )
    return m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Streamlit UI
st.title("Translation & Detection Tool")
st.write("This tool detects the language of your input text and translates it to the chosen target language.")

input_text = st.text_area("Enter text to translate:")
target_language = st.text_input("Target language code (e.g., 'en' for English):", value="en")

if st.button("Translate"):
    if input_text:
        detected_lang = detect_language(input_text)
        st.write(f"**Detected language:** {detected_lang}")

        try:
            translation = translate_text(input_text, src_lang=detected_lang, tgt_lang=target_language)
            st.write("**Translation:**")
            st.write(translation)
        except Exception as e:
            st.error(f"Translation error: {e}")
    else:
        st.warning("Please enter some text to translate.")
