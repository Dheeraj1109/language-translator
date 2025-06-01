import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load M2M100 model - better for multilingual translation
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

def translate_text(text, src_lang, tgt_lang):
    lang_codes = {
        "English": "en",
        "French": "fr",
        "German": "de",
        "Spanish": "es"
    }
    
    tokenizer.src_lang = lang_codes[src_lang]
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(lang_codes[tgt_lang])
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Streamlit UI
st.title("Multi-Language Translator")

source_text = st.text_area("Enter text to translate:")
source_lang = st.selectbox("Source Language", ["English", "French", "German", "Spanish"])
target_lang = st.selectbox("Target Language", ["English", "French", "German", "Spanish"])

if st.button("Translate"):
    if source_text:
        translation = translate_text(source_text, source_lang, target_lang)
        st.success(f"Translated Text: {translation}")
    else:
        st.error("Please enter text to translate!")