from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load offline T5-base model
tokenizer_t5 = AutoTokenizer.from_pretrained("models/t5-base")
model_t5 = AutoModelForSeq2SeqLM.from_pretrained("models/t5-base")

# Load offline English-to-Spanish model
tokenizer_es = AutoTokenizer.from_pretrained("models/opus-mt-en-es")
model_es = AutoModelForSeq2SeqLM.from_pretrained("models/opus-mt-en-es")


# T5-based generic translator (e.g., English to French/German/etc.)
def translate_with_t5(input_text, src_lang, tgt_lang):
    task_prefix = f"translate {src_lang} to {tgt_lang}: "
    formatted_input = task_prefix + input_text

    inputs = tokenizer_t5(
        formatted_input, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    outputs = model_t5.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    return tokenizer_t5.decode(outputs[0], skip_special_tokens=True)

# Helsinki-NLP specialized English to Spanish translation
def translate_to_spanish(input_text):
    inputs = tokenizer_es(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model_es.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    return tokenizer_es.decode(outputs[0], skip_special_tokens=True)

# Example usage
source_text = "How are you? I am fine."

translations = {
    "French": translate_with_t5(source_text, "English", "French"),
    "German": translate_with_t5(source_text, "English", "German"),
    "Spanish": translate_to_spanish(source_text),  # uses dedicated model
}

# Print results
for lang, translated_text in translations.items():
    print(f"Translated to {lang}: {translated_text}")
