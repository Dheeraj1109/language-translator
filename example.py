from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load a pre-trained T5 model for translation
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

def translate_with_t5(input_text, src_lang, tgt_lang):
    # T5 uses task prefixes to define translation tasks
    task_prefix = f"translate {src_lang} to {tgt_lang}: "
    input_text = task_prefix + input_text

    # Tokenize and encode input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Generate translation
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_text

# Example usage
source_text = "Hello my name is shashank"
source_language = "French"
target_language = "English"

translated_text = translate_with_t5(source_text, source_language, target_language)
print(f"Translated Text: {translated_text}")
