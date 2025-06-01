from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Download and save T5-base model (for English to French, German, etc.)
tokenizer_t5 = AutoTokenizer.from_pretrained("t5-base")
model_t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

tokenizer_t5.save_pretrained("models/t5-base")
model_t5.save_pretrained("models/t5-base")

# Download and save English to Spanish model
tokenizer_es = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
model_es = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")

tokenizer_es.save_pretrained("models/opus-mt-en-es")
model_es.save_pretrained("models/opus-mt-en-es")
