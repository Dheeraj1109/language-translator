import tkinter as tk
from tkinter import ttk
from transformers import MarianMTModel, MarianTokenizer

# Language to model mapping
language_models = {
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Spanish": "Helsinki-NLP/opus-mt-en-es"
}

# Load all models and tokenizers up front
models = {}
tokenizers = {}

for lang, model_name in language_models.items():
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    models[lang] = model
    tokenizers[lang] = tokenizer

def translate():
    input_text = input_box.get("1.0", tk.END).strip()
    target_lang = language_var.get()

    if not input_text:
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "Please enter text to translate.")
        return

    tokenizer = tokenizers[target_lang]
    model = models[target_lang]

    inputs = tokenizer([input_text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    output = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, output)

# GUI setup
root = tk.Tk()
root.title("Offline Translator")
root.geometry("500x400")

tk.Label(root, text="Enter English Text:").pack()
input_box = tk.Text(root, height=5)
input_box.pack()

tk.Label(root, text="Select Target Language:").pack()
language_var = tk.StringVar(value="French")
dropdown = ttk.Combobox(root, textvariable=language_var, values=list(language_models.keys()))
dropdown.pack()

tk.Button(root, text="Translate", command=translate).pack(pady=10)

tk.Label(root, text="Translated Text:").pack()
output_box = tk.Text(root, height=5)
output_box.pack()

root.mainloop()
