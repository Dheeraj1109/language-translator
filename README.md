# Real-Time Language Translation using NMT

## Project Overview
This project is a real-time language translation system using **Neural Machine Translation (NMT)** models from the **Hugging Face Transformers library**. It utilizes **Google T5 (Text-to-Text Transfer Transformer)** for multilingual translation and **Helsinki-NLP's OPUS-MT** models for specific language pairs.

## Features
- Translates text from English to multiple languages, including **French, German, and Spanish**.
- Uses **Google T5 (t5-base)** for general multilingual translation.
- Uses **Helsinki-NLP OPUS-MT** for specific language pairs (e.g., English to Spanish).
- Streamlit-based UI for user-friendly interaction (coming soon!).

## Technologies Used
- Python
- Hugging Face **Transformers**
- **T5-base** for multilingual translation
- **Helsinki-NLP OPUS-MT** for specific translations
- **Streamlit** for UI (upcoming feature)

## Installation
### **1. Clone the Repository**
```sh
git clone https://github.com/yourusername/Real-Time-Language-Translation-NMT.git
cd Real-Time-Language-Translation-NMT
```

### **2. Set Up a Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **3. Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4. Install Additional Libraries (if needed)**
```sh
pip install transformers sacremoses sentencepiece
```

## Usage
Run the Python script to translate text:
```sh
python translation.py
```

## Code Structure
```
Real-Time-Language-Translation-NMT/
├── translation.py          # Main script for translation
├── app.py                   # Streamlit UI (coming soon)
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation
└── venv/                   # Virtual environment (not included in repo)
```

## Example Output
```sh
Translated to French: Comment vous êtes-vous ?
Translated to German: Wie läuft es Ihnen?
Translated to Spanish: ¿Cómo estás?
```

## Streamlit UI (Upcoming Feature)
A **web-based UI** using Streamlit will be added to allow users to input text and select target languages for translation in an interactive interface.

To run the Streamlit UI (once implemented):
```sh
streamlit run ui.py
```


