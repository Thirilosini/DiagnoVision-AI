# 🩺 DiagnoVision AI

An AI-powered medical imaging assistant built with **Streamlit**, **Agno Agents**, and **Google Gemini Vision**, equipped with real-time clinical research capabilities via **DuckDuckGo**.

Repository: [Thirilosini/DiagnoVision-AI](https://github.com/Thirilosini/DiagnoVision-AI.git)

---

## 🚀 Features

- **Automated Modality & Region Detection**: Analyzes X-rays, CT scans, MRIs, and Ultrasounds.
- **Systematic Diagnostic Assessment**: Generates primary findings, ranked differential diagnoses, and confidence estimates.
- **Patient-Friendly Translations**: Translates clinical findings into clear language with visual analogies.
- **Clinical Research References**: Searches real-time medical literature and treatment pathways via DuckDuckGo.
- **Secure Key Handling**: Support for `.env` configuration and sidebar key input with masked display.
- **Report Export**: Download diagnostic reports directly in Markdown format.

---

## 🛠️ Setup Instructions

### 1. Create and Activate a Virtual Environment

```bash
# In the project directory:
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt):
.\venv\Scripts\activate.bat
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Copy `.env.example` to `.env` and set your Google Gemini API key:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(You can also enter your API key directly in the sidebar inside the Streamlit app.)*

### 4. Run the Application

```bash
streamlit run app.py
```

---

## ⚠️ Medical Disclaimer

*This application is for educational and investigational purposes only. It is not intended to replace professional medical advice, clinical diagnosis, or treatment.*
