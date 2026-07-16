

# 🎓 Emotion-Aware Learning Assistant

A dual-model (BiLSTM + BERT) system that detects a student's emotional state — **Bored, Confident, Confused, Curious, or Frustrated** — from free-text input, and generates personalized, AI-powered learning guidance in response.

---

## 📖 Overview

Students often struggle silently: getting stuck on a problem, disengaging from repetitive material, or gaining confidence — without any of it being visible to an instructor or platform. This project detects that emotional state directly from what a student types, and responds with tailored guidance instead of generic, one-size-fits-all content.

**How it works, end to end:**

1. Student selects their field of study and describes a problem or challenge.
2. Two independently trained models — a **BiLSTM** neural network and a fine-tuned **BERT** transformer — each classify the emotion behind the text.
3. A keyword-enhancement layer boosts predictions when explicit emotional language is present, and a mixed-emotion detector flags secondary emotions above a 15% confidence threshold.
4. The detected emotion, confidence, field, and problem text are sent to an AI guidance engine (**Gemini**, with automatic fallback to **OpenRouter → DeepSeek → Grok**, and a static template as a last resort) to generate a supportive, field-specific response.
5. Every interaction is logged to CSV, and a live analytics dashboard tracks emotional patterns across the session.

---

## ✨ Features

- **Dual-model emotion classification** — BiLSTM (fast, ~4.7M params) and BERT (higher accuracy, 85% test accuracy) running side by side
- **Domain-adapted models** — both models fine-tuned specifically on student/academic-context language
- **Mixed-emotion detection** — surfaces secondary emotions (e.g. "Curious + Confused") instead of forcing a single label
- **Keyword-based confidence boosting** — sharpens predictions when explicit emotional language appears
- **4-provider AI fallback chain** — Gemini → OpenRouter → DeepSeek → Grok → offline template, so a response is always returned
- **Session history & CSV persistence** — every interaction logged for future analysis or retraining
- **Interactive analytics dashboard** — Plotly-powered charts across Emotions, Fields, and Summary tabs
- **Configurable settings** — toggle AI responses, CSV logging, and detailed score breakdowns on the fly

---

## 🏗️ Project Structure

```
emotion-detection/
├── data/
│   ├── combined_preprocessed.csv       # Cleaned, tokenized training data
│   ├── padded_sequences.npy            # Tokenized BiLSTM input sequences
│   └── student_domain_data.csv         # Synthetic student-context fine-tuning data
├── models/
│   ├── bltsm/
│   │   ├── bilstm_student_adaptive.keras
│   │   ├── tokenizer.pickle
│   │   └── label_encoder.pickle
│   ├── bert_emotion_model_final/       # Base fine-tuned BERT
│   └── bert_student_adaptive/          # Domain-adapted BERT (best model)
├── src/
│   ├── preprocessing.py                # Text cleaning + tokenization
│   ├── train.py                        # BiLSTM training script
│   ├── bilstm_predictor.py             # BiLSTM inference class
│   ├── bert_model.py                   # BERT inference class (with class weighting)
│   ├── keyword_enhancement.py          # Keyword-based probability boosting
│   ├── mixed_emotion.py                # Mixed-emotion detection + emoji/response mapping
│   ├── unified_schema.py               # Shared prediction schema + validation
│   ├── gemini_helper.py                # AI response engine + 4-provider fallback
│   └── predict.py                      # Standalone prediction CLI
├── streamlit_app.py                    # Main application entry point
├── emotion_response_examples.csv       # Logged interactions
├── emotion_response_mapping.csv        # Emotion → response mapping
├── requirements.txt
├── .env                                # API keys (not committed)
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit, Plotly |
| Classical/Sequential Model | TensorFlow / Keras — Bidirectional LSTM |
| Transformer Model | PyTorch + Hugging Face Transformers — fine-tuned `bert-base-uncased` |
| AI Response Generation | Google Gemini 2.5 Flash → OpenRouter → DeepSeek → Grok (xAI) |
| Data Processing | pandas, NumPy, scikit-learn, NLTK |
| Training Infrastructure | Kaggle (dual T4 GPU) |
| Local Inference | Python 3.11, NVIDIA GPU (CUDA 12.1) |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11 (required — Keras 3.13.2 needs Python ≥3.11)
- An NVIDIA GPU with CUDA support (optional but recommended for faster BERT inference)
- API key(s) for at least one AI provider (Gemini recommended; others optional as fallback)

### 1. Clone and enter the project
```bash
cd emotion-detection
```

### 2. Create and activate a virtual environment
```bash
py -3.11 -m venv py-venv
source py-venv/Scripts/activate     # Windows (Git Bash)
# or: source py-venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt --only-binary=:all:
```

### 4. Configure API keys
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
GROK_API_KEY=your_grok_key_here
```
Only `GEMINI_API_KEY` is required to get started — the others are optional fallbacks. Get keys at:
- Gemini: https://aistudio.google.com/apikey
- OpenRouter: https://openrouter.ai (Keys section, free tier available)
- DeepSeek: https://platform.deepseek.com
- Grok: https://console.x.ai

### 5. Verify the model files are in place
```
models/bltsm/bilstm_student_adaptive.keras
models/bltsm/tokenizer.pickle
models/bltsm/label_encoder.pickle
models/bert_student_adaptive/  (config.json, model.safetensors, tokenizer.json, tokenizer_config.json)
```

---

## ▶️ Running the Application

```bash
streamlit run streamlit_app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

### Quick test
1. Select a field (e.g. "Computer Science")
2. Type a problem, or click one of the **Quick Examples**
3. Click **🔍 Get AI Learning Help**
4. Review the detected emotion, confidence scores, and AI-generated guidance
5. Check the **Learning Analytics** section at the bottom after a few interactions

### Running components individually
```bash
python src/predict.py              # BERT + keyword-enhanced prediction, CLI test
python src/bilstm_predictor.py     # BiLSTM prediction, CLI test
python src/bert_model.py           # BERT with class weighting, CLI test
python src/gemini_helper.py        # Test the AI fallback chain directly
```

---

## 📊 Model Performance

| Metric | BiLSTM | BERT |
|---|---|---|
| Overall Accuracy | 74.7% | **85.0%** |
| Bored F1 | 1.00 | 1.00 |
| Confident F1 | 0.91 | 0.99 |
| Confused F1 | 0.46 | 0.54 |
| Curious F1 | 0.50 | 0.68 |
| Frustrated F1 | 0.85 | 0.93 |

Both models were further validated on hand-written sentences never seen during training — after domain-adaptive fine-tuning, BERT correctly classified 4/4 novel test sentences (up from 1/4 before adaptation).

---

## 🖼️ Screenshots

Add screenshots here to give readers a quick visual sense of the app before they run it. Suggested set:

- [ ] **Main input screen** — field selector, problem text area, and Quick Examples
- [ ] **Prediction results** — BiLSTM vs. BERT side-by-side comparison with confidence bars
- [ ] **AI-generated response** — the guidance panel with the Analysis Details expander open
- [ ] **Analytics dashboard** — all three tabs (Emotions, Fields, Summary)
- [ ] **Sidebar dashboard** — model status, interaction count, recent sessions

To add an image, save it into a `docs/screenshots/` folder and reference it like this:

```markdown
![Main input screen](docs/screenshots/main-input.png)
```

---

## ⚠️ Known Limitations

- The **Bored** class has no organic source data — it is entirely synthetically generated, unlike the other four classes which come from real datasets (GoEmotions, EmpatheticDialogues, ISEAR).
- Domain-adaptation training data (student-context sentences) is template-generated rather than collected from real students, which may limit generalization to genuinely novel phrasing.
- Free-tier AI provider quotas (especially Gemini) can be exhausted quickly under heavy testing — the fallback chain mitigates but does not eliminate this.
- CSV-based persistence is suitable for prototyping but should be replaced with a proper database for production-scale usage.

---
