"""
BiLSTM Emotion Predictor
- Supports fast, lightweight PyTorch BiLSTM inference (models/bltsm/bilstm_model.pt).
- Gracefully falls back to TensorFlow / Keras (models/bltsm/bilstm_student_adaptive.keras) if needed.
- Standardized 5-class probability distribution with full schema compliance.
"""

import os
import sys
import json
import pickle
import numpy as np
import re

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.join(os.path.dirname(__file__)))
from unified_schema import validate_prediction

# Fast default English stopwords to avoid slow NLTK network initialization
DEFAULT_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

MAX_SEQ_LEN = 80
EMBEDDING_DIM = 128
LSTM_UNITS = 128


class BiLSTMTorchModel:
    """Lightweight PyTorch BiLSTM implementation for instant loading (<0.02s)."""
    def __init__(self, model_path, vocab_path, config_path=None):
        import torch
        import torch.nn as nn

        class _BiLSTMModule(nn.Module):
            def __init__(self, vocab_size, embedding_dim, lstm_units, num_classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.lstm = nn.LSTM(
                    embedding_dim, lstm_units, batch_first=True,
                    bidirectional=True, dropout=0.0
                )
                self.dropout1 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(lstm_units * 2, 64)
                self.relu = nn.ReLU()
                self.dropout2 = nn.Dropout(0.3)
                self.fc2 = nn.Linear(64, num_classes)

            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, (hidden, _) = self.lstm(embedded)
                h_forward = hidden[0]
                h_backward = hidden[1]
                combined = torch.cat((h_forward, h_backward), dim=1)
                x = self.dropout1(combined)
                x = self.relu(self.fc1(x))
                x = self.dropout2(x)
                return self.fc2(x)

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.device = torch.device("cpu")
        self.torch = torch
        self.model = _BiLSTMModule(len(self.vocab), EMBEDDING_DIM, LSTM_UNITS, 5)
        
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_probs(self, text_tokens):
        # Convert tokens to indexed sequence padded to MAX_SEQ_LEN
        indices = [self.vocab.get(t, 1) for t in text_tokens][:MAX_SEQ_LEN]
        if len(indices) < MAX_SEQ_LEN:
            indices = indices + [0] * (MAX_SEQ_LEN - len(indices))
        
        inp = self.torch.tensor([indices], dtype=self.torch.long, device=self.device)
        with self.torch.no_grad():
            logits = self.model(inp)
            probs = self.torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs


class EmotionPredictor:
    """Unified BiLSTM Predictor supporting PyTorch fast loading and Keras fallback."""

    def __init__(
        self,
        model_path="models/bltsm/bilstm_model.pt",
        tokenizer_path="models/bltsm/tokenizer.pickle",
        label_encoder_path="models/bltsm/label_encoder.pickle",
        vocab_path="models/bltsm/vocab.json",
    ):
        print("📦 Loading BiLSTM model...")
        self.stopwords = DEFAULT_STOPWORDS
        self.classes = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]
        self.use_torch = False
        self.torch_predictor = None
        self.keras_model = None
        self.tokenizer = None

        # 1. Try ultra-fast PyTorch loading
        torch_model_path = "models/bltsm/bilstm_model.pt"
        if os.path.exists(torch_model_path) and os.path.exists(vocab_path):
            try:
                self.torch_predictor = BiLSTMTorchModel(torch_model_path, vocab_path)
                self.use_torch = True
                print("⚡ Loaded PyTorch BiLSTM (Instant & Lightweight)")
            except Exception as e:
                print(f"⚠️ PyTorch BiLSTM fallback: {e}")
                self.use_torch = False

        # 2. Fallback to Keras if PyTorch is not available and tensorflow exists
        if not self.use_torch:
            try:
                keras_path = "models/bltsm/bilstm_student_adaptive.keras"
                if os.path.exists(keras_path):
                    import tensorflow as tf
                    from keras.src.utils.sequence_utils import pad_sequences
                    self.pad_sequences = pad_sequences
                    try:
                        self.keras_model = tf.keras.models.load_model(keras_path)
                    except Exception:
                        self.keras_model = tf.keras.models.load_model(keras_path, compile=False)

                    with open(tokenizer_path, "rb") as f:
                        self.tokenizer = pickle.load(f)

                    if os.path.exists(label_encoder_path):
                        with open(label_encoder_path, "rb") as f:
                            le = pickle.load(f)
                            self.classes = list(le.classes_)

                    print("✅ Loaded Keras BiLSTM model")
            except Exception as ke:
                print(f"⚠️ Keras fallback note: {ke}")

        print("✅ BiLSTM ready. Classes:", self.classes)

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        tokens = re.findall(r"\b[a-zA-Z]+\b", text)
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        return " ".join(tokens)

    def predict(self, text: str) -> dict:
        cleaned = self.clean_text(text)
        tokens = cleaned.split() if cleaned.strip() else text.lower().split()

        if not tokens:
            return {
                "emotion": "Confused",
                "confidence": 0.5,
                "scores": {cls: round(1 / len(self.classes), 4) for cls in self.classes},
                "cleaned_text": cleaned,
            }

        if self.use_torch and self.torch_predictor:
            probs = self.torch_predictor.predict_probs(tokens)
        elif self.keras_model and self.tokenizer:
            sequence = self.tokenizer.texts_to_sequences([cleaned])
            if not sequence or not sequence[0]:
                probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            else:
                padded = self.pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
                probs = self.keras_model.predict(padded, verbose=0)
                probs = np.array(probs).flatten()
        else:
            probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        if len(probs) != len(self.classes):
            probs = np.resize(probs, len(self.classes))
            probs = probs / np.sum(probs)

        # Apply keyword boost
        from keyword_enhancement import apply_keyword_boost
        boosted_probs, _ = apply_keyword_boost(text, probs, self.classes)

        pred_idx = int(np.argmax(boosted_probs))
        predicted_emotion = self.classes[pred_idx]
        confidence = float(boosted_probs[pred_idx])

        result = {
            "emotion": predicted_emotion,
            "confidence": round(confidence, 4),
            "scores": {cls: round(float(p), 4) for cls, p in zip(self.classes, boosted_probs)},
            "cleaned_text": cleaned,
        }

        validate_prediction(result)
        return result


if __name__ == "__main__":
    predictor = EmotionPredictor()
    test_sentences = [
        "why does this equation flip when i move the variable to the other side",
        "i already know all of this, can we move faster",
        "i totally understand this topic now",
        "why does this keep failing no matter what i try",
    ]

    for s in test_sentences:
        res = predictor.predict(s)
        print(f"'{s}' -> {res['emotion']} ({res['confidence']:.1%}) | {res['scores']}")