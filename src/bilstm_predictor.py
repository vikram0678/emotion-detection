
import tensorflow as tf

import json
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords


from keras.src.utils.sequence_utils import pad_sequences as pad_sequences
from unified_schema import validate_prediction

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

MAX_SEQ_LEN = 80


class EmotionPredictor:
    """Loads the trained BiLSTM model and predicts emotion probabilities via softmax."""

    def __init__(
        self,
        model_path="models/bltsm/bilstm_student_adaptive.keras",
        tokenizer_path="models/bltsm/tokenizer.pickle",
        label_encoder_path="models/bltsm/label_encoder.pickle",
    ):
        print("📦 Loading BiLSTM model...")
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception:
            # Fallback for models saved with a custom loss function (e.g. focal loss)
            self.model = tf.keras.models.load_model(model_path, compile=False)

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.classes = list(self.label_encoder.classes_)
        self.stopwords = set(stopwords.words("english"))

        print("✅ BiLSTM loaded. Classes:", self.classes)

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        return " ".join(tokens)

    def predict(self, text: str) -> dict:
        cleaned = self.clean_text(text)

        if not cleaned.strip():
            cleaned = text.lower()

        sequence = self.tokenizer.texts_to_sequences([cleaned])

        # If the cleaned text produced no recognizable tokens, return a safe fallback
        if not sequence or not sequence[0]:
            return {
                "emotion": "Confused",
                "confidence": 0.5,
                "scores": {cls: round(1 / len(self.classes), 4) for cls in self.classes},
                "cleaned_text": cleaned,
            }

        padded = pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

        probs = self.model.predict(padded, verbose=0)
        probs = np.array(probs).flatten()

        # Safety check: make sure output length matches number of classes
        if len(probs) != len(self.classes):
            probs = np.resize(probs, len(self.classes))

        # Softmax normalization (in case raw output isn't already a clean probability dist)
        # probs = np.exp(probs - np.max(probs))  # subtract max for numerical stability
        # probs = probs / np.sum(probs)

        pred_idx = int(np.argmax(probs))
        predicted_emotion = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        result = {
            "emotion": predicted_emotion,
            "confidence": round(confidence, 4),
            "scores": {cls: round(float(p), 4) for cls, p in zip(self.classes, probs)},
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
        result = predictor.predict(s)
        print(f"'{s}'")
        print(f"  -> {result['emotion']} ({result['confidence']:.2%})")
        print(f"  scores: {result['scores']}\n")