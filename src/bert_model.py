import os
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unified_schema import validate_prediction


class BERTEmotionClassifier:
    """Loads the fine-tuned BERT model and applies class weighting +
    keyword-based confidence/confusion adjustments to sharpen predictions."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_labels = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]
        self.id2label = {i: label for i, label in enumerate(self.emotion_labels)}

    def load_model(self, model_path="models/bert_student_adaptive"):
        print("📦 Loading BERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load label mappings if available (optional override of default order)
        label_path = os.path.join(model_path, "label_mappings.pkl")
        if os.path.exists(label_path):
            with open(label_path, "rb") as f:
                mappings = pickle.load(f)
            if "id2label" in mappings:
                self.id2label = mappings["id2label"]
                self.emotion_labels = [mappings["id2label"][i] for i in range(len(mappings["id2label"]))]

        print("✅ BERT loaded on", self.device, "| Classes:", self.emotion_labels)

    def predict(self, text):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # Base class weights (tuned per class — reduces Confused over-triggering,
        # boosts Confident slightly since it's historically underrepresented)
        # Order: Bored, Confident, Confused, Curious, Frustrated
        class_weights = np.array([1.2, 1.8, 0.6, 1.0, 1.4])

        # --- Keyword-based adjustments ---
        text_lower = text.lower()
        confidence_keywords = ["comfortable", "confident", "easy", "clear", "understand", "got it", "makes sense"]
        confusion_keywords = ["confused", "unclear", "lost", "don't understand", "puzzled"]

        if any(keyword in text_lower for keyword in confidence_keywords):
            class_weights[1] *= 2.5   # Boost Confident
            class_weights[2] *= 0.3   # Reduce Confused
        elif any(keyword in text_lower for keyword in confusion_keywords):
            class_weights[2] *= 2.0   # Boost Confused

        weighted_probs = probs * class_weights
        pred_id = int(np.argmax(weighted_probs))
        emotion = self.id2label[pred_id]

        normalized_sum = np.sum(weighted_probs)

        result =  {
            "emotion": emotion,
            "confidence": float(weighted_probs[pred_id] / normalized_sum),
            "scores": {self.id2label[i]: float(weighted_probs[i] / normalized_sum) for i in range(len(self.emotion_labels))},
            "cleaned_text": text.strip(),
        }

        validate_prediction(result)
        return result


if __name__ == "__main__":
    classifier = BERTEmotionClassifier()
    classifier.load_model()

    test_sentences = [
        "why does this equation flip when i move the variable to the other side",
        "i already know all of this, can we move faster",
        "i totally understand this topic now",
        "why does this keep failing no matter what i try",
        "i'm confident this makes sense now, got it",
        "this is so confusing, i'm completely lost",
    ]

    for s in test_sentences:
        result = classifier.predict(s)
        print(f"'{s}'")
        print(f"  -> {result['emotion']} ({result['confidence']:.2%})")
        print(f"  scores: {result['scores']}\n")