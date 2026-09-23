import os
import sys

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unified_schema import validate_prediction


class BERTEmotionClassifier:
    """Loads the fine-tuned BERT model (preferring ONNX INT8 or PyTorch INT8) and applies
    class weighting + keyword-based confidence/confusion adjustments to sharpen predictions."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.ort_session = None
        self.use_onnx = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_labels = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]
        self.id2label = {i: label for i, label in enumerate(self.emotion_labels)}

    def load_model(self, model_path="models/minilm_student_adaptive"):
        if not os.path.exists(model_path):
            model_path = "models/bert_student_adaptive"
        self.model_path = model_path
        print(f"📦 Loading transformer model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        onnx_path = os.path.join(model_path, "model_quantized.onnx")
        quantized_pt_path = os.path.join(model_path, "pytorch_model_quantized.pt")

        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                print(f"⚡ Found ONNX INT8 Quantized model ({model_size_mb:.1f} MB). Loading ultra-fast ONNX Runtime...")
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                self.ort_session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
                self.use_onnx = True
                self.device = torch.device("cpu")
            except Exception as e:
                print(f"⚠️ ONNX load fallback: {e}")
                self.use_onnx = False

        if not self.use_onnx:
            if os.path.exists(quantized_pt_path):
                print("⚡ Found Quantized INT8 PyTorch model. Loading lightweight model...")
                self.model = torch.load(quantized_pt_path, map_location=torch.device("cpu"))
                self.device = torch.device("cpu")
                self.model.eval()
            else:
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

        print(f"✅ Transformer loaded ({'ONNX INT8' if self.use_onnx else 'PyTorch'}) on", self.device, "| Classes:", self.emotion_labels)

    def predict(self, text):
        if not self.use_onnx and self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self.use_onnx:
            inputs = self.tokenizer(text, return_tensors="np", truncation=True, padding=True, max_length=80)
            ort_inputs = {
                k: v.astype(np.int64)
                for k, v in inputs.items()
                if k in [inp.name for inp in self.ort_session.get_inputs()]
            }
            outputs = self.ort_session.run(None, ort_inputs)
            logits = outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
        else:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # Base class weights (MiniLM is fully balanced; BERT base has legacy calibration)
        if "minilm" in getattr(self, "model_path", "").lower():
            class_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            class_weights = np.array([1.2, 1.8, 0.6, 1.0, 1.4])

        weighted_probs = probs * class_weights
        weighted_probs = weighted_probs / np.sum(weighted_probs)

        # Standardized 10x keyword enhancement
        from keyword_enhancement import apply_keyword_boost
        boosted_probs, _ = apply_keyword_boost(text, weighted_probs, self.emotion_labels)

        pred_id = int(np.argmax(boosted_probs))
        emotion = self.emotion_labels[pred_id]
        confidence = float(boosted_probs[pred_id])

        result = {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "scores": {self.emotion_labels[i]: round(float(boosted_probs[i]), 4) for i in range(len(self.emotion_labels))},
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