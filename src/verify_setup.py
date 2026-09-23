import sys
import os
import io
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. Package versions
import numpy as np
import sklearn
import torch
import transformers
try:
    import onnxruntime as ort
    ort_ver = ort.__version__
except Exception:
    ort_ver = "Not installed"

print(f"[1/4] Dependency Check:", flush=True)
print(f"  • PyTorch:        {torch.__version__}", flush=True)
print(f"  • ONNX Runtime:   {ort_ver}", flush=True)
print(f"  • Transformers:   {transformers.__version__}", flush=True)
print(f"  • scikit-learn:   {sklearn.__version__}", flush=True)
print(f"  • NumPy:          {np.__version__}", flush=True)

# 2. Test BiLSTM Predictor
print(f"\n[2/4] Testing PyTorch BiLSTM Predictor:", flush=True)
t0 = time.time()
from bilstm_predictor import EmotionPredictor
bilstm = EmotionPredictor(
    model_path="models/bltsm/bilstm_model.pt",
    vocab_path="models/bltsm/vocab.json"
)
load_time_bilstm = (time.time() - t0) * 1000
print(f"  • BiLSTM Load Time: {load_time_bilstm:.1f} ms", flush=True)

t0 = time.time()
sample_text = "I am confused about how recursion unwinds and returns values"
res_b = bilstm.predict(sample_text)
pred_time_bilstm = (time.time() - t0) * 1000
print(f"  • BiLSTM Prediction ({pred_time_bilstm:.2f} ms): {res_b['emotion']} ({res_b['confidence']:.1%})", flush=True)

# 3. Test MiniLM ONNX Transformer
print(f"\n[3/4] Testing MiniLM ONNX INT8 Classifier:", flush=True)
t0 = time.time()
from bert_model import BERTEmotionClassifier
bert = BERTEmotionClassifier()
bert.load_model("models/bert_student_adaptive")
load_time_bert = (time.time() - t0) * 1000
print(f"  • MiniLM Load Time: {load_time_bert:.1f} ms", flush=True)

t0 = time.time()
res_t = bert.predict(sample_text)
pred_time_bert = (time.time() - t0) * 1000
print(f"  • MiniLM Prediction ({pred_time_bert:.2f} ms): {res_t['emotion']} ({res_t['confidence']:.1%})", flush=True)

# 4. Test Pedagogical Agent Response Generator
print(f"\n[4/4] Testing Pedagogical Agent Generator:", flush=True)
from gemini_helper import get_gemini_response
t0 = time.time()
agent_out = get_gemini_response(
    field="Computer Science",
    problem=sample_text,
    emotion=res_t['emotion'],
    confidence=res_t['confidence'],
    use_ai=False,
    persona="Socratic Mentor"
)
gen_time = (time.time() - t0) * 1000
print(f"  • Agent Response Generated in: {gen_time:.1f} ms", flush=True)

print("\n" + "=" * 60, flush=True)
print("✅ ALL ENGINES & MODELS VERIFIED - SYSTEM IS ULTRA-SMOOTH & READY!", flush=True)
print("=" * 60, flush=True)