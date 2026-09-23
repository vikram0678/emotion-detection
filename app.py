from flask import Flask, request, jsonify, render_template_string
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from src.keyword_enhancement import apply_keyword_boost
import pandas as pd
import os
import sys

# Ensure src directory is in sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.bert_model import BERTEmotionClassifier

app = Flask(__name__)

MODEL_PATH = "models/bert_student_adaptive"
RESPONSE_MAPPING_PATH = "emotion_response_mapping.csv"

classifier = BERTEmotionClassifier()
classifier.load_model(MODEL_PATH)

# --- Load canned responses per emotion, if available ---
response_map = {}
if os.path.exists(RESPONSE_MAPPING_PATH):
    df = pd.read_csv(RESPONSE_MAPPING_PATH)
    response_map = dict(zip(df["emotion"], df["response"]))
    print("✅ Loaded response mapping for:", list(response_map.keys()))
else:
    print("⚠️ No emotion_response_mapping.csv found — responses will be generic")


def predict_emotion(text):
    result = classifier.predict(text)
    return result["emotion"], result["confidence"]


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Emotion Detector</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 60px auto; padding: 0 20px; }
        textarea { width: 100%; height: 100px; font-size: 16px; padding: 10px; box-sizing: border-box; }
        button { margin-top: 10px; padding: 10px 20px; font-size: 16px; cursor: pointer; }
        #result { margin-top: 20px; padding: 15px; border-radius: 8px; background: #f0f0f0; display: none; }
        #emotion { font-size: 22px; font-weight: bold; }
        #confidence { color: #555; }
        #response { margin-top: 10px; font-style: italic; }
    </style>
</head>
<body>
    <h2>Student Emotion Detector</h2>
    <p>Type something a student might say, then click Detect.</p>
    <textarea id="textInput" placeholder="e.g. I don't understand this formula at all"></textarea><br>
    <button onclick="detectEmotion()">Detect Emotion</button>

    <div id="result">
        <div id="emotion"></div>
        <div id="confidence"></div>
        <div id="response"></div>
    </div>

    <script>
        async function detectEmotion() {
            const text = document.getElementById("textInput").value;
            if (!text.trim()) return;

            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text })
            });
            const data = await res.json();

            document.getElementById("result").style.display = "block";
            document.getElementById("emotion").innerText = "Emotion: " + data.emotion;
            document.getElementById("confidence").innerText = "Confidence: " + (data.confidence * 100).toFixed(1) + "%";
            document.getElementById("response").innerText = data.response ? ("💬 " + data.response) : "";
        }
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    emotion, confidence = predict_emotion(text)
    response = response_map.get(emotion, "")

    return jsonify({
        "emotion": emotion,
        "confidence": confidence,
        "response": response
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)