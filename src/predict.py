import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from keyword_enhancement import apply_keyword_boost, clean_text_keep_emotion

MODEL_PATH = "models/bert_student_adaptive"

print("📦 Loading final BERT model (student domain-adapted)...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id2label = model.config.id2label
classes = [id2label[i] for i in range(len(id2label))]
print("✅ Model loaded on", device, "| Classes:", classes)


def predict_emotion(text, use_keyword_boost=True):
    """
    Predicts emotion for raw text.
    Returns: (predicted_emotion, confidence, raw_probs_dict, boosted_probs_dict)
    """
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=80, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    raw_probs_dict = dict(zip(classes, probs.round(4)))

    if use_keyword_boost:
        boosted_probs, keyword_scores = apply_keyword_boost(text, probs, classes)
    else:
        boosted_probs, keyword_scores = probs, {}

    boosted_probs_dict = dict(zip(classes, boosted_probs.round(4)))

    pred_idx = boosted_probs.argmax()
    predicted_emotion = classes[pred_idx]
    confidence = float(boosted_probs[pred_idx])

    return predicted_emotion, confidence, raw_probs_dict, boosted_probs_dict


if __name__ == "__main__":
    test_sentences = [
        "why does this equation flip when i move the variable to the other side",
        "i already know all of this, can we move faster",
        "i totally understand this topic now",
        "why does this keep failing no matter what i try",
        "this is so frustrating, nothing works no matter what I try",
        "wow that's really interesting, I wonder how it works",
    ]

    for s in test_sentences:
        emotion, conf, raw, boosted = predict_emotion(s)
        print(f"'{s}'")
        print(f"  Raw model:      {raw}")
        print(f"  After keywords: {boosted}")
        print(f"  -> Final: {emotion} ({conf:.2%})\n")