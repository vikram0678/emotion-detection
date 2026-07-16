from typing import TypedDict, Dict


class EmotionPrediction(TypedDict):
    """
    Standard prediction schema returned by BOTH BiLSTMPredictor.predict()
    and BERTEmotionClassifier.predict(). Any new model added to this
    project must also return this exact shape.
    """
    emotion: str          # predicted emotion label, e.g. "Confused"
    confidence: float     # confidence of the predicted emotion, 0.0-1.0
    scores: Dict[str, float]  # all 5 emotions with their individual scores, summing to ~1.0
    cleaned_text: str     # the text after preprocessing/cleaning


REQUIRED_KEYS = {"emotion", "confidence", "scores", "cleaned_text"}
VALID_EMOTIONS = {"Bored", "Confident", "Confused", "Curious", "Frustrated"}


def validate_prediction(result: dict) -> bool:
    """
    Checks that a model's output dict matches the unified schema.
    Raises a clear error if something is missing or malformed —
    useful when adding a new model or debugging a broken prediction.
    """
    missing = REQUIRED_KEYS - result.keys()
    if missing:
        raise ValueError(f"Prediction missing required keys: {missing}")

    if result["emotion"] not in VALID_EMOTIONS:
        raise ValueError(f"Unexpected emotion label: {result['emotion']}")

    if not (0.0 <= result["confidence"] <= 1.0):
        raise ValueError(f"Confidence out of range: {result['confidence']}")

    if set(result["scores"].keys()) != VALID_EMOTIONS:
        raise ValueError(f"Scores dict has unexpected classes: {result['scores'].keys()}")

    score_sum = sum(result["scores"].values())
    if not (0.95 <= score_sum <= 1.05):  # small tolerance for floating point rounding
        raise ValueError(f"Scores don't sum to ~1.0 (got {score_sum:.4f})")

    return True


def combine_predictions(bilstm_result: dict, bert_result: dict) -> dict:
    """
    Merges BiLSTM and BERT predictions into a single unified result —
    useful for ensembling or side-by-side comparison/logging.
    Averages the two models' scores for a simple ensemble prediction.
    """
    validate_prediction(bilstm_result)
    validate_prediction(bert_result)

    combined_scores = {
        emotion: (bilstm_result["scores"][emotion] + bert_result["scores"][emotion]) / 2
        for emotion in VALID_EMOTIONS
    }

    ensemble_emotion = max(combined_scores, key=combined_scores.get)

    return {
        "bilstm": bilstm_result,
        "bert": bert_result,
        "ensemble": {
            "emotion": ensemble_emotion,
            "confidence": combined_scores[ensemble_emotion],
            "scores": combined_scores,
            "cleaned_text": bert_result["cleaned_text"],
        },
    }


if __name__ == "__main__":
    # Quick self-test with fake data matching the real schema
    fake_bilstm = {
        "emotion": "Frustrated",
        "confidence": 0.96,
        "scores": {"Bored": 0.001, "Confident": 0.004, "Confused": 0.027, "Curious": 0.008, "Frustrated": 0.960},
        "cleaned_text": "keep failing no matter try",
    }
    fake_bert = {
        "emotion": "Frustrated",
        "confidence": 0.999,
        "scores": {"Bored": 0.0001, "Confident": 0.0001, "Confused": 0.0002, "Curious": 0.0004, "Frustrated": 0.9992},
        "cleaned_text": "why does this keep failing no matter what i try",
    }

    print("✅ BiLSTM schema valid:", validate_prediction(fake_bilstm))
    print("✅ BERT schema valid:", validate_prediction(fake_bert))

    combined = combine_predictions(fake_bilstm, fake_bert)
    print("\n✅ Combined result:")
    print(combined["ensemble"])