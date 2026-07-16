import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)


# ============================================
# 1. TEXT CLEANING (preserves emotion-carrying punctuation)
# ============================================
def clean_text_keep_emotion(text):
    """
    Cleans text but PRESERVES punctuation that signals emotion (! and ,).
    Only strips basic filler words (the, a, an) instead of full stopwords,
    since words like 'not', 'no', 'never' can flip emotional meaning.
    """
    text = str(text).lower()
    # Keep letters, spaces, ! and , — strip everything else
    text = re.sub(r"[^a-zA-Z\s,!]", " ", text)
    tokens = word_tokenize(text)

    skip_words = {"the", "a", "an"}
    tokens = [t for t in tokens if t not in skip_words and len(t) > 1]

    return " ".join(tokens) if tokens else text


# ============================================
# 2. EMOTION KEYWORD DICTIONARY
# ============================================
EMOTION_KEYWORDS = {
    "Frustrated": [
        "frustrated", "frustrating", "annoying", "angry", "hate", "difficult",
        "stuck", "wrong answer", "keep getting", "unnecessarily complicated",
        "tried everything", "nothing works", "sick of", "annoyed"
    ],
    "Curious": [
        "why", "how", "what", "curious", "wonder", "interested", "learn",
        "know more", "want to know", "explore", "could we", "what happens",
        "intuition", "what if", "how does"
    ],
    "Confident": [
        "easy", "amazing", "great", "excellent", "good", "awesome", "perfect",
        "solved", "got it", "clear now", "finally", "move ahead",
        "understand clearly", "makes sense now", "confident"
    ],
    "Bored": [
        "boring", "bored", "tired", "repetitive", "dull", "not engaging",
        "didn't engage", "didn't feel engaging", "not interesting", "too basic",
        "losing interest", "already know", "seen this before"
    ],
    "Confused": [
        "confused", "lost", "unclear", "don't understand", "don't make sense",
        "missing", "incomplete", "unsure", "not fully confident", "no idea",
        "doesn't make sense"
    ],
}

# Words that get the strongest boost (very explicit, unambiguous emotion words)
HIGH_WEIGHT_TRIGGERS = {"frustrated", "frustrating", "curious", "confident", "bored", "boring", "confused"}


# ============================================
# 3. KEYWORD SCORING
# ============================================
def score_emotions_by_keywords(text_lower):
    """Scores each emotion based on keyword matches. Explicit words get 10x weight."""
    emotion_scores = {}

    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                if keyword in HIGH_WEIGHT_TRIGGERS:
                    score += 10
                else:
                    score += 2
        emotion_scores[emotion] = score

    return emotion_scores


# ============================================
# 4. BOOST + RENORMALIZE MODEL PROBABILITIES
# ============================================
def apply_keyword_boost(text, probs, classes):
    """
    Takes the model's raw probability array and boosts it based on keyword matches.

    Args:
        text: original raw text (string)
        probs: numpy array of model probabilities, one per class
        classes: list of class names in the SAME order as probs

    Returns:
        boosted_probs: numpy array, same shape as probs, renormalized to sum to 1
        emotion_scores: dict of keyword scores per emotion (for debugging/inspection)
    """
    text_lower = text.lower()
    emotion_scores = score_emotions_by_keywords(text_lower)

    probs = np.array(probs, dtype=float).copy()
    max_score = max(emotion_scores.values())

    if max_score > 0:
        winning_emotions = [e for e, s in emotion_scores.items() if s == max_score]

        for emotion, score in emotion_scores.items():
            if emotion not in classes:
                continue
            idx = classes.index(emotion)

            if score == max_score:
                # Strong boost for the winning emotion(s)
                probs[idx] *= (1 + score * 3.0)
            elif max_score >= 5:
                # Suppress others when there's a strong, confident keyword signal
                probs[idx] *= 0.01

        probs = probs / np.sum(probs)  # renormalize so probabilities sum to 1

    return probs, emotion_scores


# ============================================
# 5. QUICK STANDALONE TEST
# ============================================
if __name__ == "__main__":
    classes = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

    # Simulate a case where the model's raw prediction was uncertain/wrong
    fake_model_probs = [0.25, 0.20, 0.20, 0.20, 0.15]  # nearly uniform, unsure

    test_text = "this is so frustrating, nothing works no matter what I try"

    boosted, scores = apply_keyword_boost(test_text, fake_model_probs, classes)

    print("Keyword scores:", scores)
    print("Original probs:", dict(zip(classes, fake_model_probs)))
    print("Boosted probs: ", dict(zip(classes, boosted.round(4))))
    print("Final prediction:", classes[np.argmax(boosted)])