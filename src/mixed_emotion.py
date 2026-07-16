# Emoji + supportive response + suggested action per emotion.
# Used both for the AI-response UI and as the offline fallback template.
EMOTION_RESPONSES = {
    "Bored": {
        "emoji": "😐",
        "response": "Let's make this more engaging. Here are some interactive exercises...",
        "action": "Show interactive content",
    },
    "Confident": {
        "emoji": "💪",
        "response": "Great! You're making excellent progress! Ready for the next challenge?",
        "action": "Suggest advanced content",
    },
    "Confused": {
        "emoji": "😕",
        "response": "I see you might be confused. Let me break this down step-by-step...",
        "action": "Show detailed explanation",
    },
    "Curious": {
        "emoji": "🧐",
        "response": "Excellent question! Here's more in-depth information...",
        "action": "Provide research papers & advanced materials",
    },
    "Frustrated": {
        "emoji": "😤",
        "response": "I understand this is frustrating! Let's try a simpler approach...",
        "action": "Suggest alternative learning path",
    },
}


def get_mixed_emotions(scores, threshold=0.15):
    """
    Takes a dict of {emotion: score} and returns a list of (emotion, score) tuples
    for every emotion at or above the threshold, sorted highest first.
    """
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_emotions[0]
    mixed = [primary]

    for emotion, score in sorted_emotions[1:]:
        if score >= threshold:
            mixed.append((emotion, score))

    return mixed if len(mixed) > 1 else [primary]