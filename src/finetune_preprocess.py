import pandas as pd

student_templates = {
    "Confused": [
        "I don't understand this formula at all",
        "Wait, how did we get this answer, I'm lost",
        "Can you explain that step again, it doesn't make sense",
        "I have no idea what the professor just said",
        "This assignment instructions are so unclear",
        "I keep getting a different answer than the textbook",
        "None of this is making sense to me right now",
        "I'm stuck on this problem and don't know where to start",
    ],
    "Curious": [
        "Wait, why does this formula work like that",
        "I wonder how this concept applies to real projects",
        "That's interesting, can we go deeper into this topic",
        "I want to know more about how this algorithm works",
        "How does this relate to what we learned last week",
        "This is fascinating, what happens if we change this variable",
        "I'd like to explore this idea further after class",
        "What would happen if we applied this to a different problem",
    ],
    "Confident": [
        "I've got this assignment figured out completely",
        "I understand this topic really well now",
        "I'm sure I'll do well on this exam",
        "This makes total sense, I can explain it to others",
        "I feel ready for the presentation tomorrow",
        "I solved the whole problem set without help",
        "I know exactly how to approach this project",
        "I'm confident I answered all the quiz questions correctly",
    ],
    "Frustrated": [
        "I've tried this problem five times and still can't solve it",
        "This assignment is taking forever and nothing works",
        "I'm so annoyed, the code keeps throwing the same error",
        "Why does this keep failing no matter what I try",
        "I'm sick of redoing this same section over and over",
        "This professor's instructions never make sense",
        "I can't believe I have three assignments due tomorrow",
        "Nothing about this group project is going smoothly",
    ],
    "Bored": [
        "This lecture is so slow, I already know all of this",
        "I keep zoning out during this class, nothing is new",
        "This textbook chapter is putting me to sleep",
        "Same review material again, I've heard this five times",
        "I can't focus on this presentation, it's dragging on",
        "This homework is repetitive and uninteresting",
        "I'm just waiting for this class to end already",
        "Nothing about this topic holds my attention today",
    ],
}

fillers = ["", " honestly", " right now", " during class", " today", " again", " seriously"]

rows = []
for emotion, templates in student_templates.items():
    texts = []
    for t in templates:
        for f in fillers:
            texts.append((t + f).strip())
    target_per_class = 400  # smaller this time — fine-tuning needs less data
    texts = (texts * ((target_per_class // len(texts)) + 1))[:target_per_class]
    for txt in texts:
        rows.append({"text": txt, "emotion": emotion})

student_df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
student_df["label"] = student_df["emotion"]  # reuse label2id from BERT setup

print("✅ Student data:", student_df.shape)
print(student_df["emotion"].value_counts())