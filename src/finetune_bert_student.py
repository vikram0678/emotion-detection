import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import Dataset as HFDataset

# ============================================
# STEP 1: Load the base BERT model (already trained on your 3 datasets)
# ============================================
MODEL_PATH = "models/bert_emotion_model_final"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

print("📦 Loading base BERT model...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

id2label = model.config.id2label
label2id = model.config.label2id
print("✅ Classes:", id2label)

# ============================================
# STEP 2: Student-domain synthetic data (with expanded Bored templates)
# ============================================
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
        # Expanded: "already know it, move faster" flavor of boredom
        "I already know all of this, can we move faster",
        "We've covered this before, let's skip ahead",
        "This is basic stuff I learned a while ago",
        "Can we speed through this part, nothing new here",
        "I already get this concept, no need to repeat it",
        "This is too easy and slow for what I already know",
        "Let's move on, I've mastered this part already",
        "I finished this a while ago, waiting for something new",
    ],
}

fillers = ["", " honestly", " right now", " during class", " today", " again", " seriously"]

rows = []
for emotion, templates in student_templates.items():
    texts = []
    for t in templates:
        for f in fillers:
            texts.append((t + f).strip())
    target_per_class = 400
    texts = (texts * ((target_per_class // len(texts)) + 1))[:target_per_class]
    for txt in texts:
        rows.append({"text": txt, "emotion": emotion})

student_df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
student_df["label"] = student_df["emotion"].map(label2id)

print("\n✅ Student data generated:", student_df.shape)
print(student_df["emotion"].value_counts())

# ============================================
# STEP 3: Tokenize + split
# ============================================
train_df, val_df = train_test_split(
    student_df, test_size=0.2, random_state=42, stratify=student_df["label"]
)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=80)

train_ds = HFDataset.from_pandas(train_df[["text", "label"]].rename(columns={"label": "labels"}))
val_ds = HFDataset.from_pandas(val_df[["text", "label"]].rename(columns={"label": "labels"}))

train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("\n✅ Tokenized - Train:", len(train_ds), "Val:", len(val_ds))

# ============================================
# STEP 4: Fine-tune (small LR, few epochs)
# ============================================
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 3

print("\n🚀 Fine-tuning on student domain (local GPU)...")
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - train"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} - val"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(batch["labels"].cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1}: train_loss={sum(train_losses)/len(train_losses):.4f}, val_acc={val_acc:.4f}")

# ============================================
# STEP 5: Save the adapted model
# ============================================
SAVE_PATH = "models/bert_student_adaptive"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"\n✅ Saved adapted model to {SAVE_PATH}")

# ============================================
# STEP 6: Sanity check on real, hand-written sentences
# ============================================
test_sentences = [
    "why does this equation flip when i move the variable to the other side",
    "i already know all of this, can we move faster",
    "i totally understand this topic now",
    "why does this keep failing no matter what i try",
]

print("\n🔍 Testing on genuinely new sentences:")
model.eval()
for s in test_sentences:
    inputs = tokenizer(s, padding="max_length", truncation=True, max_length=80, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        conf = probs[0][pred_id].item()
    print(f"'{s}' -> {id2label[pred_id]} ({conf:.2%})")

print("\n✅ DONE")