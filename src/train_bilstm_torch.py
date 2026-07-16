import re
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import nltk
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
english_stopwords = set(stopwords.words('english'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)
if torch.cuda.is_available():
    print("✅ GPU:", torch.cuda.get_device_name(0))

MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 80
EMBEDDING_DIM = 128
LSTM_UNITS = 128
BATCH_SIZE = 64
EPOCHS = 10

# ============================================
# 1. TEXT CLEANING
# ============================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in english_stopwords and len(t) > 1]
    return " ".join(tokens)


# ============================================
# 2. LOAD DATA
# ============================================
print("\n📦 Loading combined dataset...")
combined_df = pd.read_csv("data/combined_preprocessed.csv")
combined_df = combined_df.dropna(subset=["clean_text", "emotion"])
combined_df["clean_text"] = combined_df["clean_text"].astype(str)

label_encoder = LabelEncoder()
combined_df["label"] = label_encoder.fit_transform(combined_df["emotion"])
num_classes = len(label_encoder.classes_)
print("✅ Classes:", list(label_encoder.classes_))


# ============================================
# 3. BUILD VOCABULARY (replaces Keras Tokenizer)
# ============================================
print("\n🔤 Building vocabulary...")
counter = Counter()
for text in combined_df["clean_text"]:
    counter.update(text.split())

# Reserve index 0 for padding, 1 for unknown/OOV
most_common = counter.most_common(MAX_VOCAB_SIZE - 2)
vocab = {"<PAD>": 0, "<OOV>": 1}
for i, (word, _) in enumerate(most_common, start=2):
    vocab[word] = i

print("✅ Vocab size:", len(vocab))


def text_to_sequence(text, vocab, max_len=MAX_SEQ_LEN):
    tokens = text.split()
    seq = [vocab.get(t, vocab["<OOV>"]) for t in tokens]
    seq = seq[:max_len]
    seq = seq + [0] * (max_len - len(seq))  # pad with 0 (post-padding)
    return seq


# ============================================
# 4. DATASET CLASS
# ============================================
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = text_to_sequence(self.texts[idx], self.vocab)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# ============================================
# 5. TRAIN/VAL/TEST SPLIT
# ============================================
X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    combined_df["clean_text"].tolist(), combined_df["label"].tolist(),
    test_size=0.3, random_state=42, stratify=combined_df["label"]
)
X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n✅ Train: {len(X_train_text)}  Val: {len(X_val_text)}  Test: {len(X_test_text)}")

train_dataset = EmotionDataset(X_train_text, y_train, vocab)
val_dataset = EmotionDataset(X_val_text, y_val, vocab)
test_dataset = EmotionDataset(X_test_text, y_test, vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================
# 6. CLASS WEIGHTS (for imbalance)
# ============================================
class_weights_array = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float32).to(device)
print("✅ Class weights:", dict(zip(label_encoder.classes_, class_weights_array.round(3))))


# ============================================
# 7. MODEL DEFINITION (BiLSTM, matches original architecture)
# ============================================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, lstm_units, batch_first=True,
            bidirectional=True, dropout=0.0
        )
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_units * 2, 64)  # *2 because bidirectional
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)                 # (batch, seq_len, embed_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)   # hidden: (2, batch, lstm_units)
        # Concatenate final forward and backward hidden states
        h_forward = hidden[0]
        h_backward = hidden[1]
        combined = torch.cat((h_forward, h_backward), dim=1)  # (batch, lstm_units*2)

        x = self.dropout1(combined)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits


model = BiLSTMClassifier(len(vocab), EMBEDDING_DIM, LSTM_UNITS, num_classes).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n✅ Model created. Total parameters: {total_params:,}")


# ============================================
# 8. FOCAL LOSS
# ============================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor)
optimizer = Adam(model.parameters(), lr=1e-3)


# ============================================
# 9. TRAINING LOOP
# ============================================
print("\n🚀 Training on", device, "...")

best_val_loss = float("inf")
patience = 3
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for seqs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - train"):
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seqs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses, val_preds, val_labels_list = [], [], []
    with torch.no_grad():
        for seqs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} - val"):
            seqs, labels = seqs.to(device), labels.to(device)
            logits = model(seqs)
            loss = criterion(logits, labels)
            val_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels_list.extend(labels.cpu().numpy())

    train_loss_avg = np.mean(train_losses)
    val_loss_avg = np.mean(val_losses)
    val_acc = np.mean(np.array(val_preds) == np.array(val_labels_list))

    print(f"Epoch {epoch+1}: train_loss={train_loss_avg:.4f}, val_loss={val_loss_avg:.4f}, val_acc={val_acc:.4f}")

    # Early stopping
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        patience_counter = 0
        torch.save(model.state_dict(), "models/bltsm/bilstm_best_weights.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered")
            break


# ============================================
# 10. LOAD BEST WEIGHTS + EVALUATE ON TEST SET
# ============================================
model.load_state_dict(torch.load("models/bltsm/bilstm_best_weights.pt"))
model.eval()

test_preds, test_labels_list = [], []
with torch.no_grad():
    for seqs, labels in tqdm(test_loader, desc="Test"):
        seqs, labels = seqs.to(device), labels.to(device)
        logits = model(seqs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_labels_list.extend(labels.cpu().numpy())

print("\n📊 Test Classification Report:")
print(classification_report(test_labels_list, test_preds, target_names=label_encoder.classes_))


# ============================================
# 11. SAVE EVERYTHING (portable, no version issues)
# ============================================
torch.save(model.state_dict(), "models/bltsm/bilstm_model.pt")

with open("models/bltsm/vocab.json", "w") as f:
    json.dump(vocab, f)

with open("models/bltsm/label_encoder.pickle", "wb") as f:
    pickle.dump(label_encoder, f)

config = {
    "vocab_size": len(vocab),
    "embedding_dim": EMBEDDING_DIM,
    "lstm_units": LSTM_UNITS,
    "num_classes": num_classes,
    "max_seq_len": MAX_SEQ_LEN,
    "classes": list(label_encoder.classes_),
}
with open("models/bltsm/model_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n✅ Saved: bilstm_model.pt, vocab.json, label_encoder.pickle, model_config.json")
print("✅ DONE")