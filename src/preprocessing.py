import pandas as pd
import random
# --- GoEmotions: multi-label -> single label ---
go_df = pd.read_csv("data/go_emotions_dataset.csv")

emotion_cols = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]
emotion_cols = [c for c in emotion_cols if c in go_df.columns]

go_df["emotion_count"] = go_df[emotion_cols].sum(axis=1)
go_df = go_df[go_df["emotion_count"] == 1]
go_df["emotion"] = go_df[emotion_cols].idxmax(axis=1)
go_df = go_df[["text", "emotion"]]
print("✅ GoEmotions cleaned:", go_df.shape)

# --- EmpatheticDialogues ---
emp_df = pd.read_csv("data/emotion-emotion_69k.csv")
emp_df = emp_df.rename(columns={"Situation": "text"})[["text", "emotion"]]
print("✅ EmpatheticDialogues cleaned:", emp_df.shape)

# --- ISEAR ---
isear_df = pd.read_csv("data/eng_dataset.csv")
isear_df = isear_df.rename(columns={"sentiment": "emotion", "content": "text"})[["text", "emotion"]]
print("✅ ISEAR cleaned:", isear_df.shape)

# --- Combine + map to your 5 target classes ---
label_map = {
    "confusion": "Confused",
    "curiosity": "Curious",
    "confident": "Confident",
    "anger": "Frustrated",
    "annoyance": "Frustrated",
    "angry": "Frustrated",
    "annoyed": "Frustrated",
    "furious": "Frustrated",
}

combined_df = pd.concat([go_df, emp_df, isear_df], ignore_index=True)
combined_df["emotion"] = combined_df["emotion"].str.lower().str.strip()
combined_df["mapped_class"] = combined_df["emotion"].map(label_map)

combined_df = combined_df.dropna(subset=["mapped_class"])
combined_df = combined_df[["text", "mapped_class"]].rename(columns={"mapped_class": "emotion"})

combined_df.to_csv("data/combined_mapped.csv", index=False)

print("\n✅ Combined + mapped:", combined_df.shape)
print(combined_df["emotion"].value_counts())


bored_templates = [
    "This lecture is so slow, I already know all of this",
    "I can't focus anymore, this class is dragging on forever",
    "Can we just skip ahead, this is nothing new",
    "I've heard this explanation like five times already",
    "This is so repetitive, when does something new happen",
    "I keep zoning out, nothing here is interesting",
    "Same old content again, I'm losing interest fast",
    "This part of the course feels like it never ends",
    "Nothing about this topic is engaging me at all",
    "I'm just staring at the screen, this is dull",
    "Why is this taking so long, I want to move on",
    "I don't feel like paying attention anymore",
    "This is the most boring topic we've covered so far",
    "I keep checking the time, this class won't end",
    "None of this feels relevant or interesting to me",
]

fillers = ["", " honestly", " right now", " today", " again", " seriously", " ugh"]
bored_texts = []
for t in bored_templates:
    for f in fillers:
        bored_texts.append((t + f).strip())

# Aim for a count roughly similar to your smaller real classes (~2000-3000)
target_count = 3000
bored_texts = (bored_texts * ((target_count // len(bored_texts)) + 1))[:target_count]
random.shuffle(bored_texts)

bored_df = pd.DataFrame({"text": bored_texts, "emotion": "Bored"})
print("✅ Synthetic Bored data:", bored_df.shape)

combined_df = pd.concat([combined_df, bored_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print("\n✅ FINAL combined dataset:", combined_df.shape)
print(combined_df["emotion"].value_counts())


import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

nltk.download('punkt')
nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in english_stopwords and len(t) > 1]
    return " ".join(tokens)

print("\n✂️ Cleaning text...")
combined_df["clean_text"] = combined_df["text"].apply(clean_text)

MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 80

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(combined_df["clean_text"])

sequences = tokenizer.texts_to_sequences(combined_df["clean_text"])
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

print("✅ Tokenization complete:", padded_sequences.shape)

# Save artifacts
np.save("data/padded_sequences.npy", padded_sequences)
combined_df.to_csv("data/combined_preprocessed.csv", index=False)

with open("models/tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ PREPROCESSING COMPLETE")