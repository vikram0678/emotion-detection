import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

combined_df = pd.read_csv("data/combined_preprocessed.csv")
combined_df = combined_df.dropna(subset=["clean_text"])
combined_df["clean_text"] = combined_df["clean_text"].astype(str)

MAX_VOCAB_SIZE = 30000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(combined_df["clean_text"])

label_encoder = LabelEncoder()
label_encoder.fit(combined_df["emotion"])

with open("models/bltsm/tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/bltsm/label_encoder.pickle", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Saved tokenizer.pickle and label_encoder.pickle to models/bltsm/")
print("✅ Vocab size:", len(tokenizer.word_index))
print("✅ Classes:", list(label_encoder.classes_))