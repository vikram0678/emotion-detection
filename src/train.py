import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

# --- Load preprocessed data ---
combined_df = pd.read_csv("data/combined_preprocessed.csv")
padded_sequences = np.load("data/padded_sequences.npy")

# --- Encode emotion labels to integers ---
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(combined_df["emotion"])
num_classes = len(label_encoder.classes_)

print("✅ Classes:", list(label_encoder.classes_))

# --- Train/test split (stratified to preserve class ratios) ---
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42, stratify=labels
)

print("✅ Train shape:", X_train.shape, "Test shape:", X_test.shape)

# --- Class weights (helps minority classes like Bored/Confident) ---
class_weights_array = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights = dict(enumerate(class_weights_array))
print("✅ Class weights:", class_weights)

# --- Focal Loss (gamma=2.0) ---
def focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true_onehot * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return loss_fn

MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 80
EMBEDDING_DIM = 128
LSTM_UNITS = 128

model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LEN),
    Bidirectional(LSTM(LSTM_UNITS)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss=focal_loss(gamma=2.0),
    metrics=["accuracy"]
)

model.summary()

print("\n🚀 Starting training on CPU (this may take a while)...")

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    class_weight=class_weights,
    verbose=1
)

print("✅ Training complete")

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_))


import pickle

model.save("models/bltsm/bilstm_emotion_model.keras")

with open("models/bltsm/label_encoder.pickle", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model saved to models/bltsm/")