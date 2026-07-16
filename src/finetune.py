import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Load baseline model + label encoder ---
model = tf.keras.models.load_model(
    "models/bltsm/bilstm_emotion_model.keras",
    custom_objects={"loss_fn": None},  # focal loss custom object may need re-attaching, see note below
    compile=False
)

with open("models/bltsm/label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# --- Load student domain data ---
student_df = pd.read_csv("data/student_preprocessed.csv")
padded = np.load("data/student_padded_sequences.npy")
labels = label_encoder.transform(student_df["emotion"])

X_train, X_val, y_train, y_val = train_test_split(
    padded, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- Recompile with a LOW learning rate (fine-tuning, not training from scratch) ---
def focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(label_encoder.classes_))
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true_onehot * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return loss_fn

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # small LR to avoid overwriting prior learning
    loss=focal_loss(gamma=2.0),
    metrics=["accuracy"]
)

# --- Fine-tune for a few epochs only ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=64,
    verbose=1
)

# --- Evaluate ---
y_pred = np.argmax(model.predict(X_val), axis=1)
print("\n📊 Fine-tuned model report:")
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

# --- Save adapted model ---
model.save("models/bltsm/bilstm_student_adaptive.keras")
print("✅ Saved as bilstm_student_adaptive.keras")