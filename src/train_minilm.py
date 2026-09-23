"""
MiniLM-L6 Fine-Tuning & Ultra-Compression Pipeline
- Fine-tunes Microsoft MiniLM-L6 (22.7M params) on student emotion data.
- Saves FP16 Safetensors (~45MB).
- Exports and Quantizes to ONNX INT8 (~22.5MB).
- Benchmarks accuracy, size, and CPU inference speed.
"""

import os
import sys
import time

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Label mappings
EMOTIONS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for i, e in enumerate(EMOTIONS)}


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=80):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def get_file_size_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    elif os.path.isdir(path):
        total = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(path)
            for f in files
        )
        return total / (1024 * 1024)
    return 0


def train_and_compress_minilm():
    base_model_name = "nreimers/MiniLM-L6-H384-uncased"
    output_dir = "models/minilm_student_adaptive"
    os.makedirs(output_dir, exist_ok=True)

    print("\n==========================================")
    print(f"[1/4] LOADING DATA & TOKENIZER: {base_model_name}")
    print("==========================================")

    data_path = "data/student_domain_data.csv"
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "emotion"])
    df["label"] = df["emotion"].map(LABEL2ID)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].values, df["label"].values, test_size=0.15, random_state=42, stratify=df["label"].values
    )
    print(f"[i] Dataset loaded: {len(train_texts)} train, {len(val_texts)} validation samples")

    tokenizer = None
    model = None
    for attempt in range(5):
        try:
            print(f"[*] Downloading / loading tokenizer and model (attempt {attempt+1}/5)...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=5,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
            print("[+] Successfully loaded model and tokenizer!")
            break
        except Exception as e:
            print(f"[!] Attempt {attempt+1} failed: {e}. Retrying in 3 seconds...")
            time.sleep(3)

    if model is None:
        raise RuntimeError("Failed to load MiniLM after multiple attempts.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Training device: {device}")
    model.to(device)

    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    epochs = 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_acc:.2%} | Time: {time.time() - t0:.1f}s")

    print("\n[+] Validation Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS, digits=4))

    print("\n==========================================")
    print("[3/4] SAVING FP16 SAFETENSORS (~45 MB)")
    print("==========================================")
    # Save as float16 safetensors
    model.eval()
    model_fp16 = model.half()
    model_fp16.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    print(f"[+] FP16 model saved to: {safetensors_path}")
    print(f"[i] FP16 Safetensors size: {get_file_size_mb(safetensors_path):.2f} MB")

    print("\n==========================================")
    print("[4/4] EXPORTING & QUANTIZING TO ONNX INT8 (~22 MB)")
    print("==========================================")
    model_fp32 = AutoModelForSequenceClassification.from_pretrained(output_dir, torch_dtype=torch.float32)
    model_fp32.eval()

    onnx_fp32_path = os.path.join(output_dir, "model.onnx")
    onnx_int8_path = os.path.join(output_dir, "model_quantized.onnx")

    dummy_text = "I am confused about this formula"
    dummy_inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=80)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }

    if "token_type_ids" in dummy_inputs:
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
        inputs_tuple = (dummy_inputs["input_ids"], dummy_inputs["attention_mask"], dummy_inputs["token_type_ids"])
    else:
        inputs_tuple = (dummy_inputs["input_ids"], dummy_inputs["attention_mask"])

    print("[*] Exporting to ONNX...")
    torch.onnx.export(
        model_fp32,
        inputs_tuple,
        onnx_fp32_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    print("[*] Applying Dynamic INT8 Quantization...")
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        weight_type=QuantType.QInt8,
    )
    if os.path.exists(onnx_fp32_path):
        os.remove(onnx_fp32_path)

    int8_size = get_file_size_mb(onnx_int8_path)
    print(f"[+] Ultra-compact ONNX INT8 saved to: {onnx_int8_path}")
    print(f"[i] FINAL MODEL SIZE: {int8_size:.2f} MB")
    print(f"[SUCCESS] Total reduction vs original BERT (418MB): {((418 - int8_size) / 418) * 100:.1f}% reduction!")


def test_minilm_inference():
    print("\n==========================================")
    print("[TESTING PREDICTIONS ON ULTRA-LIGHTWEIGHT MINILM]")
    print("==========================================")

    model_dir = "models/minilm_student_adaptive"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    onnx_path = os.path.join(model_dir, "model_quantized.onnx")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

    test_sentences = [
        "I'm really confused about how recursion works in tree traversal.",
        "Debugging this pointer segmentation fault is super frustrating!",
        "I feel confident that I solved all the calculus integration problems.",
        "This lecture is so slow and boring, I already know this stuff.",
        "I wonder how neural networks optimize weights with backpropagation, so curious!",
    ]

    for s in test_sentences:
        t0 = time.time()
        inputs = tokenizer(s, return_tensors="np", truncation=True, padding=True, max_length=80)
        ort_inputs = {
            k: v.astype(np.int64)
            for k, v in inputs.items()
            if k in [inp.name for inp in session.get_inputs()]
        }
        outputs = session.run(None, ort_inputs)
        logits = outputs[0][0]
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        pred_id = int(np.argmax(probs))
        latency_ms = (time.time() - t0) * 1000

        print(f"Prompt: \"{s}\"")
        print(f"  -> Predicted: {EMOTIONS[pred_id]} ({probs[pred_id]:.1%}) [Latency: {latency_ms:.2f} ms]")


if __name__ == "__main__":
    train_and_compress_minilm()
    test_minilm_inference()
