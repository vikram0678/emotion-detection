import tensorflow as tf
import numpy as np
import sklearn
import transformers
import torch

print("✅ Verifying installation...")
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"PyTorch: {torch.__version__}")

# GPU check — TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print(f"\n🔥 TF GPUs detected: {len(gpus)}")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  - {gpu}")

# GPU check — PyTorch
print(f"\n🔥 Torch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - {torch.cuda.get_device_name(0)}")

if gpus and torch.cuda.is_available():
    print("\n✅ Local environment is STABLE - GPU ready for both TF and PyTorch!")
else:
    print("\n⚠️ GPU not detected in one or both frameworks — check driver/CUDA setup.")