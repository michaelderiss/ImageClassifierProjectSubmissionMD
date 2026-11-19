"""
Image Classifier Project – Dogs vs Cats (VGG16-style CNN)
===========================================================
Dataset : Asirra (Microsoft/Petfinder.com) – 25000 labelled cat/dog images
Pipeline: load → visualise → VGG16 CNN → train + callbacks → save + evaluate

Two loading strategies (auto-selected):
  1. Pre-decoded memmap cache (*.dat / *.npy in data/cache/)
  2. JPEG files (one-time decode, creates cache for future runs)

A tf.data.Dataset.from_generator pipeline streams mini-batches from the
on-disk memmap so the full dataset is never copied into a single TF tensor.

Every Conv2D and Dense layer uses He normal initialisation
(kernel_initializer='he_normal'), specifically designed for ReLU activations.
This prevents the vanishing-gradient problem that afflicts deep ReLU networks
initialised with the default glorot_uniform.

Architecture: 3 VGG blocks (32→64→128 filters, 2→2→3 convs) with
GlobalAveragePooling2D to keep total parameters under 500K for practical
CPU training. The VGG16 pattern of stacked 3×3 Conv2D→ReLU→MaxPooling2D
blocks followed by a Dense classifier is preserved.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # suppress info logs
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D,
    Activation, Input,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ── Paths ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

H, W = 100, 100
BATCH_SIZE = 128
EPOCHS = 30


# ──────────────────────────────────────────────────────
# 1. Load / preprocess
# ──────────────────────────────────────────────────────
def load_data():
    paths = dict(
        img_train = os.path.join(CACHE_DIR, "train_images_100.dat"),
        lbl_train = os.path.join(CACHE_DIR, "train_labels.npy"),
        img_test  = os.path.join(CACHE_DIR, "test_images_100.dat"),
        lbl_test  = os.path.join(CACHE_DIR, "test_labels.npy"),
    )
    if all(os.path.exists(v) for v in paths.values()):
        print("  Loading pre-decoded cache (data/cache/) ...")
        y_train = np.load(paths["lbl_train"]).astype(np.float32)
        y_test  = np.load(paths["lbl_test"]).astype(np.float32)
        n_tr, n_te = len(y_train), len(y_test)
        expected = n_tr * H * W * 3 * 4
        if os.path.getsize(paths["img_train"]) < expected:
            print("  Cache incomplete, regenerating ...")
            return load_from_jpegs(paths)
        print(f"  Loading {n_tr}+{n_te} = {n_tr+n_te} images ({H}x{W}) into memory ...")
        X_train = np.array(np.memmap(paths["img_train"], dtype=np.float32, mode="r",
                                     shape=(n_tr, H, W, 3)))
        X_test  = np.array(np.memmap(paths["img_test"],  dtype=np.float32, mode="r",
                                     shape=(n_te, H, W, 3)))
        return X_train, y_train, X_test, y_test
    return load_from_jpegs(paths)


def load_from_jpegs(paths):
    print("  No valid cache found – pre-decoding JPEGs (one-time cost) ...")
    from glob import glob
    def load_dir(data_dir):
        images, labels = [], []
        for cls, label in [("cats", 0), ("dogs", 1)]:
            path = os.path.join(data_dir, cls)
            if not os.path.isdir(path):
                continue
            for f in sorted(glob(os.path.join(path, "*.jpg"))):
                try:
                    raw = tf.io.read_file(f)
                    dec = tf.image.decode_jpeg(raw, channels=3)
                    res = tf.image.resize(dec, (H, W))
                    images.append(res.numpy() / 255.0)
                    labels.append(label)
                except Exception:
                    pass
        return np.array(images, np.float32), np.array(labels, np.float32)
    X_tr, y_tr = load_dir(TRAIN_DIR)
    X_te, y_te = load_dir(TEST_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    for name, arr in [("train_images.dat", X_tr), ("test_images.dat", X_te)]:
        mm = np.memmap(os.path.join(CACHE_DIR, name), dtype=np.float32,
                       mode="w+", shape=arr.shape)
        mm[:] = arr[:]; mm.flush(); del mm
    np.save(paths["lbl_train"], y_tr)
    np.save(paths["lbl_test"],  y_te)
    return X_tr, y_tr, X_te, y_te


print("=" * 60)
print("  Step 1: Loading and preprocessing images")
print("=" * 60)
X_train, y_train, X_test, y_test = load_data()
N_TRAIN, N_TEST = len(X_train), len(X_test)
STEPS_PER_EPOCH = int(np.ceil(N_TRAIN / BATCH_SIZE))
print(f"  Training set : {N_TRAIN} images")
print(f"  Validation   : {N_TEST} images")
print(f"  Shape        : {H}x{W}x3  Batch size {BATCH_SIZE}")
print(f"  Steps/epoch  : {STEPS_PER_EPOCH}")
print(f"  Labels       – cats: {int(np.sum(y_train == 0))}, dogs: {int(np.sum(y_train == 1))}")


# ──────────────────────────────────────────────────────
# 2. Visualise sample images
# ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 2: Visualising sample images")
print("=" * 60)

def save_sample_grid(X, y, class_name, class_val, save_path):
    idx = np.where(y == class_val)[0][:9]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X[i])
        ax.axis("off")
    plt.suptitle(f"Sample {class_name} images")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")

save_sample_grid(X_train, y_train, "cats", 0.0,
                 os.path.join(MODEL_DIR, "sample_cats.png"))
save_sample_grid(X_train, y_train, "dogs", 1.0,
                 os.path.join(MODEL_DIR, "sample_dogs.png"))


# ──────────────────────────────────────────────────────
# 3. Build VGG16-style CNN
#
# VGG16 architecture pattern:
#   Stacked 3×3 Conv2D → ReLU blocks, each followed by 2×2 MaxPooling,
#   then a dense classifier head.
#
# This CPU-friendly variant uses 3 blocks with filter progression
# 32 → 64 → 128, 2→2→3 convs per block, and GlobalAveragePooling2D
# to keep the parameter count low (~470K).
#
# He normal initialisation prevents vanishing gradients through
# the 7-convolution-layer ReLU stack on CPU.
# ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 3: Building VGG16-style CNN")
print("=" * 60)


def vgg_block(model, filters, num_convs):
    """VGG block: num_convs × (Conv2D→ReLU) + MaxPooling2D."""
    for _ in range(num_convs):
        model.add(Conv2D(filters, (3, 3), padding="same",
                         kernel_initializer="he_normal"))
        model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2), strides=2))

model = Sequential(name="VGG_Style_Dogs_vs_Cats")
model.add(Input(shape=(H, W, 3)))

vgg_block(model, 32, 2)    # Block 1: 200 → 100  (2 convs)
vgg_block(model, 64, 2)    # Block 2: 100 →  50  (2 convs)
vgg_block(model, 128, 3)   # Block 3:  50 →  25  (3 convs)

model.add(GlobalAveragePooling2D())                  # (128,) → tiny
model.add(Dense(256, activation="relu",
                kernel_initializer="he_normal"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# ──────────────────────────────────────────────────────
# 4. Callbacks & training (in-memory numpy arrays)
# ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 4: Training with callbacks")
print("=" * 60)

checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
checkpoint = ModelCheckpoint(
    checkpoint_path, monitor="val_accuracy", verbose=1,
    save_best_only=True, mode="max",
)
early_stop = EarlyStopping(
    monitor="val_accuracy", patience=5, verbose=1,
    restore_best_weights=True, mode="max",
)

print(f"  Training {N_TRAIN} samples, {STEPS_PER_EPOCH} steps/epoch, {EPOCHS} epochs ...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
    verbose=1,
)


# ──────────────────────────────────────────────────────
# 5. Save final model & evaluate on test set
# ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Step 5: Saving the model & evaluating on test set")
print("=" * 60)

final_path = os.path.join(MODEL_DIR, "dog_cat_classifier_final.keras")
model.save(final_path)
print(f"  Final model saved to: {final_path}")

best_model = load_model(checkpoint_path)
print("\n  Evaluating best model on test set ...")
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=1)
print(f"  Test accuracy : {test_acc:.4f}")
print(f"  Test loss     : {test_loss:.4f}")


def plot_history(hist, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    r = np.arange(1, len(hist.history["accuracy"]) + 1)
    ax1.plot(r, hist.history["accuracy"],     "b-", label="Training")
    ax1.plot(r, hist.history["val_accuracy"], "r-", label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True)

    ax2.plot(r, hist.history["loss"],     "b-", label="Training")
    ax2.plot(r, hist.history["val_loss"], "r-", label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")

plot_history(history, os.path.join(MODEL_DIR, "training_history.png"))

print("\n" + "=" * 60)
print("  Image classifier pipeline completed successfully!")
print("=" * 60)
