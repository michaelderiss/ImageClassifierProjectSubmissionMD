"""Quick speed benchmark for in-memory training."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

print("Loading data into memory ...")
y_train = np.load("data/cache/train_labels.npy").astype(np.float32)
n_tr = len(y_train)
X_train = np.array(
    np.memmap("data/cache/train_images.dat", dtype=np.float32, mode="r",
              shape=(n_tr, 200, 200, 3))
)
print(f"  X_train: {X_train.shape}, {X_train.nbytes/1e9:.1f} GB")

y_test = np.load("data/cache/test_labels.npy").astype(np.float32)
n_te = len(y_test)
X_test = np.array(
    np.memmap("data/cache/test_images.dat", dtype=np.float32, mode="r",
              shape=(n_te, 200, 200, 3))
)
print(f"  X_test:  {X_test.shape}, {X_test.nbytes/1e9:.1f} GB")

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input((200, 200, 3)),
    tf.keras.layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, kernel_initializer="he_normal", activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
print(f"Params: {model.count_params()}")
print(f"Steps/epoch: {int(np.ceil(n_tr / 128))}")

# Benchmark 1 epoch
t0 = time.time()
model.fit(X_train, y_train, batch_size=128, epochs=1, verbose=2,
          validation_data=(X_test, y_test))
t1 = time.time()

epoch_s = t1 - t0
ps = epoch_s / int(np.ceil(n_tr / 128))
print(f"\n1 epoch: {epoch_s:.1f}s ({ps:.2f}s/step)")
print(f"30 epochs: {epoch_s*30/60:.1f} min")
print(f"Val acc: {model.history.history['val_accuracy'][0]:.4f}")