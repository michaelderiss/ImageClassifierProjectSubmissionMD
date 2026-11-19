"""Resize memmap cache from 200x200 to 100x100 for faster CPU training."""
import os
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

def resize_cache(src_path, dst_path, labels_path, new_size):
    """Read memmap, resize each image, write new memmap."""
    y = np.load(labels_path)
    n = len(y)
    H, W = new_size
    
    # Read original memmap
    orig = np.memmap(src_path, dtype=np.float32, mode="r", shape=(n, 200, 200, 3))
    
    print(f"Resizing {n} images from 200x200 to {H}x{W} ...")
    
    # Create output memmap
    mm = np.memmap(dst_path, dtype=np.float32, mode="w+", shape=(n, H, W, 3))
    
    # Process in chunks
    chunk = 500
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        batch = orig[start:end]
        resized = tf.image.resize(batch, (H, W)).numpy()
        mm[start:end] = resized
        if (start + chunk) % 2000 == 0 or start + chunk >= n:
            print(f"  ... {end}/{n}")
    
    mm.flush()
    del mm
    print(f"  Done. Saved to {dst_path}")
    expected_mb = n * H * W * 3 * 4 / 1024 / 1024
    actual_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f"  Expected: {expected_mb:.0f} MB, Actual: {actual_mb:.0f} MB")

base = "data/cache"
resize_cache(f"{base}/train_images.dat", f"{base}/train_images_100.dat",
             f"{base}/train_labels.npy", (100, 100))
resize_cache(f"{base}/test_images.dat", f"{base}/test_images_100.dat",
             f"{base}/test_labels.npy", (100, 100))
print("\nDone! 100x100 cache ready.")