from models.nerf_model import NeRF
from utils.rendering import render_rays
from utils.rays import get_rays

import json
import tensorflow as tf
import yaml
import imageio
import numpy as np
import os

# -------------------------
# Create checkpoint folder
# -------------------------
os.makedirs("checkpoints", exist_ok=True)

# -------------------------
# Load config
# -------------------------
with open("configs/nerf_config.yaml") as f:
    config = yaml.safe_load(f)

N_ITERS = config["training"]["iterations"]
BATCH   = config["training"]["batch_size"]
LR      = config["training"]["learning_rate"]
N_SAMPLES = config["rendering"]["n_samples"]

# -------------------------
# Dataset path
# -------------------------
DATASET_PATH = "data/nerf-dataset/nerf_synthetic/nerf_synthetic/chair"

# -------------------------
# Load dataset metadata
# -------------------------
with open(os.path.join(DATASET_PATH, "transforms_train.json"), "r") as f:
    meta = json.load(f)

images = []
poses = []

# -------------------------
# Load images and poses
# -------------------------
for frame in meta["frames"]:

    img_path = os.path.join(DATASET_PATH, frame["file_path"] + ".png")

    img = imageio.v2.imread(img_path) / 255.0

    # remove alpha channel if present
    if img.shape[-1] == 4:
        img = img[..., :3]

    images.append(img)

    pose = np.array(frame["transform_matrix"], dtype=np.float32)
    poses.append(pose)

images = np.array(images, dtype=np.float32)
poses = np.array(poses, dtype=np.float32)

# -------------------------
# Camera parameters
# -------------------------
H, W = images[0].shape[:2]
focal = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])

# -------------------------
# Initialize model
# -------------------------
model = NeRF()
optimizer = tf.keras.optimizers.Adam(LR)

# -------------------------
# Loss function
# -------------------------
def loss_fn(pred, target):
    return tf.reduce_mean((pred - target) ** 2)

# -------------------------
# PSNR metric
# -------------------------
def compute_psnr(mse):
    return -10.0 * tf.math.log(mse) / tf.math.log(10.0)

# -------------------------
# Precompute rays
# -------------------------
all_rays_o = []
all_rays_d = []
all_rgb = []

for img, pose in zip(images, poses):

    ro, rd = get_rays(H, W, focal, pose)

    all_rays_o.append(ro.reshape(-1, 3))
    all_rays_d.append(rd.reshape(-1, 3))
    all_rgb.append(img.reshape(-1, 3))

all_rays_o = np.concatenate(all_rays_o, axis=0)
all_rays_d = np.concatenate(all_rays_d, axis=0)
all_rgb    = np.concatenate(all_rgb, axis=0)

# convert to tensors
all_rays_o = tf.convert_to_tensor(all_rays_o, dtype=tf.float32)
all_rays_d = tf.convert_to_tensor(all_rays_d, dtype=tf.float32)
all_rgb    = tf.convert_to_tensor(all_rgb, dtype=tf.float32)

print("Total rays:", len(all_rays_o))

# -------------------------
# Training loop
# -------------------------
for step in range(N_ITERS):

    idx = np.random.randint(0, len(all_rays_o), size=BATCH)

    ro = tf.gather(all_rays_o, idx)
    rd = tf.gather(all_rays_d, idx)
    gt = tf.gather(all_rgb, idx)

    with tf.GradientTape() as tape:

        pred = render_rays(model, ro, rd, N_SAMPLES)

        loss = loss_fn(pred, gt)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # compute PSNR
    psnr = compute_psnr(loss)

    # -------------------------
    # Logging
    # -------------------------
    if step % 200 == 0:
        print(
            f"Step {step:05d} | Loss: {loss.numpy():.6f} | PSNR: {psnr.numpy():.2f}"
        )

    # -------------------------
    # Save checkpoint
    # -------------------------
    if step % 1000 == 0 and step > 0:
        model.save_weights("checkpoints/nerf_checkpoint.weights.h5")

# -------------------------
# Final save
# -------------------------
model.save_weights("checkpoints/nerf_checkpoint.weights.h5")

print("Training complete.")