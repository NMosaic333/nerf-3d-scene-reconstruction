from models.nerf_model import NeRF
from utils.rendering import render_image
from utils.rays import get_rays

import tensorflow as tf
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
import os

# -------------------------
# Load config
# -------------------------
with open("configs/nerf_config.yaml") as f:
    config = yaml.safe_load(f)

N_SAMPLES = config["rendering"]["n_samples"]

# -------------------------
# Load model
# -------------------------
model = NeRF()
model.load_weights("checkpoints/nerf_checkpoint.weights.h5")

# -------------------------
# Metrics
# -------------------------
def compute_psnr(pred, gt):
    mse = tf.reduce_mean((pred - gt) ** 2)
    return -10.0 * tf.math.log(mse) / tf.math.log(10.0)

def compute_ssim(pred, gt):
    pred = tf.reshape(pred, (H, W, 3))
    gt = tf.reshape(gt, (H, W, 3))

    pred = tf.expand_dims(pred, 0)
    gt = tf.expand_dims(gt, 0)

    return tf.image.ssim(pred, gt, max_val=1.0)

# -------------------------
# Dataset path
# -------------------------
DATA_DIR = "data/nerf-dataset/nerf_synthetic/nerf_synthetic/chair"

# -------------------------
# Load metadata
# -------------------------
with open(os.path.join(DATA_DIR, "transforms_test.json")) as f:
    meta = json.load(f)

test_poses = []
test_images = []

for frame in meta["frames"]:

    pose = np.array(frame["transform_matrix"], dtype=np.float32)
    test_poses.append(pose)

    img_path = os.path.join(DATA_DIR, frame["file_path"] + ".png")
    image = plt.imread(img_path)[..., :3]  # remove alpha channel

    test_images.append(image)

test_poses = np.array(test_poses)
test_images = np.array(test_images)

# -------------------------
# Camera parameters
# -------------------------
camera_angle_x = float(meta["camera_angle_x"])

H, W = test_images[0].shape[:2]
focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

# -------------------------
# Evaluation
# -------------------------
psnr_values = []
ssim_values = []

for i, (img, pose) in enumerate(zip(test_images, test_poses)):

    rays_o, rays_d = get_rays(H, W, focal, pose)

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    rays_o = tf.convert_to_tensor(rays_o, dtype=tf.float32)
    rays_d = tf.convert_to_tensor(rays_d, dtype=tf.float32)

    pred = render_image(model, rays_o, rays_d)

    gt = img.reshape(-1, 3)

    psnr = compute_psnr(pred, gt)
    ssim = compute_ssim(pred, gt)

    psnr_values.append(psnr.numpy())
    ssim_values.append(ssim.numpy())

    print(f"Test image {i:03d}: PSNR={psnr:.2f}, SSIM={ssim:.3f}")

print("\nAverage Results:")
print(f"PSNR = {np.mean(psnr_values):.2f}")
print(f"SSIM = {np.mean(ssim_values):.3f}")