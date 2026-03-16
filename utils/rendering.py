import tensorflow as tf
from utils.rays import get_rays
def sample_points(rays_o, rays_d, N):
    # Force casting everything to float32
    rays_o = tf.cast(rays_o, tf.float32)
    rays_d = tf.cast(rays_d, tf.float32)

    t_vals = tf.linspace(2.0, 6.0, N)
    t_vals = tf.cast(t_vals, tf.float32)

    pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., None]
    return pts, t_vals

def render_rays(model, rays_o, rays_d, N=64):
    B = tf.shape(rays_o)[0]

    # Sample points
    pts, t_vals = sample_points(rays_o, rays_d, N)

    # Flatten for MLP
    pts_flat = tf.reshape(pts, (-1, 3))
    dirs = tf.repeat(rays_d[:, None, :], N, axis=1)
    dirs_flat = tf.reshape(dirs, (-1, 3))

    # Run MLP
    raw = model(pts_flat, dirs_flat)

    # ❗ CRITICAL FIX: reshape back
    raw = tf.reshape(raw, (B, N, 4))

    # RGB + sigma
    rgb = tf.sigmoid(raw[..., :3])           # (B, N, 3)
    sigma = tf.nn.relu(raw[..., 3])          # (B, N)

    # Volume rendering
    deltas = t_vals[1:] - t_vals[:-1]
    deltas = tf.concat([deltas, [1e-3]], axis=0)
    deltas = tf.reshape(deltas, (1, N))

    alpha = 1.0 - tf.exp(-sigma * deltas)

    eps = 1e-10
    trans = tf.math.cumprod(1.0 - alpha + eps, axis=-1, exclusive=True)

    weights = alpha * trans                  # (B, N)

    # Match rgb shape
    weights = tf.expand_dims(weights, -1)    # (B, N, 1)

    # Weighted sum
    rgb_map = tf.reduce_sum(weights * rgb, axis=1)  # (B, 3)

    return rgb_map

def render_rays_chunked(model, rays_o, rays_d, N=96, chunk=4096):
    all_rgb = []

    for i in range(0, rays_o.shape[0], chunk):
        ro_chunk = rays_o[i: i+chunk]
        rd_chunk = rays_d[i: i+chunk]

        rgb_chunk = render_rays(model, ro_chunk, rd_chunk, N)
        all_rgb.append(rgb_chunk)

    return tf.concat(all_rgb, axis=0)

def render_image(model, pose):
    ro, rd = get_rays(H, W, focal, pose)
    ro = ro.reshape(-1, 3)
    rd = rd.reshape(-1, 3)

    rgb = render_rays_chunked(model, ro, rd, N=96, chunk=4096)
    return rgb.numpy().reshape(H, W, 3)