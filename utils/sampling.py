import tensorflow as tf

def sample_points(rays_o, rays_d, N):

    rays_o = tf.cast(rays_o, tf.float32)
    rays_d = tf.cast(rays_d, tf.float32)

    t_vals = tf.linspace(2.0, 6.0, N)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., None]

    return pts, t_vals