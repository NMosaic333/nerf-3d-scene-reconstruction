import numpy as np

def get_rays(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

    dirs = np.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -np.ones_like(i)
    ], -1)

    rays_d = (dirs[..., None, :] * c2w[:3, :3]).sum(-1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))

    return rays_o, rays_d