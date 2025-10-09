import numpy as np


def transform(pc, extrinsic_mat):
    zup = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype="f")  # Z_UP
    return np.dot(extrinsic_mat @ zup, pc.T).T


def random_y_rotation_matrix():
    pitch = np.deg2rad(np.random.uniform(-90, 90))

    # Rotation around Y-axis for pitch in Z-up coordinates
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    return R_y
