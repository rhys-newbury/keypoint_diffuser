#!/usr/bin/env python3
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageChops


# -------------------------------------------------------------
# helpers
# -------------------------------------------------------------
def safe_faces(faces):
    faces = np.array(faces)
    if faces.ndim == 1:
        faces = np.stack(faces, axis=0)
    return faces.astype(np.int32)


def make_kp_spheres(kps_centered, colors, size):
    """
    kps_centered: (K,3) AFTER centering mesh
    colors: (K,3) in 0..1 from the npz
    size: bbox size → use to scale sphere radius
    """
    # scale relative to object, the 2.8 is just to keep close to earlier guesses
    radius = 0.07 * (size / 2.8)
    sphere_tm = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere_meshes = []
    for kp, col in zip(kps_centered, colors):
        col = np.clip(col, 0.0, 1.0)
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[col[0], col[1], col[2], 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.5,
        )
        m = pyrender.Mesh.from_trimesh(sphere_tm, material=mat, smooth=True)
        T = np.eye(4)
        T[:3, 3] = kp  # we'll shift z later
        sphere_meshes.append((m, T))
    return sphere_meshes


# -------------------------------------------------------------
# render ONE chair (transparent bg) with keypoints
# -------------------------------------------------------------
def render_one_rgba(verts, faces, kps, colors, w=800, h=800):
    faces = safe_faces(faces)

    # 1) build mesh
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # 2) center by bbox (same as your golden version)
    vmin = tm.vertices.min(axis=0)
    vmax = tm.vertices.max(axis=0)
    center = 0.5 * (vmin + vmax)
    size = np.linalg.norm(vmax - vmin)
    if size == 0:
        size = 1.0

    tm.apply_translation(-center)              # mesh → origin

    kps_centered = kps - center[None, :]       # kps → origin too

    rot_y = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])
    tm.apply_transform(rot_y)
    kps_centered = (rot_y[:3, :3] @ kps_centered.T).T

    # 3) transparent-ish chair
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 0.4],  # adjust alpha here
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    pr_mesh = pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True)

    # 4) scene
    scene = pyrender.Scene(
        bg_color=[0, 0, 0, 0],              # transparent background
        ambient_light=[0.4, 0.4, 0.4, 1.0],
    )

    # add mesh a bit in front, like before
    mesh_pose = np.eye(4)
    mesh_pose[2, 3] = -0.5

    # 4b) add keypoints, but shift them by the same -0.5 in z
    kp_meshes = make_kp_spheres(kps_centered, colors, size)
    for m, T in kp_meshes:
        T2 = T.copy()
        T2[2, 3] -= 0.5      # SAME offset as mesh
        scene.add(m, pose=T2)

    scene.add(pr_mesh, pose=mesh_pose)

    # 5) camera (same as golden)
    dist = size * 1.8
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = dist
    scene.add(cam, pose=cam_pose)

    # 6) lights
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), pose=cam_pose)
    scene.add(pyrender.PointLight(color=np.ones(3), intensity=5.0), pose=cam_pose)
    scene.ambient_light = np.array([0.2, 0.2, 0.2, 1.0])

    # 7) render to RGBA
    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()

    # --- UN-PREMULTIPLY ONCE ---
    arr = color.astype(np.float32) / 255.0
    alpha = arr[..., 3:4]
    alpha = np.clip(alpha, 1e-6, 1.0)
    arr[..., :3] /= alpha
    arr = np.clip(arr, 0.0, 1.0)

    img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGBA")
    return img


def crop_rgba(im: Image.Image, i=None):
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (0, 0, 0, 0))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, 0)
    box = diff.getbbox()
    if i is not None:
        print(f"[crop] {i}: {box}")
    return im.crop(box) if box else im


def stitch_rgba(images):
    # assume all images are straight RGBA
    imgs = [img.convert("RGBA") for img in images]

    max_w = max(im.width for im in imgs)
    max_h = max(im.height for im in imgs)

    panel = Image.new("RGBA", (max_w * len(imgs), max_h), (0, 0, 0, 0))

    for i, im in enumerate(imgs):
        slot = Image.new("RGBA", (max_w, max_h), (0, 0, 0, 0))
        off = ((max_w - im.width) // 2, (max_h - im.height) // 2)
        slot.paste(im, off, im)
        panel.paste(slot, (i * max_w, 0), slot)

    return panel


def main():
    data = np.load("chairkeypoints.npz", allow_pickle=True)
    verts_list = data["verts"]
    faces_list = data["faces"]
    kps_list = data["kps"]
    colors = data["colors"]

    tiles = []
    for i in range(len(verts_list)):
        print(f"[INFO] rendering {i+1}/{len(verts_list)} ...")
        img = render_one_rgba(
            verts_list[i],
            faces_list[i],
            kps_list[i],
            colors,
            w=800,
            h=800,
        )
        img = crop_rgba(img, i)
        tiles.append(img)

    panel = stitch_rgba(tiles)
    white_bg = Image.new("RGBA", panel.size, (255, 255, 255, 255))
    white_bg.paste(panel, (0, 0), panel)

    # Option 1: save RGBA (still transparent-capable)
    white_bg.save("chairkeypoints_panel_pyrender.png")
    print("[✓] saved -> chairkeypoints_panel_pyrender.png")


if __name__ == "__main__":
    main()
