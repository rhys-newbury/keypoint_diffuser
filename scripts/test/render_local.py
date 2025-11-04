#!/usr/bin/env python3
import numpy as np
import open3d as o3d
from PIL import Image, ImageChops


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def build_o3d_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def make_kp_spheres(kps, colors, radius=0.02, resolution=12):
    geoms = []
    for i, kp in enumerate(kps):
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        sph.translate(kp)
        col = np.clip(colors[i], 0.0, 1.0)
        sph.paint_uniform_color(col.tolist())
        sph.compute_vertex_normals()
        geoms.append(sph)
    return geoms


def make_point_cloud(points, color=(0, 0, 0)):
    """Build Open3D point cloud geometry from Nx3 array."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    col = np.tile(np.array(color, dtype=np.float32)[None, :], (len(points), 1))
    pc.colors = o3d.utility.Vector3dVector(col)
    return pc

def build_o3d_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def mesh_to_lineset(mesh: o3d.geometry.TriangleMesh, color=(0, 0, 0)):
    """
    Convert a triangle mesh to a wireframe LineSet.
    """
    triangles = np.asarray(mesh.triangles)
    edges = set()
    for tri in triangles:
        i0, i1, i2 = tri
        edges.add(tuple(sorted((i0, i1))))
        edges.add(tuple(sorted((i1, i2))))
        edges.add(tuple(sorted((i2, i0))))
    edges = np.array(list(edges), dtype=np.int32)
    pastel_red = np.array([0.94, 0.7, 0.7], dtype=np.float64)
    line_set = o3d.geometry.LineSet()
    line_set.points = mesh.vertices
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(pastel_red[None, :], (edges.shape[0], 1))
    )
    return line_set


def make_kp_spheres(kps, colors, radius=0.02, resolution=12):
    geoms = []
    for i, kp in enumerate(kps):
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        sph.translate(kp)
        col = np.clip(colors[i], 0.0, 1.0)
        sph.paint_uniform_color(col.tolist())
        sph.compute_vertex_normals()
        geoms.append(sph)
    return geoms


def make_point_cloud(points, color=(0, 0, 0)):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    col = np.tile(np.array(color, dtype=np.float32)[None, :], (len(points), 1))
    pc.colors = o3d.utility.Vector3dVector(col)
    return pc

# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------
def render_one_legacy(
    verts,
    faces,
    kps,
    colors,
    pc,
    center,
    extent,
    width=800,
    height=800,
    target_tris=50000,
    wire_color=(0.0, 0.0, 0.0),
    show_solid=False,  # 👈 set to True if you want solid + wire
):
    """
    Renders (wireframe) mesh + keypoints (+ optional PC) using Open3D legacy visualizer.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=True)

    # --- main mesh ---
    mesh = build_o3d_mesh(verts, faces)

    # decimate
    # if len(mesh.triangles) > target_tris:
    #     mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    #     mesh.remove_degenerate_triangles()
    #     mesh.remove_duplicated_triangles()
    #     mesh.remove_duplicated_vertices()
    #     mesh.remove_non_manifold_edges()
    # mesh.compute_vertex_normals()

    # make wireframe
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))  # rotate +90° around X-axis
    mesh.rotate(R, center=(0, 0, 0))

    wire = mesh_to_lineset(mesh, color=wire_color)
    vis.add_geometry(wire)

    # pastel_red = [0.94, 0.7, 0.7]  # soft rose tone
    # mesh.paint_uniform_color(pastel_red)
    # vis.add_geometry(mesh)

    # add to scene
    # if show_solid:
    #     mesh.paint_uniform_color([0.85, 0.85, 0.85])
    #     vis.add_geometry(mesh)

    # --- OPTIONAL: point cloud (commented out) ---
    # pc_geom = make_point_cloud(pc, color=(0.05, 0.05, 0.05))
    # vis.add_geometry(pc_geom)

    # --- keypoints ---
    kps = (R @ kps.T).T

    kp_geoms = make_kp_spheres(kps, colors, radius=0.08, resolution=16)
    for g in kp_geoms:
        vis.add_geometry(g)

    # --- camera ---
    ctr = vis.get_view_control()

    # 👇 move the eye BELOW the object (negative Y)
    # eye = center - np.array([0.0, extent * 3.5, 0.0])

    # lookat = center
    # up = np.array([0.0, 0.0, -1.0])  # flip up vector since we’re looking upward

    # ctr.set_lookat(lookat.tolist())
    # ctr.set_front((lookat - eye).tolist())
    # ctr.set_up(up.tolist())
    # ctr.set_zoom(0.9)

    ctr = vis.get_view_control()

    # 👁️ Position the camera in front of the object (along +X)
    eye = center + np.array([extent * 3.5, 0.0, 0.0])   # move camera forward
    lookat = center                                     # look toward the center
    up = np.array([0.0, 0.0, 1.0])                      # Z is still "up"

    ctr.set_lookat(lookat.tolist())
    ctr.set_front((lookat - eye).tolist())               # direction camera faces
    ctr.set_up(up.tolist())
    ctr.set_zoom(0.9)


    # ctr = vis.get_view_control()
    # eye = center + np.array([0.0, extent * 3.5, 0.0])
    # lookat = center
    # up = np.array([0.0, 0.0, 1.0])
    # ctr.set_lookat(lookat.tolist())
    # ctr.set_front((lookat - eye).tolist())
    # ctr.set_up(up.tolist())
    # ctr.set_zoom(0.9)


    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)

    vis.run()
    vis.destroy_window()

    img_np = (255 * np.asarray(img)).astype(np.uint8)
    return Image.fromarray(img_np)

def crop_whitespace(im: Image.Image, bg_color=(255, 255, 255), tol=5):
    """
    Crop away background near bg_color (RGB).
    Returns cropped image.
    """
    if im.mode != "RGB":
        im = im.convert("RGB")
    bg = Image.new("RGB", im.size, bg_color)
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, 0)
    bbox = diff.getbbox()
    # import pdb; pdb.set_trace()
    return im.crop(bbox) if bbox else im


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # 1. load exported data
    data = np.load("mugkeypoints.npz", allow_pickle=True)
    verts_list = data["verts"]
    faces_list = data["faces"]
    kps_list = data["kps"]
    pcs_list = data["pcs"]  # NEW
    colors = data["colors"]
    lo = data["bounds_lo"]
    hi = data["bounds_hi"]

    # shared camera data
    center = 0.5 * (lo + hi)
    extent = np.linalg.norm(hi - lo)

    # 2. render each sample
    tiles = []
    for i in range(len(verts_list)):
        im = render_one_legacy(
            verts_list[i],
            faces_list[i],
            kps_list[i],
            colors,
            pcs_list[i],  # <── point cloud here
            center,
            extent,
            width=800,
            height=800,
        )
        tiles.append(im)

    # 3. stitch into horizontal panel

    # --- After rendering tiles ---

    # 1. crop whitespace from each
    cropped_tiles = [crop_whitespace(im, bg_color=(255, 255, 255), tol=5) for im in tiles]

    # 2. find the maximum cropped width/height
    max_w = max(im.width for im in cropped_tiles)
    max_h = max(im.height for im in cropped_tiles)

    # 3. pad each cropped image to same size (centered)
    padded_tiles = []
    for im in cropped_tiles:
        padded = Image.new("RGB", (max_w, max_h), (255, 255, 255))
        offset = ((max_w - im.width) // 2, (max_h - im.height) // 2)
        padded.paste(im, offset)
        padded_tiles.append(padded)

    # 4. stitch side-by-side
    panel_w = max_w * len(padded_tiles)
    panel_h = max_h
    panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    for i, im in enumerate(padded_tiles):
        panel.paste(im, (i * max_w, 0))

    panel.save("guitarkeypoints_panel_o3d.png")


if __name__ == "__main__":
    main()
