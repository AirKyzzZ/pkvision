"""Pyrender-based SMPL mesh renderer.

Renders SMPL meshes from GVHMR output as video overlays or standalone 3D views.
Two modes:
  - incam: overlay mesh on original video using camera intrinsics
  - global: third-person view with ground plane (for verification)

Usage:
    renderer = MeshRenderer(1920, 1080)
    # Single frame overlay
    img = renderer.render_incam_frame(verts[0], faces, K, background=video_frame)
    # Full video
    renderer.render_video(verts, faces, K, video_path="input.mp4", output_path="render.mp4")
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh


# Mesh appearance
MESH_COLOR = [0.55, 0.65, 0.80, 1.0]  # Soft blue-gray (Blender-like default)
GROUND_COLOR_A = [0.25, 0.25, 0.28, 1.0]  # Dark tile
GROUND_COLOR_B = [0.30, 0.30, 0.33, 1.0]  # Slightly lighter tile


class MeshRenderer:
    """Offscreen SMPL mesh renderer using pyrender."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._renderer = pyrender.OffscreenRenderer(width, height)

    def __del__(self):
        try:
            self._renderer.delete()
        except Exception:
            pass

    # ── In-Camera Rendering (overlay on video) ─────────────────────

    def render_incam_frame(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        K: np.ndarray,
        background: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render mesh in camera space, optionally overlay on background.

        GVHMR's incam params place the mesh in OpenCV camera coordinates
        (X-right, Y-down, Z-forward). Pyrender uses OpenGL (X-right, Y-up, Z-backward).
        We flip Y and Z to convert.

        Args:
            vertices: (6890, 3) SMPL vertices in camera space.
            faces: (13776, 3) face indices.
            K: (3, 3) camera intrinsic matrix from GVHMR.
            background: Optional (H, W, 3) BGR image to overlay on.

        Returns:
            (H, W, 3) BGR rendered image.
        """
        # Convert OpenCV camera space -> OpenGL camera space
        verts_gl = vertices.copy()
        verts_gl[:, 1] *= -1  # flip Y
        verts_gl[:, 2] *= -1  # flip Z

        # Build trimesh with uniform color
        mesh_tri = trimesh.Trimesh(vertices=verts_gl, faces=faces, process=False)
        mesh_tri.visual.vertex_colors = np.tile(
            np.array(MESH_COLOR) * 255, (len(verts_gl), 1)
        ).astype(np.uint8)

        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_tri, smooth=True)

        # Build scene
        scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],
            ambient_light=[0.3, 0.3, 0.3],
        )
        scene.add(mesh_pyrender)

        # Camera from intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy,
            znear=0.01, zfar=100.0,
        )
        # Camera at origin, looking down -Z (OpenGL default)
        scene.add(camera, pose=np.eye(4))

        # Lighting: key light from front-above + fill from the side
        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        key_pose = np.eye(4)
        key_pose[:3, :3] = _rotation_matrix_from_vectors([0, 0, -1], [0.2, -0.8, -0.6])
        scene.add(key_light, pose=key_pose)

        fill_light = pyrender.DirectionalLight(color=[0.8, 0.85, 0.9], intensity=2.0)
        fill_pose = np.eye(4)
        fill_pose[:3, :3] = _rotation_matrix_from_vectors([0, 0, -1], [-0.5, -0.3, -0.7])
        scene.add(fill_light, pose=fill_pose)

        # Render
        color, depth = self._renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        # color: (H, W, 4) uint8 RGBA

        # Convert to BGR for OpenCV
        rendered_bgr = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = color[:, :, 3].astype(np.float32) / 255.0

        if background is not None:
            # Resize background if needed
            bg = background
            if bg.shape[:2] != (self.height, self.width):
                bg = cv2.resize(bg, (self.width, self.height))

            # Alpha composite
            alpha_3ch = alpha[:, :, np.newaxis]
            result = (alpha_3ch * rendered_bgr.astype(np.float32)
                      + (1.0 - alpha_3ch) * bg.astype(np.float32))
            return result.astype(np.uint8)
        else:
            return rendered_bgr

    # ── Global 3D View ─────────────────────────────────────────────

    def render_global_frame(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        camera_distance: float = 4.0,
        camera_elevation: float = 20.0,
        camera_azimuth: float = 135.0,
        center: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render mesh from a third-person view with ground plane.

        Args:
            vertices: (6890, 3) SMPL vertices in world space (Y-up).
            faces: (13776, 3) face indices.
            camera_distance: Distance from mesh center.
            camera_elevation: Camera elevation angle in degrees.
            camera_azimuth: Camera azimuth angle in degrees.
            center: (3,) center point to look at. Auto-computed if None.

        Returns:
            (H, W, 3) BGR rendered image.
        """
        # Auto-compute center from mesh
        if center is None:
            center = vertices.mean(axis=0)

        # Build person mesh
        mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh_tri.visual.vertex_colors = np.tile(
            np.array(MESH_COLOR) * 255, (len(vertices), 1)
        ).astype(np.uint8)

        # Build ground plane (checkerboard)
        ground = _make_ground_plane(center, size=6.0)

        scene = pyrender.Scene(
            bg_color=[18, 18, 22, 255],  # Dark background
            ambient_light=[0.25, 0.25, 0.25],
        )
        scene.add(pyrender.Mesh.from_trimesh(mesh_tri, smooth=True))
        if ground is not None:
            scene.add(pyrender.Mesh.from_trimesh(ground, smooth=False))

        # Camera
        camera = pyrender.PerspectiveCamera(yfov=np.radians(45), aspectRatio=self.width / self.height)
        cam_pose = _orbit_camera_pose(center, camera_distance, camera_elevation, camera_azimuth)
        scene.add(camera, pose=cam_pose)

        # Lighting
        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        key_pose = np.eye(4)
        key_pose[:3, :3] = _rotation_matrix_from_vectors([0, 0, -1], [0.3, -0.7, -0.5])
        scene.add(key_light, pose=key_pose)

        fill_light = pyrender.DirectionalLight(color=[0.6, 0.65, 0.75], intensity=2.5)
        fill_pose = np.eye(4)
        fill_pose[:3, :3] = _rotation_matrix_from_vectors([0, 0, -1], [-0.4, -0.4, 0.6])
        scene.add(fill_light, pose=fill_pose)

        # Render
        color, _ = self._renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        return cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)

    # ── Video Rendering ────────────────────────────────────────────

    def render_video(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        output_path: str,
        K: np.ndarray | None = None,
        video_path: str | None = None,
        mode: str = "incam",
        fps: float = 30.0,
        camera_distance: float = 4.0,
    ) -> str:
        """Render full sequence to video.

        Args:
            vertices: (T, 6890, 3) mesh vertices.
            faces: (13776, 3) face indices.
            output_path: Output video path (.mp4).
            K: Camera intrinsics (required for mode="incam").
            video_path: Original video for background (mode="incam").
            mode: "incam" (overlay) or "global" (third-person).
            fps: Output video frame rate.
            camera_distance: For global mode.

        Returns:
            Path to rendered video.
        """
        T = len(vertices)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Open source video if provided
        cap = None
        if video_path and mode == "incam":
            cap = cv2.VideoCapture(video_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))

        # For global mode, compute stable camera target from trajectory center
        if mode == "global":
            trajectory_center = vertices[:, :, :].mean(axis=(0, 1))
            # Find the ground level (minimum Y across all frames)
            ground_y = vertices[:, :, 1].min()
            trajectory_center[1] = ground_y + 1.0  # Look at roughly pelvis height

        for i in range(T):
            # Read background frame
            bg = None
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    bg = frame

            if mode == "incam":
                if K is None:
                    raise ValueError("Camera intrinsics K required for incam mode")
                img = self.render_incam_frame(vertices[i], faces, K, background=bg)
            else:
                img = self.render_global_frame(
                    vertices[i], faces,
                    camera_distance=camera_distance,
                    center=trajectory_center if mode == "global" else None,
                )

            writer.write(img)

            if (i + 1) % 30 == 0 or i == T - 1:
                print(f"  Rendering: {i+1}/{T} frames", end="\r")

        print()
        writer.release()
        if cap is not None:
            cap.release()

        return output_path


# ── Utilities ──────────────────────────────────────────────────────


def _rotation_matrix_from_vectors(vec_from: list, vec_to: list) -> np.ndarray:
    """Rotation matrix that rotates vec_from to vec_to."""
    a = np.array(vec_from, dtype=np.float64)
    b = np.array(vec_to, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.abs(c + 1.0) < 1e-8:
        # 180-degree rotation
        return -np.eye(3)

    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return R


def _orbit_camera_pose(
    target: np.ndarray,
    distance: float,
    elevation_deg: float,
    azimuth_deg: float,
) -> np.ndarray:
    """Compute a 4x4 camera pose matrix orbiting around a target point."""
    elev = np.radians(elevation_deg)
    azim = np.radians(azimuth_deg)

    # Camera position in spherical coordinates
    x = distance * np.cos(elev) * np.sin(azim)
    y = distance * np.sin(elev)
    z = distance * np.cos(elev) * np.cos(azim)
    cam_pos = target + np.array([x, y, z])

    # Look-at matrix
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_norm

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # OpenGL: camera looks along -Z, so forward = -Z
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_pos

    return pose


def _make_ground_plane(
    center: np.ndarray,
    size: float = 6.0,
    tile_size: float = 0.5,
) -> trimesh.Trimesh | None:
    """Create a checkerboard ground plane mesh at ground level.

    The ground is placed at the minimum Y of the center minus a small offset.
    """
    ground_y = center[1] - 1.0  # Below the mesh center

    # Build a grid of tiles
    n_tiles = int(size / tile_size)
    half = size / 2.0
    cx, cz = center[0], center[2]

    verts = []
    face_list = []
    colors = []

    for ix in range(n_tiles):
        for iz in range(n_tiles):
            x0 = cx - half + ix * tile_size
            z0 = cz - half + iz * tile_size
            x1 = x0 + tile_size
            z1 = z0 + tile_size

            vi = len(verts)
            verts.extend([
                [x0, ground_y, z0],
                [x1, ground_y, z0],
                [x1, ground_y, z1],
                [x0, ground_y, z1],
            ])
            face_list.extend([
                [vi, vi + 1, vi + 2],
                [vi, vi + 2, vi + 3],
            ])

            # Checkerboard coloring
            is_light = (ix + iz) % 2 == 0
            c = GROUND_COLOR_B if is_light else GROUND_COLOR_A
            c_uint8 = [int(x * 255) for x in c]
            colors.extend([c_uint8] * 4)

    if not verts:
        return None

    mesh = trimesh.Trimesh(
        vertices=np.array(verts, dtype=np.float32),
        faces=np.array(face_list, dtype=np.int32),
        process=False,
    )
    mesh.visual.vertex_colors = np.array(colors, dtype=np.uint8)
    return mesh
