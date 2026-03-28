"""SMPL mesh generation from GVHMR output.

Loads SMPLx model, runs forward pass on saved GVHMR parameters,
projects to SMPL topology via sparse matrix, and exports meshes.

Usage:
    gen = SMPLMeshGenerator()
    gen.load_gvhmr("outputs/demo/backflip/hmr4d_results.pt")
    verts_incam, faces = gen.generate(space="incam")
    verts_global, _    = gen.generate(space="global")
    gen.export_obj(verts_incam[0], faces, "frame_0.obj")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh

# Default paths (GVHMR installation on this machine)
DEFAULT_MODEL_PATH = "C:/Users/pc/GVHMR/inputs/checkpoints/body_models"
DEFAULT_SPARSE_PATH = "C:/Users/pc/GVHMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt"


class SMPLMeshGenerator:
    """Generates SMPL mesh vertices from GVHMR hmr4d_results.pt output.

    Pipeline: SMPLx params -> SMPLx forward pass -> 10475 verts -> sparse project -> 6890 SMPL verts
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        sparse_path: str = DEFAULT_SPARSE_PATH,
        device: str | None = None,
    ):
        import smplx as smplx_lib

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load SMPLx model (for forward pass - produces 10475 vertices)
        self.smplx_model = smplx_lib.create(
            model_path,
            model_type="smplx",
            gender="neutral",
            num_betas=10,
            use_pca=False,
            flat_hand_mean=False,
        ).to(self.device)

        with torch.no_grad():
            pass  # model is in inference mode
        self.smplx_model.requires_grad_(False)

        # Load sparse projection matrix: (6890, 10475)
        self.smplx2smpl = torch.load(
            sparse_path, map_location=self.device, weights_only=False
        )

        # Load SMPL model just for its face topology (13776 triangles)
        smpl_model = smplx_lib.create(
            model_path, model_type="smpl", gender="neutral"
        )
        self.faces = smpl_model.faces.astype(np.int32)  # (13776, 3)

        # GVHMR data (populated by load_gvhmr)
        self._data = None
        self._K = None

    def load_gvhmr(self, pt_path: str) -> dict:
        """Load GVHMR hmr4d_results.pt.

        Returns the raw data dict for inspection.
        """
        self._data = torch.load(pt_path, map_location="cpu", weights_only=False)

        # Camera intrinsics
        if "K_fullimg" in self._data:
            self._K = self._data["K_fullimg"][0].numpy()  # (3, 3) - first frame

        return self._data

    @property
    def num_frames(self) -> int:
        if self._data is None:
            return 0
        return self._data["smpl_params_global"]["global_orient"].shape[0]

    @property
    def K(self) -> np.ndarray | None:
        """Camera intrinsic matrix (3x3) from GVHMR."""
        return self._K

    def generate(
        self,
        space: str = "incam",
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate SMPL mesh vertices from loaded GVHMR output.

        Args:
            space: "incam" (camera-space, for video overlay) or
                   "global" (world-space with gravity alignment, for analysis/3D view).
            batch_size: Process this many frames at a time to avoid OOM.

        Returns:
            (vertices, faces) where vertices is (T, 6890, 3) float32 numpy,
            faces is (13776, 3) int32 numpy.
        """
        if self._data is None:
            raise RuntimeError("No GVHMR data loaded. Call load_gvhmr() first.")

        key = f"smpl_params_{space}"
        if key not in self._data:
            raise ValueError(f"No '{key}' in GVHMR output. Available: {list(self._data.keys())}")

        params = self._data[key]
        T = params["global_orient"].shape[0]

        all_verts = []
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            B = end - start
            batch = {}
            for k, v in params.items():
                if not isinstance(v, torch.Tensor):
                    continue
                if v.shape[0] == 1 and T > 1:
                    batch[k] = v.expand(B, *v.shape[1:]).to(self.device)
                else:
                    batch[k] = v[start:end].to(self.device)

            # SMPLx forward needs all pose params — GVHMR only saves body+global.
            # Provide explicit zeros for jaw, eyes, hands so they broadcast correctly.
            if "jaw_pose" not in batch:
                batch["jaw_pose"] = torch.zeros(B, 3, device=self.device)
            if "leye_pose" not in batch:
                batch["leye_pose"] = torch.zeros(B, 3, device=self.device)
            if "reye_pose" not in batch:
                batch["reye_pose"] = torch.zeros(B, 3, device=self.device)
            if "left_hand_pose" not in batch:
                batch["left_hand_pose"] = torch.zeros(B, 45, device=self.device)
            if "right_hand_pose" not in batch:
                batch["right_hand_pose"] = torch.zeros(B, 45, device=self.device)
            if "expression" not in batch:
                batch["expression"] = torch.zeros(B, 10, device=self.device)

            with torch.no_grad():
                smplx_out = self.smplx_model(**batch)

            # SMPLx vertices -> SMPL vertices via sparse multiply
            smplx_verts = smplx_out.vertices  # (B, 10475, 3)
            smpl_verts = torch.stack(
                [self.smplx2smpl @ v for v in smplx_verts]
            )  # (B, 6890, 3)

            all_verts.append(smpl_verts.cpu().numpy())

        vertices = np.concatenate(all_verts, axis=0).astype(np.float32)  # (T, 6890, 3)
        return vertices, self.faces

    def export_obj(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        output_path: str,
    ) -> str:
        """Export a single frame's mesh as .obj file.

        Args:
            vertices: (6890, 3) vertex positions for one frame.
            faces: (13776, 3) face indices.
            output_path: Path for the .obj file.
        """
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_path)
        return output_path

    def export_sequence(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        output_dir: str,
        every_n: int = 1,
        fmt: str = "obj",
    ) -> list[str]:
        """Export multiple frames as mesh files.

        Args:
            vertices: (T, 6890, 3) all frames.
            faces: (13776, 3) face indices.
            output_dir: Directory for output files.
            every_n: Export every N-th frame (1 = all frames).
            fmt: Export format ("obj", "ply", "glb", "stl").

        Returns:
            List of exported file paths.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for i in range(0, len(vertices), every_n):
            path = str(out_dir / f"frame_{i:05d}.{fmt}")
            mesh = trimesh.Trimesh(vertices=vertices[i], faces=faces, process=False)
            mesh.export(path)
            paths.append(path)

        return paths
