"""
GNM Face Tracker — Real-time face tracking with Google's GNM parametric head model.
====================================================================================

Uses MediaPipe Face Mesh for real-time landmark detection and GNM for 3D face
reconstruction.  The pipeline has two stages:

1. **Identity Fit** — The user sits with a neutral expression.  We optimise
   GNM's 253 identity parameters so the model's 68 standard landmarks match
   the user's MediaPipe-tracked landmarks.

2. **Real-time Tracking** — Each frame we estimate the 383 expression
   parameters from live landmarks and generate the deformed GNM mesh.

Press 'q' to quit | 'r' to re-fit identity | 'f' to toggle fullscreen GNM view.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import WebcamManager

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
FaceMesh = mp_face_mesh.FaceMesh

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NUM_MP_LANDMARKS = 478

# ---------------------------------------------------------------------------
# MediaPipe → iBUG 68 mapping
# ---------------------------------------------------------------------------
# Maps the 68 standard iBUG facial landmarks to MediaPipe Face Mesh indices.
# These are the canonical correspondences used across the community.
# MediaPipe uses 478 landmarks (with iris refinement).
#
# iBUG 68 layout:
#   0-16  : Jawline (17 pts)
#   17-21 : Right eyebrow (5 pts)
#   22-26 : Left eyebrow (5 pts)
#   27-30 : Nose bridge (4 pts)
#   31-35 : Nose bottom (5 pts)
#   36-41 : Right eye (6 pts)
#   42-47 : Left eye (6 pts)
#   48-59 : Outer mouth (12 pts)
#   60-67 : Inner mouth (8 pts)
# ---------------------------------------------------------------------------

IBUG68_TO_MEDIAPIPE: dict[int, int] = {
    # Jawline
    0: 234, 1: 93, 2: 132, 3: 58, 4: 172, 5: 136, 6: 150,
    7: 176, 8: 148, 9: 152, 10: 377, 11: 400, 12: 378, 13: 379,
    14: 365, 15: 397, 16: 288,
    # Right eyebrow
    17: 70, 18: 63, 19: 105, 20: 66, 21: 107,
    # Left eyebrow
    22: 336, 23: 296, 24: 334, 25: 293, 26: 300,
    # Nose bridge
    27: 168, 28: 6, 29: 197, 30: 195,
    # Nose bottom
    31: 5, 32: 4, 33: 1, 34: 19, 35: 94,
    # Right eye
    36: 33, 37: 246, 38: 161, 39: 160, 40: 159, 41: 158,
    # Left eye
    42: 362, 43: 398, 44: 384, 45: 385, 46: 386, 47: 387,
    # Outer mouth
    48: 61, 49: 185, 50: 40, 51: 39, 52: 37, 53: 0,
    54: 267, 55: 269, 56: 270, 57: 409, 58: 291, 59: 308,
    # Inner mouth
    60: 415, 61: 310, 62: 311, 63: 312, 64: 13, 65: 82, 66: 81, 67: 80,
}

# MediaPipe → iBUG (reverse mapping)
MEDIAPIPE_TO_IBUG68: dict[int, int] = {
    v: k for k, v in IBUG68_TO_MEDIAPIPE.items()
}


# ---------------------------------------------------------------------------
# Helper: rigid Procrustes alignment
# ---------------------------------------------------------------------------

def procrustes_align(
    source: np.ndarray,
    target: np.ndarray,
    use_scaling: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align *source* to *target* via similarity transform (rotation, translation,
    and optionally uniform scale).

    Args:
        source: (N, 3) points to align.
        target: (N, 3) reference points.
        use_scaling: If True, estimate uniform scale as well.

    Returns:
        (aligned_source, rotation_matrix, scale)
    """
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid

    # Optimal rotation via SVD
    h = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    if use_scaling:
        num = np.sum(tgt_centered * (src_centered @ r))
        den = np.sum(src_centered ** 2)
        scale = num / max(den, 1e-12)
    else:
        scale = 1.0

    aligned = scale * (source - src_centroid) @ r + tgt_centroid
    return aligned, r, scale


# ---------------------------------------------------------------------------
# Simple mesh renderer (software rasteriser)
# ---------------------------------------------------------------------------

class SimpleMeshRenderer:
    """Software rasteriser for rendering the GNM mesh to a BGR image.

    Uses flat shading with per-face normals and depth-sorted triangle
    rasterisation via OpenCV's fillPoly — fast enough for real-time use.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
        fov_y: float = 45.0,
    ):
        self.width, self.height = image_size
        fov_rad = np.radians(fov_y)
        fx = self.width / (2.0 * np.tan(fov_rad / 2.0))
        fy = fx
        cx = self.width / 2.0
        cy = self.height / 2.0
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    def _project(
        self, vertices: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Project vertices to 2D.  Returns (u, v, depth, cam_points)."""
        rotmat, _ = cv2.Rodrigues(rvec)
        cam = vertices @ rotmat.T + tvec.reshape(1, 3)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        z = cam[:, 2]
        z_safe = np.where(np.abs(z) < 1e-6, np.copysign(1e-6, z), z)
        u = fx * cam[:, 0] / z_safe + cx
        v = fy * cam[:, 1] / z_safe + cy
        return u, v, z_safe, cam

    def render(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        vertex_normals: np.ndarray | None = None,
        rvec: np.ndarray | None = None,
        tvec: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render the mesh to a BGR image using flat shading.

        Args:
            vertices: (V, 3) vertex positions.
            triangles: (T, 3) triangle indices.
            vertex_normals: (V, 3) per-vertex normals (unused — computed from faces).
            rvec: (3,) axis-angle rotation.
            tvec: (3,) translation.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if rvec is None:
            rvec = np.zeros(3, dtype=np.float32)
        if tvec is None:
            tvec = np.array([0.0, 0.05, 1.8], dtype=np.float32)

        u, v, z, cam = self._project(vertices, rvec, tvec)
        img = np.full((self.height, self.width, 3), 230, dtype=np.uint8)
        rotmat, _ = cv2.Rodrigues(rvec)

        # Light direction in camera space
        light_dir = np.array([0.15, -0.25, 0.95])
        light_dir = light_dir / np.linalg.norm(light_dir)

        # Base skin tone BGR
        base = np.array([185, 145, 115], dtype=np.float32)
        ambient = 0.3

        T = len(triangles)
        tri_depth = np.empty(T, dtype=np.float32)
        tri_colours = np.empty((T, 3), dtype=np.uint8)

        # Compute face normals, shading, and depth in one pass
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / np.maximum(norms, 1e-8)

        # Face normals in camera space
        fn_cam = face_normals @ rotmat.T
        lambert = np.clip(np.dot(fn_cam, light_dir), 0.0, 1.0)
        shade = ambient + (1.0 - ambient) * lambert

        # Depth per triangle
        tri_depth = z[triangles].mean(axis=1)

        # Colour per triangle
        tri_colours = np.clip(base * shade[:, None], 0, 255).astype(np.uint8)

        # Sort farthest → nearest (painter's algorithm)
        order = np.argsort(-tri_depth)

        for tri_idx in order:
            tri = triangles[tri_idx]
            pts = np.stack([u[tri], v[tri]], axis=1).astype(np.int32)

            # Skip off-screen
            if (pts[:, 0].min() >= self.width or pts[:, 0].max() < 0 or
                    pts[:, 1].min() >= self.height or pts[:, 1].max() < 0):
                continue

            # Skip back-facing (face normal points away from camera)
            if fn_cam[tri_idx, 2] >= 0:
                continue

            colour = tuple(int(c) for c in tri_colours[tri_idx])
            cv2.fillPoly(img, [pts], colour)

        return img

    @staticmethod
    def _compute_vertex_normals(
        vertices: np.ndarray, triangles: np.ndarray
    ) -> np.ndarray:
        """Compute per-vertex normals by averaging adjacent face normals."""
        vn = np.zeros_like(vertices)
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
        fn = fn / np.maximum(fn_len, 1e-8)
        np.add.at(vn, triangles[:, 0], fn)
        np.add.at(vn, triangles[:, 1], fn)
        np.add.at(vn, triangles[:, 2], fn)
        norms = np.linalg.norm(vn, axis=1, keepdims=True)
        vn = vn / np.maximum(norms, 1e-8)
        return vn


# ---------------------------------------------------------------------------
# GNM Face Tracker
# ---------------------------------------------------------------------------

class GNMFaceTracker:
    """Real-time face tracking with GNM.

    Usage::

        tracker = GNMFaceTracker()
        tracker.run()
    """

    def __init__(self):
        self.gnm = None
        self.face_mesh: Optional[FaceMesh] = None
        self.identity: Optional[np.ndarray] = None   # (253,)
        self.expression: Optional[np.ndarray] = None  # (383,)

        # GNM 68-landmark 3D positions on the template mesh
        self._gnm68_template: Optional[np.ndarray] = None  # (68, 3)

        # Pre-computed regressor: MP 68 landmarks → expression params
        self._expr_regressor: Optional[np.ndarray] = None  # (E, 68*3)

        # Cached landmark definitions
        self._lm_indices_cache: Optional[np.ndarray] = None
        self._lm_weights_cache: Optional[np.ndarray] = None
        self._num_cached_lm: int = 0

        # Fallback
        self._fallback_landmark_vertex_indices: np.ndarray | None = None

        # State
        self._show_fullscreen_gnm = False
        self._fps_window: list[float] = []
        self._fps = 0.0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_gnm(self) -> bool:
        """Load GNM and extract the 68 landmark positions."""
        print("[GNM] Loading GNM Head model ...")
        try:
            from gnm.shape import gnm_numpy
            from gnm.shape.gnm_landmarks import (
                GNMLandmarksType,
                load_landmarks,
                check_body_part_compatibility,
            )
            from gnm.shape.gnm_landmarks import GNMLandmarksDataNotLinkedError

            self.gnm = gnm_numpy.GNM.from_local(
                version=gnm_numpy.GNMMajorVersion.V3,
                variant=gnm_numpy.GNMVariant.HEAD,
            )
            print(f"[GNM] Loaded.  V={self.gnm.num_vertices}  "
                  f"I={self.gnm.identity_dim}  E={self.gnm.expression_dim}")

            # Load the 68 sparse head landmarks
            print("[GNM] Loading HEAD_SPARSE_68 landmarks ...")
            try:
                landmarks_config = load_landmarks(GNMLandmarksType.HEAD_SPARSE_68)
                self._extract_68_landmark_positions(landmarks_config)
                print(f"[GNM] 68 landmark positions extracted "
                      f"({len(self._gnm68_template)} points).")
            except GNMLandmarksDataNotLinkedError:
                print("[GNM] WARNING: Landmark data not linked.  "
                      "Using fallback vertex indices.")
                self._use_fallback_landmarks()

            return True

        except ImportError as e:
            print(f"[GNM] ERROR: {e}")
            print("  To install GNM:")
            print("    git clone https://github.com/google/GNM.git")
            print("    cd GNM/gnm/shape && pip install -e .")
            return False
        except Exception as e:
            print(f"[GNM] ERROR loading model: {e}")
            return False

    def _extract_68_landmark_positions(self, landmarks_config) -> None:
        """Compute the 3D position of each of the 68 landmarks on the GNM template.

        Each landmark is defined by multiple (vertex_index, weight) pairs.
        We compute the weighted sum of the template vertex positions.
        """
        indices = landmarks_config.indices   # (68, K) — K vertex indices per landmark
        weights = landmarks_config.weights   # (68, K) — corresponding weights
        template = self.gnm.template_vertex_positions  # (V, 3)

        num_lm, num_pairs = indices.shape
        positions = np.zeros((num_lm, 3), dtype=np.float32)
        for i in range(num_lm):
            pos = np.zeros(3, dtype=np.float32)
            for j in range(num_pairs):
                vtx_idx = indices[i, j]
                w = weights[i, j]
                if w > 0 and vtx_idx >= 0:
                    pos += w * template[vtx_idx]
            positions[i] = pos
        self._gnm68_template = positions

    def _use_fallback_landmarks(self) -> None:
        """Use a subset of GNM vertices as approximate landmarks.

        Chooses vertices that are likely to be facial feature points
        based on their positions on the template mesh.
        """
        template = self.gnm.template_vertex_positions
        v = template

        # Heuristic: pick vertices at extremal positions along cardinal axes
        # This is NOT anatomically accurate but serves as a fallback.
        nose_tip = np.argmax(v[:, 2])  # front-most
        chin = np.argmin(v[:, 1])      # bottom-most (y points down in GNM?)

        # Approximate: use the 68 most "extreme" vertices
        # Better than nothing, but the landmark data file is strongly preferred
        centroid = v.mean(axis=0)
        dists = np.linalg.norm(v - centroid, axis=1)
        extreme_indices = np.argsort(-dists)[:68]
        extreme_indices = np.sort(extreme_indices)

        self._gnm68_template = template[extreme_indices]
        self._fallback_landmark_vertex_indices = extreme_indices

        print(f"[GNM] Using {len(extreme_indices)} extreme vertices as "
              f"fallback landmarks (nose_tip={nose_tip}, chin={chin}).")

    def _setup_mediapipe(self):
        """Initialise MediaPipe Face Mesh."""
        print("[MediaPipe] Initialising Face Mesh ...")
        self.face_mesh = FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,        # includes iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.6,
        )

    # ------------------------------------------------------------------
    # Identity fitting
    # ------------------------------------------------------------------

    def fit_identity_interactive(self) -> bool:
        """Interactive identity fitting via webcam capture.

        The user sees a live preview.  Press SPACE to capture a neutral
        frame; press 'q' to skip.
        """
        print("\n" + "=" * 60)
        print("  Identity Fit")
        print("=" * 60)
        print("  Look straight at the camera with a NEUTRAL expression.")
        print("  Press SPACE to capture  |  'q' to skip\n")

        with WebcamManager(width=IMAGE_WIDTH, height=IMAGE_HEIGHT) as cam:
            captured_lm = None
            while True:
                success, frame = cam.read()
                if not success:
                    continue

                display = frame.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                status = "Face: DETECTED" if results.multi_face_landmarks else "Face: --"
                cv2.putText(display, status, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if results.multi_face_landmarks else (0, 0, 255), 2)
                cv2.putText(display, "SPACE = capture  |  q = skip",
                            (20, display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
                cv2.imshow("Identity Fit", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0]
                    captured_lm = np.array(
                        [[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32
                    )
                    print("[Identity Fit] Frame captured.  Optimising ...")
                    break
                elif key == ord("q"):
                    break

        cv2.destroyWindow("Identity Fit")

        if captured_lm is None:
            print("[Identity Fit] Skipped — no frame captured.")
            return False

        self.identity = self._optimise_identity(captured_lm)
        # Pre-compute the expression regressor with the fitted identity
        self._build_expression_regressor()
        return True

    def _optimise_identity(self, mp_landmarks_3d: np.ndarray) -> np.ndarray:
        """Optimise GNM identity parameters to match the user's landmarks.

        Works in GNM space: converts MediaPipe landmarks via a similarity
        transform, then optimises the 253 identity coefficients.

        The fitting iterates a few times because the similarity transform
        and identity parameters are coupled (the transform should only model
        pose/scale, not face shape).

        Args:
            mp_landmarks_3d: (478, 3) MediaPipe 3D landmarks from the user.

        Returns:
            (253,) optimised identity parameters.
        """
        template_68 = self._gnm68_template.astype(np.float64)  # (68, 3) in GNM space

        # Gather user's corresponding 68 landmarks in MP space
        user_68_mp = np.zeros((68, 3), dtype=np.float64)
        for ibug_idx, mp_idx in IBUG68_TO_MEDIAPIPE.items():
            if mp_idx < len(mp_landmarks_3d):
                user_68_mp[ibug_idx] = mp_landmarks_3d[mp_idx]

        # Prepare GNM data
        id_basis = self.gnm.vertex_identity_basis.astype(np.float64)  # (I, V, 3)
        template_v = self.gnm.template_vertex_positions.astype(np.float64)  # (V, 3)

        # Load landmark vertex indices and weights
        try:
            from gnm.shape.gnm_landmarks import (
                GNMLandmarksType,
                load_landmarks,
            )
            lm_config = load_landmarks(GNMLandmarksType.HEAD_SPARSE_68)
            lm_indices = lm_config.indices.astype(np.int32)   # (68, K)
            lm_weights = lm_config.weights.astype(np.float64)  # (68, K)
        except Exception:
            fb = np.array(self._fallback_landmark_vertex_indices or [], dtype=np.int32)
            lm_indices = fb.reshape(-1, 1)
            lm_weights = np.ones((len(fb), 1), dtype=np.float64)

        # Template landmark positions in GNM space: (68, 3)
        template_lm = np.zeros((68, 3), dtype=np.float64)
        for i in range(68):
            for j in range(lm_indices.shape[1]):
                idx = lm_indices[i, j]
                w = lm_weights[i, j]
                if w > 0 and idx >= 0:
                    template_lm[i] += w * template_v[idx]

        # Build the landmark identity basis: (I, 68, 3)
        lm_id_basis = np.zeros((253, 68, 3), dtype=np.float64)
        for i in range(68):
            for j in range(lm_indices.shape[1]):
                idx = lm_indices[i, j]
                w = lm_weights[i, j]
                if w > 0 and idx >= 0:
                    lm_id_basis[:, i, :] += w * id_basis[:, idx, :]

        # --- Iterative fitting ---
        # The similarity transform (MP→GNM space) and identity params interact.
        # We alternate: estimate transform → fit identity → update → repeat.
        identity = np.zeros(253, dtype=np.float64)

        for iteration in range(3):
            # Current GNM landmarks with current identity
            current_lm = template_lm + np.einsum(
                "i,ivm->vm", identity, lm_id_basis
            )

            # Estimate similarity transform: MP → GNM space
            user_aligned, rot, scale = procrustes_align(
                user_68_mp, current_lm, use_scaling=True
            )

            # Apply transform to all user landmarks
            user_in_gnm = scale * (user_68_mp - user_68_mp.mean(axis=0)) @ rot
            user_in_gnm += current_lm.mean(axis=0)
            target = user_in_gnm.astype(np.float64)  # (68, 3)

            # Optimise identity
            def loss(id_vec: np.ndarray) -> float:
                predicted = template_lm + np.einsum("i,ivm->vm", id_vec, lm_id_basis)
                diff = predicted - target
                reg = 0.0005 * np.sum(id_vec ** 2)
                return float(np.sum(diff ** 2) + reg)

            def grad(id_vec: np.ndarray) -> np.ndarray:
                predicted = template_lm + np.einsum("i,ivm->vm", id_vec, lm_id_basis)
                diff = (predicted - target).ravel()
                jac = lm_id_basis.reshape(253, -1)
                return (2.0 * jac @ diff + 2.0 * 0.0005 * id_vec).astype(np.float64)

            result = minimize(
                loss,
                x0=identity,
                jac=grad,
                method="L-BFGS-B",
                options={"maxiter": 100, "disp": False},
            )
            identity = result.x.copy()
            print(f"  [Fit iter {iteration + 1}] loss={loss(identity):.4f}  "
                  f"scale={scale:.4f}")

        print(f"[Identity Fit] Done.  Final loss={loss(identity):.4f}")
        return identity.astype(np.float32)

    def _build_expression_regressor(self) -> None:
        """Pre-compute the least-squares regressor for expression estimation.

        Builds a matrix M: (E, 68*3) such that::

            expression = M @ residual_flat

        where residual_flat is the (aligned_mp_lm - template_lm - identity_lm)
        flattened to (68*3,).

        Uses regularised least squares:  min ||E@expr - residual||² + λ||expr||²
        """
        try:
            from gnm.shape.gnm_landmarks import (
                GNMLandmarksType,
                load_landmarks,
            )
            lm_config = load_landmarks(GNMLandmarksType.HEAD_SPARSE_68)
            lm_indices = lm_config.indices.astype(np.int32)
            lm_weights = lm_config.weights.astype(np.float64)
        except Exception:
            fb = np.array(self._fallback_landmark_vertex_indices or [], dtype=np.int32)
            lm_indices = fb.reshape(-1, 1)
            lm_weights = np.ones((len(fb), 1), dtype=np.float64)

        num_lm = lm_indices.shape[0]
        expr_basis = self.gnm.expression_basis.astype(np.float64)  # (E, V, 3)
        E_dim = self.gnm.expression_dim  # 383

        # Expression basis at each landmark: (E, num_lm, 3)
        lm_expr_basis = np.zeros((E_dim, num_lm, 3), dtype=np.float64)
        for i in range(num_lm):
            for j in range(lm_indices.shape[1]):
                idx = lm_indices[i, j]
                w = lm_weights[i, j]
                if w > 0 and idx >= 0:
                    lm_expr_basis[:, i, :] += w * expr_basis[:, idx, :]

        # Solve: expr = (J^T J + λI)^(-1) J^T @ residual
        jac = lm_expr_basis.reshape(E_dim, -1).T  # (num_lm*3, E)
        reg = 0.005
        lhs = jac.T @ jac + reg * np.eye(E_dim)
        self._expr_regressor = np.linalg.solve(lhs, jac.T).astype(np.float32)
        # Shape: (E, num_lm*3)
        print(f"[GNM] Expression regressor ready: {self._expr_regressor.shape}")

        # Cache landmark data
        self._lm_indices_cache = lm_indices
        self._lm_weights_cache = lm_weights
        self._num_cached_lm = num_lm

    # ------------------------------------------------------------------
    # Per-frame expression estimation
    # ------------------------------------------------------------------

    def estimate_expression(self, mp_landmarks_3d: np.ndarray) -> np.ndarray:
        """Estimate expression parameters from one frame of MediaPipe landmarks.

        Converts landmarks to GNM space using the 68-point similarity
        transform, subtracts the identity contribution, and solves for
        expression via the pre-computed regressor.

        Args:
            mp_landmarks_3d: (478, 3) MediaPipe 3D landmarks.

        Returns:
            (383,) expression parameters.
        """
        if (self.identity is None or self._expr_regressor is None
                or self._lm_indices_cache is None):
            return np.zeros(self.gnm.expression_dim, dtype=np.float32)

        # Gather user 68 landmarks in MP space
        user_68_mp = np.zeros((68, 3), dtype=np.float64)
        for ibug_idx, mp_idx in IBUG68_TO_MEDIAPIPE.items():
            if mp_idx < len(mp_landmarks_3d):
                user_68_mp[ibug_idx] = mp_landmarks_3d[mp_idx]

        # Compute identity-contributed landmark positions in GNM space
        template_v = self.gnm.template_vertex_positions.astype(np.float64)
        id_basis = self.gnm.vertex_identity_basis.astype(np.float64)
        identity_d = self.identity.astype(np.float64)
        num_lm = self._lm_indices_cache.shape[0]

        id_lm_gnm = np.zeros((num_lm, 3), dtype=np.float64)
        for i in range(num_lm):
            for j in range(self._lm_indices_cache.shape[1]):
                idx = self._lm_indices_cache[i, j]
                w = self._lm_weights_cache[i, j]
                if w > 0 and idx >= 0:
                    pos = template_v[idx] + np.dot(
                        identity_d, id_basis[:, idx, :]
                    )
                    id_lm_gnm[i] += w * pos

        # Align MP landmarks → GNM space
        user_aligned_gnm, _, _ = procrustes_align(
            user_68_mp, id_lm_gnm, use_scaling=True
        )

        # Residual = what the expression needs to explain
        residual = user_aligned_gnm - id_lm_gnm  # (num_lm, 3)

        # Solve via pre-computed regressor
        expr = self._expr_regressor @ residual.ravel().astype(np.float32)
        return expr.astype(np.float32)

    # ------------------------------------------------------------------
    # GNM forward pass
    # ------------------------------------------------------------------

    def generate_mesh(
        self,
        identity: np.ndarray | None = None,
        expression: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate the GNM head mesh for given parameters.

        Args:
            identity: (253,) identity params.  Uses fitted identity if None.
            expression: (383,) expression params.  Uses zero if None.

        Returns:
            (17821, 3) vertex positions.
        """
        if identity is None:
            identity = self.identity if self.identity is not None else np.zeros(
                self.gnm.identity_dim, dtype=np.float32
            )
        if expression is None:
            expression = (
                self.expression
                if self.expression is not None
                else np.zeros(self.gnm.expression_dim, dtype=np.float32)
            )

        return self.gnm(
            identity,
            expression,
            np.zeros((self.gnm.num_joints, 3), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full GNM face-tracking application."""
        # --- Load GNM ---
        if not self._load_gnm():
            return

        # --- Setup MediaPipe ---
        self._setup_mediapipe()

        # --- Identity Fit ---
        if not self.fit_identity_interactive():
            print("[WARNING] Using template (zero) identity.")
            self.identity = np.zeros(self.gnm.identity_dim, dtype=np.float32)
            self._build_expression_regressor()

        self.expression = np.zeros(self.gnm.expression_dim, dtype=np.float32)

        # --- Real-time loop ---
        print("\n" + "=" * 60)
        print("  Real-time Tracking")
        print("=" * 60)
        print("  'q' = quit  |  'r' = re-fit identity  "
              "|  'f' = toggle GNM view\n")

        renderer = SimpleMeshRenderer()
        t_last = time.time()
        need_refit = False

        with WebcamManager(width=IMAGE_WIDTH, height=IMAGE_HEIGHT) as cam:
            while True:
                success, frame = cam.read()
                if not success:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    face_lm = results.multi_face_landmarks[0]
                    lm_3d = np.array(
                        [[p.x, p.y, p.z] for p in face_lm.landmark],
                        dtype=np.float32,
                    )

                    # Estimate expression (in GNM space)
                    t0 = time.time()
                    self.expression = self.estimate_expression(lm_3d)
                    expr_time = (time.time() - t0) * 1000

                    # Generate GNM mesh
                    t0 = time.time()
                    vertices = self.generate_mesh()
                    mesh_time = (time.time() - t0) * 1000

                    # --- Left panel: webcam with MediaPipe overlay ---
                    display_left = frame.copy()
                    mp_drawing.draw_landmarks(
                        display_left,
                        face_lm,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 180, 0), thickness=1
                        ),
                    )

                    # --- Right panel: GNM mesh render ---
                    vn = self.gnm.compute_vertex_normals(vertices)
                    display_right = renderer.render(
                        vertices, self.gnm.triangles, vn,
                    )

                    # HUD
                    cv2.putText(display_right, "GNM Head",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2)
                    cv2.putText(display_left,
                                f"Expr: {expr_time:.0f}ms  Mesh: {mesh_time:.0f}ms",
                                (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 220, 0), 1)

                    if self._show_fullscreen_gnm:
                        display = display_right
                    else:
                        display = np.hstack([display_left, display_right])

                else:
                    display = frame.copy()
                    cv2.putText(display, "No face detected",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 2)

                # FPS counter
                dt = time.time() - t_last
                t_last = time.time()
                self._fps_window.append(dt)
                if len(self._fps_window) > 30:
                    self._fps_window.pop(0)
                self._fps = 1.0 / max(sum(self._fps_window) / len(self._fps_window), 1e-6)
                cv2.putText(display, f"FPS: {self._fps:.0f}",
                            (display.shape[1] - 110, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.imshow("GNM Face Tracker", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    need_refit = True
                    break
                elif key == ord("f"):
                    self._show_fullscreen_gnm = not self._show_fullscreen_gnm

        cv2.destroyAllWindows()

        # Handle re-fit request (restart the pipeline)
        if need_refit:
            self.run()  # recursive — clean restart
        else:
            print("[GNM Face Tracker] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the GNM Face Tracker application."""
    tracker = GNMFaceTracker()
    tracker.run()


if __name__ == "__main__":
    main()
