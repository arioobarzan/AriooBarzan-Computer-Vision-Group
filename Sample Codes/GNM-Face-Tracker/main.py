"""
GNM Face Tracker — Real-time face tracking with Google's GNM parametric head model.
====================================================================================

Uses MediaPipe Face Landmarker (Tasks API) for real-time landmark detection and
GNM for 3D face reconstruction.  The pipeline has two stages:

1. **Identity Fit** — Auto-captures a neutral face in the first 2 seconds,
   then optimises GNM's 253 identity parameters via L-BFGS-B.

2. **Real-time Tracking** — Each frame estimates the 383 expression parameters
   from live landmarks and renders the deformed GNM mesh.

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
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    FACE_MODEL_FILENAME,
    WebcamManager,
    clear_mediapipe_cache,
    get_model_path,
    suppress_gpu_warnings,
)

# ---------------------------------------------------------------------------
# Environment & model
# ---------------------------------------------------------------------------
clear_mediapipe_cache()
suppress_gpu_warnings()
MODEL_PATH = get_model_path(FACE_MODEL_FILENAME)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NUM_MP_LANDMARKS = 478

# ---------------------------------------------------------------------------
# MediaPipe → iBUG 68 mapping
# ---------------------------------------------------------------------------
# Stable landmarks for Procrustes alignment — rigid facial structure that
# doesn't deform with expression (jawline, nose bridge, outer eye corners).
_STABLE_IBUG = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,  # jawline
    27, 28, 29, 30,                                                # nose bridge
    36, 39, 42, 45,                                                # outer eye corners
]

IBUG68_TO_MEDIAPIPE: dict[int, int] = {
    0: 234, 1: 93, 2: 132, 3: 58, 4: 172, 5: 136, 6: 150,
    7: 176, 8: 148, 9: 152, 10: 377, 11: 400, 12: 378, 13: 379,
    14: 365, 15: 397, 16: 288,
    17: 70, 18: 63, 19: 105, 20: 66, 21: 107,
    22: 336, 23: 296, 24: 334, 25: 293, 26: 300,
    27: 168, 28: 6, 29: 197, 30: 195,
    31: 5, 32: 4, 33: 1, 34: 19, 35: 94,
    36: 33, 37: 246, 38: 161, 39: 160, 40: 159, 41: 158,
    42: 362, 43: 398, 44: 384, 45: 385, 46: 386, 47: 387,
    48: 61, 49: 185, 50: 40, 51: 39, 52: 37, 53: 0,
    54: 267, 55: 269, 56: 270, 57: 409, 58: 291, 59: 308,
    60: 415, 61: 310, 62: 311, 63: 312, 64: 13, 65: 82, 66: 81, 67: 80,
}


# ---------------------------------------------------------------------------
# Procrustes alignment (rigid only — no scale)
# ---------------------------------------------------------------------------

def procrustes_rigid(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align *source* to *target* via rotation + translation (no scale).

    Using scale would absorb expression changes — rigid alignment preserves
    the scale difference between MP space and GNM space, which is handled
    by the expression regressor.
    """
    src_c = source.mean(axis=0)
    tgt_c = target.mean(axis=0)
    src_centered = source - src_c
    tgt_centered = target - tgt_c

    h = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    aligned = (source - src_c) @ r + tgt_c
    return aligned, r


# ---------------------------------------------------------------------------
# Fast flat-shaded mesh renderer
# ---------------------------------------------------------------------------

class FastMeshRenderer:
    """Flat-shaded mesh renderer with batched fillPoly for speed.

    Renders the GNM mesh using per-face Lambertian shading, depth-sorted
    in batches of similar colour to minimise fillPoly calls (~8 instead of
    22K).  Flips the Y axis to account for the GNM coordinate system
    (Y-up) vs OpenCV image space (Y-down).
    """

    # Colour bins for batching
    _NUM_BINS = 8

    def __init__(self, image_size=(IMAGE_WIDTH, IMAGE_HEIGHT), fov_y=45.0):
        self.width, self.height = image_size
        fov_rad = np.radians(fov_y)
        fx = self.width / (2.0 * np.tan(fov_rad / 2.0))
        self.K = np.array(
            [[fx, 0, self.width / 2], [0, fx, self.height / 2], [0, 0, 1]],
            dtype=np.float32,
        )

    def render(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        rvec=None,
        tvec=None,
    ) -> np.ndarray:
        """Render the mesh with flat shading on a dark background.

        Args:
            vertices: (V, 3) vertex positions in GNM space (Y-up, Z-forward).
            triangles: (T, 3) triangle indices.
            rvec: (3,) axis-angle rotation vector.
            tvec: (3,) translation vector.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if rvec is None:
            rvec = np.zeros(3, dtype=np.float32)
        if tvec is None:
            # GNM face points +Z.  Camera at positive Z looking towards origin.
            # Negative tvec[2] brings camera in front of face (Z convention flip).
            tvec = np.array([0.0, 0.05, -1.8], dtype=np.float32)

        rotmat, _ = cv2.Rodrigues(rvec)
        cam = vertices @ rotmat.T + tvec.reshape(1, 3)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        # NEGATE Z — camera looks along -Z in GNM world (face at origin, camera in front)
        # NEGATE Y — GNM has Y-up, OpenCV image has Y-down
        z = -cam[:, 2]
        z_safe = np.where(np.abs(z) < 1e-6, np.copysign(1e-6, z), z)
        u = fx * cam[:, 0] / z_safe + cx
        v = fy * (-cam[:, 1]) / z_safe + cy

        # --- Face normals (GNM space) ---
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        f_norms = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(f_norms, axis=1, keepdims=True)
        f_norms = f_norms / np.maximum(norms, 1e-8)

        # Face normals in CAMERA space
        fn_cam = f_norms @ rotmat.T

        # Back-face culling mask
        front_facing = fn_cam[:, 2] < 0

        # --- Shading ---
        # Key light from upper-right-front, fill light from lower-left
        key_light = np.array([0.4, -0.3, 1.0])
        key_light = key_light / np.linalg.norm(key_light)
        fill_light = np.array([-0.3, 0.5, 0.7])
        fill_light = fill_light / np.linalg.norm(fill_light)

        key_lambert = np.clip(np.sum(fn_cam * key_light, axis=1), 0.0, 1.0)
        fill_lambert = np.clip(np.sum(fn_cam * fill_light, axis=1), 0.0, 1.0)

        ambient = 0.12
        shade = ambient + 0.65 * key_lambert + 0.23 * fill_lambert

        # Base skin tone BGR
        base = np.array([185, 145, 115], dtype=np.float32)
        colours = np.clip(base * shade[:, None], 0, 255).astype(np.uint8)
        # Convert each triangle's colour to a single integer key for binning
        grey = (colours[:, 0].astype(np.int32) * 3
                + colours[:, 1].astype(np.int32) * 7
                + colours[:, 2].astype(np.int32) * 5) // 15

        # --- Depth sorting ---
        tri_z = z[triangles].mean(axis=1)

        # --- Render ---
        img = np.full((self.height, self.width, 3), 40, dtype=np.uint8)

        # Bin triangles by depth and colour for batched fillPoly
        # Sort by depth first
        depth_order = np.argsort(-tri_z)  # far → near

        # Process in bins: partition depth_order into _NUM_BINS equal groups
        # by grey level within each depth slice
        bin_size = max(1, len(depth_order) // (self._NUM_BINS * 2))
        for batch_start in range(0, len(depth_order), bin_size):
            batch_end = min(batch_start + bin_size, len(depth_order))
            batch = depth_order[batch_start:batch_end]

            # Group by colour within this batch
            batch_grey = grey[batch]
            # Use all triangles in one fillPoly call per distinct grey bin
            unique_greys = np.unique(batch_grey)
            for g in unique_greys:
                mask = batch_grey == g
                idxs = batch[mask]
                # Filter for front-facing
                idxs = idxs[front_facing[idxs]]
                if len(idxs) == 0:
                    continue
                pts = np.stack([u[triangles[idxs]], v[triangles[idxs]]], axis=2)
                pts = pts.astype(np.int32)
                color = tuple(int(c) for c in colours[idxs[0]])
                cv2.fillPoly(img, pts, color)

        return img


# ---------------------------------------------------------------------------
# GNM Face Tracker
# ---------------------------------------------------------------------------

class GNMFaceTracker:
    """Real-time face tracking with GNM."""

    def __init__(self):
        self.gnm = None
        self.detector: Optional[FaceLandmarker] = None
        self.identity: Optional[np.ndarray] = None
        self.expression: Optional[np.ndarray] = None

        self._gnm68_template: Optional[np.ndarray] = None
        self._expr_regressor: Optional[np.ndarray] = None
        self._lm_indices_cache: Optional[np.ndarray] = None
        self._lm_weights_cache: Optional[np.ndarray] = None
        self._num_cached_lm: int = 0
        self._fallback_landmark_vertex_indices: np.ndarray | None = None
        self._skin_triangles: Optional[np.ndarray] = None

        self._show_fullscreen_gnm = False
        self._fps_window: list[float] = []
        self._fps = 0.0

        # Stabilise expression with temporal smoothing
        self._expr_smooth: Optional[np.ndarray] = None
        self._expr_alpha = 0.35  # smoothing factor (lower = faster response)
        self._expr_gain: float = 20.0  # expression amplification gain

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_gnm(self) -> bool:
        """Load GNM and the 68 sparse landmarks."""
        print("[GNM] Loading ...")
        try:
            from gnm.shape import gnm_numpy
            from gnm.shape.gnm_landmarks import (
                GNMLandmarksType, load_landmarks,
            )
            from gnm.shape.gnm_landmarks import GNMLandmarksDataNotLinkedError

            self.gnm = gnm_numpy.GNM.from_local(
                version=gnm_numpy.GNMMajorVersion.V3,
                variant=gnm_numpy.GNMVariant.HEAD,
            )
            print(f"[GNM] V={self.gnm.num_vertices}  "
                  f"I={self.gnm.identity_dim}  E={self.gnm.expression_dim}")

            # Pre-extract skin exterior triangles for rendering
            skin_idx = self.gnm.triangle_indices_for_group("skin_exterior")
            self._skin_triangles = self.gnm.triangles[skin_idx]
            print(f"[GNM] Skin triangles: {len(self._skin_triangles)}")

            try:
                lm_config = load_landmarks(GNMLandmarksType.HEAD_SPARSE_68)
                self._extract_68_landmark_positions(lm_config)
                print(f"[GNM] {len(self._gnm68_template)} landmarks loaded.")
            except GNMLandmarksDataNotLinkedError:
                self._use_fallback_landmarks()

            return True
        except ImportError as e:
            print(f"[GNM] ERROR: {e}")
            print("  git clone https://github.com/google/GNM.git")
            print("  cd GNM/gnm/shape && pip install -e .")
            return False

    def _extract_68_landmark_positions(self, lm_config) -> None:
        indices = lm_config.indices
        weights = lm_config.weights
        template = self.gnm.template_vertex_positions
        num_lm, K = indices.shape
        positions = np.zeros((num_lm, 3), dtype=np.float32)
        for i in range(num_lm):
            for j in range(K):
                idx = indices[i, j]
                w = weights[i, j]
                if w > 0 and idx >= 0:
                    positions[i] += w * template[idx]
        self._gnm68_template = positions

    def _use_fallback_landmarks(self) -> None:
        template = self.gnm.template_vertex_positions
        centroid = template.mean(axis=0)
        dists = np.linalg.norm(template - centroid, axis=1)
        idx = np.sort(np.argsort(-dists)[:68])
        self._gnm68_template = template[idx]
        self._fallback_landmark_vertex_indices = idx
        print(f"[GNM] Fallback: {len(idx)} extreme vertices.")

    def _setup_mediapipe(self):
        """Initialise MediaPipe Face Landmarker."""
        print("[MediaPipe] Initialising ...")
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.6,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Identity fitting (auto-capture)
    # ------------------------------------------------------------------

    def _auto_fit_identity(self) -> bool:
        """Auto-capture and fit identity from the first stable face.

        Waits up to 3 seconds for a face.  Once detected, waits for the
        landmarks to stabilise (low frame-to-frame variance), then captures
        and optimises identity.
        """
        print("\n" + "=" * 60)
        print("  Identity Fit — auto-capturing neutral face ...")
        print("=" * 60)

        with WebcamManager(width=IMAGE_WIDTH, height=IMAGE_HEIGHT) as cam:
            prev_lm = None
            stable_count = 0
            start_time = time.time()
            captured_lm = None

            while True:
                success, frame = cam.read()
                if not success:
                    continue

                display = frame.copy()
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = int(time.time() * 1000)
                result = self.detector.detect_for_video(mp_image, ts)

                elapsed = time.time() - start_time

                if result.face_landmarks:
                    face_lm = result.face_landmarks[0]
                    lm_3d = np.array(
                        [[p.x, p.y, p.z] for p in face_lm], dtype=np.float32
                    )

                    # Check stability
                    if prev_lm is not None:
                        diff = np.abs(lm_3d - prev_lm).mean()
                        if diff < 0.003:  # stable
                            stable_count += 1
                        else:
                            stable_count = max(0, stable_count - 1)

                    prev_lm = lm_3d.copy()

                    # Draw landmarks on display
                    for lm in face_lm:
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.circle(display, (px, py), 1, (0, 220, 0), -1)

                    # Status
                    if stable_count >= 8:
                        captured_lm = lm_3d
                        status = "CAPTURED!"
                        colour = (0, 255, 255)
                    elif stable_count > 0:
                        status = f"Hold still... {stable_count}/8"
                        colour = (0, 255, 200)
                    else:
                        status = f"Face detected — hold neutral ({elapsed:.0f}s)"
                        colour = (0, 200, 0)
                else:
                    status = "Looking for face..."
                    colour = (0, 0, 255)
                    stable_count = 0

                cv2.putText(display, status, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
                cv2.putText(display, "q = skip | auto-capture when stable",
                            (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (150, 150, 150), 1)
                cv2.imshow("Identity Fit", display)

                key = cv2.waitKey(1) & 0xFF
                if captured_lm is not None or key == ord("q"):
                    break

                # Timeout after 5 seconds with no face
                if elapsed > 5 and stable_count == 0:
                    break

        cv2.destroyWindow("Identity Fit")

        if captured_lm is None:
            print("[Identity Fit] Skipped.")
            return False

        print("[Identity Fit] Optimising identity parameters ...")
        self.identity = self._optimise_identity(captured_lm)
        self._build_expression_regressor()
        return True

    # ------------------------------------------------------------------
    # Identity optimisation
    # ------------------------------------------------------------------

    def _optimise_identity(self, mp_landmarks_3d: np.ndarray) -> np.ndarray:
        """Optimise 253 identity parameters via iterative L-BFGS-B."""
        template_68 = self._gnm68_template.astype(np.float64)

        # Gather user 68 landmarks in MP space
        user_68_mp = np.zeros((68, 3), dtype=np.float64)
        for ibug_idx, mp_idx in IBUG68_TO_MEDIAPIPE.items():
            if mp_idx < len(mp_landmarks_3d):
                user_68_mp[ibug_idx] = mp_landmarks_3d[mp_idx]

        id_basis = self.gnm.vertex_identity_basis.astype(np.float64)
        template_v = self.gnm.template_vertex_positions.astype(np.float64)

        try:
            from gnm.shape.gnm_landmarks import (
                GNMLandmarksType, load_landmarks,
            )
            lm_config = load_landmarks(GNMLandmarksType.HEAD_SPARSE_68)
            lm_indices = lm_config.indices.astype(np.int32)
            lm_weights = lm_config.weights.astype(np.float64)
        except Exception:
            fb = np.array(
                self._fallback_landmark_vertex_indices or [], dtype=np.int32
            )
            lm_indices = fb.reshape(-1, 1)
            lm_weights = np.ones((len(fb), 1), dtype=np.float64)

        num_lm = lm_indices.shape[0]

        # Template landmarks in GNM space
        template_lm = np.zeros((num_lm, 3), dtype=np.float64)
        for i in range(num_lm):
            for j in range(lm_indices.shape[1]):
                idx = lm_indices[i, j]
                w = lm_weights[i, j]
                if w > 0 and idx >= 0:
                    template_lm[i] += w * template_v[idx]

        # Landmark identity basis: (253, num_lm, 3)
        lm_id_basis = np.zeros((253, num_lm, 3), dtype=np.float64)
        for i in range(num_lm):
            for j in range(lm_indices.shape[1]):
                idx = lm_indices[i, j]
                w = lm_weights[i, j]
                if w > 0 and idx >= 0:
                    lm_id_basis[:, i, :] += w * id_basis[:, idx, :]

        identity = np.zeros(253, dtype=np.float64)

        for iteration in range(3):
            current_lm = template_lm + np.einsum(
                "i,ivm->vm", identity, lm_id_basis
            )
            # Use similarity (with scale) for identity fit — this is correct
            # because we WANT to absorb the overall MP→GNM scale difference
            # into the transform, not into identity params
            user_aligned, rot, scale = _procrustes_similarity(
                user_68_mp, current_lm
            )
            user_in_gnm = scale * (user_68_mp - user_68_mp.mean(axis=0)) @ rot
            user_in_gnm += current_lm.mean(axis=0)
            target = user_in_gnm.astype(np.float64)

            def loss(v):
                p = template_lm + np.einsum("i,ivm->vm", v, lm_id_basis)
                return float(np.sum((p - target) ** 2) + 0.0005 * np.sum(v ** 2))

            def grad(v):
                p = template_lm + np.einsum("i,ivm->vm", v, lm_id_basis)
                d = (p - target).ravel()
                j = lm_id_basis.reshape(253, -1)
                return (2 * j @ d + 2 * 0.0005 * v).astype(np.float64)

            result = minimize(
                loss, x0=identity, jac=grad, method="L-BFGS-B",
                options={"maxiter": 80},
            )
            identity = result.x.copy()
            print(f"  Iter {iteration + 1}: loss={loss(identity):.4f}")

        print(f"[Identity Fit] Done.  Final loss={loss(identity):.4f}")
        return identity.astype(np.float32)

    # ------------------------------------------------------------------
    # Expression regressor
    # ------------------------------------------------------------------

    def _build_expression_regressor(self) -> None:
        """Pre-compute expression regressor matrix."""
        try:
            from gnm.shape.gnm_landmarks import (
                GNMLandmarksType, load_landmarks,
            )
            lm_config = load_landmarks(GNMLandmarksType.HEAD_SPARSE_68)
            lm_indices = lm_config.indices.astype(np.int32)
            lm_weights = lm_config.weights.astype(np.float64)
        except Exception:
            fb = np.array(
                self._fallback_landmark_vertex_indices or [], dtype=np.int32
            )
            lm_indices = fb.reshape(-1, 1)
            lm_weights = np.ones((len(fb), 1), dtype=np.float64)

        num_lm = lm_indices.shape[0]
        expr_basis = self.gnm.expression_basis.astype(np.float64)
        E_dim = self.gnm.expression_dim

        lm_expr_basis = np.zeros((E_dim, num_lm, 3), dtype=np.float64)
        for i in range(num_lm):
            for j in range(lm_indices.shape[1]):
                idx = lm_indices[i, j]
                w = lm_weights[i, j]
                if w > 0 and idx >= 0:
                    lm_expr_basis[:, i, :] += w * expr_basis[:, idx, :]

        jac = lm_expr_basis.reshape(E_dim, -1).T
        # Very weak regularisation — expression changes in GNM space are
        # tiny (sub-mm), so we need the regressor to be sensitive enough
        # to amplify them into visible expression parameters.
        reg = 0.0001
        lhs = jac.T @ jac + reg * np.eye(E_dim)
        self._expr_regressor = np.linalg.solve(lhs, jac.T).astype(np.float32)
        self._expr_gain = 20.0  # amplify expression for visible mesh deformation
        self._lm_indices_cache = lm_indices
        self._lm_weights_cache = lm_weights
        self._num_cached_lm = num_lm
        print(f"[GNM] Regressor: {self._expr_regressor.shape}")

    # ------------------------------------------------------------------
    # Per-frame expression estimation
    # ------------------------------------------------------------------

    def estimate_expression(self, mp_landmarks_3d: np.ndarray) -> np.ndarray:
        """Estimate expression from MP landmarks.

        Uses STABLE landmarks (jawline, nose bridge, outer eye corners)
        for Procrustes alignment — this prevents expression-driven
        landmark movement from being absorbed by the alignment transform.
        The full 68-landmark residual then drives expression parameters.
        """
        if (self.identity is None or self._expr_regressor is None
                or self._lm_indices_cache is None):
            return np.zeros(self.gnm.expression_dim, dtype=np.float32)

        # Gather user 68 landmarks in MP space
        user_68_mp = np.zeros((68, 3), dtype=np.float64)
        for ibug_idx, mp_idx in IBUG68_TO_MEDIAPIPE.items():
            if mp_idx < len(mp_landmarks_3d):
                user_68_mp[ibug_idx] = mp_landmarks_3d[mp_idx]

        # Identity landmarks in GNM space (all 68)
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

        # --- Align using ONLY stable landmarks ---
        # Stable = rigid facial structure (jaw, nose bridge, outer eye corners).
        # This prevents mouth/eyebrow expression from being absorbed by the
        # alignment transform.
        stable_user = user_68_mp[_STABLE_IBUG]
        stable_gnm = id_lm_gnm[_STABLE_IBUG]
        _, rot = procrustes_rigid(stable_user, stable_gnm)

        # Apply the SAME rotation to ALL 68 user landmarks
        user_c = user_68_mp.mean(axis=0)
        gnm_c = stable_gnm.mean(axis=0)
        user_aligned = (user_68_mp - user_c) @ rot + gnm_c

        # Residual on all 68 landmarks
        residual = user_aligned - id_lm_gnm
        expr_raw = self._expr_regressor @ residual.ravel().astype(np.float32)

        # Amplify for visible deformation; clamp to GNM typical range
        expr_raw *= self._expr_gain
        expr_raw = np.clip(expr_raw, -3.0, 3.0)

        # Temporal smoothing
        if self._expr_smooth is None:
            self._expr_smooth = expr_raw.copy()
        else:
            self._expr_smooth = (
                self._expr_alpha * self._expr_smooth
                + (1 - self._expr_alpha) * expr_raw
            )

        return self._expr_smooth.copy()

    # ------------------------------------------------------------------
    # GNM forward pass
    # ------------------------------------------------------------------

    def generate_mesh(
        self,
        identity: np.ndarray | None = None,
        expression: np.ndarray | None = None,
    ) -> np.ndarray:
        if identity is None:
            identity = (
                self.identity
                if self.identity is not None
                else np.zeros(self.gnm.identity_dim, dtype=np.float32)
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
    # Landmark helpers
    # ------------------------------------------------------------------

    def _compute_gnm_landmark_positions(
        self, vertices: np.ndarray
    ) -> np.ndarray:
        """Compute the 3D position of each GNM 68 landmark on the deformed mesh.

        Uses the cached landmark indices and barycentric weights.
        """
        num_lm = self._lm_indices_cache.shape[0]
        positions = np.zeros((num_lm, 3), dtype=np.float32)
        for i in range(num_lm):
            for j in range(self._lm_indices_cache.shape[1]):
                idx = self._lm_indices_cache[i, j]
                w = self._lm_weights_cache[i, j]
                if w > 0 and idx >= 0:
                    positions[i] += w * vertices[idx]
        return positions

    @staticmethod
    def _draw_gnm_landmarks(
        img: np.ndarray,
        lm_3d: np.ndarray,
        rvec: np.ndarray,
        renderer,
    ) -> None:
        """Draw the 68 GNM landmarks as coloured dots on the rendered image.

        Colour key:
          - Green: jawline (0-16)
          - Yellow: eyebrows (17-26)
          - Cyan: nose (27-35)
          - Magenta: eyes (36-47)
          - Orange: mouth (48-67)
        """
        rotmat, _ = cv2.Rodrigues(rvec)
        tvec = np.array([0.0, 0.05, -1.8], dtype=np.float32)
        cam = lm_3d @ rotmat.T + tvec.reshape(1, 3)
        fx, fy = renderer.K[0, 0], renderer.K[1, 1]
        cx, cy = renderer.K[0, 2], renderer.K[1, 2]
        z = -cam[:, 2]
        z_safe = np.where(np.abs(z) < 1e-6, np.copysign(1e-6, z), z)
        u = fx * cam[:, 0] / z_safe + cx
        v = fy * (-cam[:, 1]) / z_safe + cy

        # Region colours (BGR)
        regions = [
            (0, 16, (0, 255, 80)),     # jawline: green
            (17, 26, (0, 230, 230)),   # eyebrows: yellow
            (27, 35, (230, 230, 0)),   # nose: cyan
            (36, 47, (230, 0, 230)),   # eyes: magenta
            (48, 67, (0, 140, 255)),   # mouth: orange
        ]
        for start, end, colour in regions:
            pts = np.stack([u[start:end+1], v[start:end+1]], axis=1)
            pts = pts.astype(np.int32)
            for pt in pts:
                if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                    cv2.circle(img, tuple(pt), 2, colour, -1)
            # Draw connecting lines
            if len(pts) > 1:
                cv2.polylines(img, [pts], True, colour, 1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full face-tracking application."""
        if not self._load_gnm():
            return

        self._setup_mediapipe()

        # Identity fit (auto-capture)
        if not self._auto_fit_identity():
            print("[WARNING] Using template identity.")
            self.identity = np.zeros(self.gnm.identity_dim, dtype=np.float32)
            self._build_expression_regressor()

        self.expression = np.zeros(self.gnm.expression_dim, dtype=np.float32)
        self._expr_smooth = None

        print("\n" + "=" * 60)
        print("  Real-time Tracking")
        print("=" * 60)
        print("  'q' = quit  |  'r' = re-fit  |  'f' = GNM fullscreen\n")

        renderer = FastMeshRenderer()
        t_last = time.time()
        need_refit = False

        with WebcamManager(width=IMAGE_WIDTH, height=IMAGE_HEIGHT) as cam:
            while True:
                success, frame = cam.read()
                if not success:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = int(time.time() * 1000)
                result = self.detector.detect_for_video(mp_image, ts)

                if result.face_landmarks:
                    face_lm = result.face_landmarks[0]
                    lm_3d = np.array(
                        [[p.x, p.y, p.z] for p in face_lm], dtype=np.float32
                    )

                    # Expression estimation
                    t0 = time.time()
                    self.expression = self.estimate_expression(lm_3d)
                    expr_time = (time.time() - t0) * 1000

                    # GNM forward pass
                    t0 = time.time()
                    vertices = self.generate_mesh()
                    mesh_time = (time.time() - t0) * 1000

                    # Left: webcam + landmark dots
                    display_left = frame.copy()
                    h, w = frame.shape[:2]
                    for lm in face_lm:
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.circle(display_left, (px, py), 1, (0, 220, 0), -1)

                    # Right: GNM mesh with slight Y-rotation for 3D depth
                    angle = np.sin(time.time() * 0.3) * 0.35  # gentle sway
                    rvec_3d = np.array([0.0, angle, 0.0], dtype=np.float32)
                    display_right = renderer.render(
                        vertices, self._skin_triangles, rvec=rvec_3d,
                    )

                    # --- Draw GNM 68 landmarks on the rendered mesh ---
                    # Compute current landmark positions on the deformed GNM mesh
                    gnm_lm_3d = self._compute_gnm_landmark_positions(vertices)
                    self._draw_gnm_landmarks(
                        display_right, gnm_lm_3d, rvec_3d, renderer,
                    )

                    # Expression magnitude HUD — show as a bar
                    expr_mag = float(np.abs(self.expression).mean())
                    expr_max = float(np.abs(self.expression).max())
                    bar_w = int(np.clip(expr_mag * 200, 0, 200))
                    cv2.rectangle(display_right, (10, 30), (10 + bar_w, 42),
                                  (0, 200, 100), -1)
                    cv2.putText(display_right,
                                f"Expr: {expr_mag:.3f}  max:{expr_max:.2f}",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (200, 200, 200), 1)

                    # HUD
                    cv2.putText(display_left,
                                f"Expr: {expr_time:.0f}ms | Mesh: {mesh_time:.0f}ms",
                                (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 220, 0), 1)

                    display = (
                        display_right if self._show_fullscreen_gnm
                        else np.hstack([display_left, display_right])
                    )
                else:
                    display = frame.copy()
                    cv2.putText(display, "No face", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                # FPS
                t_now = time.time()
                dt = t_now - t_last
                t_last = t_now
                self._fps_window.append(dt)
                if len(self._fps_window) > 30:
                    self._fps_window.pop(0)
                self._fps = 1.0 / max(
                    sum(self._fps_window) / len(self._fps_window), 1e-6
                )
                cv2.putText(display, f"FPS: {self._fps:.0f}",
                            (display.shape[1] - 100, 22),
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
        if self.detector:
            self.detector.close()

        if need_refit:
            self.run()
        else:
            print("[GNM Face Tracker] Done.")


# ---------------------------------------------------------------------------
# Helpers (module-level)
# ---------------------------------------------------------------------------

def _procrustes_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Similarity alignment (rotation + translation + uniform scale)."""
    src_c = source.mean(axis=0)
    tgt_c = target.mean(axis=0)
    src_cen = source - src_c
    tgt_cen = target - tgt_c
    h = src_cen.T @ tgt_cen
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    num = np.sum(tgt_cen * (src_cen @ r))
    den = np.sum(src_cen ** 2)
    scale = num / max(den, 1e-12)
    aligned = scale * (source - src_c) @ r + tgt_c
    return aligned, r, scale


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    tracker = GNMFaceTracker()
    tracker.run()


if __name__ == "__main__":
    main()
