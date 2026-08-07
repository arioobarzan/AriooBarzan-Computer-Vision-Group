"""
GNM Face Tracker -- Real-time 3D avatar synced to your face.
============================================================

Uses MediaPipe Face Landmarker for head-pose and landmark detection,
and Google's GNM parametric head model for a clean 3D avatar that
mirrors your head rotation, mouth, and facial expressions in real time.

Stages:
1. **Identity Fit** -- auto-captures a neutral frame, optimises GNM identity.
2. **Real-time Tracking** -- head pose + expression → animated 3D avatar.
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

clear_mediapipe_cache()
suppress_gpu_warnings()
MODEL_PATH = get_model_path(FACE_MODEL_FILENAME)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
RENDER_SCALE = 2  # render at 2× resolution for anti-aliasing

# Stable iBUG landmarks for Procrustes (rigid face structure)
_STABLE_IBUG = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    27, 28, 29, 30,
    36, 39, 42, 45,
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
# Procrustes
# ---------------------------------------------------------------------------

def procrustes_rigid(src: np.ndarray, tgt: np.ndarray):
    """Rigid alignment (rotation + translation)."""
    sc, tc = src.mean(axis=0), tgt.mean(axis=0)
    h = (src - sc).T @ (tgt - tc)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    return (src - sc) @ r + tc, r

def _procrustes_similarity(src, tgt):
    """Similarity alignment (rotation + scale + translation)."""
    sc, tc = src.mean(axis=0), tgt.mean(axis=0)
    scn, tcn = src - sc, tgt - tc
    h = scn.T @ tcn
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    s = np.sum(tcn * (scn @ r)) / max(np.sum(scn ** 2), 1e-12)
    return s * (src - sc) @ r + tc, r, s

# ---------------------------------------------------------------------------
# Clean 3D Avatar Renderer (MSAA + Gouraud shading)
# ---------------------------------------------------------------------------

class AvatarRenderer:
    """High-quality 3D avatar renderer.

    - Renders at 2× resolution then downsamples (anti-aliasing).
    - Uses Gouraud (smooth) shading with per-vertex normals.
    - Draws on a subtle gradient background.
    """

    def __init__(self, image_size=(IMAGE_WIDTH, IMAGE_HEIGHT), fov_y=45.0):
        self.w, self.h = image_size
        self.rw, self.rh = self.w * RENDER_SCALE, self.h * RENDER_SCALE
        fov = np.radians(fov_y)
        fx = self.rw / (2.0 * np.tan(fov / 2.0))
        self.K = np.array(
            [[fx, 0, self.rw / 2], [0, fx, self.rh / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        self._bg = self._make_background()

    def _make_background(self) -> np.ndarray:
        """Subtle dark radial gradient background."""
        ys = np.linspace(0, 1, self.rh)
        xs = np.linspace(0, 1, self.rw)
        xv, yv = np.meshgrid(xs, ys)
        r = np.sqrt((xv - 0.5) ** 2 + (yv - 0.35) ** 2)
        bg = (25 + r * 50).astype(np.uint8)
        return np.dstack([bg, bg, bg])

    def render(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        vertex_normals: np.ndarray,
        rvec=None,
        tvec=None,
    ) -> np.ndarray:
        """Render avatar at display resolution.

        Args:
            vertices: (V, 3) GNM-space vertices.
            triangles: (T, 3) triangle indices.
            vertex_normals: (V, 3) per-vertex normals.
            rvec: (3,) axis-angle world→camera rotation.
            tvec: (3,) world→camera translation.

        Returns:
            (H, W, 3) uint8 BGR image at display resolution.
        """
        if rvec is None:
            rvec = np.zeros(3, dtype=np.float32)
        if tvec is None:
            tvec = np.array([0.0, 0.05, -1.8], dtype=np.float32)

        rotmat, _ = cv2.Rodrigues(rvec)
        cam = vertices @ rotmat.T + tvec.reshape(1, 3)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        z = -cam[:, 2]
        z_safe = np.where(np.abs(z) < 1e-6, np.copysign(1e-6, z), z)
        u = fx * cam[:, 0] / z_safe + cx
        v = fy * (-cam[:, 1]) / z_safe + cy  # Y-up → Y-down

        # --- Per-vertex Lambertian shading ---
        vn_cam = vertex_normals @ rotmat.T
        vn_cam = vn_cam / np.maximum(
            np.linalg.norm(vn_cam, axis=1, keepdims=True), 1e-8
        )

        key_light = np.array([0.4, -0.3, 1.0])
        key_light = key_light / np.linalg.norm(key_light)
        fill_light = np.array([-0.3, 0.5, 0.7])
        fill_light = fill_light / np.linalg.norm(fill_light)

        key = np.clip(np.dot(vn_cam, key_light), 0.0, 1.0)
        fill = np.clip(np.dot(vn_cam, fill_light), 0.0, 1.0)
        ambient = 0.10
        v_shade = ambient + 0.65 * key + 0.25 * fill  # per-vertex shade

        base = np.array([190, 148, 118], dtype=np.float32)  # skin BGR
        v_colours = np.clip(base * v_shade[:, None], 0, 255).astype(np.float32)

        # --- Depth sort ---
        tri_z = z[triangles].mean(axis=1)
        order = np.argsort(-tri_z)

        # Face normals for back-face culling
        fn_cam = np.cross(
            cam[triangles[:, 1]] - cam[triangles[:, 0]],
            cam[triangles[:, 2]] - cam[triangles[:, 0]],
        )

        # --- Rasterise at 2× resolution ---
        img = self._bg.copy()

        # Process in depth batches for painter's algorithm
        T = len(triangles)
        batch = 4000
        for start in range(0, T, batch):
            end = min(start + batch, T)
            batch_idx = order[start:end]

            # Back-face cull for this batch
            bf = batch_idx[fn_cam[batch_idx, 2] < 0]
            if len(bf) == 0:
                continue

            # Build per-triangle polygon list
            tris = triangles[bf]
            # Vertex colours at triangle corners → Gouraud shading
            # We draw each triangle with 3-colour interpolation by splitting
            # into smaller colour bands (simple approach: use mean colour)
            for ti in range(len(bf)):
                tri = tris[ti]
                # Bounding-box check
                uu = u[tri]
                vv = v[tri]
                if (uu.min() >= self.rw or uu.max() < 0
                        or vv.min() >= self.rh or vv.max() < 0):
                    continue

                pts = np.stack([uu, vv], axis=1).astype(np.int32)
                # Average vertex colour for this triangle
                # (true Gouraud would interpolate, but this is fast and looks OK)
                mean_c = v_colours[tri].mean(axis=0)
                colour = tuple(int(c) for c in np.clip(mean_c, 0, 255))
                cv2.fillPoly(img, [pts], colour)

        # Downsample to display resolution
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return img


# ---------------------------------------------------------------------------
# GNM Face Tracker
# ---------------------------------------------------------------------------

class GNMFaceTracker:
    """Real-time 3D face avatar."""

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

        # Expression state
        self._expr_smooth: Optional[np.ndarray] = None
        self._expr_alpha = 0.35
        self._expr_gain: float = 20.0

        # Head pose state
        self._head_rvec: np.ndarray = np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _load_gnm(self) -> bool:
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

            skin_idx = self.gnm.triangle_indices_for_group("skin_exterior")
            self._skin_triangles = self.gnm.triangles[skin_idx]
            print(f"[GNM] Skin: {len(self._skin_triangles)} tris")

            try:
                lm_config = load_landmarks(GNMLandmarksType.HEAD_SPARSE_68)
                self._extract_68_landmark_positions(lm_config)
                print(f"[GNM] {len(self._gnm68_template)} landmarks.")
            except GNMLandmarksDataNotLinkedError:
                self._use_fallback_landmarks()

            return True
        except ImportError as e:
            print(f"[GNM] ERROR: {e}")
            print("  git clone https://github.com/google/GNM.git")
            print("  cd GNM/gnm/shape && pip install -e .")
            return False

    def _extract_68_landmark_positions(self, lm_config) -> None:
        idx = lm_config.indices
        w = lm_config.weights
        tpl = self.gnm.template_vertex_positions
        n, K = idx.shape
        pos = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            for j in range(K):
                vi, wi = idx[i, j], w[i, j]
                if wi > 0 and vi >= 0:
                    pos[i] += wi * tpl[vi]
        self._gnm68_template = pos

    def _use_fallback_landmarks(self) -> None:
        tpl = self.gnm.template_vertex_positions
        c = tpl.mean(axis=0)
        d = np.linalg.norm(tpl - c, axis=1)
        ix = np.sort(np.argsort(-d)[:68])
        self._gnm68_template = tpl[ix]
        self._fallback_landmark_vertex_indices = ix
        print(f"[GNM] Fallback: {len(ix)} vertices.")

    def _setup_mediapipe(self):
        print("[MediaPipe] Initialising ...")
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.6,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,  # ← head pose!
        )
        self.detector = FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Identity fit
    # ------------------------------------------------------------------

    def _auto_fit_identity(self) -> bool:
        print("\n" + "=" * 60)
        print("  Identity Fit -- auto-capturing neutral face ...")
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
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = int(time.time() * 1000)
                result = self.detector.detect_for_video(mp_img, ts)
                elapsed = time.time() - start_time

                if result.face_landmarks:
                    face_lm = result.face_landmarks[0]
                    lm_3d = np.array(
                        [[p.x, p.y, p.z] for p in face_lm], dtype=np.float32
                    )
                    if prev_lm is not None:
                        diff = np.abs(lm_3d - prev_lm).mean()
                        stable_count = (
                            stable_count + 1 if diff < 0.003
                            else max(0, stable_count - 1)
                        )
                    prev_lm = lm_3d.copy()

                    for lm in face_lm:
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.circle(display, (px, py), 1, (0, 220, 0), -1)

                    if stable_count >= 8:
                        captured_lm = lm_3d
                        status = "CAPTURED!"
                        colour = (0, 255, 255)
                    elif stable_count > 0:
                        status = f"Hold still... {stable_count}/8"
                        colour = (0, 255, 200)
                    else:
                        status = f"Face detected ({elapsed:.0f}s)"
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
                if elapsed > 5 and stable_count == 0:
                    break

        cv2.destroyWindow("Identity Fit")

        if captured_lm is None:
            print("[Identity Fit] Skipped.")
            return False

        print("[Identity Fit] Optimising ...")
        self.identity = self._optimise_identity(captured_lm)
        self._build_expression_regressor()
        return True

    def _optimise_identity(self, mp_lm: np.ndarray) -> np.ndarray:
        t68 = self._gnm68_template.astype(np.float64)
        u68 = np.zeros((68, 3), dtype=np.float64)
        for ib, mp_i in IBUG68_TO_MEDIAPIPE.items():
            if mp_i < len(mp_lm):
                u68[ib] = mp_lm[mp_i]

        ib = self.gnm.vertex_identity_basis.astype(np.float64)
        tv = self.gnm.template_vertex_positions.astype(np.float64)

        try:
            from gnm.shape.gnm_landmarks import GNMLandmarksType, load_landmarks
            lc = load_landmarks(GNMLandmarksType.HEAD_SPARSE_68)
            li = lc.indices.astype(np.int32)
            lw = lc.weights.astype(np.float64)
        except Exception:
            fb = np.array(self._fallback_landmark_vertex_indices or [], dtype=np.int32)
            li = fb.reshape(-1, 1)
            lw = np.ones((len(fb), 1), dtype=np.float64)

        nl = li.shape[0]
        tl = np.zeros((nl, 3), dtype=np.float64)
        lbi = np.zeros((253, nl, 3), dtype=np.float64)
        for i in range(nl):
            for j in range(li.shape[1]):
                vi, wi = li[i, j], lw[i, j]
                if wi > 0 and vi >= 0:
                    tl[i] += wi * tv[vi]
                    lbi[:, i, :] += wi * ib[:, vi, :]

        identity = np.zeros(253, dtype=np.float64)
        for it in range(3):
            cl = tl + np.einsum("i,ivm->vm", identity, lbi)
            ua, r, s = _procrustes_similarity(u68, cl)
            u_gnm = s * (u68 - u68.mean(axis=0)) @ r + cl.mean(axis=0)
            tgt = u_gnm.astype(np.float64)

            def loss(v):
                p = tl + np.einsum("i,ivm->vm", v, lbi)
                return float(np.sum((p - tgt) ** 2) + 0.0005 * np.sum(v ** 2))

            def grad(v):
                p = tl + np.einsum("i,ivm->vm", v, lbi)
                d = (p - tgt).ravel()
                j = lbi.reshape(253, -1)
                return (2 * j @ d + 2 * 0.0005 * v).astype(np.float64)

            r = minimize(loss, x0=identity, jac=grad, method="L-BFGS-B",
                         options={"maxiter": 80})
            identity = r.x.copy()
            print(f"  Iter {it + 1}: loss={loss(identity):.4f}")

        print(f"[Identity Fit] Done. loss={loss(identity):.4f}")
        return identity.astype(np.float32)

    # ------------------------------------------------------------------
    # Expression regressor
    # ------------------------------------------------------------------

    def _build_expression_regressor(self):
        try:
            from gnm.shape.gnm_landmarks import GNMLandmarksType, load_landmarks
            lc = load_landmarks(GNMLandmarksType.HEAD_SPARSE_68)
            li = lc.indices.astype(np.int32)
            lw = lc.weights.astype(np.float64)
        except Exception:
            fb = np.array(self._fallback_landmark_vertex_indices or [], dtype=np.int32)
            li = fb.reshape(-1, 1)
            lw = np.ones((len(fb), 1), dtype=np.float64)

        nl = li.shape[0]
        eb = self.gnm.expression_basis.astype(np.float64)
        ED = self.gnm.expression_dim

        leb = np.zeros((ED, nl, 3), dtype=np.float64)
        for i in range(nl):
            for j in range(li.shape[1]):
                vi, wi = li[i, j], lw[i, j]
                if wi > 0 and vi >= 0:
                    leb[:, i, :] += wi * eb[:, vi, :]

        jac = leb.reshape(ED, -1).T
        reg = 0.0001
        lhs = jac.T @ jac + reg * np.eye(ED)
        self._expr_regressor = np.linalg.solve(lhs, jac.T).astype(np.float32)
        self._expr_gain = 20.0
        self._lm_indices_cache = li
        self._lm_weights_cache = lw
        self._num_cached_lm = nl
        print(f"[GNM] Regressor: {self._expr_regressor.shape}")

    # ------------------------------------------------------------------
    # Expression estimation
    # ------------------------------------------------------------------

    def estimate_expression(self, mp_lm: np.ndarray) -> np.ndarray:
        if (self.identity is None or self._expr_regressor is None
                or self._lm_indices_cache is None):
            return np.zeros(self.gnm.expression_dim, dtype=np.float32)

        u68 = np.zeros((68, 3), dtype=np.float64)
        for ib, mp_i in IBUG68_TO_MEDIAPIPE.items():
            if mp_i < len(mp_lm):
                u68[ib] = mp_lm[mp_i]

        tv = self.gnm.template_vertex_positions.astype(np.float64)
        ib = self.gnm.vertex_identity_basis.astype(np.float64)
        idv = self.identity.astype(np.float64)
        nl = self._lm_indices_cache.shape[0]

        i_gnm = np.zeros((nl, 3), dtype=np.float64)
        for i in range(nl):
            for j in range(self._lm_indices_cache.shape[1]):
                vi = self._lm_indices_cache[i, j]
                wi = self._lm_weights_cache[i, j]
                if wi > 0 and vi >= 0:
                    i_gnm[i] += wi * (tv[vi] + np.dot(idv, ib[:, vi, :]))

        # Align on stable landmarks only
        su = u68[_STABLE_IBUG]
        si = i_gnm[_STABLE_IBUG]
        _, rot = procrustes_rigid(su, si)
        ua = (u68 - u68.mean(axis=0)) @ rot + si.mean(axis=0)

        res = ua - i_gnm
        ex = self._expr_regressor @ res.ravel().astype(np.float32)
        ex = np.clip(ex * self._expr_gain, -3.0, 3.0)

        if self._expr_smooth is None:
            self._expr_smooth = ex.copy()
        else:
            self._expr_smooth = (
                self._expr_alpha * self._expr_smooth
                + (1 - self._expr_alpha) * ex
            )
        return self._expr_smooth.copy()

    # ------------------------------------------------------------------
    # Head pose extraction from MediaPipe transform matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_head_rotation(transform_matrix) -> np.ndarray:
        """Extract axis-angle rotation from MediaPipe face transform.

        MediaPipe gives a 4×4 column-major matrix.  We extract the 3×3
        rotation, convert to axis-angle, and remap to GNM coordinate system:
          MediaPipe: X=right, Y=down, Z=forward
          GNM:       X=right, Y=up,   Z=forward
        The Y-flip accounts for the Y-axis difference.
        """
        m = np.array(transform_matrix, dtype=np.float32).reshape(4, 4)
        r_mp = m[:3, :3].copy()

        # Flip MP Y (down) → GNM Y (up)
        flip_y = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
        r_gnm = flip_y @ r_mp @ flip_y

        rvec, _ = cv2.Rodrigues(r_gnm)
        rvec = rvec.ravel().astype(np.float32)
        # Pitch correction: MediaPipe Y-down → GNM Y-up inverts the pitch
        # direction.  Negate the X (pitch) component so tilting head up
        # makes the avatar look up, not down.
        rvec[0] *= -1.0
        return rvec

    # ------------------------------------------------------------------
    # GNM forward pass
    # ------------------------------------------------------------------

    def generate_mesh(self, identity=None, expression=None) -> np.ndarray:
        if identity is None:
            identity = (self.identity if self.identity is not None
                        else np.zeros(self.gnm.identity_dim, dtype=np.float32))
        if expression is None:
            expression = (self.expression if self.expression is not None
                          else np.zeros(self.gnm.expression_dim, dtype=np.float32))
        return self.gnm(identity, expression,
                        np.zeros((self.gnm.num_joints, 3), dtype=np.float32),
                        np.zeros(3, dtype=np.float32))

    # ------------------------------------------------------------------
    # Landmark overlay helpers
    # ------------------------------------------------------------------

    def _compute_gnm_lm_positions(self, verts: np.ndarray) -> np.ndarray:
        nl = self._lm_indices_cache.shape[0]
        pos = np.zeros((nl, 3), dtype=np.float32)
        for i in range(nl):
            for j in range(self._lm_indices_cache.shape[1]):
                vi = self._lm_indices_cache[i, j]
                wi = self._lm_weights_cache[i, j]
                if wi > 0 and vi >= 0:
                    pos[i] += wi * verts[vi]
        return pos

    @staticmethod
    def _draw_gnm_lm(img, lm3, rvec, renderer):
        rotmat, _ = cv2.Rodrigues(rvec)
        tv = np.array([0.0, 0.05, -1.8], dtype=np.float32)
        cam = lm3 @ rotmat.T + tv.reshape(1, 3)
        fx, fy = renderer.K[0, 0] / RENDER_SCALE, renderer.K[1, 1] / RENDER_SCALE
        cx, cy = renderer.K[0, 2] / RENDER_SCALE, renderer.K[1, 2] / RENDER_SCALE
        z = -cam[:, 2]
        zs = np.where(np.abs(z) < 1e-6, np.copysign(1e-6, z), z)
        u = fx * cam[:, 0] / zs + cx
        v = fy * (-cam[:, 1]) / zs + cy

        regions = [
            (0, 16, (0, 220, 80)), (17, 26, (0, 210, 210)),
            (27, 35, (210, 210, 0)), (36, 47, (210, 0, 210)),
            (48, 67, (0, 120, 240)),
        ]
        for s, e, c in regions:
            pts = np.stack([u[s:e+1], v[s:e+1]], axis=1).astype(np.int32)
            for pt in pts:
                if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                    cv2.circle(img, tuple(pt), 1, c, -1)
            if len(pts) > 1:
                cv2.polylines(img, [pts], True, c, 1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        if not self._load_gnm():
            return
        self._setup_mediapipe()

        if not self._auto_fit_identity():
            print("[WARNING] Using template identity.")
            self.identity = np.zeros(self.gnm.identity_dim, dtype=np.float32)
            self._build_expression_regressor()

        self.expression = np.zeros(self.gnm.expression_dim, dtype=np.float32)
        self._expr_smooth = None

        print("\n" + "=" * 60)
        print("  Real-time Tracking -- head pose + expression synced")
        print("=" * 60)
        print("  q=quit | r=re-fit | f=fullscreen\n")

        renderer = AvatarRenderer()
        t_last = time.time()
        need_refit = False

        with WebcamManager(width=IMAGE_WIDTH, height=IMAGE_HEIGHT) as cam:
            while True:
                success, frame = cam.read()
                if not success:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = int(time.time() * 1000)
                result = self.detector.detect_for_video(mp_img, ts)

                if result.face_landmarks:
                    face_lm = result.face_landmarks[0]
                    lm_3d = np.array(
                        [[p.x, p.y, p.z] for p in face_lm], dtype=np.float32
                    )

                    # --- Head pose ---
                    if result.facial_transformation_matrixes:
                        self._head_rvec = self._extract_head_rotation(
                            result.facial_transformation_matrixes[0]
                        )

                    # --- Expression ---
                    t0 = time.time()
                    self.expression = self.estimate_expression(lm_3d)
                    expr_ms = (time.time() - t0) * 1000

                    # --- GNM mesh ---
                    t0 = time.time()
                    verts = self.generate_mesh()
                    mesh_ms = (time.time() - t0) * 1000

                    # --- Left: webcam ---
                    left = frame.copy()
                    h, w = frame.shape[:2]
                    for lm in face_lm:
                        cv2.circle(left, (int(lm.x * w), int(lm.y * h)),
                                   1, (0, 220, 0), -1)

                    # --- Right: 3D avatar with real head pose ---
                    vn = self.gnm.compute_vertex_normals(verts)
                    right = renderer.render(
                        verts, self._skin_triangles, vn,
                        rvec=self._head_rvec,
                    )
                    # Landmark overlay
                    gnm_lm = self._compute_gnm_lm_positions(verts)
                    self._draw_gnm_lm(right, gnm_lm, self._head_rvec, renderer)

                    # Expression bar
                    em = float(np.abs(self.expression).mean())
                    bw = int(np.clip(em * 200, 0, 200))
                    cv2.rectangle(right, (10, 30), (10 + bw, 42),
                                  (0, 200, 100), -1)
                    cv2.putText(right, f"Expr: {em:.3f}",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (200, 200, 200), 1)

                    cv2.putText(left, f"E:{expr_ms:.0f}ms M:{mesh_ms:.0f}ms",
                                (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 220, 0), 1)

                    disp = right if self._show_fullscreen_gnm else np.hstack([left, right])
                else:
                    disp = frame.copy()
                    cv2.putText(disp, "No face", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                # FPS
                tn = time.time()
                dt = tn - t_last
                t_last = tn
                self._fps_window.append(dt)
                if len(self._fps_window) > 30:
                    self._fps_window.pop(0)
                self._fps = 1.0 / max(sum(self._fps_window) / len(self._fps_window), 1e-6)
                cv2.putText(disp, f"FPS:{self._fps:.0f}",
                            (disp.shape[1] - 90, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cv2.imshow("GNM Face Tracker", disp)

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


def main():
    GNMFaceTracker().run()


if __name__ == "__main__":
    main()
