from typing import Optional, Tuple

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_H
from qrcode.util import pattern_position
from PIL import Image, ImageFilter, ImageOps

DEFAULT_ACCENT = "#1a1a1a"
DEFAULT_BACKGROUND = "#ffffff"
FINDER_ZONE = 8  # finder pattern (7x7) plus its 1-module separator ring
MIN_VERSION = 14  # ~73x73 modules minimum: at low module counts, each module's
# forced color is a large, individually-obvious block; at this resolution the
# same forcing reads as fine grain instead of blotches, which is what actually
# keeps a real (detailed, busy-background) photo recognizable.

# Escalating (low_force, high_force) pairs. low_force controls how much a module
# that already agrees with the photo gets nudged toward pure black/white;
# high_force controls how hard a conflicting module gets corrected. We render
# at the most photo-preserving level first and only fall back to stronger
# (uglier but safer) levels if the result doesn't actually decode.
FORCE_LADDER = [
    (0.05, 0.45),
    (0.10, 0.60),
    (0.15, 0.75),
    (0.15, 0.85),
    (0.25, 0.90),
    (0.35, 0.95),
    (0.50, 1.0),
    (0.75, 1.0),
    (1.0, 1.0),  # failsafe: zero photo influence, functionally a plain QR
]

# Some (version, error-correction) encodings of otherwise-ordinary data turn
# out to be unreliable for certain scanners regardless of styling (observed:
# version 6 at ERROR_CORRECT_H). Bumping the version sidesteps that, so it's
# tried as an outer fallback layer alongside the force ladder.
VERSION_ESCALATION_OFFSETS = [0, 2, 4, 6, 10]


def _hex_to_rgb(value: Optional[str], fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    value = (value or "").strip().lstrip("#")
    if len(value) != 6:
        return fallback
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback


def _median_luminance(gray_img: Image.Image) -> float:
    hist = gray_img.histogram()
    total = sum(hist)
    cum = 0
    for level, count in enumerate(hist):
        cum += count
        if cum >= total / 2:
            return float(level)
    return 127.0


def _alignment_centers(version: int, n: int):
    positions = pattern_position(version)
    first, last = positions[0], positions[-1]
    corners = {(first, first), (first, last), (last, first)}
    centers = []
    for r in positions:
        for c in positions:
            if (r, c) in corners:
                continue
            centers.append((r, c))
    return centers


def _crisp_mask(n: int, version: int, alignment_centers) -> np.ndarray:
    """Finder, timing, alignment, and version-info patterns must stay
    undistorted — these are the structural landmarks (and metadata) a
    scanner needs to locate and correctly sample the grid, and they aren't
    protected by error correction the way ordinary data modules are."""
    mask = np.zeros((n, n), dtype=bool)
    mask[:FINDER_ZONE, :FINDER_ZONE] = True
    mask[:FINDER_ZONE, n - FINDER_ZONE :] = True
    mask[n - FINDER_ZONE :, :FINDER_ZONE] = True
    mask[6, :] = True
    mask[:, 6] = True

    if version >= 7:
        # Two redundant version-info blocks near the top-right and bottom-left
        # finder patterns, just outside the finder+separator zone. They're
        # only BCH-protected for up to 3 bit errors.
        lo, hi = n - 11, n - 9
        mask[:6, lo : hi + 1] = True
        mask[lo : hi + 1, :6] = True

    for r, c in alignment_centers:
        r0, r1 = max(0, r - 2), min(n, r + 3)
        c0, c1 = max(0, c - 2), min(n, c + 3)
        mask[r0:r1, c0:c1] = True

    return mask


def _build_qr_matrix(data: str, box_size: int, border: int, min_version: int):
    """Make a QRCode, boosted to at least min_version for more resolution."""
    probe = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_H, box_size=box_size, border=border)
    probe.add_data(data)
    probe.make(fit=True)
    version = max(probe.version, min_version)

    qr = qrcode.QRCode(version=version, error_correction=ERROR_CORRECT_H, box_size=box_size, border=border)
    qr.add_data(data)
    qr.make(fit=False)
    return qr


def _prepare_photo_render(qr, photo: Image.Image, accent, background, box_size: int, border: int) -> dict:
    """Do the expensive, force-independent work once per (qr version, photo,
    color) combination: blur/resize the photo down to module resolution and
    precompute per-module statistics. The force ladder then just re-blends
    these precomputed arrays, which is cheap enough to try every rung."""
    n = qr.modules_count
    alignment_centers = _alignment_centers(qr.version, n)
    data_size = n * box_size
    full_size = data_size + 2 * border * box_size
    offset = border * box_size

    grayscale = photo.convert("L")
    # A real photo carries far more fine detail (skin texture, background
    # clutter, fabric patterns) than a ~70x70-module grid can represent.
    # Without smoothing it first, that detail doesn't shrink gracefully — it
    # aliases into per-module noise that looks nothing like the source. Blur
    # radius is sized against module count (n), not final canvas pixels,
    # since the effective information resolution is one value per module.
    downsample_ratio = min(grayscale.size) / n
    blur_radius = max(2.0, downsample_ratio * 0.6)
    grayscale = grayscale.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    grayscale = ImageOps.autocontrast(grayscale, cutoff=1)
    fitted = ImageOps.fit(grayscale, (data_size, data_size), Image.LANCZOS)
    duotone = ImageOps.colorize(fitted, black=accent, white=background).convert("RGB")

    fitted_arr = np.asarray(fitted, dtype=np.float64)
    duotone_arr = np.asarray(duotone, dtype=np.float64)

    local_lum_grid = fitted_arr.reshape(n, box_size, n, box_size).mean(axis=(1, 3))
    avg_color_grid = duotone_arr.reshape(n, box_size, n, box_size, 3).mean(axis=(1, 3))

    modules_arr = np.array(qr.modules, dtype=bool)
    crisp_mask = _crisp_mask(n, qr.version, alignment_centers)

    threshold = _median_luminance(fitted)
    light_side_range = max(255.0 - threshold, 1.0)
    dark_side_range = max(threshold, 1.0)

    accent_arr = np.array(accent, dtype=np.float64)
    background_arr = np.array(background, dtype=np.float64)
    target_grid = np.where(modules_arr[..., None], accent_arr, background_arr)

    # How much does each module conflict with what it needs to be? 0 = the
    # photo already agrees with the required polarity, 1 = fully opposed.
    # Each direction is normalized by its own actual tonal range around the
    # threshold, so a narrow-range (e.g. mostly-dark) photo still produces
    # full-strength conflict signal instead of being swamped by the other side.
    conflict = np.where(
        modules_arr,
        np.clip((local_lum_grid - threshold) / light_side_range, 0.0, 1.0),
        np.clip((threshold - local_lum_grid) / dark_side_range, 0.0, 1.0),
    )

    return {
        "box_size": box_size,
        "data_size": data_size,
        "full_size": full_size,
        "offset": offset,
        "avg_color_grid": avg_color_grid,
        "target_grid": target_grid,
        "conflict": conflict,
        "crisp_mask": crisp_mask,
        "background": background_arr,
    }


def _apply_force(prepared: dict, low_force: float, high_force: float) -> Image.Image:
    force = low_force + (high_force - low_force) * prepared["conflict"]
    color_grid = prepared["avg_color_grid"] + (prepared["target_grid"] - prepared["avg_color_grid"]) * force[..., None]
    color_grid = np.where(prepared["crisp_mask"][..., None], prepared["target_grid"], color_grid)
    color_grid = np.clip(color_grid, 0, 255)

    box_size = prepared["box_size"]
    upscaled = np.repeat(np.repeat(color_grid, box_size, axis=0), box_size, axis=1)

    full_size = prepared["full_size"]
    offset = prepared["offset"]
    data_size = prepared["data_size"]
    canvas = np.empty((full_size, full_size, 3), dtype=np.float64)
    canvas[:, :] = prepared["background"]
    canvas[offset : offset + data_size, offset : offset + data_size] = upscaled

    return Image.fromarray(canvas.astype(np.uint8), "RGB")


_QR_DETECTOR = cv2.QRCodeDetector()


def _decodes_to(img: Image.Image, expected: str) -> bool:
    """Server-side scannability check: render to an array and try to decode
    it exactly like a real scanner would, so we never ship a QR code that
    doesn't actually work."""
    arr = np.array(img.convert("RGB"))[:, :, ::-1].copy()
    try:
        decoded, _points, _ = _QR_DETECTOR.detectAndDecode(arr)
    except cv2.error:
        return False
    return decoded == expected


def build_qr_image(
    data: str,
    photo: Optional[Image.Image] = None,
    accent_color: Optional[str] = None,
    background_color: Optional[str] = None,
    box_size: int = 10,
    border: int = 4,
    min_version: int = MIN_VERSION,
) -> Image.Image:
    """Build a scannable QR code.

    Without a photo: a plain QR in the given (or default) colors.

    With a photo: the data modules are rendered as a duotone version of the
    photo. Contrast is added adaptively — a module whose underlying photo
    pixel already leans the right way (dark module over a naturally dark
    area, or light module over a naturally light area) is barely touched, so
    the photo stays recognizable; a module that conflicts with its pixel
    gets pushed toward the required color so the code still scans. Finder,
    timing, alignment, and version-info patterns (the structural landmarks
    and metadata a scanner needs) are left completely crisp.

    Every candidate render is verified by actually decoding it before it's
    returned. If the most photo-preserving version doesn't decode cleanly,
    we automatically re-render with progressively stronger correction until
    one does — so a generated code is never handed back broken.
    """
    accent = _hex_to_rgb(accent_color, _hex_to_rgb(DEFAULT_ACCENT, (26, 26, 26)))
    background = _hex_to_rgb(background_color, _hex_to_rgb(DEFAULT_BACKGROUND, (255, 255, 255)))

    best = None
    for offset in VERSION_ESCALATION_OFFSETS:
        qr = _build_qr_matrix(data, box_size, border, min_version + offset)

        if photo is None:
            candidate = qr.make_image(fill_color=accent, back_color=background).convert("RGB")
            best = candidate
            if _decodes_to(candidate, data):
                return candidate
            continue

        prepared = _prepare_photo_render(qr, photo, accent, background, box_size, border)
        for low_force, high_force in FORCE_LADDER:
            candidate = _apply_force(prepared, low_force, high_force)
            best = candidate
            if _decodes_to(candidate, data):
                return candidate

    return best
