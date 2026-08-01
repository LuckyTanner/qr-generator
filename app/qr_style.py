from typing import Optional, Tuple

import qrcode
from qrcode.constants import ERROR_CORRECT_H
from PIL import Image, ImageOps, ImageStat

DEFAULT_ACCENT = "#1a1a1a"
DEFAULT_BACKGROUND = "#ffffff"
FINDER_ZONE = 8  # finder pattern (7x7) plus its 1-module separator ring


def _hex_to_rgb(value: Optional[str], fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    value = (value or "").strip().lstrip("#")
    if len(value) != 6:
        return fallback
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback


def _lerp(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _in_finder_zone(row: int, col: int, n: int) -> bool:
    if row < FINDER_ZONE and col < FINDER_ZONE:
        return True
    if row < FINDER_ZONE and col >= n - FINDER_ZONE:
        return True
    if row >= n - FINDER_ZONE and col < FINDER_ZONE:
        return True
    return False


def build_qr_image(
    data: str,
    photo: Optional[Image.Image] = None,
    accent_color: Optional[str] = None,
    background_color: Optional[str] = None,
    box_size: int = 10,
    border: int = 4,
    photo_blend: float = 0.5,
) -> Image.Image:
    """Build a scannable QR code.

    Without a photo: a plain QR in the given (or default) colors.
    With a photo: the data modules are rendered as a duotone version of the
    photo, each pixel biased toward the module's dark/light color so the
    code keeps enough contrast to scan. Finder patterns (the three corner
    squares that scanners lock onto first) are left crisp and undistorted.
    """
    accent = _hex_to_rgb(accent_color, _hex_to_rgb(DEFAULT_ACCENT, (26, 26, 26)))
    background = _hex_to_rgb(background_color, _hex_to_rgb(DEFAULT_BACKGROUND, (255, 255, 255)))

    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_H,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)
    modules = qr.modules
    n = qr.modules_count

    if photo is None:
        img = qr.make_image(fill_color=accent, back_color=background)
        return img.convert("RGB")

    data_size = n * box_size
    full_size = data_size + 2 * border * box_size

    grayscale = ImageOps.autocontrast(photo.convert("L"))
    fitted = ImageOps.fit(grayscale, (data_size, data_size), Image.LANCZOS)
    duotone = ImageOps.colorize(fitted, black=accent, white=background).convert("RGB")

    canvas = Image.new("RGB", (full_size, full_size), background)
    offset = border * box_size
    canvas.paste(duotone, (offset, offset))

    for row in range(n):
        for col in range(n):
            dark = modules[row][col]
            target = accent if dark else background
            x0 = offset + col * box_size
            y0 = offset + row * box_size

            if _in_finder_zone(row, col, n):
                color = target
            else:
                region = duotone.crop(
                    (col * box_size, row * box_size, (col + 1) * box_size, (row + 1) * box_size)
                )
                avg = tuple(round(v) for v in ImageStat.Stat(region).mean)
                color = _lerp(avg, target, photo_blend)

            canvas.paste(
                Image.new("RGB", (box_size, box_size), color),
                (x0, y0, x0 + box_size, y0 + box_size),
            )

    return canvas
