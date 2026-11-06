"""
certificate.py
Renders a printable "Coral Health Assessment Certificate" (A4 landscape)
for one analysed photograph, as PNG and PDF bytes.

Pure Pillow drawing - no network, no external assets - so it works offline
in a viva. Fonts fall back to Pillow's built-in scalable font if DejaVu is
not installed.
"""

from __future__ import annotations

import io
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 2480, 1754                       # A4 landscape at 300 dpi-ish
DPI = 212                               # -> exactly A4 when printed

INK = (18, 44, 66)
MUTED = (84, 108, 128)
GOLD = (196, 160, 82)
TEAL = (18, 118, 140)
CORAL = (255, 127, 102)
PAPER = (253, 251, 245)

_FONT_DIRS = ["/usr/share/fonts/truetype/dejavu/", "C:/Windows/Fonts/"]
_FONT_FILES = {"serif": ["DejaVuSerif.ttf", "georgia.ttf"],
               "serif-bold": ["DejaVuSerif-Bold.ttf", "georgiab.ttf"],
               "sans": ["DejaVuSans.ttf", "segoeui.ttf"],
               "sans-bold": ["DejaVuSans-Bold.ttf", "segoeuib.ttf"]}


def _font(kind: str, size: int):
    for d in _FONT_DIRS:
        for f in _FONT_FILES[kind]:
            try:
                return ImageFont.truetype(d + f, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def _center(draw, y, text, font, fill=INK, cx=W // 2):
    w = draw.textlength(text, font=font)
    draw.text((cx - w / 2, y), text, font=font, fill=fill)


# ----------------------------------------------------------------------
# BACKGROUND: ocean gradient, light rays, bubbles, reef silhouette
# ----------------------------------------------------------------------
def _ocean_background() -> Image.Image:
    y = np.linspace(0, 1, H)[:, None]
    top, mid, bot = np.array([14, 110, 150]), np.array([8, 62, 96]), np.array([3, 26, 44])
    col = np.where(y < .5, top + (mid - top) * (y / .5), mid + (bot - mid) * ((y - .5) / .5))
    img = np.repeat(col[:, None, :], W, axis=1).reshape(H, W, 3)
    bg = Image.fromarray(img.astype(np.uint8))

    rays = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(rays)
    rng = random.Random(7)
    for _ in range(9):
        x = rng.uniform(0, W); spread = rng.uniform(60, 180)
        d.polygon([(x - 30, 0), (x + 30, 0), (x + spread * 3, H), (x - spread, H)],
                  fill=rng.randint(18, 34))
    rays = rays.filter(ImageFilter.GaussianBlur(40))
    fade = Image.linear_gradient("L").resize((W, H)).point(lambda v: 255 - v)
    bg = Image.composite(Image.new("RGB", (W, H), (200, 240, 255)), bg,
                         Image.fromarray((np.asarray(rays) * (np.asarray(fade) / 255)).astype(np.uint8)))

    d = ImageDraw.Draw(bg, "RGBA")
    for _ in range(70):                                   # bubbles
        x, yy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(4, 16)
        d.ellipse((x - r, yy - r, x + r, yy + r), outline=(255, 255, 255, 90), width=2)
        d.ellipse((x - r * .4, yy - r * .5, x - r * .05, yy - r * .15), fill=(255, 255, 255, 110))
    for layer, (shade, base) in enumerate([((6, 40, 60, 255), H - 150), ((3, 24, 38, 255), H - 90)]):
        pts, x = [(0, H)], 0                               # reef silhouette
        while x <= W:
            h = rng.uniform(40, 160) * (1.3 if rng.random() < .25 else 1)
            pts += [(x, base), (x + rng.uniform(10, 30), base - h), (x + rng.uniform(30, 60), base)]
            x += rng.uniform(45, 110)
        pts.append((W, H))
        d.polygon(pts, fill=shade)
    return bg


def _paste_framed(canvas, bgr, box, caption, font):
    x0, y0, x1, y1 = box
    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    s = min((x1 - x0) / rgb.width, (y1 - y0 - 60) / rgb.height)   # up- or down-scale
    rgb = rgb.resize((max(1, int(rgb.width * s)), max(1, int(rgb.height * s))), Image.LANCZOS)
    px = x0 + (x1 - x0 - rgb.width) // 2
    d = ImageDraw.Draw(canvas)
    d.rectangle((px - 10, y0 - 10, px + rgb.width + 10, y0 + rgb.height + 10), fill="white",
                outline=(210, 205, 190), width=3)
    canvas.paste(rgb, (px, y0))
    _center(d, y0 + rgb.height + 24, caption, font, MUTED, cx=(x0 + x1) // 2)


def _seal(d, cx, cy, r):
    for i, (rr, w) in enumerate([(r, 6), (r - 16, 2), (r - 30, 2)]):
        d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=GOLD, width=w)
    for k in range(36):                                   # scalloped edge
        a = 2 * math.pi * k / 36
        d.ellipse((cx + (r + 8) * math.cos(a) - 7, cy + (r + 8) * math.sin(a) - 7,
                   cx + (r + 8) * math.cos(a) + 7, cy + (r + 8) * math.sin(a) + 7), fill=GOLD)
    _center(d, cy - 52, "AI", _font("serif-bold", 56), GOLD, cx=cx)
    _center(d, cy + 10, "ASSESSED", _font("sans-bold", 22), GOLD, cx=cx)
    _center(d, cy + 40, "MSc IT", _font("sans", 18), GOLD, cx=cx)


# ----------------------------------------------------------------------
def render(name: str, original_bgr, explained_bgr, result: dict) -> Image.Image:
    """result keys: label, confidence, time, assessment_id, model,
    model_accuracy (str), agreement (str|None), site (str|None),
    enhanced (bool), authors (str), image_name."""
    img = _ocean_background()
    d = ImageDraw.Draw(img, "RGBA")

    m = 90                                                # paper panel
    d.rounded_rectangle((m + 12, m + 16, W - m + 12, H - m - 90 + 16), 36, fill=(0, 0, 0, 90))
    d.rounded_rectangle((m, m, W - m, H - m - 90), 36, fill=PAPER)
    d.rounded_rectangle((m + 26, m + 26, W - m - 26, H - m - 116), 26, outline=GOLD, width=5)
    d.rounded_rectangle((m + 40, m + 40, W - m - 40, H - m - 130), 20, outline=TEAL, width=2)

    _center(d, 170, "CORAL REEF HEALTH MONITORING", _font("sans-bold", 34), TEAL)
    _center(d, 222, "Certificate of Assessment", _font("serif-bold", 92), INK)
    d.line((W // 2 - 360, 345, W // 2 + 360, 345), fill=GOLD, width=4)
    _center(d, 378, "This certifies that the underwater photograph submitted by",
            _font("serif", 38), MUTED)
    _center(d, 430, name, _font("serif-bold", 84), INK)
    _center(d, 540, f"was analysed on {result['time']}", _font("serif", 38), MUTED)

    cap = _font("sans", 26)
    _paste_framed(img, original_bgr, (220, 640, 900, 1250),
                  f"Submitted image · {result['image_name'][:30]}", cap)
    _paste_framed(img, explained_bgr, (950, 640, 1580, 1250),
                  "Grad-CAM · what drove the decision", cap)

    x0, y0, x1, y1 = 1650, 630, 2270, 1300               # result panel
    healthy = result["label"].lower() == "healthy"
    accent = (34, 160, 120) if healthy else (196, 150, 40)
    d.rounded_rectangle((x0, y0, x1, y1), 24, fill=(242, 247, 248), outline=accent, width=4)
    _center(d, y0 + 30, "ASSESSMENT RESULT", _font("sans-bold", 28), MUTED, cx=(x0 + x1) // 2)
    size = 96                                              # shrink verdict to fit
    while d.textlength(result["label"].upper(), font=_font("serif-bold", size)) > (x1 - x0) - 80:
        size -= 4
    _center(d, y0 + 75 + (96 - size) // 2, result["label"].upper(),
            _font("serif-bold", size), accent, cx=(x0 + x1) // 2)
    meaning = ("Normal coral pigmentation observed" if healthy
               else "Signs of bleaching (loss of pigmentation)")
    _center(d, y0 + 195, meaning, _font("sans", 26), MUTED, cx=(x0 + x1) // 2)

    bx0, bx1, by = x0 + 50, x1 - 50, y0 + 255             # confidence bar
    d.rounded_rectangle((bx0, by, bx1, by + 26), 13, fill=(220, 228, 232))
    d.rounded_rectangle((bx0, by, bx0 + (bx1 - bx0) * result["confidence"], by + 26), 13, fill=accent)
    _center(d, by + 38, f"Confidence {result['confidence']:.1%}", _font("sans-bold", 30), INK,
            cx=(x0 + x1) // 2)

    rows = [("Model", result["model"]), ("Measured accuracy", result["model_accuracy"])]
    if result.get("agreement"):
        rows.append(("Model agreement", result["agreement"]))
    rows.append(("Enhancement", "Applied" if result["enhanced"] else "Not applied"))
    if result.get("site"):
        rows.append(("Survey site", result["site"]))
    lab, val = _font("sans", 22), _font("sans-bold", 23)   # label above value
    yy = by + 82
    for k, v in rows:
        d.text((bx0, yy), k.upper(), font=lab, fill=MUTED)
        while d.textlength(v, font=val) > (bx1 - bx0) and len(v) > 4:
            v = v[:-2].rstrip() + "…"
        d.text((bx0, yy + 26), v, font=val, fill=INK)
        yy += 58

    _seal(d, 330, 1395, 100)                              # footer
    sig = _font("serif", 30)
    for cx, who in [(960, result["authors"].split("·")[0].strip()),
                    (1520, result["authors"].split("·")[-1].strip())]:
        d.line((cx - 250, 1390, cx + 250, 1390), fill=INK, width=2)
        _center(d, 1402, who, sig, INK, cx=cx)
        _center(d, 1444, "Researcher", _font("sans", 24), MUTED, cx=cx)
    idf = _font("sans-bold", 34)
    rx = W - m - 90                                       # right-aligned ID
    d.text((rx - d.textlength("ASSESSMENT ID", font=_font("sans", 22)), 1345),
           "ASSESSMENT ID", font=_font("sans", 22), fill=MUTED)
    d.text((rx - d.textlength(result["assessment_id"], font=idf), 1375),
           result["assessment_id"], font=idf, fill=INK)

    _center(d, 1492, "Automated research-prototype assessment. Not a substitute for an expert "
            "field survey; low-confidence results should be reviewed by a marine biologist.",
            _font("sans", 21), MUTED)
    return img


def to_png(img: Image.Image) -> bytes:
    b = io.BytesIO(); img.save(b, "PNG", optimize=True, dpi=(DPI, DPI)); return b.getvalue()


def to_pdf(img: Image.Image) -> bytes:
    b = io.BytesIO(); img.convert("RGB").save(b, "PDF", resolution=DPI); return b.getvalue()
