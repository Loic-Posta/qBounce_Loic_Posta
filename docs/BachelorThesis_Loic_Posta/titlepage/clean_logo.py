#!/usr/bin/env python3
"""Make a logo's white background transparent and trim the surrounding margin.

A logo saved as an opaque PNG paints a near-white rectangle onto the page --
here the ATI logo's background sat at 253 rather than 255, which reads as a
faint grey frame. Adding an alpha channel and cropping to the ink fixes both.

    python3 clean_logo.py <in.png> [out.png]
"""
import sys
import cv2
import numpy as np

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else src
im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
if im is None:
    sys.exit(f"cannot read {src}")
if im.ndim == 2:
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
if im.shape[2] == 3:
    im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)

# Anything near-white becomes fully transparent; 245 keeps antialiased edges.
white = (im[:, :, :3] > 245).all(axis=2)
im[:, :, 3] = np.where(white, 0, 255).astype(np.uint8)

ys, xs = np.where(im[:, :, 3] > 0)
if len(ys):
    im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

cv2.imwrite(dst, im)
print(f"{dst}: {im.shape[1]}x{im.shape[0]}, transparent background, trimmed")
