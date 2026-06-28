"""
Multi-scale crop tiler.

The curator often rhymes a sports photo with a *detail* of a large painting, not
the whole canvas. So instead of embedding only the full artwork, we slide windows
of several sizes across it and embed each region. A query then matches the best
region, and we can report which crop of which artwork won.

iter_crops(img) yields (box, crop) where box is (left, top, right, bottom) in the
original pixel coords -- handy for drawing/exporting the matching region later.
"""


def iter_crops(img, scales=(0.6, 0.4), overlap=0.4, min_px=80):
    """
    Yield (box, crop) over the full image plus a grid of sub-windows.

    The full image is always emitted once. Then for each scale we slide a square
    window of side (short_side * scale) across the image.

    scales  -- sub-window sides as fractions of the image's short side.
    overlap -- fraction two adjacent windows share (0.4 = step by 60% of a window).
    min_px  -- skip windows smaller than this (avoids useless tiny crops).
    """
    w, h = img.size
    short = min(w, h)
    seen = {(0, 0, w, h)}
    yield (0, 0, w, h), img  # always include the whole artwork
    for s in scales:
        if s >= 0.99:
            continue  # full image already emitted
        win = int(round(short * s))
        if win < min_px:
            continue
        step = max(1, int(round(win * (1 - overlap))))
        xs = list(range(0, max(1, w - win) + 1, step))
        ys = list(range(0, max(1, h - win) + 1, step))
        if xs[-1] != w - win:
            xs.append(max(0, w - win))
        if ys[-1] != h - win:
            ys.append(max(0, h - win))
        for top in ys:
            for left in xs:
                box = (left, top, min(left + win, w), min(top + win, h))
                if box in seen:
                    continue
                seen.add(box)
                yield box, img.crop(box)
