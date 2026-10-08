"""Dynamic-programming seam (pure numpy/cv2), copied from seam_in_box.py for reuse without side imports."""
import cv2
import numpy as np

STEP = 4


def _dp(G, valid, K):
    """G (H, X) gains, valid mask; path t(x) maximizing sum G[t(x), x] with |t(x) - t(x-1)| <= K."""
    H, X = G.shape
    NEG = -1e9
    V = np.where(valid[:, 0], G[:, 0], NEG)
    back = np.zeros((H, X), np.int32)
    idx = np.arange(H)
    for x in range(1, X):
        best, arg = np.full(H, NEG), idx.copy()
        for d in range(-K, K + 1):                     # sliding-window max over the previous column
            src = idx + d
            ok = (src >= 0) & (src < H)
            cand = np.full(H, NEG); cand[ok] = V[src[ok]]
            better = cand > best
            best[better], arg[better] = cand[better], src[better]
        V = np.where(valid[:, x], G[:, x] + best, NEG)
        back[:, x] = arg
    t = np.zeros(X, np.int32); t[-1] = int(V.argmax())
    for x in range(X - 1, 0, -1):
        t[x - 1] = back[t[x], x]
    return t


def seam_polygon(P, thr, K):
    """P (H, W) crop probability -> polygon (n, 2) in crop coordinates, or None."""
    H, W = P.shape
    X = W // STEP
    if X < 2:
        return None
    Pc = P[:, :X * STEP].reshape(H, X, STEP).mean(axis=2)
    colmax = Pc.max(axis=0)
    on = np.where(colmax >= thr)[0]
    if len(on) < 2:
        return None
    a, b = on[0], on[-1] + 1                            # horizontal extent of the line
    Pc = Pc[:, a:b]
    X = b - a
    blur = cv2.GaussianBlur(Pc, (1, 0), sigmaX=0.1, sigmaY=max(1.0, H / 12))
    yc = blur.argmax(axis=0).astype(float)
    weak = Pc.max(axis=0) < thr                         # word gaps: interpolate the line center
    if weak.any() and (~weak).any():
        yc[weak] = np.interp(np.where(weak)[0], np.where(~weak)[0], yc[~weak])
    yc = np.round(cv2.medianBlur(yc.astype(np.float32).reshape(1, -1), 5).ravel()).astype(int).clip(0, H - 1)
    S = Pc - thr
    Cs = np.vstack([np.zeros((1, X)), np.cumsum(S, axis=0)])        # Cs[y] = sum S[:y]
    rows = np.arange(H)[:, None]
    cols = np.arange(X)
    top_gain = Cs[yc + 1, cols][None] - Cs[rows, cols[None]]         # sum S[t..yc]
    bot_gain = Cs[rows + 1, cols[None]] - Cs[yc, cols][None]         # sum S[yc..b]
    t = _dp(top_gain, rows <= yc[None], K)
    u = _dp(bot_gain, rows >= yc[None], K)
    xs = (a + np.arange(X)) * STEP
    top = [(x, y) for i, (x, y) in enumerate(zip(xs, t)) for x in (x, x + STEP)]
    bot = [(x, y + 1) for i, (x, y) in enumerate(zip(xs, u)) for x in (x, x + STEP)]
    return np.array(top + bot[::-1], np.int32)


