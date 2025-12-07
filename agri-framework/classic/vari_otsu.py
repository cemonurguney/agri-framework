from pathlib import Path
import argparse, os, csv, glob
import cv2
import numpy as np



# ---------- Preprocess ----------

def gray_world(img):
    img = img.astype(np.float32)
    b, g, r = cv2.split(img)
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    k = (mb + mg + mr) / 3.0
    kb = k / (mb + 1e-6)
    kg = k / (mg + 1e-6)
    kr = k / (mr + 1e-6)
    out = cv2.merge([b * kb, g * kg, r * kr])
    return np.clip(out, 0, 255).astype(np.uint8)

def clahe_rgb(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.createCLAHE(2.0, (8, 8)).apply(y)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

# ---------- Indeksler ----------

def compute_exg(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(img)
    exg = 2 * g - r - b
    return exg

def compute_vari(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(img)
    denom = g + r - b
    denom = np.where(np.abs(denom) < 1e-3, 1e-3, denom)
    v = (g - r) / denom
    return np.clip(v, -1.0, 1.0)

def norm_to_uint8(arr):
    a_min, a_max = float(arr.min()), float(arr.max())
    return ((arr - a_min) / (a_max - a_min + 1e-6) * 255.0).astype(np.uint8)

# ---------- K-means fallback ----------

def kmeans_veg_mask(img_bgr, exg):
    h, w, _ = img_bgr.shape

    target_w = 256
    scale = target_w / float(w)
    small_w = target_w
    small_h = int(h * scale)
    small_img = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    exg_small = cv2.resize(exg, (small_w, small_h), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(small_img, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    feat = np.stack([L, a, b, norm_to_uint8(exg_small)], axis=-1)
    Z = feat.reshape((-1, 4)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    K = 3
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    veg_cluster = np.argmax(centers[:, -1])
    mask_small = (labels.reshape((small_h, small_w)) == veg_cluster).astype(np.uint8)

    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask

# ---------- Ana maske üretici ----------

def make_mask(img_bgr):
    img_pp = clahe_rgb(gray_world(img_bgr))

    exg = compute_exg(img_pp)
    vari = compute_vari(img_pp)

    exg_u8 = norm_to_uint8(exg)
    exg_blur = cv2.GaussianBlur(exg_u8, (5, 5), 0)
    _, m1 = cv2.threshold(exg_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m1 = (m1 > 0).astype(np.uint8)
    r1 = m1.mean()

    if 0.02 < r1 < 0.7:
        base_mask = m1
    else:
        vari_u8 = norm_to_uint8(vari)
        vari_blur = cv2.GaussianBlur(vari_u8, (5, 5), 0)
        _, m2 = cv2.threshold(vari_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m2 = (m2 > 0).astype(np.uint8)
        r2 = m2.mean()

        if 0.02 < r2 < 0.7:
            base_mask = m2
        else:
            km = kmeans_veg_mask(img_pp, exg)
            base_mask = km

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(base_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = 50
    clean = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1

    return (clean * 255).astype(np.uint8), float(clean.mean())

# ---------- Pipeline ----------

def process(in_dir, out_dir, csv_path, save_masks_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    if save_masks_dir:
        os.makedirs(save_masks_dir, exist_ok=True)

    rows = [["filename", "green_area_ratio(0-1)"]]
    image_paths = sorted(
        glob.glob(os.path.join(in_dir, "*.jpg"))  +
        glob.glob(os.path.join(in_dir, "*.JPG"))  +
        glob.glob(os.path.join(in_dir, "*.jpeg")) +
        glob.glob(os.path.join(in_dir, "*.JPEG")) 
    )
    if not image_paths:
        print(f"[!] '{in_dir}' içinde .jpg/.png yok.")
        return

    for ip in image_paths:
        img = cv2.imread(ip)
        if img is None:
            print(f"[!] Okunamadı, geçiliyor: {ip}")
            continue

        mask, ratio = make_mask(img)

        overlay = img.copy()
        overlay[mask > 0] = (
            0.55 * overlay[mask > 0] + 0.45 * np.array([0, 0, 255])
        ).astype(np.uint8)
        edges = cv2.Canny(mask, 0, 1)
        overlay[edges > 0] = [0, 0, 255]
        vis = np.hstack([img, overlay])

        out_path = os.path.join(out_dir, os.path.basename(ip))
        cv2.imwrite(out_path, vis)

        rows.append([os.path.basename(ip), f"{ratio:.4f}"])
        print(f"[✓] {os.path.basename(ip)}  alan_oranı={ratio:.4f}  → {out_path}")

        if save_masks_dir:
            mp = os.path.join(
                save_masks_dir,
                os.path.basename(ip).rsplit(".", 1)[0] + ".png"
            )
            cv2.imwrite(mp, mask)

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[✓] CSV yazıldı → {csv_path}")

# ---------- main(args) ----------

def main(args):
    process(
        args.in_dir,
        args.out_dir,
        args.csv,
        save_masks_dir=(args.save_masks_dir if getattr(args, "save_masks_dir", "") else None),
    )

# ---------- CLI ----------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Girdi resim klasörü (örn. data/images)")
    ap.add_argument("--out_dir", required=True, help="Bindirme çıktı klasörü (örn. outputs/pseudo)")
    ap.add_argument("--csv", required=True, help="Özet CSV yolu (örn. outputs/pseudo/area.csv)")
    ap.add_argument("--save_masks_dir", default="", help="Opsiyonel: ikili maskeleri kaydet (örn. data/masks_pseudo)")
    args = ap.parse_args()
    main(args)
