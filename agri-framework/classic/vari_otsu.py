import argparse, os, csv, glob
import cv2
import numpy as np
# White balance (gray-world, hızlı)
def gray_world(img):
    b,g,r = cv2.split(img.astype(np.float32))
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    kg, kb, kr = (mb+mg+mr)/(3*mg+1e-6), (mb+mg+mr)/(3*mb+1e-6), (mb+mg+mr)/(3*mr+1e-6)
    out = cv2.merge([b*kb, g*kg, r*kr]); return np.clip(out,0,255).astype(np.uint8)

# CLAHE (Y kanalında)
def clahe_rgb(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(ycrcb)
    y = cv2.createCLAHE(2.0,(8,8)).apply(y)
    return cv2.cvtColor(cv2.merge([y,cr,cb]), cv2.COLOR_YCrCb2BGR)

def vari(img_bgr: np.ndarray) -> np.ndarray:
    # VARI = (G - R) / (G + R - B)
    b, g, r = cv2.split(img_bgr.astype(np.float32) + 1e-6)
    return (g - r) / (g + r - b + 1e-6)

def process(in_dir, out_dir, csv_path, save_masks_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    if save_masks_dir:
        os.makedirs(save_masks_dir, exist_ok=True)

    rows = [["filename", "green_area_ratio(0-1)"]]
    image_paths = sorted(glob.glob(os.path.join(in_dir, "*.jpg")) +
                         glob.glob(os.path.join(in_dir, "*.png")))

    if not image_paths:
        print(f"[!] '{in_dir}' içinde .jpg/.png yok.")
        return

    for ip in image_paths:
        img = cv2.imread(ip)
        if img is None:
            print(f"[!] Okunamadı, geçiliyor: {ip}")
            continue

        # 1) VARI hesapla
        v = vari(img)

        # 2) 0–255’e ölçekle
        v_min, v_max = float(v.min()), float(v.max())
        v_norm = ((v - v_min) / (v_max - v_min + 1e-6) * 255).astype(np.uint8)

        # 3) (İsteğe bağlı) hafif blur; NOT: v_norm TANIMLANDIKTAN SONRA
        v_norm = cv2.GaussianBlur(v_norm, (5, 5), 0)

        # 4) Otsu eşik
        _, mask = cv2.threshold(v_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5) Morfoloji temizliği
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 6) Bindirme
        overlay = img.copy()
        overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
        edges = cv2.Canny(mask, 0, 1)
        overlay[edges > 0] = [0, 0, 255]
        vis = np.hstack([img, overlay])

        out_path = os.path.join(out_dir, os.path.basename(ip))
        cv2.imwrite(out_path, vis)

        ratio = float((mask > 0).sum()) / float(mask.size)
        rows.append([os.path.basename(ip), f"{ratio:.4f}"])
        print(f"[✓] {os.path.basename(ip)}  alan_oranı={ratio:.4f}  → {out_path}")

        if save_masks_dir:
            mp = os.path.join(save_masks_dir, os.path.basename(ip).rsplit(".", 1)[0] + ".png")
            cv2.imwrite(mp, mask)

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[✓] CSV yazıldı → {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Girdi resim klasörü (örn. data/images)")
    ap.add_argument("--out_dir", required=True, help="Bindirme çıktı klasörü (örn. outputs/samples)")
    ap.add_argument("--csv", required=True, help="Özet CSV yolu (örn. outputs/area.csv)")
    ap.add_argument("--save_masks_dir", default="", help="Opsiyonel: ikili maskeleri kaydet (örn. data/masks)")
    args = ap.parse_args()

    process(args.in_dir, args.out_dir, args.csv,
            save_masks_dir=(args.save_masks_dir if args.save_masks_dir else None))
