from pathlib import Path
import os, argparse
import cv2
import numpy as np
import torch
import albumentations as A
import segmentation_models_pytorch as smp


def iou_f1(prob, y, thr=0.6, eps=1e-6):
    """
    prob: (1,1,H,W) tensör, 0-1 ya da 0/1
    y   : (1,1,H,W) tensör, 0/1
    """
    p = (prob > thr).float()
    inter = (p * y).sum()
    union = p.sum() + y.sum() - inter
    iou = ((inter + eps) / (union + eps)).item()
    prec = (inter / (p.sum() + eps)).item() if p.sum() > 0 else 0.0
    rec = (inter / (y.sum() + eps)).item() if y.sum() > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec + eps)) if (prec + rec) > 0 else 0.0
    return iou, f1


def build_val_aug(size):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT_101),
        A.CenterCrop(size, size),
    ])


def exg_vari_mask(rgb, vari_q=0.65):
    """
    Adaptif ExG + VARI maskeleme.
    - ExG için Otsu threshold
    - VARI için quantile (varsayılan en üst ~%35)

    rgb: HxWx3, uint8 (0-255)
    çıktı: 0/1 uint8
    """
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    exg = 2 * g - r - b
    vari = (g - r) / (g + r - b + 1e-6)

    # --- ExG: Otsu ile adaptif eşik ---
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, exg_bin = cv2.threshold(
        exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    m_exg = exg_bin > 0

    # --- VARI: üst quantile ---
    vari_flat = vari.reshape(-1)
    vari_flat = vari_flat[np.isfinite(vari_flat)]
    if vari_flat.size == 0:
        m_vari = np.zeros_like(vari, dtype=bool)
    else:
        thr_vari = np.quantile(vari_flat, vari_q)
        m_vari = vari > thr_vari

    m = (m_exg & m_vari).astype(np.uint8)
    return m


def postprocess_mask(m, prob=None, min_area=50, prob_thr=None):
    """
    m: (H,W) uint8 {0,1}
    - open + close
    - küçük componentleri sil
    - (opsiyonel) component ortalama prob < prob_thr ise sil
    """
    m = (m > 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    if prob is not None and prob_thr is not None:
        # prob: HxW (0-1)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                m[labels == i] = 0
                continue

            mask_i = labels == i
            mean_p = float(prob[mask_i].mean()) if mask_i.any() else 0.0
            if mean_p < prob_thr:
                m[mask_i] = 0
    else:
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                m[labels == i] = 0

    return m


def main(args):
    # run.py'den gelmeyen kullanımda da patlamasın diye getattr
    run_dir = getattr(args, "run_dir", None)
    test_tag = getattr(args, "test_tag", "test")
    out_dir_arg = getattr(args, "out_dir", "outputs/pred_dl")

    # Threshold ve min_area defaultlarını güvene al
    thr = getattr(args, "thr", 0.6)
    min_area = getattr(args, "min_area", 80)  # biraz yukarı aldım

    # Eğer run_dir verilmiş ve out_dir default ise, pred klasörünü run içine koy
    if run_dir is not None and out_dir_arg == "outputs":
        out_dir = Path(run_dir) / f"pred_{test_tag}"
    elif run_dir is not None and out_dir_arg == "outputs/pred_dl":
        out_dir = Path(run_dir) / f"pred_{test_tag}"
    else:
        out_dir = Path(out_dir_arg)

    out_dir.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(
        encoder_name="timm-efficientnet-b3",
        encoder_weights=None,  # zaten weight yükleyeceğiz
        in_channels=3,
        classes=1
    ).to(dev)

    try:
        sd = torch.load(args.model, map_location=dev, weights_only=True)
    except TypeError:
        sd = torch.load(args.model, map_location=dev)
    model.load_state_dict(sd)
    model.eval()

    exts = (".jpg", ".png", ".jpeg")
    img_files = sorted([
        f for f in os.listdir(args.img_dir)
        if f.lower().endswith(exts)
    ])

    has_masks = args.mask_dir is not None and os.path.isdir(args.mask_dir)
    ious, f1s = [], []

    aug = build_val_aug(args.size)

    with torch.no_grad():
        for name in img_files:
            ip = os.path.join(args.img_dir, name)
            im_bgr = cv2.imread(ip)
            im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

            gt = None
            if has_masks:
                base = os.path.splitext(name)[0] + ".png"
                mp = os.path.join(args.mask_dir, base)
                if os.path.isfile(mp):
                    gt = cv2.imread(mp, 0)

            if gt is not None:
                data = aug(image=im, mask=gt)
                r, gt_t = data["image"], data["mask"]
            else:
                data = aug(image=im)
                r, gt_t = data["image"], None

            x = torch.from_numpy(r).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            # DL çıktısı (prob)
            p = torch.sigmoid(model(x.to(dev))).cpu().numpy()[0, 0]  # HxW, 0-1
            m_dl = (p > thr).astype(np.uint8)

            # Renk bazlı ön-maske (adaptif ExG + VARI)
            m_color = exg_vari_mask(r)
            m_color = (m_color > 0).astype(np.uint8)

            # DL + renk öncülü kesişimi
            m = (m_dl & m_color).astype(np.uint8)

            # Post-process + prob filtresi
            m = postprocess_mask(m, prob=p, min_area=min_area, prob_thr=0.7)

            # Overlay
            overlay = r.copy()
            overlay[m > 0] = (
                0.55 * overlay[m > 0] + 0.45 * np.array([255, 0, 0])
            ).astype(np.uint8)
            edges = cv2.Canny((m * 255).astype(np.uint8), 0, 1)
            overlay[edges > 0] = [255, 0, 0]
            vis = np.hstack([r, overlay])

            cv2.imwrite(
                str(out_dir / name),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            )

            # Test metriği (opsiyonel, varsa)
            if gt_t is not None:
                gt_bin = torch.from_numpy(
                    (gt_t > 127).astype(np.float32)
                ).unsqueeze(0).unsqueeze(0)
                pred_t = torch.from_numpy(
                    m.astype(np.float32)
                ).unsqueeze(0).unsqueeze(0)
                iou, f1 = iou_f1(pred_t, gt_bin, thr=0.5)  # m zaten 0/1
                ious.append(iou)
                f1s.append(f1)

    # Test metriklerini run klasörüne yaz (varsa)
    if has_masks and run_dir is not None:
        import json
        test_metrics = {
            "mIoU": float(np.mean(ious)) if ious else 0.0,
            "F1": float(np.mean(f1s)) if f1s else 0.0,
            "n_samples": len(ious),
            "test_tag": test_tag,
        }
        with open(Path(run_dir) / f"test_metrics_{test_tag}.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        print("[✓] test metrics:", test_metrics)

    print("[✓] örnekler:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="data/test_images")
    ap.add_argument("--out_dir", default="outputs/pred_dl")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--model", default="outputs/model_smp.pt")

    # Yeni: run + opsiyonel test mask
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--test_tag", default="test")
    ap.add_argument("--mask_dir", default=None)

    # Gürültü kontrol parametreleri
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--min_area", type=int, default=80)

    args = ap.parse_args()
    main(args)
