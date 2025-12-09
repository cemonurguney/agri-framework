from pathlib import Path
import os, argparse, json
import cv2
import numpy as np
import torch
import albumentations as A
import segmentation_models_pytorch as smp

# Supervisely / Bonirob için foreground sınıfları
FG_CLASSES = {"weed", "crop", "sugar beet", "sugarbeet"}


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


def postprocess_mask(m, prob=None, min_area=25, prob_thr=None):
    """
    m: (H,W) uint8 {0,1}
    - hafif close + dilate
    - küçük componentleri ve düşük ortalama prob'lu componentleri sil
    """
    m = (m > 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.dilate(m, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    if prob is not None and prob_thr is not None:
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                m[labels == i] = 0
                continue

            mask_i = labels == i
            if not mask_i.any():
                m[mask_i] = 0
                continue

            mean_p = float(prob[mask_i].mean())
            if mean_p < prob_thr:
                m[mask_i] = 0
    else:
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                m[labels == i] = 0

    return m


def build_model(model_name: str):
    """Egitimdeki gibi model mimarisini secen yardimci fonksiyon."""
    encoder = "timm-efficientnet-b3"
    # Inference sırasında encoder_weights=None, çünkü load_state_dict ile yükleyeceğiz.
    kwargs = dict(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    model_name = model_name.lower()
    if model_name == "unet":
        return smp.Unet(**kwargs)
    elif model_name in ("unetpp", "unet++"):
        return smp.UnetPlusPlus(**kwargs)
    elif model_name in ("deeplabv3p", "deeplabv3plus"):
        return smp.DeepLabV3Plus(**kwargs)
    elif model_name == "fpn":
        return smp.FPN(**kwargs)
    else:
        raise ValueError(f"Desteklenmeyen model_name: {model_name}")


def load_gt_mask(mask_dir, name, dataset):
    """
    name: input image file name (örn: 001_image.png veya rgb_*.png)
    dataset: "default" | "cwfid" | "bonirob" vs
    PNG varsa onu, yoksa Supervisely .png.json okur.
    """
    if mask_dir is None or not os.path.isdir(mask_dir):
        return None

    stem = os.path.splitext(name)[0]

    if dataset == "cwfid":
        # 001_image -> 001_mask
        stem = stem.replace("_image", "_mask")

    png_path = os.path.join(mask_dir, stem + ".png")
    json_path = os.path.join(mask_dir, stem + ".png.json")

    # 1) PNG maske
    if os.path.isfile(png_path):
        gt = cv2.imread(png_path, 0)
        if gt is None:
            return None
        if dataset == "cwfid":
            # CWFiD: ot = siyah, arka plan = beyaz → ters çevir
            gt = 255 - gt
        return gt

    # 2) Supervisely JSON
    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        h = data.get("size", {}).get("height")
        w = data.get("size", {}).get("width")
        if h is None or w is None:
            return None

        fg_classes = {c.lower() for c in FG_CLASSES}
        mask = np.zeros((h, w), dtype=np.uint8)

        for obj in data.get("objects", []):
            cls = obj.get("classTitle", "").lower()
            if cls not in fg_classes:
                continue
            pts = np.array(obj["points"]["exterior"], dtype=np.int32)
            if pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts], 255)

        return mask

    return None


def main(args):
    # run.py'den gelmeyen kullanımda da patlamasın diye getattr
    run_dir = getattr(args, "run_dir", None)
    test_tag = getattr(args, "test_tag", "test")
    out_dir_arg = getattr(args, "out_dir", "outputs/pred_dl")

    # Model ismini al, yoksa varsayılan unet
    model_name = getattr(args, "model_name", "unet")

    # Dataset bilgisi (default / cwfid / bonirob vs.)
    dataset = getattr(args, "dataset", "default")

    # Defaultlar
    thr = getattr(args, "thr", 0.0)            # hybrid için pixel threshold
    min_area = getattr(args, "min_area", 25)   # minik gürültüleri kes
    prob_thr = getattr(args, "prob_thr", 0.65) # component ortalama prob eşiği

    # Eğer run_dir verilmiş ve out_dir default ise, pred klasörünü run içine koy
    if run_dir is not None and out_dir_arg == "outputs":
        out_dir = Path(run_dir) / f"pred_{test_tag}"
    elif run_dir is not None and out_dir_arg == "outputs/pred_dl":
        out_dir = Path(run_dir) / f"pred_{test_tag}"
    else:
        out_dir = Path(out_dir_arg)

    out_dir.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Building model: {model_name}...")
    model = build_model(model_name).to(dev)

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

    ious_h, f1s_h = [], []
    ious_dl, f1s_dl = [], []
    ious_c, f1s_c = [], []

    aug = build_val_aug(args.size)

    with torch.no_grad():
        for name in img_files:
            ip = os.path.join(args.img_dir, name)
            im_bgr = cv2.imread(ip)
            if im_bgr is None:
                print(f"[WARN] image read failed, skipping: {ip}")
                continue

            im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

            gt = None
            if has_masks:
                gt = load_gt_mask(args.mask_dir, name, dataset)

            if gt is not None:
                data = aug(image=im, mask=gt)
                r, gt_t = data["image"], data["mask"]
            else:
                data = aug(image=im)
                r, gt_t = data["image"], None

            x = torch.from_numpy(r).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            # DL çıktısı (prob)
            p = torch.sigmoid(model(x.to(dev))).cpu().numpy()[0, 0]  # HxW, 0-1

            # ---- DL-only ----
            m_dl_only = (p > 0.5).astype(np.uint8)
            m_dl_only_pp = postprocess_mask(m_dl_only, prob=None,
                                            min_area=min_area, prob_thr=None)

            # ---- Color-only ----
            m_color_only = exg_vari_mask(r)
            m_color_only = (m_color_only > 0).astype(np.uint8)
            m_color_only_pp = postprocess_mask(m_color_only, prob=None,
                                               min_area=min_area, prob_thr=None)

            # ---- Hybrid (DL + renk) ----
            m_dl_hybrid = (p > thr).astype(np.uint8)
            m_color = exg_vari_mask(r)
            m_color = (m_color > 0).astype(np.uint8)

            m_hybrid = (m_dl_hybrid & m_color).astype(np.uint8)
            m_hybrid_pp = postprocess_mask(m_hybrid, prob=p,
                                           min_area=min_area, prob_thr=prob_thr)

            # Overlay: hybrid'i gösteriyoruz
            overlay = r.copy()
            overlay[m_hybrid_pp > 0] = (
                0.55 * overlay[m_hybrid_pp > 0] + 0.45 * np.array([255, 0, 0])
            ).astype(np.uint8)
            edges = cv2.Canny((m_hybrid_pp * 255).astype(np.uint8), 0, 1)
            overlay[edges > 0] = [255, 0, 0]
            vis = np.hstack([r, overlay])

            cv2.imwrite(
                str(out_dir / name),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            )

            # Test metrikleri (varsa GT)
            if gt_t is not None:
                gt_bin = torch.from_numpy(
                    (gt_t > 127).astype(np.float32)
                ).unsqueeze(0).unsqueeze(0)

                # DL-only metric
                pred_dl_t = torch.from_numpy(
                    m_dl_only_pp.astype(np.float32)
                ).unsqueeze(0).unsqueeze(0)
                i_dl, f1_dl = iou_f1(pred_dl_t, gt_bin, thr=0.5)
                ious_dl.append(i_dl)
                f1s_dl.append(f1_dl)

                # Color-only metric
                pred_c_t = torch.from_numpy(
                    m_color_only_pp.astype(np.float32)
                ).unsqueeze(0).unsqueeze(0)
                i_c, f1_c = iou_f1(pred_c_t, gt_bin, thr=0.5)
                ious_c.append(i_c)
                f1s_c.append(f1_c)

                # Hybrid metric
                pred_h_t = torch.from_numpy(
                    m_hybrid_pp.astype(np.float32)
                ).unsqueeze(0).unsqueeze(0)
                i_h, f1_h = iou_f1(pred_h_t, gt_bin, thr=0.5)
                ious_h.append(i_h)
                f1s_h.append(f1_h)

    # Test metriklerini run klasörüne yaz (varsa)
    if has_masks and run_dir is not None:
        n = len(ious_h)
        test_metrics = {
            "mIoU": float(np.mean(ious_h)) if ious_h else 0.0,
            "F1": float(np.mean(f1s_h)) if f1s_h else 0.0,

            "mIoU_dl": float(np.mean(ious_dl)) if ious_dl else 0.0,
            "F1_dl": float(np.mean(f1s_dl)) if f1s_dl else 0.0,

            "mIoU_color": float(np.mean(ious_c)) if ious_c else 0.0,
            "F1_color": float(np.mean(f1s_c)) if f1s_c else 0.0,

            "mIoU_hybrid": float(np.mean(ious_h)) if ious_h else 0.0,
            "F1_hybrid": float(np.mean(f1s_h)) if f1s_h else 0.0,

            "n_samples": n,
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

    # Model ismi
    ap.add_argument("--model_name", default="unet", help="unet | unetpp | deeplabv3p | fpn")

    # run + opsiyonel test mask
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--test_tag", default="test")
    ap.add_argument("--mask_dir", default=None)

    # Gürültü kontrol parametreleri
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--min_area", type=int, default=25)
    ap.add_argument("--prob_thr", type=float, default=0.65)

    # Dataset tipi (cwfid için mask invert, isim eşleştirme vs)
    ap.add_argument(
        "--dataset",
        default="default",
        choices=["default", "cwfid", "bonirob"],
    )

    args = ap.parse_args()
    main(args)
