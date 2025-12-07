from pathlib import Path
import os, json, time, argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import albumentations as A
import segmentation_models_pytorch as smp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class SegDs(Dataset):
    def __init__(self, img_dir, mask_dir, size=512, train=True):
        exts = (".jpg", ".png", ".jpeg")
        self.imgs = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(exts) and os.path.isfile(os.path.join(img_dir, f))
        ])
        self.mask_dir = mask_dir
        self.size = size

        aug_train = A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomCrop(size, size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.3),
            A.RandomGamma(p=0.3),
        ])

        aug_val = A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT_101),
            A.CenterCrop(size, size),
        ])

        self.aug = aug_train if train else aug_val

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        ip = self.imgs[i]
        name = os.path.splitext(os.path.basename(ip))[0] + ".png"
        mp = os.path.join(self.mask_dir, name)

        im = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        m = cv2.imread(mp, 0)
        if m is None:
            m = np.zeros(im.shape[:2], np.uint8)

        data = self.aug(image=im, mask=m)
        im, m = data["image"], data["mask"]

        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        m = torch.from_numpy((m > 127).astype(np.float32)).unsqueeze(0)
        return im, m


def iou_f1(prob, y, thr=0.5, eps=1e-6):
    p = (prob > thr).float()
    inter = (p * y).sum()
    union = p.sum() + y.sum() - inter
    iou = ((inter + eps) / (union + eps)).item()
    prec = (inter / (p.sum() + eps)).item() if p.sum() > 0 else 0.0
    rec = (inter / (y.sum() + eps)).item() if y.sum() > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec + eps)) if (prec + rec) > 0 else 0.0
    return iou, f1


def build_model(model_name: str):
    """
    model_name:
      - "unet"
      - "unetpp"
      - "deeplabv3p"
      - "fpn"
    """
    encoder = "timm-efficientnet-b3"
    kwargs = dict(
        encoder_name=encoder,
        encoder_weights="imagenet",
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


def main(args):
    # Kök output klasörü
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Model tabanlı run klasörü: outputs/runs/<model_name>/<run_name>/
    model_name = args.model_name.lower()
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = root / "runs" / model_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Config kaydet
    cfg = vars(args).copy()
    cfg["model_name"] = model_name
    cfg["run_name"] = run_name
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Dataset & split
    full = SegDs(args.img_dir, args.mask_dir, size=args.size, train=True)
    n = len(full)
    if n < 3:
        raise SystemExit(f"Çok az görüntü var: {n}")

    n_tr = max(2, int(0.85 * n))
    n_va = max(1, n - n_tr)
    tr_ids, va_ids = random_split(
        range(n),
        [n_tr, n_va],
        generator=torch.Generator().manual_seed(42)
    )

    tr = torch.utils.data.Subset(
        SegDs(args.img_dir, args.mask_dir, args.size, train=True),
        tr_ids.indices
    )
    va = torch.utils.data.Subset(
        SegDs(args.img_dir, args.mask_dir, args.size, train=False),
        va_ids.indices
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pin = dev == "cuda"
    workers = 2 if dev == "cuda" else 0

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True,
                       num_workers=workers, pin_memory=pin)
    dl_va = DataLoader(va, batch_size=args.batch,
                       num_workers=workers, pin_memory=pin)

    # Model seçimi
    model = build_model(model_name).to(dev)

    loss_bce = torch.nn.BCEWithLogitsLoss()

    def loss_fn(p, t):
        p_sig = torch.sigmoid(p)
        inter = (p_sig * t).sum()
        denom = p_sig.sum() + t.sum()
        dice = 1 - (2 * inter + 1e-6) / (denom + 1e-6)
        return loss_bce(p, t) + dice

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)

    best = 0.0
    history = []

    # Global model klasörü (model_adına göre)
    models_root = root / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    # Backward compatibility: eski tek dosya
    legacy_model_path = root / "model_smp.pt"
    legacy_metrics_path = root / "metrics_smp.json"
    legacy_history_path = root / "train_history_smp.json"

    for e in range(args.epochs):
        t0 = time.time()
        model.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            p = model(x)
            loss = loss_fn(p, y)
            loss.backward()
            opt.step()

        model.eval()
        ious, f1s = [], []
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev)
                p = torch.sigmoid(model(x))
                i, f1 = iou_f1(p, y)
                ious.append(i)
                f1s.append(f1)

        miou = float(np.mean(ious)) if ious else 0.0
        mf1 = float(np.mean(f1s)) if f1s else 0.0
        dt = time.time() - t0
        sched.step()

        history.append({"epoch": e, "mIoU": miou, "F1": mf1, "time_s": dt})
        print(f"epoch {e:02d}  mIoU={miou:.3f}  F1={mf1:.3f}  time={dt:.1f}s")

        if miou > best:
            best = miou
            # 1) Eski davranış (isteğe bağlı, hala dursun)
            torch.save(model.state_dict(), legacy_model_path)
            with open(legacy_metrics_path, "w") as f:
                json.dump(
                    {"best_mIoU": best, "best_F1": mf1, "epoch": e + 1, "model": model_name},
                    f,
                    indent=2,
                )

            # 2) Run klasörü için model + metrics
            torch.save(model.state_dict(), run_dir / "model_smp.pt")
            with open(run_dir / "metrics_smp.json", "w") as f:
                json.dump(
                    {
                        "best_mIoU": best,
                        "best_F1": mf1,
                        "epoch": e + 1,
                        "model_name": model_name,
                        "run_name": run_name,
                    },
                    f,
                    indent=2,
                )

            # 3) Model adına göre global kayıt
            torch.save(model.state_dict(), models_root / f"model_{model_name}.pt")

    # History global + run altında
    with open(legacy_history_path, "w") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "train_history_smp.json", "w") as f:
        json.dump(history, f, indent=2)

    print("[✓] best mIoU:", best)
    print(f"[✓] run dir  : {run_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="data/images")
    ap.add_argument("--mask_dir", default="data/masks")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)

    # Yeni: yöntem seçimi + run adı
    ap.add_argument("--model_name", default="unet",
                    help="unet | unetpp | deeplabv3p | fpn")
    ap.add_argument("--run_name", default=None,
                    help="Run ismi (klasör), boş bırakılırsa timestamp")

    args = ap.parse_args()
    main(args)
