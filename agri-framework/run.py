import argparse
from pathlib import Path

from classic import vari_otsu
from dl import train_smp, infer_smp

# ==========================
# PATHLER
# ==========================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
MASKS_PSEUDO_DIR = DATA_DIR / "masks_pseudo"
TEST_IMAGES_DIR = DATA_DIR / "test_images"

OUTPUTS_DIR = ROOT / "outputs"
PSEUDO_VIS_DIR = OUTPUTS_DIR / "pseudo_vis"
PRED_DL_DIR = OUTPUTS_DIR / "pred_dl"


# ==========================
# KOMUT FONKSİYONLARI
# ==========================

def cmd_maskecikar(args):
    """
    Pseudo maske üretimi:
    data/images  ->  data/masks_pseudo (+ outputs/pseudo_vis, csv)
    """
    PSEUDO_VIS_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_PSEUDO_DIR.mkdir(parents=True, exist_ok=True)

    pseudo_args = argparse.Namespace(
        in_dir=str(IMAGES_DIR),
        out_dir=str(PSEUDO_VIS_DIR),
        csv=str(OUTPUTS_DIR / "pseudo_areas.csv"),
        save_masks_dir=str(MASKS_PSEUDO_DIR),
    )

    print(f"[INFO] Pseudo maske üretimi: {IMAGES_DIR} → {MASKS_PSEUDO_DIR}")
    vari_otsu.main(pseudo_args)


def cmd_train(args):
    """
    Eğitim:
    data/images + data/masks kullanarak model eğit.
    Çıktılar: outputs/ içinde model_smp.pt vs.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_args = argparse.Namespace(
        img_dir=str(IMAGES_DIR),
        mask_dir=str(MASKS_DIR),
        out_dir=str(OUTPUTS_DIR),
        size=args.size,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
    )

    print("[INFO] Eğitim başlıyor:")
    print(f"   images = {IMAGES_DIR}")
    print(f"   masks  = {MASKS_DIR}")
    print(f"   out    = {OUTPUTS_DIR}")
    train_smp.main(train_args)


def cmd_test(args):
    """
    Test / infer:
    data/test_images -> outputs/pred_dl
    """
    PRED_DL_DIR.mkdir(parents=True, exist_ok=True)

    infer_args = argparse.Namespace(
        img_dir=str(TEST_IMAGES_DIR),
        out_dir=str(PRED_DL_DIR),
        size=args.size,
        model=str(OUTPUTS_DIR / args.model),
    )

    print("[INFO] Test / infer:")
    print(f"   test_images = {TEST_IMAGES_DIR}")
    print(f"   out         = {PRED_DL_DIR}")
    print(f"   model       = {infer_args.model}")
    infer_smp.main(infer_args)


# ==========================
# ARGPARSE / SUBCOMMAND
# ==========================

def build_parser():
    parser = argparse.ArgumentParser(
        description="AGRI-FRAMEWORK: maskecikar / train / test"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # maskecikar
    p_mask = subparsers.add_parser("maskecikar", help="Pseudo maske üret")
    p_mask.set_defaults(func=cmd_maskecikar)

    # train
    p_train = subparsers.add_parser("train", help="UNet/SMP modeli eğit")
    p_train.add_argument("--size", type=int, default=384)
    p_train.add_argument("--batch", type=int, default=4)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.set_defaults(func=cmd_train)

    # test
    p_test = subparsers.add_parser("test", help="Eğitilmiş modeli test et / infer")
    p_test.add_argument("--size", type=int, default=384)
    p_test.add_argument("--model", type=str, default="model_smp.pt")
    p_test.set_defaults(func=cmd_test)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
