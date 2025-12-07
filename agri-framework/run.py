import argparse
import time
from pathlib import Path

from dl import train_smp, infer_smp


# Tek dataset preset'in var: images + masks + test_images
DATA_CFG = {
    "img_dir": "data/images",
    "mask_dir": "data/masks",
    "test_images": "data/test_images",
}


def cmd_train(args):
    img_dir = DATA_CFG["img_dir"]
    mask_dir = DATA_CFG["mask_dir"]

    model_name = args.model_name.lower()
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")

    print("[INFO] Train:")
    print(f"   model_name = {model_name}")
    print(f"   img_dir    = {img_dir}")
    print(f"   mask_dir   = {mask_dir}")
    print(f"   out_dir    = {args.out_dir}")
    print(f"   run_name   = {run_name}")
    print(f"   size       = {args.size}")
    print(f"   batch      = {args.batch}")
    print(f"   epochs     = {args.epochs}")
    print(f"   lr         = {args.lr}")

    train_args = argparse.Namespace(
        img_dir=img_dir,
        mask_dir=mask_dir,
        out_dir=args.out_dir,
        size=args.size,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        model_name=model_name,
        run_name=run_name,
    )

    train_smp.main(train_args)


def _find_latest_run(out_dir: Path, model_name: str):
    base = out_dir / "runs" / model_name
    if not base.exists():
        return None
    runs = [p for p in base.iterdir() if p.is_dir()]
    if not runs:
        return None
    runs_sorted = sorted(runs, key=lambda p: p.name)
    return runs_sorted[-1]


def cmd_test(args):
    img_dir = DATA_CFG["test_images"]

    out_root = Path(args.out_dir)
    model_name = args.model_name.lower()

    # run klasörünü bul
    if args.run_name:
        run_dir = out_root / "runs" / model_name / args.run_name
    elif args.use_latest_run:
        run_dir = _find_latest_run(out_root, model_name)
    else:
        run_dir = None

    # Model dosyası: eğer run_dir varsa oradan, yoksa arg'dan
    if run_dir is not None and args.model == "outputs/model_smp.pt":
        model_path = run_dir / "model_smp.pt"
    else:
        model_path = Path(args.model)

    print("[INFO] Test / infer:")
    print(f"   model_name = {model_name}")
    print(f"   test_imgs  = {img_dir}")
    print(f"   out_root   = {args.out_dir}")
    print(f"   model      = {model_path}")
    print(f"   run_dir    = {run_dir if run_dir is not None else '(yok / kullanılmıyor)'}")
    print(f"   size       = {args.size}")
    print(f"   test_tag   = {args.test_tag}")

    infer_args = argparse.Namespace(
        img_dir=img_dir,
        out_dir=args.out_dir,  # infer run_dir'ü görünce zaten run içini kullanacak
        size=args.size,
        model=str(model_path),
        run_dir=str(run_dir) if run_dir is not None else None,
        test_tag=args.test_tag,
        mask_dir=None,  # ileride test mask klasörü eklemek istersen burayı değiştiririz
    )

    infer_smp.main(infer_args)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    default_out = "outputs"

    # TRAIN
    p_tr = sub.add_parser("train", help="Segmentation modeli eğit")
    p_tr.set_defaults(func=cmd_train)
    p_tr.add_argument("--out_dir", default=default_out,
                      help="Çıktı kök klasörü (default: outputs)")
    p_tr.add_argument("--size", type=int, default=384)
    p_tr.add_argument("--batch", type=int, default=4)
    p_tr.add_argument("--epochs", type=int, default=20)
    p_tr.add_argument("--lr", type=float, default=3e-4)
    p_tr.add_argument("--model_name", default="unet",
                      help="unet | unetpp | deeplabv3p | fpn")
    p_tr.add_argument("--run_name", default=None,
                      help="Run klasörü ismi; boşsa timestamp")

    # TEST
    p_te = sub.add_parser("test", help="Test / inference")
    p_te.set_defaults(func=cmd_test)
    p_te.add_argument("--out_dir", default=default_out,
                      help="Çıktı kök klasörü (default: outputs)")
    p_te.add_argument("--size", type=int, default=384)
    p_te.add_argument("--model", default="outputs/model_smp.pt",
                      help="Model dosyası; run_dir varsa override edilir")
    p_te.add_argument("--model_name", default="unet",
                      help="unet | unetpp | deeplabv3p | fpn (run bulmak için)")
    p_te.add_argument("--run_name", default=None,
                      help="outputs/runs/<model_name>/<run_name> seçmek için")
    p_te.add_argument("--use_latest_run", action="store_true",
                      help="run_name yoksa, ilgili model için en son run'ı kullan")
    p_te.add_argument("--test_tag", default="test",
                      help="run_dir içindeki pred_<test_tag> klasör adı")

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
