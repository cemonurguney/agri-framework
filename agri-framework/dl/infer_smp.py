import os, argparse, cv2, numpy as np, torch
import segmentation_models_pytorch as smp

def hysteresis_mask(prob, t_low=0.45, t_high=0.70):
    low = (prob >= t_low).astype(np.uint8)
    high = (prob >= t_high).astype(np.uint8)

    n, labels = cv2.connectedComponents(low, connectivity=8)
    keep = np.zeros_like(low)
    for i in range(1, n):
        if (high[labels == i].sum() > 0):
            keep[labels == i] = 1
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="../data/images")
    ap.add_argument("--out_dir", default="../outputs/samples_dl")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--model", default="../outputs/model_smp.pt")
    ap.add_argument("--t_low", type=float, default=0.55)   # daha sıkı default
    ap.add_argument("--t_high", type=float, default=0.90)  # daha sıkı default
    ap.add_argument("--exg_thr", type=float, default=0.35)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(
        encoder_name="timm-efficientnet-b3",
        encoder_weights=None,   # SSL indirme yok
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
    imgs = [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) if f.lower().endswith(exts)]

    with torch.no_grad():
        for ip in imgs:
            name = os.path.basename(ip)

            bgr = cv2.imread(ip)
            if bgr is None:
                print("CAN'T READ:", ip)
                continue

            im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            r = cv2.resize(im, (args.size, args.size))
            x = torch.from_numpy(r).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            p = torch.sigmoid(model(x.to(dev))).cpu().numpy()[0, 0]
            print(name, "p[min,max,mean] =", float(p.min()), float(p.max()), float(p.mean()))

            # 1) model mask (histerezis)
            m = hysteresis_mask(p, t_low=args.t_low, t_high=args.t_high)

            # 2) sky kill (HSV blue)
            hsv = cv2.cvtColor(r, cv2.COLOR_RGB2HSV)
            H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            sky = ((H >= 90) & (H <= 140) & (S >= 40) & (V >= 80)).astype(np.uint8)
            m[sky == 1] = 0

            # 3) green gate (ExG)
            rgbf = r.astype(np.float32)
            R, G, B = rgbf[:, :, 0], rgbf[:, :, 1], rgbf[:, :, 2]
            exg = 2 * G - R - B
            exg = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)
            green = (exg > args.exg_thr).astype(np.uint8)

            m = (m & green).astype(np.uint8)

            # visualize
            overlay = r.copy()
            overlay[m > 0] = (0.55 * overlay[m > 0] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
            edges = cv2.Canny((m * 255).astype(np.uint8), 0, 1)
            overlay[edges > 0] = [255, 0, 0]

            vis = np.hstack([r, overlay])
            cv2.imwrite(os.path.join(args.out_dir, name), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print("[✓] örnekler:", args.out_dir)

if __name__ == "__main__":
    main()