import os, argparse, cv2, numpy as np, torch
import segmentation_models_pytorch as smp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="../data/images")
    ap.add_argument("--out_dir", default="../outputs/samples_dl")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--model", default="../outputs/model_smp.pt")
    args=ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(
        encoder_name="timm-efficientnet-b3",  # DEĞİŞTİ
        encoder_weights="imagenet",  # timm üzerinden çeker
        in_channels=3,
        classes=1
    ).to(dev)

    # PyTorch 2.5+ güvenli yükleme
    try:
        sd = torch.load(args.model, map_location=dev, weights_only=True)
    except TypeError:
        # Eski sürümlerde weights_only yoksa normal yükle (bizim dosya zaten state_dict)
        sd = torch.load(args.model, map_location=dev)
    model.load_state_dict(sd)
    model.eval()

    exts=(".jpg",".png",".jpeg")
    imgs=[os.path.join(args.img_dir,f) for f in os.listdir(args.img_dir) if f.lower().endswith(exts)]
    with torch.no_grad():
        for ip in imgs[:]:
            name=os.path.basename(ip)
            im=cv2.cvtColor(cv2.imread(ip),cv2.COLOR_BGR2RGB)
            r=cv2.resize(im,(args.size,args.size))
            x=torch.from_numpy(r).permute(2,0,1).float().unsqueeze(0)/255.0
            p=torch.sigmoid(model(x.to(dev))).cpu().numpy()[0,0]
            m=(p>0.5).astype(np.uint8)
            overlay=r.copy()
            overlay[m>0]=(0.55*overlay[m>0]+0.45*np.array([255,0,0])).astype(np.uint8)
            edges=cv2.Canny((m*255).astype(np.uint8),0,1); overlay[edges>0]=[255,0,0]
            vis=np.hstack([r, overlay])
            cv2.imwrite(os.path.join(args.out_dir,name), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print("[✓] örnekler:", args.out_dir)

if __name__=="__main__":
    main()
