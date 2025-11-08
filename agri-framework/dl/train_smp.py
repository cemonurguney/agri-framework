import os, json, time, argparse, cv2, numpy as np, torch
from torch.utils.data import DataLoader, random_split, Dataset
import albumentations as A
import segmentation_models_pytorch as smp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class SegDs(Dataset):
    def __init__(self, img_dir, mask_dir, size=512, train=True):
        self.imgs = sorted([p for p in sum([ [os.path.join(img_dir,f) for f in os.listdir(img_dir) if f.lower().endswith(ext)] for ext in (".jpg",".png",".jpeg") ], []) if os.path.isfile(p)])
        self.mask_dir = mask_dir; self.size=size
        aug_train = A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size,size,border_mode=cv2.BORDER_REFLECT_101),
            A.RandomCrop(size,size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.3),
            A.RandomGamma(p=0.3),
        ])
        aug_val = A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size,size,border_mode=cv2.BORDER_REFLECT_101),
            A.CenterCrop(size,size),
        ])
        self.aug = aug_train if train else aug_val

    def __len__(self): return len(self.imgs)

    def __getitem__(self,i):
        ip=self.imgs[i]; name=os.path.splitext(os.path.basename(ip))[0]+".png"
        mp=os.path.join(self.mask_dir,name)
        im=cv2.cvtColor(cv2.imread(ip),cv2.COLOR_BGR2RGB)
        m=cv2.imread(mp,0)
        if m is None:
            m=np.zeros(im.shape[:2],np.uint8)
        data=self.aug(image=im, mask=m)
        im, m = data["image"], data["mask"]
        im = torch.from_numpy(im).permute(2,0,1).float()/255.0
        m  = torch.from_numpy((m>127).astype(np.float32)).unsqueeze(0)
        return im, m

def iou_f1(prob, y, thr=0.5, eps=1e-6):
    p=(prob>thr).float()
    inter=(p*y).sum(); union=p.sum()+y.sum()-inter
    iou=((inter+eps)/(union+eps)).item()
    prec=(inter/(p.sum()+eps)).item()
    rec =(inter/(y.sum()+eps)).item()
    f1= (2*prec*rec/(prec+rec+eps))
    return iou, f1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="../data/images")
    ap.add_argument("--mask_dir", default="../data/masks")
    ap.add_argument("--out_dir",  default="../outputs")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    args=ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    full = SegDs(args.img_dir, args.mask_dir, size=args.size, train=True)
    n=len(full); n_tr=max(2,int(0.85*n)); n_va=max(1,n-n_tr)
    tr_ids, va_ids = random_split(range(n), [n_tr,n_va], generator=torch.Generator().manual_seed(42))
    tr = torch.utils.data.Subset(SegDs(args.img_dir,args.mask_dir,args.size,train=True), tr_ids.indices)
    va = torch.utils.data.Subset(SegDs(args.img_dir,args.mask_dir,args.size,train=False), va_ids.indices)

    dev="cuda" if torch.cuda.is_available() else "cpu"
    pin = dev=="cuda"; workers=2 if dev=="cuda" else 0
    dl_tr=DataLoader(tr,batch_size=args.batch,shuffle=True,num_workers=workers,pin_memory=pin)
    dl_va=DataLoader(va,batch_size=args.batch,num_workers=workers,pin_memory=pin)

    model = smp.Unet(
        encoder_name="timm-efficientnet-b3",  # DEĞİŞTİ
        encoder_weights="imagenet",  # timm üzerinden çeker
        in_channels=3,
        classes=1
    ).to(dev)

    loss_bce=torch.nn.BCEWithLogitsLoss()
    def loss_fn(p,t):
        p_sig=torch.sigmoid(p); inter=(p_sig*t).sum(); denom=p_sig.sum()+t.sum()
        dice=1-(2*inter+1e-6)/(denom+1e-6)
        return loss_bce(p,t)+dice

    opt=AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched=CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.1)

    best=0.0; history=[]
    for e in range(args.epochs):
        t0=time.time(); model.train()
        for x,y in dl_tr:
            x,y=x.to(dev),y.to(dev)
            opt.zero_grad(set_to_none=True)
            p=model(x); loss=loss_fn(p,y)
            loss.backward(); opt.step()
        # val
        model.eval(); ious=[]; f1s=[]
        with torch.no_grad():
            for x,y in dl_va:
                x,y=x.to(dev),y.to(dev)
                p=torch.sigmoid(model(x))
                i,f1=iou_f1(p,y); ious.append(i); f1s.append(f1)
        miou=float(np.mean(ious)) if ious else 0.0
        mf1 =float(np.mean(f1s)) if f1s else 0.0
        dt=time.time()-t0; sched.step()
        history.append({"epoch":e,"mIoU":miou,"F1":mf1,"time_s":dt})
        print(f"epoch {e:02d}  mIoU={miou:.3f}  F1={mf1:.3f}  time={dt:.1f}s")

        if miou>best:
            best=miou
            torch.save(model.state_dict(), os.path.join(args.out_dir,"model_smp.pt"))
            with open(os.path.join(args.out_dir,"metrics_smp.json"),"w") as f:
                json.dump({"best_mIoU":best,"best_F1":mf1,"epoch":e+1},f)

    with open(os.path.join(args.out_dir,"train_history_smp.json"),"w") as f:
        json.dump(history,f,indent=2)
    print("[✓] best mIoU:",best)

if __name__=="__main__":
    main()
