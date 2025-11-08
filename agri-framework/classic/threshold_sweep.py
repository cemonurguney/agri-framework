import os, glob, csv, time, cv2, numpy as np

def sweep(img, gt=None):
    v = (img - img.min())/(img.max()-img.min()+1e-6)*255
    v = v.astype(np.uint8)
    best=(None,-1.0)
    t0=time.time()
    for t in range(50, 205, 5):
        _,mask = cv2.threshold(v, t, 255, cv2.THRESH_BINARY)
        if gt is None: continue
        p=(mask>127); g=(gt>127)
        inter=(p & g).sum(); union=(p | g).sum()
        iou=inter/(union+1e-6)
        if iou>best[1]: best=(t,iou)
    dt=time.time()-t0
    return best, dt

IMG_DIR="data/images"; GT_DIR="data/gt_masks"; OUT="outputs/threshold_sweep.csv"
os.makedirs("outputs",exist_ok=True)
rows=[["filename","best_thr","best_iou","time_s"]]

for ip in sorted(glob.glob(os.path.join(IMG_DIR,"*.jpg"))+glob.glob(os.path.join(IMG_DIR,"*.png"))):
    name=os.path.basename(ip); img=cv2.imread(ip)
    # VARI haritası
    b,g,r = cv2.split(img.astype(np.float32)+1e-6)
    v = (g - r) / (g + r - b + 1e-6)
    gt_path=os.path.join(GT_DIR, name.rsplit(".",1)[0]+".png")
    gt = cv2.imread(gt_path,0) if os.path.exists(gt_path) else None
    (thr,iou),dt = sweep(v, gt)
    rows.append([name, "" if thr is None else thr, "" if iou<0 else f"{iou:.3f}", f"{dt:.3f}"])
with open(OUT,"w",newline="") as f: csv.writer(f).writerows(rows)
print("[✓] Yazıldı:", OUT)
