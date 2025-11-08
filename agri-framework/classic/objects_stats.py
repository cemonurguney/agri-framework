import os, glob, csv, cv2, numpy as np

MASK_DIR="data/masks"; OUT="outputs/objects_stats.csv"
os.makedirs("outputs",exist_ok=True)
rows=[["filename","n_objects(>min_area)","mean_area_px","median_area_px"]]

MIN_AREA=200  # küçük çöp lekeleri uçsun
for mp in sorted(glob.glob(os.path.join(MASK_DIR,"*.png"))):
    name=os.path.basename(mp); m=(cv2.imread(mp,0)>127).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = [stats[i,cv2.CC_STAT_AREA] for i in range(1,num) if stats[i,cv2.CC_STAT_AREA]>=MIN_AREA]
    if len(areas)==0:
        rows.append([name, 0, 0, 0]); continue
    rows.append([name, len(areas), int(np.mean(areas)), int(np.median(areas))])

with open(OUT,"w",newline="") as f: csv.writer(f).writerows(rows)
print("[✓] Yazıldı:", OUT)
