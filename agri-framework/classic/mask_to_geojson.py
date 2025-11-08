import os, glob, json, cv2, numpy as np

MASK_DIR="data/masks"; OUT_DIR="outputs/geojson"; os.makedirs(OUT_DIR, exist_ok=True)

# GSD: metre/piksel (yoksa 1 al)
GSD = 0.05  # örnek: 5 cm/pixel
def poly_area_m2(cnt):
    return cv2.contourArea(cnt) * (GSD**2)

def contour_to_polygon(cnt):
    # GeoJSON: [lon, lat] ister ama biz düz piksel koordinatıyla Feature basıyoruz.
    # Sunum için yeterli; gerçek koordinat isterse georeferans gerekir.
    return [[float(p[0][0]), float(p[0][1])] for p in cnt]

for mp in sorted(glob.glob(os.path.join(MASK_DIR,"*.png"))):
    name = os.path.basename(mp).rsplit(".",1)[0]
    m = (cv2.imread(mp,0)>127).astype(np.uint8)*255
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    feats=[]
    for c in cnts:
        if cv2.contourArea(c)<100: continue
        poly = contour_to_polygon(c)
        area = poly_area_m2(c)
        feats.append({
            "type":"Feature",
            "properties":{"filename":name, "area_m2": area},
            "geometry":{"type":"Polygon","coordinates":[poly]}
        })
    fc={"type":"FeatureCollection","features":feats}
    outp=os.path.join(OUT_DIR, name+".geojson")
    with open(outp,"w") as f: json.dump(fc,f)
    print("[✓] GeoJSON:", outp)
