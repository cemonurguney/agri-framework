import os, glob, cv2, numpy as np, torch
from torch.utils.data import Dataset

class SegDataset(Dataset):
    def __init__(self, img_dir="../data/images", mask_dir="../data/masks", size=384):
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                           glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_dir = mask_dir
        self.size = size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        ip = self.imgs[i]
        name = os.path.basename(ip).rsplit(".", 1)[0] + ".png"
        mp = os.path.join(self.mask_dir, name)

        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, 0)
        if mask is None:
            # Maske yoksa komple sıfır ver (acil durum)
            mask = np.zeros(img.shape[:2], np.uint8)

        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        x = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        y = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return x, y
