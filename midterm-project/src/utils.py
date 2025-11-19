from pathlib import Path
import pandas as pd
from PIL import Image
import torchvision.transforms as T

def load_and_split():
    gt = pd.read_csv("data/ISIC_2020_Training_GroundTruth.csv")
    dups = pd.read_csv("data/ISIC_2020_Training_Duplicates.csv")

    dupset = set(dups["image_name_1"]).union(set(dups["image_name_2"]))
    df = gt[~gt["image_name"].isin(dupset)]

    # resize folder
    orig = Path("data/train")
    small = Path("data/train_small")
    small.mkdir(exist_ok=True)

    for p in orig.glob("*.jpg"):
        Image.open(p).convert("RGB").resize((128,128)).save(small/p.name)

    train_df = df.sample(frac=0.8, random_state=42)
    val_df   = df.drop(train_df.index)

    return train_df, val_df, small

class ISICDataset:
    def __init__(self, df, img_dir, transform):
        self.df, self.dir, self.tr = df.reset_index(drop=True), Path(img_dir), transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        name = self.df.iloc[idx]["image_name"]
        img = Image.open(self.dir/f"{name}.jpg").convert("RGB")
        return self.tr(img), float(self.df.iloc[idx]["target"])

tfm = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5]*3,[0.5]*3)
])
