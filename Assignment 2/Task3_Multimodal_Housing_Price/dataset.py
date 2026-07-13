"""
PyTorch Dataset that pairs each house's tabular features with its image(s).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = Path(__file__).parent / "data"

TABULAR_NUMERIC = ["area", "bedrooms", "bathrooms", "floors", "year_built", "condition"]
TABULAR_CATEGORICAL = ["location"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_image_transform(img_size=128):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class HousingMultimodalDataset(Dataset):
    """
    Expects a CSV with columns:
        area, bedrooms, bathrooms, floors, year_built, location, condition,
        image_path, price
    `image_path` is relative to `data/`.
    """

    def __init__(self, df: pd.DataFrame, location_categories, img_size=128, train=True,
                 tabular_mean=None, tabular_std=None):
        self.df = df.reset_index(drop=True)
        self.transform = default_image_transform(img_size)
        self.location_categories = location_categories
        self.train = train

        numeric = self.df[TABULAR_NUMERIC].astype(float).values
        if tabular_mean is None:
            tabular_mean = numeric.mean(axis=0)
            tabular_std = numeric.std(axis=0) + 1e-6
        self.tabular_mean = tabular_mean
        self.tabular_std = tabular_std
        self.numeric_scaled = (numeric - tabular_mean) / tabular_std

        # one-hot encode location
        self.location_onehot = np.zeros((len(self.df), len(location_categories)), dtype=np.float32)
        for i, loc in enumerate(self.df["location"]):
            if loc in location_categories:
                self.location_onehot[i, location_categories.index(loc)] = 1.0

        self.tabular_features = np.concatenate(
            [self.numeric_scaled.astype(np.float32), self.location_onehot], axis=1
        )

    @property
    def tabular_dim(self):
        return self.tabular_features.shape[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = DATA_DIR / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        tabular_tensor = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        price = torch.tensor(float(row["price"]), dtype=torch.float32)
        log_price = torch.log1p(price)

        return {
            "image": img_tensor,
            "tabular": tabular_tensor,
            "price": price,
            "log_price": log_price,
        }


def load_and_split(test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(DATA_DIR / "houses.csv")
    location_categories = sorted(df["location"].unique().tolist())
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    return train_df, test_df, location_categories
