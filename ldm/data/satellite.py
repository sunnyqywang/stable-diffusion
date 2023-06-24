import glob
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

class SatelliteBase(Dataset):
    def __init__(self, size=256, image_root=".", data_root=".", load_demo=-1):

        self.image_root = image_root
        self.data_root = data_root
        self.load_demo = load_demo
        self.size = size

        self._load()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_name = self.data[item]
        image = Image.open(img_name)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = (image.astype(np.float32) / 127.5) - 1.0

        s = int((600 - self.size) // 2)
        e = int((600 + self.size) // 2)
        image = image[s:e, s:e, :]

        return {'image': image}

    def _load(self):
        image_list = glob.glob(self.image_root + "*.png")
        image_list += glob.glob(self.image_root + "*.jpg")
        self.image_df = pd.DataFrame(image_list, columns=['img_dir'])
        self.image_df['geoid'] = [img_name[img_name.rfind('/') + 1:img_name.rfind('_')] for img_name in image_list]
        self.image_df['idx'] = [int(img_name[img_name.rfind('_') + 1:img_name.rfind('.')]) for img_name in image_list]
        if self.load_demo > 0:
            raise NotImplementedError()


class SatelliteTrain(SatelliteBase):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self._filter()

    def _filter(self):

        self.image_df = self.image_df[self.image_df['idx'] < 18]
        self.data = self.image_df['img_dir'].to_numpy()


class SatelliteValidation(SatelliteBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._filter()

    def _filter(self):
        self.image_df = self.image_df[self.image_df['idx'] >= 18]
        self.data = self.image_df['img_dir'].to_numpy()
