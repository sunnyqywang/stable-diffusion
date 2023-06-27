import glob
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

class SatelliteBase(Dataset):
    def __init__(self, size=256, image_root=".", data_root=".", conditional=False):

        self.image_root = image_root
        self.data_root = data_root
        self.conditional = conditional
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

        if self.conditional:
            return {'image': image, 'condition': self.demo[item], 'image_path': img_name}
        else:
            return {'image': image}

    def _load(self):
        image_list = glob.glob(self.image_root + "*.png")
        image_list += glob.glob(self.image_root + "*.jpg")
        self.image_df = pd.DataFrame(image_list, columns=['img_dir'])
        self.image_df['geoid'] = [img_name[img_name.rfind('/') + 1:img_name.rfind('_')] for img_name in image_list]
        self.image_df['idx'] = [int(img_name[img_name.rfind('_') + 1:img_name.rfind('.')]) for img_name in image_list]

        if self.conditional:
            demo_df = pd.read_csv(self.data_root + "Census_old/demo_tract.csv")
            demo_df['pop_density'] = demo_df['tot_population'] / demo_df['area']

            for c in ['pop_density', 'pct25_34yrs', 'pct35_50yrs', 'pctover65yrs',
                      'pctwhite_alone', 'pct_nonwhite',
                      'pctblack_alone',
                      'pct_col_grad', 'avg_tt_to_work', 'inc_per_capita']:

                demo_df[c] = (demo_df[c] - demo_df[c].min()) / (demo_df[c].max() - demo_df[c].min())
                demo_df[c] = (demo_df[c] - 0.5) / 0.5

            # 7 variables version
            #     demo_df = demo_df[['pop_density','pct25_34yrs','pct35_50yrs','pctover65yrs',
            #              'pctwhite_alone','pct_nonwhite','inc_per_capita']]

            # 10 variables version
            self.image_df = pd.merge(self.image_df, demo_df, how='inner')


class SatelliteTrain(SatelliteBase):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self._filter()

    def _filter(self):

        self.image_df = self.image_df[self.image_df['idx'] < 18]
        self.data = self.image_df['img_dir'].to_numpy()

        if self.conditional:
            self.demo = self.image_df[['pop_density', 'pct25_34yrs', 'pct35_50yrs', 'pctover65yrs',
                      'pctwhite_alone', 'pct_nonwhite',
                      'pctblack_alone',
                      'pct_col_grad', 'avg_tt_to_work', 'inc_per_capita']].to_numpy()

class SatelliteValidation(SatelliteBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._filter()

    def _filter(self):
        self.image_df = self.image_df[self.image_df['idx'] >= 18]
        self.data = self.image_df['img_dir'].to_numpy()
        if self.conditional:
            self.demo = self.image_df[['pop_density', 'pct25_34yrs', 'pct35_50yrs', 'pctover65yrs',
                      'pctwhite_alone', 'pct_nonwhite',
                      'pctblack_alone',
                      'pct_col_grad', 'avg_tt_to_work', 'inc_per_capita']].to_numpy()
