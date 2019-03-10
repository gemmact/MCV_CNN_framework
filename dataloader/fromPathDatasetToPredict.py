import os
import numpy as np
from dataloader import Data_loader


class fromPathDatasetToPredict(Data_loader):

    def __init__(self, cf, path, resize,
                        preprocess=None):
        super(fromPathDatasetToPredict, self).__init__()
        self.cf = cf
        self.resize = resize
        self.preprocess = preprocess
        self.image_names = []
        for label in cf.labels:
            for im in os.listdir(os.path.join(path, label)):
                self.image_names.append(os.path.abspath(os.path.join(path, label, im)))

        self.num_images = len(self.image_names)

        if len(self.image_names) < self.num_images or self.num_images == -1:
            self.num_images = len(self.image_names)
        self.img_indexes = np.arange(len(self.image_names))
        self.indexes = self.img_indexes[:self.num_images]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_path = self.image_names[self.indexes[idx]]
        img_path_comp = img_path.split("/")
        img_name = img_path_comp[-1]
        img, shape = self.load_img(img_path, self.resize, self.cf.grayscale, order=1)
        if self.preprocess is not None:
            img = self.preprocess(img)
        return img, img_name, shape