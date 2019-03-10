import os
import torch
import numpy as np
from dataloader import Data_loader
#from sklearn.preprocessing import LabelEncoder


class fromPathDatasetClassification(Data_loader):

    def __init__(self, cf, path, resize=None,
                 preprocess=None, transform=None, valid=False):
        super(fromPathDatasetClassification, self).__init__()
        self.cf = cf
        self.resize = resize
        self.transform = transform
        self.preprocess = preprocess
        self.path = path
        self.image_names = []
        self.gt = []

        print("\t Images from {}".format(path))

        print(cf.labels)
        #cf.map_labels = {'background': 0, 'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian': 4, 'Person_sitting': 5, 'Cyclist': 6, 'Tram': 7}
        print(cf.map_labels)
        for label in cf.labels:
            for im in os.listdir(os.path.join(path, label)):
                self.image_names.append(os.path.abspath(os.path.join(path, label, im)))
                print(label)
                print(cf.map_labels[label])
                self.gt.append(int(cf.map_labels[label]))
        print(cf.labels)
        print(cf.map_labels)
        self.num_images = len(self.image_names)

        print ("\t Images found: " + str(len(self.image_names)))
        if len(self.image_names) < self.num_images or self.num_images == -1:
            self.num_images = len(self.image_names)
        self.img_indexes = np.arange(len(self.image_names))
        self.update_indexes(valid=valid)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_path = self.image_names[self.indexes[idx]]
        img = np.asarray(self.Load_image(img_path, self.resize, self.cf.grayscale))
        gt = [self.gt[self.indexes[idx]]]
        if self.transform is not None:
            img, _ = self.transform(img, None)
        if self.preprocess is not None:
            img = self.preprocess(img)
        gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()
        return img, gt

    def update_indexes(self, num_images=None, valid=False):
        if self.cf.shuffle and not valid:
            np.random.shuffle(self.img_indexes)
        if num_images is not None:
            if len(self.image_names) < self.num_images or num_images == -1:
                self.num_images = len(self.image_names)
            else:
                self.num_images = num_images
        self.indexes = self.img_indexes[:self.num_images]