from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as standard_transforms
import sys

sys.path.append('../')
from utils.transformations.input_preprocess import preproces_input
from utils.transformations.tensor import ToTensor, ToTensorPil
from utils.transformations.crop import CropSegSem, CropObjDet
from utils.transformations.flip import RandomHorizontalFlipSegSem, RandomHorizontalFlipObjDet
from utils.transformations.random_distort import Random_distort
from utils.transformations.composition import ComposeSemSeg, ComposeObjDet, ComposeResize
from utils.transformations.resize import Resize
from fromFileDatasetSegmentation import fromFileDatasetSegmentation
from fromFileDatasetClassification import fromFileDatasetClassification
from fromPathDatasetClassification import fromPathDatasetClassification
from fromFileDatasetDetection import fromFileDatasetDetection
from fromFileDatasetToPredict import fromFileDatasetToPredict
from fromPathDatasetToPredict import fromPathDatasetToPredict


class Dataloader_Builder(object):
    def __init__(self, cf, model):
        self.cf = cf
        self.model = model
        self.loader_set = None
        # Compose preprocesing function for dataloaders
        if self.cf.problem_type == 'detection':
            self.img_preprocessing = standard_transforms.Compose([Random_distort(self.cf), preproces_input(self.cf),
                                                                  standard_transforms.ToTensor()])
            self.test_img_preprocessing = standard_transforms.Compose([preproces_input(self.cf),
                                                                  standard_transforms.ToTensor()])
            self.train_transformation = ComposeObjDet([CropObjDet(self.cf), RandomHorizontalFlipObjDet(self.cf)])
            self.resize = ComposeResize([Resize(self.cf)])
        else:
            self.img_preprocessing = standard_transforms.Compose([Random_distort(self.cf), preproces_input(self.cf),
                                                                  ToTensor()])
            self.test_img_preprocessing = standard_transforms.Compose([preproces_input(self.cf),
                                                                  ToTensor()])
            self.train_transformation = ComposeSemSeg([CropSegSem(self.cf),
                                                       RandomHorizontalFlipSegSem(self.cf)])

    def build_train(self):
        if self.cf.problem_type =='segmentation':
            self.train_set = fromFileDatasetSegmentation(self.cf, self.cf.train_images_txt, self.cf.train_gt_txt,
                                        self.cf.train_samples, self.cf.resize_image_train,
                                preprocess=self.img_preprocessing, transform=self.train_transformation)
        elif self.cf.problem_type =='classification':
            if self.cf.train_path is not None:
                self.train_set = fromPathDatasetClassification(self.cf, self.cf.train_path, self.cf.resize_image_train,
                                                               preprocess=self.img_preprocessing,
                                                               transform=self.train_transformation)
            else:
                self.train_set = fromFileDatasetClassification(self.cf, self.cf.train_images_txt, self.cf.train_gt_txt,
                                                          self.cf.train_samples, self.cf.resize_image_train,
                                            preprocess=self.img_preprocessing, transform=self.train_transformation)
        elif self.cf.problem_type =='detection':
            self.train_set = fromFileDatasetDetection(self.cf, self.cf.train_images_txt, self.cf.train_gt_txt,
                                                      self.cf.train_samples, self.cf.resize_image_train,
                                                      preprocess=self.img_preprocessing,
                                                      transform=self.train_transformation,
                                                      box_coder=self.model.box_coder,
                                                      resize_process=self.resize)
        self.train_loader = DataLoader(self.train_set, batch_size=self.cf.train_batch_size, num_workers=4)

    def build_valid(self, valid_samples, images_txt, gt_txt, resize_image, batch_size):
        if self.cf.problem_type == 'segmentation':
            self.loader_set = fromFileDatasetSegmentation(self.cf, images_txt, gt_txt,
                                                          valid_samples, resize_image,
                                                          preprocess=self.img_preprocessing, transform=None,
                                                          valid=True)
        elif self.cf.problem_type == 'classification':
            if self.cf.valid_path is not None:
                self.loader_set = fromPathDatasetClassification(self.cf, self.cf.valid_path, resize_image,
                                                                preprocess=self.img_preprocessing, transform=None,
                                                                valid=True)
            else:
                self.loader_set = fromFileDatasetClassification(self.cf, images_txt, gt_txt,
                                                                valid_samples, resize_image,
                                                                preprocess=self.img_preprocessing, transform=None,
                                                                valid=True)
        elif self.cf.problem_type == 'detection':
            self.train_transformation = ComposeObjDet([Resize(self.cf)])
            self.loader_set = fromFileDatasetDetection(self.cf, images_txt, gt_txt,
                                                       valid_samples, resize_image,
                                                       preprocess=self.img_preprocessing,
                                                       transform=None,
                                                       valid=True,
                                                       resize_process=self.resize)
        self.loader = DataLoader(self.loader_set, batch_size=batch_size, num_workers=4)

    def build_predict(self):
        if self.cf.problem_type == 'classification' and self.cf.test_path is not None:
            self.predict_set = fromPathDatasetToPredict(self.cf, self.cf.test_path, self.cf.resize_image_test,
                                                   preprocess=self.test_img_preprocessing)
        else:
            self.predict_set = fromFileDatasetToPredict(self.cf, self.cf.test_images_txt,
                                                   self.cf.test_samples, self.cf.resize_image_test,
                                                   preprocess=self.test_img_preprocessing)
        self.predict_loader = DataLoader(self.predict_set, batch_size=1, num_workers=4)
