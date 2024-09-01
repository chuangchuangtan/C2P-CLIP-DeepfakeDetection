import torch
import numpy as np
import torchvision.datasets as datasets
from tqdm import tqdm
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from PIL import ImageFile, Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')
import os


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class ImageFolder2(datasets.DatasetFolder):
    def __init__(
        self,  root: str,  transform: Optional[Callable] = None, ):
        super().__init__(
            root,
            transform=transform,
            extensions=IMG_EXTENSIONS,
            loader = pil_loader )
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path

def get_dataset(opt):
    dset_lst = []
    depth=0
    def get_01(root_path, depth):
        if depth>10: print('='*10, f'\ndataset search depth max 10\n', '='*10); return
        classes = os.listdir(root_path)
        if '0_real' in classes and '1_fake' in classes and len(classes)==2:
            dset_lst.append( dataset_folder(opt, root_path))
            return
        else:
            depth+=1
            for cls in classes:
                get_01(root_path + '/' + cls, depth=depth)
    get_01(opt.dataroot, depth=depth)
    assert len(dset_lst) > 0
    if len(dset_lst) == 1: return dset_lst[0]
    else:  return torch.utils.data.ConcatDataset(dset_lst)

def dataset_folder(opt, root_path):
    # if opt.tencrop:
        # cropfuc = transforms.TenCrop(opt.cropSize)
    # else:
        # cropfuc = transforms.Lambda(lambda img: [ transforms.CenterCrop(opt.cropSize)(img) ] )
    return ImageFolder2(   root = root_path,
                        transform = transforms.Compose([
                            transforms.Resize( opt.loadSize ),
                            transforms.CenterCrop(opt.cropSize),
                            transforms.ToTensor(),
                            # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                            # transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
                            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                        ]))


def create_dataloader(opt):
    dataset = get_dataset(opt)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              sampler=None,
                                              num_workers=8)
    return data_loader
