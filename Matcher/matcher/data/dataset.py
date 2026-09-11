r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from .coco import DatasetCOCO
from .pascal import DatasetPASCAL
from .fss import DatasetFSS

# Optional datasets have extra dependencies (notably Detectron2).  Keep
# imports lazy/optional so semantic FSS/COCO/Pascal experiments do not require
# installing dependencies that are only used by PACO-Part/LVIS.
try:
    from .paco_part import DatasetPACOPart
except ModuleNotFoundError:
    DatasetPACOPart = None

try:
    from .pascal_part import DatasetPASCALPart
except ModuleNotFoundError:
    DatasetPASCALPart = None

try:
    from .lvis import DatasetLVIS
except ModuleNotFoundError:
    DatasetLVIS = None

try:
    from .dis import DatasetDIS
except ModuleNotFoundError:
    DatasetDIS = None
import numpy as np 

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'coco': DatasetCOCO,
            'pascal': DatasetPASCAL,
            'fss': DatasetFSS,
            'paco_part': DatasetPACOPart,
            'pascal_part': DatasetPASCALPart,
            'lvis': DatasetLVIS,
            'dis' : DatasetDIS
        }

        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0
        np.random.seed(42)
        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader
