r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):
        # Dataset implementations are imported on demand in build_dataloader.
        # This keeps optional dependencies (notably detectron2 for PACO-Part)
        # out of unrelated experiments.
        cls.datasets = {
            'coco': ('.coco', 'DatasetCOCO'),
            'pascal': ('.pascal', 'DatasetPASCAL'),
            'fss': ('.fss', 'DatasetFSS'),
            'paco_part': ('.paco_part', 'DatasetPACOPart'),
            'pascal_part': ('.pascal_part', 'DatasetPASCALPart'),
            'lvis': ('.lvis', 'DatasetLVIS'),
            'dis': ('.dis', 'DatasetDIS'),
        }
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        if img_size is None:         
            cls.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        else: 
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
        if benchmark not in cls.datasets:
            raise KeyError(f'Unknown benchmark: {benchmark}')

        module_name, class_name = cls.datasets[benchmark]
        from importlib import import_module
        dataset_cls = getattr(import_module(module_name, package=__package__), class_name)

        dataset = dataset_cls(cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader
