# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Keep optional training/evaluation data components from blocking inference
# imports such as ``dinov2.data.transforms``.  This Matcher checkout does not
# include the ImageNet dataset package used by ``loaders.py``; the FSS/COCO/
# Pascal inference path does not need it.
try:
    from .adapters import DatasetWithEnumeratedTargets
except ModuleNotFoundError:
    DatasetWithEnumeratedTargets = None

try:
    from .loaders import make_data_loader, make_dataset, SamplerType
except ModuleNotFoundError:
    make_data_loader = None
    make_dataset = None
    SamplerType = None

try:
    from .collate import collate_data_and_cast
except ModuleNotFoundError:
    collate_data_and_cast = None

try:
    from .masking import MaskingGenerator
except ModuleNotFoundError:
    MaskingGenerator = None

try:
    from .augmentations import DataAugmentationDINO
except ModuleNotFoundError:
    DataAugmentationDINO = None
