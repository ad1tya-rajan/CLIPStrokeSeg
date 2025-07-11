from monai.transforms import (
    AsDiscrete,
    # AddChanneld,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
    ResampleToMatchd,
    ConcatItemsd
)

import collections.abc
import math
import pickle
import shutil
import sys
import os
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py
import glob


import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..") 
from utils.utils import get_key

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey
    
    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img['name'])
            self.data_dic[key].append(img)
        
        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)
    
    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i
    
    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        # data_index = int(index / self.__len__() * self.datasetnum[set_index])
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_statis()
    
    def data_statis(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0

        for img in self.data:
            key = get_key(img['name'])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f'the dataset {key} has no data'
            self.data_num.append(item)
        
        self.datasetlen = len(self.datasetkey)
    
    def index_uniform(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        # print(post_index, self.__len__())
        return self._transform(post_index)

class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        post_label_pth = d['post_label']
        with h5py.File(post_label_pth, 'r') as hf:
            data = hf['post_label'][()]
        d['post_label'] = data[0]
        return d

class RandZoomd_select(RandZoomd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']):
            return d
        d = super().__call__(d)
        return d


class RandCropByPosNegLabeld_select(RandCropByPosNegLabeld):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if key in ['10_03', '10_07', '10_08', '04']:
            return d
        d = super().__call__(d)
        return d

class RandCropByLabelClassesd_select(RandCropByLabelClassesd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if key not in ['10_03', '10_07', '10_08', '04']:
            return d
        d = super().__call__(d)
        return d

class Compose_Select(Compose):
    def __call__(self, input_):
        name = input_['name']
        key = get_key(name)
        for index, _transform in enumerate(self.transforms):
            # for RandCropByPosNegLabeld and RandCropByLabelClassesd case
            if (key in ['10_03', '10_07', '10_08', '04']) and (index == 8):
                continue
            elif (key not in ['10_03', '10_07', '10_08', '04']) and (index == 9):
                continue
            # for RandZoomd case
            if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']) and (index == 7):
                continue
            input_ = apply_transform(_transform, input_, self.map_items, self.unpack_items, self.log_stats)
        return input_

"""
@TODO:
- modify get_loader to support ISLES MRI data
- ignore using LoadImage5hd, use LoadImaged instead
- update data dict creation to ignore post_label since ISLES MRI data does not have post_label
- support MRI path and modality in data_dicts creation
- modify datasetkey since we are not doing multi-class segmentation
"""

def get_loader(args):
    # ---------- Transforms ----------
    train_transforms = Compose([
        LoadImaged(keys=["flair", "adc", "dwi", "label"]),
        EnsureChannelFirstd(keys=["flair", "adc", "dwi", "label"]),
        ResampleToMatchd(keys=["flair", "adc", "dwi"], key_dst="label", mode="bilinear"),
        ConcatItemsd(keys=["flair", "adc", "dwi"], name="image", dim=0),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min, a_max=args.a_max,
            b_min=args.b_min, b_max=args.b_max,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=2, neg=1, num_samples=args.num_samples,
            allow_smaller=True
        ),
        RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["flair", "adc", "dwi", "label"]),
        EnsureChannelFirstd(keys=["flair", "adc", "dwi", "label"]),
        ResampleToMatchd(keys=["flair", "adc", "dwi"], key_dst="label", mode="bilinear"),
        ConcatItemsd(keys=["flair", "adc", "dwi"], name="image", dim=0),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min, a_max=args.a_max,
            b_min=args.b_min, b_max=args.b_max,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ])

    # ---------- Helper for parsing ISLES txt files ----------
    def parse_split_file(split_name):
        file_path = os.path.join(args.data_txt_path, f"{args.dataset_list[0]}_{split_name}.txt")
        samples = []
        with open(file_path, 'r') as f:
            for line in f:
                image_str, label_path = line.strip().split()
                flair_path, adc_path, dwi_path = image_str.split(',')
                flair_path = os.path.join(args.data_root_path, flair_path)
                adc_path = os.path.join(args.data_root_path, adc_path)
                dwi_path = os.path.join(args.data_root_path, dwi_path)
                label_path = os.path.join(args.data_root_path, label_path)
                name = os.path.basename(label_path).split('_msk')[0]
                samples.append({
                    'flair': flair_path,
                    'adc': adc_path,
                    'dwi': dwi_path,
                    'label': label_path,
                    'name': name
                })
        return samples


    # ---------- Phase-specific logic ----------
    if args.phase == 'train':
        train_dicts = parse_split_file('train')
        print(f"Loaded {len(train_dicts)} training samples")
        dataset = CacheDataset(data=train_dicts, transform=train_transforms, cache_rate=args.cache_rate) \
            if args.cache_dataset else Dataset(data=train_dicts, transform=train_transforms)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=list_data_collate)
        return loader, None

    elif args.phase == 'validation':
        val_dicts = parse_split_file('val')
        print(f"Loaded {len(val_dicts)} validation samples")
        dataset = CacheDataset(data=val_dicts, transform=val_transforms, cache_rate=args.cache_rate) \
            if args.cache_dataset else Dataset(data=val_dicts, transform=val_transforms)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return loader, val_transforms

    elif args.phase == 'test':
        test_dicts = parse_split_file('test')
        print(f"Loaded {len(test_dicts)} test samples")
        dataset = CacheDataset(data=test_dicts, transform=val_transforms, cache_rate=args.cache_rate) \
            if args.cache_dataset else Dataset(data=test_dicts, transform=val_transforms)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return loader, val_transforms

    else:
        raise ValueError(f"Unknown phase: {args.phase}")


def get_loader_without_gt(args):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ]
    )

    ## test dict part
    nii_files = glob.glob(args.data_root_path + '/**.nii.gz')
    data_dicts_test = [{'image': image, 'name': image.split('/')[-1].split('.')[0]}
                for image in nii_files]
    print('test len {}'.format(len(data_dicts_test)))
    
    if args.cache_dataset:
        test_dataset = CacheDataset(data=data_dicts_test, transform=val_transforms, cache_rate=args.cache_rate)
    else:
        test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
    return test_loader, val_transforms


if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()