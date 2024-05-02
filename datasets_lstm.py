import os
from glob import glob
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from pydantic import BaseSettings, BaseModel
from PIL import Image
from typing import Any, Iterable, Dict, List
import itertools
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

class DataConfig(BaseSettings):
    video_matting_root:str
    mode:str
    downsample_width:int = 640
    downsample_height:int = 480
    components:List[str] = ['com', 'result', 'pha']
    window_size:int = 2
    class Config:
        case_sensitive = True


class FrameOnDisk(BaseModel):
    video_id:str
    frame_id:str
    components:Dict[str,str]

class InFrame(BaseModel):
    image:Tensor
    target:Tensor
    base_pred:Tensor
    class Config:
        arbitrary_types_allowed = True

def get_id(filename:str):
    parts = filename.split('/')
    basename = parts[-1].split('.')[0]
    # print(parts)
    return parts[1], basename

def make_frame_components(filenames:Iterable[str], config:DataConfig):
    components = dict()
    for filename in filenames:
        parts = filename.split("/")
        k = parts[2]
        if k in config.components:
            components[k] = filename
    # print(components)
    return components

def load_frame_metadata(config:DataConfig)->List[FrameOnDisk]:
    pattern = f'{config.mode}*/**/*.png'
    filelist = sorted(glob(pattern, root_dir=config.video_matting_root, recursive=True), key=get_id)
    frame_files = []
    for ((video_id, frame_id), filenames) in itertools.groupby(filelist, get_id):
        frame_files.append(
            FrameOnDisk(video_id=video_id,
                        frame_id=frame_id,
                        components=make_frame_components(filenames, config)))
    return frame_files

def iter_segments(all_frame:List[FrameOnDisk], window_size):
    for video_id, video_seq in itertools.groupby(all_frame, key=lambda x: x.video_id):
        video_seq = list(video_seq)
        # print(video_id)
        # print(video_seq)
        # print(len(video_seq))
        for i in range(len(video_seq) - window_size + 1):
            segment = video_seq[i:i+window_size]
            yield segment

def load_image(config:DataConfig, filename:str, toTensor)->Tensor:
    filename = os.path.join(config.video_matting_root, filename)
    # print(filename)
    with open(filename, 'rb') as f:
        im = Image.open(f)
        im = im.resize((config.downsample_width, config.downsample_height))
        im = toTensor(im)
        return im

class FrameDataset(pl.LightningDataModule):
    def __init__(self, config:DataConfig):
        super().__init__()
        self.config = config
        self.segment_metdata = list(iter_segments(load_frame_metadata(config), config.window_size))
        self.toTensor = ToTensor()
    def __len__(self):
        return len(self.segment_metdata)
    
    def load_single_segment(self, segment:List[FrameOnDisk]):
        # (segment, channel, h, w)
        images = torch.stack([load_image(self.config, frame.components['com'], self.toTensor) for frame in segment])
        # (1, h, w)
        base_pred = load_image(self.config, segment[-1].components['result'], self.toTensor)
        # (1, h, w)
        target = load_image(self.config, segment[-1].components['pha'], self.toTensor)
 
        # base_pred = base_pred.unsqueeze(1)

        individual_tensors = torch.split(images, 1, dim=0)
        concatenated_images = []
        concatenated_image = torch.stack([individual_tensors[0].squeeze(0), individual_tensors[1].squeeze(0), individual_tensors[2].squeeze(0), individual_tensors[3].squeeze(0)], dim=1)
        # print(concatenated_image.shape)
        # print((concatenated_image.squeeze(0)).shape)
        # print(base_pred.shape)
        # print(target.shape)
        return{

            'input': concatenated_image.squeeze(0),
            'base': base_pred,
            'output': target
        }
    
    def __getitem__(self, index):
        segment = self.segment_metdata[index]
        if isinstance(index, int):
            return self.load_single_segment(segment)
        else:
            raise Exception('Batch not supported')

def data_loader(train_ds, valid_ds, test_ds, batch_size, shuffle=True):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }
