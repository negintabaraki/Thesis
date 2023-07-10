import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import ImageGrid
sys.path.append('segmentation-models')
import torch
import argparse
import torch.nn as nn
import net
import cv2
import os
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F
import numpy as np
import time
import pickle
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from networks.transforms import trimap_transform, normalise_image
from networks.models import build_model
from deploy_gen_trimap import inference_img_whole
from numpy import asarray
from numpy import savez_compressed
from numpy import load
import argparse
import os
import pims
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
from torch.nn import BatchNorm2d
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

# -------------------- here are my hyper-parameters --------------------
video_dir = 'data/train'
background_dir = 'data/Backgrounds/'
num_frames = 200       
num_samples = 20
n = 4
vis = False
#defien your desired ouput directory here
out_dir = 'data/output'
# you can define if you want teh backgrounds to be shuffled for each frame or not. 
# for creating training and validation data you need this flag to be True and for creating testing data this flag should be False.
is_random = True
recreate = True
# ----------------------------------------------------------------------

def gen_dataset(imgdir):
    sample_set = []
    img_ids = os.listdir(imgdir)
    img_ids.sort()
    cnt = len(img_ids)
    cur = 1
    for img_id in img_ids:
        img_name = os.path.join(imgdir, img_id)      
        assert(os.path.exists(img_name))
        sample_set.append(img_name)
    return sample_set
def my_torch_load(fname):
    try:
        ckpt = torch.load(fname)
        return ckpt
    except Exception as e:
        print("Load Error:{}\nTry Load Again...".format(e))
        class C:
            pass
        def c_load(ss):
            return pickle.load(ss, encoding='latin1')
        def c_unpickler(ss):
            return pickle.Unpickler(ss, encoding='latin1')
        c = C
        c.load = c_load
        c.Unpickler = c_unpickler
        ckpt = torch.load(args.resume, encoding='latin1')
        return ckpt
def read_image(name):
    return (cv2.imread(name) / 255.0)[:, :, ::-1]

def fixed_background(video_dir, background_dir, num_frames, num_samples, out_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--resize', type=int, default=None, nargs=2)
    parser.add_argument('--extension', type=str, default='.png')
    args = parser.parse_args()
        
    random.seed(10)

    videomatte_filenames = [(clipname, sorted(os.listdir(os.path.join(video_dir, 'fgr', clipname)))) 
                            for clipname in sorted(os.listdir(os.path.join(video_dir, 'fgr')))]

    background_filenames = os.listdir(background_dir)
    random.shuffle(background_filenames)

    for i in range(num_samples):
        
        clipname, framenames = videomatte_filenames[i % len(videomatte_filenames)]
        
        out_path = os.path.join(out_dir, str(i).zfill(4))
        os.makedirs(os.path.join(out_path, 'fgr'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'pha'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'com'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'bgr'), exist_ok=True)
        
        x = random.randint(0, 190)

        with Image.open(os.path.join(background_dir, background_filenames[x])) as bgr:
            bgr = bgr.convert('RGB')

        
        base_t = random.choice(range(len(framenames) - num_frames))

        for t in tqdm(range(num_frames), desc=str(i).zfill(4)):

            x = random.randint(0, 190)

            with Image.open(os.path.join(video_dir, 'fgr', clipname, framenames[base_t + t])) as fgr, \
                Image.open(os.path.join(video_dir, 'pha', clipname, framenames[base_t + t])) as pha:
                
                fgr = fgr.convert('RGB')
                pha = pha.convert('L')
                
                if args.resize is not None:
                    fgr = fgr.resize(args.resize, Image.BILINEAR)
                    pha = pha.resize(args.resize, Image.BILINEAR)
                    
                
                if i // len(videomatte_filenames) % 2 == 1:
                    fgr = fgr.transpose(Image.FLIP_LEFT_RIGHT)
                    pha = pha.transpose(Image.FLIP_LEFT_RIGHT)
                
                fgr.save(os.path.join(out_path, 'fgr', str(t).zfill(4) + args.extension))
                pha.save(os.path.join(out_path, 'pha', str(t).zfill(4) + args.extension))
            
            if t == 0:
                bgr = bgr.resize(fgr.size, Image.BILINEAR)
                bgr.save(os.path.join(out_path, 'bgr', str(t).zfill(4) + args.extension))
            else:
                os.symlink(str(0).zfill(4) + args.extension, os.path.join(out_path, 'bgr', str(t).zfill(4) + args.extension))
            
            pha = np.asarray(pha).astype(float)[:, :, None] / 255
            com = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(bgr) * (1 - pha)))
            com.save(os.path.join(out_path, 'com', str(t).zfill(4) + args.extension))
def rand_background(video_dir, background_dir, num_frames, num_samples, output_dir):

    parser = argparse.ArgumentParser()
    parser.add_argument('--resize', type=int, default=None, nargs=2)
    parser.add_argument('--extension', type=str, default='.png')
    args = parser.parse_args()

    
    random.seed(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    videomatte_filenames = [(clipname, sorted(os.listdir(os.path.join(video_dir, 'fgr', clipname)))) 
                            for clipname in sorted(os.listdir(os.path.join(video_dir, 'fgr')))]

    background_filenames = os.listdir(background_dir)
    random.shuffle(background_filenames)

    for i in range(num_samples):
        
        clipname, framenames = videomatte_filenames[i % len(videomatte_filenames)]
        
        out_path = os.path.join(output_dir, str(i).zfill(4))
        os.makedirs(os.path.join(out_path, 'fgr'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'pha'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'com'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'bgr'), exist_ok=True)

        base_t = random.choice(range(len(framenames) - num_frames))
        for t in tqdm(range(num_frames), desc=str(i).zfill(4)):
            x = random.randint(0, 190)
            with Image.open(os.path.join(video_dir, 'fgr', clipname, framenames[base_t + t])) as fgr, \
                Image.open(os.path.join(video_dir, 'pha', clipname, framenames[base_t + t])) as pha:
                with Image.open(os.path.join(background_dir, background_filenames[random.randint(0, len(background_filenames) - 1)])) as bgr:
                    bgr = bgr.convert('RGB')
                    bgr = bgr.resize(fgr.size, Image.BILINEAR)

                fgr = fgr.convert('RGB')
                pha = pha.convert('L')

                if args.resize is not None:
                    fgr = fgr.resize(args.resize, Image.BILINEAR)
                    pha = pha.resize(args.resize, Image.BILINEAR)
                    bgr = bgr.resize(fgr.size, Image.BILINEAR)  
                    
                if i // len(videomatte_filenames) % 2 == 1:
                    fgr = fgr.transpose(Image.FLIP_LEFT_RIGHT)
                    pha = pha.transpose(Image.FLIP_LEFT_RIGHT)
                
                fgr.save(os.path.join(out_path, 'fgr', str(t).zfill(4) + args.extension))
                pha.save(os.path.join(out_path, 'pha', str(t).zfill(4) + args.extension))
            
                pha = np.asarray(pha).astype(float)[:, :, None] / 255
                com = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(bgr) * (1 - pha)))
                com.save(os.path.join(out_path, 'com', str(t).zfill(4) + args.extension))

def generate_bg(video_dir, background_dir, num_frames, num_samples, out_dir, is_random):
    if is_random == True:
        rand_background(video_dir, background_dir, num_frames, num_samples, out_dir)
    else:
        fixed_background(video_dir, background_dir, num_frames, num_samples, out_dir)

def generate_afgan_masks(root_dir):
    
    seg_path = "matting-pretrained-weights/p3m_seg_branch_UnetPlusPlus.ckpt"
    trimap_path = "matting-pretrained-weights/p3m_tri_branch.ckpt"
    weights = "matting-pretrained-weights//FBA.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    directories = []
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            com_folder = os.path.join(root, dir_name, "com")
            if os.path.isdir(com_folder):
                directories.append(os.path.join(root, dir_name))

    directories.sort()

    for directory in directories:
        print(directory)

        imgDir = os.path.join(directory, "com/")
        outAlphaDir = os.path.join(directory, "mask")
        outSegDir = os.path.join(directory, "seg")
        outTriDir = os.path.join(directory, "trimap")
        resultDir = os.path.join(directory, "result")

        list = []
        dataset = gen_dataset(imgDir)

        if not os.path.exists(outAlphaDir):
            os.makedirs(outAlphaDir)
        if not os.path.exists(outSegDir):
            os.makedirs(outSegDir)
        if not os.path.exists(outTriDir):
            os.makedirs(outTriDir)


        cnt = len(dataset)
        cur = 0
        t0 = time.time()
        
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.cuda = True
        args.stage = 1
        args.crop_or_resize = "whole"
        args.max_size = 1600
    
        seg_model = net.P3MSegBranchUPP()
        seg_model = seg_model.load_from_checkpoint(checkpoint_path = seg_path, map_location=device)
        seg_model.eval()
        seg_model = seg_model.cuda()
        
        trimap_model = net.P3MTrimapGenerationBranch()
        trimap_model = trimap_model.load_from_checkpoint(checkpoint_path = trimap_path, map_location=device)
        trimap_model.eval()
        trimap_model = trimap_model.cuda()

        model = build_model(weights)
        model.eval().cuda()

        for img_path in dataset:
            _, image_id = os.path.split(img_path)
            
            original_img = cv2.imread(img_path, 0)

            img = read_image(img_path)

            img_info = (img_path.split('/')[-1], img.shape[0], img.shape[1])

            cur += 1
            print('[{}/{}] {}'.format(cur, cnt, img_info[0]))

            with torch.no_grad():
                torch.cuda.empty_cache()
                origin_pred_mattes, seg, trimap = inference_img_whole(args, model, seg_model, trimap_model, img, original_img)

            pred_mattes = (origin_pred_mattes * 255).astype(np.uint8)
            seg = (seg * 255).astype(np.uint8)
            trimap = (trimap * 255).astype(np.uint8)

            pred_mattes[trimap == 255] = 255
            pred_mattes[trimap == 0  ] = 0

            seg_id = os.path.splitext(image_id)[0] + ".png"
            tri_id = os.path.splitext(image_id)[0] + ".png"

            cv2.imwrite(os.path.join(outAlphaDir, image_id), pred_mattes)
            cv2.imwrite(os.path.join(outSegDir, seg_id), seg)
            cv2.imwrite(os.path.join(outTriDir, tri_id), trimap)


            origin_pred_mattes[trimap == 255] = 1.
            origin_pred_mattes[trimap == 0  ] = 0.

            origin_pred_mattes = (origin_pred_mattes * 255).astype(np.uint8)
            res = origin_pred_mattes.copy()

            res[trimap == 255] = 255
            res[trimap == 0  ] = 0

            if not os.path.exists(resultDir):
                os.makedirs(resultDir)
                
            cv2.imwrite(os.path.join(resultDir, img_info[0]), res)

generate_bg(video_dir, background_dir, num_frames, num_samples, out_dir, is_random)
generate_afgan_masks(out_dir)
