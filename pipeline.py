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
out_dir = 'data/output_test_2'
is_random = True
recreate = True
downsample = True
train_portion = 0.8
valid_portion = 0.15
test_portion = 0.05
batch_size = 4
learning_rate = 0.001
check_loss = 10.00
epochs = 500
# ----------------------------------------------------------------------

train_size = int(num_frames * num_samples * train_portion)
valid_size = int(num_frames * num_samples * valid_portion)
test_size = int(num_frames * num_samples * test_portion)
len_video = num_frames -1
total = ((num_frames-1) * num_samples) -1

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

class MyDataset(Dataset):
    def __init__(self, data_path, downsample):
        self.data_path = data_path
        self.downsample = downsample
        
    def __len__(self):
        return total
        
    def __getitem__(self, idx):
        folder_idx = idx // len_video 
        # print(folder_idx)
        subfolder_idx = idx % len_video
        # print(subfolder_idx)
        frame_folder = f"{self.data_path}/{folder_idx:04d}/com"
        gt_folder = f"{self.data_path}/{folder_idx:04d}/pha"
        pred_folder = f"{self.data_path}/{folder_idx:04d}/result"
        
        frame1_path = f"{frame_folder}/{subfolder_idx:04d}.png"
        frame2_path = f"{frame_folder}/{subfolder_idx + 1:04d}.png"
        pred2_path = f"{pred_folder}/{subfolder_idx + 1:04d}.png"
        gt_path = f"{gt_folder}/{subfolder_idx + 1:04d}.png"
        
        frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
        pred2 = cv2.imread(pred2_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        h, w = 2160, 3840

        if self.downsample == True:
            new_h = int(h * 0.25)
            new_w = int(w * 0.25)
        
            frame1 = cv2.resize(frame1, (new_w, new_h))
            frame2 = cv2.resize(frame2, (new_w, new_h))
            pred2 =  cv2.resize(pred2, (new_w, new_h))
            gt = cv2.resize(gt, (new_w, new_h))

        x = np.stack([frame1, frame2, pred2], axis=2)
        x = x.transpose(2, 0, 1)
        x = torch.tensor(x, dtype=torch.float32) / 255.0
        
        y = torch.tensor(gt, dtype=torch.float32) / 255.0
        
        return x, y

class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64, 128)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)
    def forward(self, x):
        blockOutputs = []
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        return blockOutputs
    
class Decoder(Module):

    def __init__(self, channels=(128, 64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)])
    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        return x
    def crop(self, encFeatures, x):
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        return encFeatures
    
class UNet(Module):
    def __init__(self, encChannels=(3, 16, 32, 64, 128),
                 decChannels=(128, 64, 32, 16),
                 nbClasses=1, retainDim=True,
                 outSize=(540,  960)):
        super().__init__()
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.sigmoid = nn.Sigmoid() # Add a sigmoid layer
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        encFeatures = self.encoder(x)
        decFeatures = self.decoder(encFeatures[::-1][0],encFeatures[::-1][1:])
        map = self.sigmoid(self.head(decFeatures))
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
        return map

def generate_random_numbers(num_frames, num_samples, n):
    upper_limit = ((num_frames - 1) * num_samples) - 1
    random_numbers = []
    for _ in range(n):
        random_numbers.append(random.randint(0, upper_limit))
    print(random_numbers)
    return random_numbers

def visualize(n, dataset):
    indices = generate_random_numbers(num_frames, num_samples, n)
    fig = plt.figure(figsize=(24, 12))
    grid = ImageGrid(fig, 111, nrows_ncols=(n, 4), axes_pad=0.3)
    for i, idx in enumerate(indices):
        x, y = dataset[idx]
        ax = grid[i*4]
        ax.imshow(x[0,...], cmap='gray')
        ax.set_title(f'Frame {idx}')
        ax = grid[i*4+1]
        ax.imshow(x[1,...], cmap='gray')
        ax.set_title(f'Frame {idx+1}')
        ax = grid[i*4+2]
        ax.imshow(x[2,...], cmap='gray')
        ax.set_title(f'Predicted frame {idx+1}')
        ax = grid[i*4+3]
        ax.imshow(y, cmap='gray')
        ax.set_title(f'GT {idx+1}')
    plt.savefig("dataloader_example.png")
    plt.imshow

def create_dataloader(out_dir, vis):
    dataset = MyDataset(out_dir, downsample)

    if vis == True:
        visualize(n, dataset)

    x, y = dataset[1] 
    print("Shape of input x:", x.shape) 
    print("Shape of output y:", y.shape) 

    print("Train size: ", range(train_size))
    print("Validation size: ",range(train_size, train_size+valid_size))
    print("Test size: ",range(train_size+valid_size, total))

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    valid_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size+valid_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size+valid_size, train_size+valid_size+test_size))

    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_set = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  

    return train_set, valid_set, test_set

def create_dataset(video_dir, background_dir, num_frames, num_samples, out_dir, is_random, recreate, downsample, vis):
    if recreate == True:
        generate_bg(video_dir, background_dir, num_frames, num_samples, out_dir, is_random)
        generate_afgan_masks(out_dir)
        return create_dataloader(out_dir, vis)
    else:
        return create_dataloader(out_dir, vis)
 
def train_model(epochs, vis):

    train_set, valid_set, test_set =  create_dataset(video_dir, background_dir, num_frames, num_samples, out_dir, is_random, recreate, downsample, vis)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    unet = UNet().to(DEVICE)
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"MyModel has {num_params} trainable parameters")
    lossFunc = nn.BCELoss()
    opt = optim.Adam(unet.parameters(), lr= learning_rate)
    trainSteps = len(train_set) /batch_size
    validSteps = len(valid_set) / batch_size
    H = {"train_loss": [], "test_loss": []}
    print("[INFO] training the network...")
    startTime = time.time()
    checkloss = 10.0
    for e in tqdm(range(epochs)):
        unet.train()
        totalTrainLoss = 0
        totalTestLoss = 0
        for (i, (x, y)) in enumerate(train_set):
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            print(i)
            pred = unet(x)
            loss = lossFunc(pred, y.unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            totalTrainLoss += loss
        with torch.no_grad():
            unet.eval()
            for (x, y) in valid_set:
                (x, y) = (x.to(DEVICE), y.to(DEVICE))
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y.unsqueeze(1))
                
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValidLoss = totalTestLoss / validSteps

        if avgValidLoss < checkloss:
            checkloss = avgValidLoss
            print(checkloss)
            print(avgValidLoss)
            torch.save(unet, 'u-net_model.pt')
            
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgValidLoss.cpu().detach().numpy())
        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgValidLoss))
    endTime = time.time()

    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    torch.save(unet, 'u-net_model_f.pt')


generate_bg(video_dir, background_dir, num_frames, num_samples, out_dir, is_random)
# june_20th = 'data/output_test/train'
# generate_afgan_masks(june_20th)