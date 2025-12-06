import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from math import pi, sqrt, ceil
import torch.nn.functional as F
import numpy as np
from matplotlib.path import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms.functional import center_crop, resize
from torchvision.io import read_image
from torch.nn.functional import interpolate
import decord
decord.bridge.set_bridge('torch')
import glob
import clip
from PIL import Image

# Patch-based Video dataset
class PatchVideoDataSet(Dataset):
    """
    Dataset that extracts patches from each frame.
    For a frame, we divide it into 8 patches and return:
    - Frame index (normalized)
    - Patch position (normalized x, y coordinates)
    - RGB patch data
    - CLIP embedding for that patch
    """
    def __init__(self, args):
        if os.path.isfile(args.data_path):
            self.video = decord.VideoReader(args.data_path)
            self.image_paths = [args.data_path] * len(self.video)
        else:
            self.image_paths = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]
            self.video = self.image_paths

        # Resize and crop parameters
        self.crop_list, self.resize_list = args.crop_list, args.resize_list
        first_frame = self.img_transform(self.img_load(0))
        self.frame_height, self.frame_width = first_frame.shape[-2:]
        self.final_size = self.frame_height * self.frame_width
        
        # Number of patches per frame (8 patches: 2x4 grid)
        self.num_patches_h = 2
        self.num_patches_w = 4
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch size
        self.patch_h = self.frame_height // self.num_patches_h
        self.patch_w = self.frame_width // self.num_patches_w
        
        self.clip_manager = CLIPManager(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_embeddings_cache = {}

    def img_load(self, idx):
        if isinstance(self.video, list):
            img = read_image(self.video[idx])
        else:
            img = self.video[idx].permute(-1,0,1)
        return img / 255.

    def img_transform(self, img):
        if self.crop_list != '-1': 
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
            if 'last' not in self.crop_list:
                img = center_crop(img, (crop_h, crop_w))
        if self.resize_list != '-1':
            if '_' in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split('_')]
                img = interpolate(img.unsqueeze(0), (resize_h, resize_w), mode='bicubic').squeeze(0)
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw, interpolation='bicubic')
        if 'last' in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))
        return img

    def __len__(self):
        # Total number of samples = num_frames * num_patches_per_frame
        return len(self.video) * self.num_patches

    def __getitem__(self, idx):
        # Determine which frame and which patch
        frame_idx = idx // self.num_patches
        patch_idx = idx % self.num_patches
        
        # Load and transform the full frame
        tensor_image = self.img_transform(self.img_load(frame_idx))
        
        # Compute patch position in the grid
        patch_row = patch_idx // self.num_patches_w
        patch_col = patch_idx % self.num_patches_w
        
        # Extract the patch from the frame
        y_start = patch_row * self.patch_h
        y_end = y_start + self.patch_h
        x_start = patch_col * self.patch_w
        x_end = x_start + self.patch_w
        
        patch_img = tensor_image[:, y_start:y_end, x_start:x_end]
        
        # Normalized frame index
        norm_frame_idx = float(frame_idx) / len(self.video)
        
        # Normalized patch position (center of the patch)
        norm_patch_x = (x_start + self.patch_w / 2) / self.frame_width
        norm_patch_y = (y_start + self.patch_h / 2) / self.frame_height
        
        # Get CLIP embedding for this patch
        clip_embed = None
        if frame_idx in self.clip_embeddings_cache:
            # Use cached CLIP embeddings
            all_patch_embeds = self.clip_embeddings_cache[frame_idx]
            clip_embed = all_patch_embeds[patch_idx]
        else:
            # Extract CLIP embeddings for all patches in this frame
            image_path = self.image_paths[frame_idx]
            if isinstance(self.video, decord.VideoReader):
                temp_img_path = f"/tmp/frame_{frame_idx}.png"
                img_to_save = self.img_load(frame_idx).permute(1, 2, 0).cpu().numpy() * 255
                Image.fromarray(img_to_save.astype('uint8')).save(temp_img_path)
                image_path = temp_img_path
            
            # Get CLIP embeddings for all patches
            all_patch_embeds = self.clip_manager.get_patch_embeddings_grid(
                image_path, self.num_patches_h, self.num_patches_w
            )
            self.clip_embeddings_cache[frame_idx] = all_patch_embeds
            clip_embed = all_patch_embeds[patch_idx]
            
            if isinstance(self.video, decord.VideoReader):
                os.remove(image_path)
        
        # Combine frame index and patch position as input coordinates
        # Input: [norm_frame_idx, norm_patch_x, norm_patch_y]
        input_coords = torch.tensor([norm_frame_idx, norm_patch_x, norm_patch_y], dtype=torch.float32)
        
        sample = {
            'img': patch_img,  # [3, patch_h, patch_w]
            'frame_idx': frame_idx,
            'patch_idx': patch_idx,
            'input_coords': input_coords,  # [3] - (t, x, y)
            'clip_embed': clip_embed,  # [512]
        }
        
        return sample


# Video dataset (keep original for compatibility)
class VideoDataSet(Dataset):
    def __init__(self, args):
        if os.path.isfile(args.data_path):
            self.video = decord.VideoReader(args.data_path)
            self.image_paths = [args.data_path] * len(self.video)
        else:
            self.image_paths = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]
            self.video = self.image_paths

        self.crop_list, self.resize_list = args.crop_list, args.resize_list
        first_frame = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)
        self.clip_manager = CLIPManager(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_embeddings_cache = {}

    def img_load(self, idx):
        if isinstance(self.video, list):
            img = read_image(self.video[idx])
        else:
            img = self.video[idx].permute(-1,0,1)
        return img / 255.

    def img_transform(self, img):
        if self.crop_list != '-1': 
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
            if 'last' not in self.crop_list:
                img = center_crop(img, (crop_h, crop_w))
        if self.resize_list != '-1':
            if '_' in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split('_')]
                img = interpolate(img.unsqueeze(0), (resize_h, resize_w), mode='bicubic').squeeze(0)
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw, interpolation='bicubic')
        if 'last' in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))
        return img

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        tensor_image = self.img_transform(self.img_load(idx))
        norm_idx = float(idx) / len(self.video)
        
        clip_embeds, clip_coords = None, None
        if idx in self.clip_embeddings_cache:
            clip_embeds, clip_coords = self.clip_embeddings_cache[idx]
        else:
            image_path = self.image_paths[idx]
            if isinstance(self.video, decord.VideoReader):
                temp_img_path = f"/tmp/frame_{idx}.png"
                img_to_save = self.img_load(idx).permute(1, 2, 0).cpu().numpy() * 255
                Image.fromarray(img_to_save.astype('uint8')).save(temp_img_path)
                image_path = temp_img_path

            clip_embeds, clip_coords = self.clip_manager.get_clip_embeddings(image_path)
            if clip_embeds is not None:
                self.clip_embeddings_cache[idx] = (clip_embeds, clip_coords)

            if isinstance(self.video, decord.VideoReader):
                os.remove(image_path)

        sample = {'img': tensor_image, 'idx': idx, 'norm_idx': norm_idx}
        if clip_embeds is not None:
            sample['clip_embeds'] = clip_embeds
            sample['clip_coords'] = torch.tensor(clip_coords, dtype=torch.float32)
        
        return sample


class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        conv = UpConv if kargs['dec_block'] else DownConv
        self.conv = conv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], strd=kargs['strd'], ks=kargs['ks'], 
            conv_type=kargs['conv_type'], bias=kargs['bias'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def Quantize_tensor(img_embed, quant_bit):
    out_min = img_embed.min(dim=1, keepdim=True)[0]
    out_max = img_embed.max(dim=1, keepdim=True)[0]
    scale = (out_max - out_min) / 2 ** quant_bit
    img_embed = ((img_embed - out_min) / scale).round()
    img_embed = out_min + scale * img_embed  
    return img_embed


def OutImg(x, out_bias='tanh'):
    if out_bias == 'sigmoid':
        return torch.sigmoid(x)
    elif out_bias == 'tanh':
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)


class DualHeadHNeRV(nn.Module):
    """
    Dual-head HNeRV model that takes (frame_idx, patch_x, patch_y) as input
    and outputs both RGB patch and CLIP embedding.
    """
    def __init__(self, args):
        super().__init__()
        self.embed = args.embed
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split('_')]
        enc_blks, dec_blks = [int(x) for x in args.num_blks.split('_')]

        # Input: 3 coordinates (frame_idx, patch_x, patch_y)
        if 'pe' in self.embed:
            input_dim = 3  # (t, x, y)
            ch_in = 2 * int(args.embed.split('_')[-1]) * input_dim  # PE for each coordinate
            self.pe_embed = PositionEncoding(args.embed, input_dim=input_dim)
            self.encoder = nn.Identity()
            self.fc_h, self.fc_w = [int(x) for x in args.fc_hw.split('_')]
        else:
            raise NotImplementedError("For patch-based dual-head model, please use positional encoding (pe)")

        # Shared decoder layers
        decoder_layers = []        
        ngf = args.fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = NeRVBlock(dec_block=False, conv_type='conv', ngf=ch_in, new_ngf=out_f, ks=0, strd=1, 
            bias=True, norm=args.norm, act=args.act)
        decoder_layers.append(decoder_layer1)
        
        for i, strd in enumerate(args.dec_strds):                         
            reduction = sqrt(strd) if args.reduce==-1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            for j in range(dec_blks):
                cur_blk = NeRVBlock(dec_block=True, conv_type=args.conv_type[1], ngf=ngf, new_ngf=new_ngf, 
                    ks=min(ks_dec1+2*i, ks_dec2), strd=1 if j else strd, bias=True, norm=args.norm, act=args.act)
                decoder_layers.append(cur_blk)
                ngf = new_ngf
        
        self.decoder = nn.ModuleList(decoder_layers)
        
        # Dual heads
        # RGB head: outputs RGB patch
        self.rgb_head = nn.Conv2d(ngf, 3, 3, 1, 1)
        
        # CLIP head: outputs CLIP embedding (512-dim)
        clip_dim = args.clip_dim if hasattr(args, 'clip_dim') else 512
        self.clip_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(),
            nn.Linear(ngf, ngf * 2),
            nn.ReLU(inplace=True),
            nn.Linear(ngf * 2, clip_dim)
        )
        
        self.out_bias = args.out_bias

    def forward(self, input_coords, input_embed=None):
        """
        Args:
            input_coords: [batch, 3] - (frame_idx, patch_x, patch_y)
            input_embed: Optional pre-computed embedding
        Returns:
            rgb_out: [batch, 3, H, W] - RGB patch
            clip_out: [batch, 512] - CLIP embedding
            embed_list: List of intermediate embeddings
            dec_time: Decoding time
        """
        if input_embed is not None:
            img_embed = input_embed
        else:
            # Apply positional encoding to input coordinates
            img_embed = self.pe_embed(input_coords).float()

        embed_list = [img_embed]
        dec_start = time.time()
        
        # First decoder layer
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        embed_list.append(output)
        
        # Remaining decoder layers
        for layer in self.decoder[1:]:
            output = layer(output) 
            embed_list.append(output)

        # RGB head
        rgb_out = OutImg(self.rgb_head(output), self.out_bias)
        
        # CLIP head
        clip_out = self.clip_head(output)
        # Normalize CLIP embedding
        clip_out = F.normalize(clip_out, dim=-1)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start

        return rgb_out, clip_out, embed_list, dec_time


class HNeRV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = args.embed
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split('_')]
        enc_blks, dec_blks = [int(x) for x in args.num_blks.split('_')]

        # BUILD Encoder LAYERS
        if 'pe' in self.embed:
            ch_in = 2 * int(args.embed.split('_')[-1])
            self.pe_embed = PositionEncoding(args.embed)
            self.encoder = nn.Identity()
            self.fc_h, self.fc_w = [int(x) for x in args.fc_hw.split('_')]
        else:
            enc_dim1, enc_dim2 = [int(x) for x in args.enc_dim.split('_')]
            c_out_list = [enc_dim1] * len(args.enc_strds)
            c_out_list[-1] = enc_dim2
            c_in_list = [enc_dim1] * len(args.enc_strds)
            
            ch_in = 3
            self.encoder = ConvNeXt(stage_blocks=enc_blks, strds=args.enc_strds, dims=c_out_list,
                drop_path_rate=0, in_chans=ch_in) if args.conv_type[0] == 'convnext' else ...
            
            if args.conv_type[0] != 'convnext':
                c_in_list[0] = ch_in
                encoder_layers = []
                for c_in, c_out, strd in zip(c_in_list, c_out_list, args.enc_strds):
                    encoder_layers.append(NeRVBlock(dec_block=False, conv_type=args.conv_type[0], ngf=c_in,
                     new_ngf=c_out, ks=ks_enc, strd=strd, bias=True, norm=args.norm, act=args.act))
                self.encoder = nn.Sequential(*encoder_layers)

            hnerv_hw = np.prod(args.enc_strds) // np.prod(args.dec_strds)
            self.fc_h, self.fc_w = hnerv_hw, hnerv_hw
            ch_in = enc_dim2

        # BUILD Decoder LAYERS  
        decoder_layers = []        
        ngf = args.fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = NeRVBlock(dec_block=False, conv_type='conv', ngf=ch_in, new_ngf=out_f, ks=0, strd=1, 
            bias=True, norm=args.norm, act=args.act)
        decoder_layers.append(decoder_layer1)
        for i, strd in enumerate(args.dec_strds):                         
            reduction = sqrt(strd) if args.reduce==-1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            for j in range(dec_blks):
                cur_blk = NeRVBlock(dec_block=True, conv_type=args.conv_type[1], ngf=ngf, new_ngf=new_ngf, 
                    ks=min(ks_dec1+2*i, ks_dec2), strd=1 if j else strd, bias=True, norm=args.norm, act=args.act)
                decoder_layers.append(cur_blk)
                ngf = new_ngf
        
        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(ngf, 3, 3, 1, 1) 
        self.out_bias = args.out_bias
        
        self.predict_clip = args.predict_clip if hasattr(args, 'predict_clip') else False
        if self.predict_clip:
            clip_dim = args.clip_dim if hasattr(args, 'clip_dim') else 512
            self.clip_head = nn.Sequential(
                nn.Conv2d(ngf, ngf * 2, 1, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(ngf * 2, clip_dim, 1, 1, 0)
            )

    def forward(self, input, input_embed=None, encode_only=False):
        if input_embed != None:
            img_embed = input_embed
        else:
            if 'pe' in self.embed:
                input = self.pe_embed(input[:,None]).float()
            img_embed = self.encoder(input)

        embed_list = [img_embed]
        dec_start = time.time()
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        embed_list.append(output)
        for layer in self.decoder[1:]:
            output = layer(output) 
            embed_list.append(output)

        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start

        clip_out = None
        if self.predict_clip:
            clip_out = self.clip_head(output)
            b, c, h, w = clip_out.shape
            clip_out = clip_out.view(b, c, -1)
            clip_out = F.normalize(clip_out, dim=1)
            clip_out = clip_out.view(b, c, h, w)

        return  img_out, embed_list, dec_time, clip_out


class HNeRVDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fc_h, self.fc_w = [torch.tensor(x) for x in [model.fc_h, model.fc_w]]
        self.out_bias = model.out_bias
        self.decoder = model.decoder
        self.head_layer = model.head_layer

    def forward(self, img_embed):
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        for layer in self.decoder[1:]:
            output = layer(output) 
        output = self.head_layer(output)

        return  OutImg(output, self.out_bias)


###################################  Basic layers  ###################################
class PositionEncoding(nn.Module):
    def __init__(self, pe_embed, input_dim=1):
        super(PositionEncoding, self).__init__()
        self.pe_embed = pe_embed
        self.input_dim = input_dim
        if 'pe' in pe_embed:
            lbase, levels = [float(x) for x in pe_embed.split('_')[-2:]]
            self.pe_bases = lbase ** torch.arange(int(levels)) * pi

    def forward(self, pos):
        """
        Args:
            pos: [batch, input_dim] or [batch, 1] tensor
        Returns:
            pe_embed: [batch, 2*levels*input_dim, 1, 1] tensor
        """
        if 'pe' in self.pe_embed:
            # pos shape: [batch, input_dim]
            if pos.dim() == 2 and pos.size(1) > 1:
                # Multi-dimensional input
                pe_list = []
                for i in range(pos.size(1)):
                    value_list = pos[:, i:i+1] * self.pe_bases.to(pos.device)
                    pe_list.append(torch.sin(value_list))
                    pe_list.append(torch.cos(value_list))
                pe_embed = torch.cat(pe_list, dim=-1)
            else:
                # Single-dimensional input
                value_list = pos * self.pe_bases.to(pos.device)
                pe_embed = torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)
            return pe_embed.view(pos.size(0), -1, 1, 1)
        else:
            return pos


class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)


def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = Sin()
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class DownConv(nn.Module):
    def __init__(self, **kargs):
        super(DownConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        if kargs['conv_type'] == 'pshuffel':
            self.downconv = nn.Sequential(
                nn.PixelUnshuffle(strd) if strd !=1 else nn.Identity(),
                nn.Conv2d(ngf * strd**2, new_ngf, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias'])
            )
        elif kargs['conv_type'] == 'conv':
            self.downconv = nn.Conv2d(ngf, new_ngf, ks+strd, strd, ceil(ks / 2), bias=kargs['bias'])
        elif kargs['conv_type'] == 'interpolate':
            self.downconv = nn.Sequential(
                nn.Upsample(scale_factor=1. / strd, mode='bilinear',),
                nn.Conv2d(ngf, new_ngf, ks+strd, 1, ceil((ks + strd -1) / 2), bias=kargs['bias'])
            )
        
    def forward(self, x):
        return self.downconv(x)


class UpConv(nn.Module):
    def __init__(self, **kargs):
        super(UpConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        if  kargs['conv_type']  == 'pshuffel':
            self.upconv = nn.Sequential(
                nn.Conv2d(ngf, new_ngf * strd * strd, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias']),
                nn.PixelShuffle(strd) if strd !=1 else nn.Identity(),
            )
        elif  kargs['conv_type']  == 'conv':
            self.upconv = nn.ConvTranspose2d(ngf, new_ngf, ks+strd, strd, ceil(ks / 2))
        elif  kargs['conv_type']  == 'interpolate':
            self.upconv = nn.Sequential(
                nn.Upsample(scale_factor=strd, mode='bilinear',),
                nn.Conv2d(ngf, new_ngf, strd + ks, 1, ceil((ks + strd -1) / 2), bias=kargs['bias'])
            )

    def forward(self, x):
        return self.upconv(x)


###################################  Transform input  ###################################
def RandomMask(height, width, points_num, scale=(0, 1)):
    polygon = [(x, y) for x,y in zip(np.random.randint(height * scale[0], height * scale[1], size=points_num), 
                             np.random.randint(width * scale[0], width * scale[1], size=points_num))]
    poly_path=Path(polygon)

    x, y = np.mgrid[:height, :width]
    coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))
    mask = poly_path.contains_points(coors).reshape(height, width)
    return 1 - torch.from_numpy(mask).float()


class TransformInput(nn.Module):
    def __init__(self, args):
        super(TransformInput, self).__init__()
        self.vid = args.vid
        if 'inpaint' in self.vid:
            self.inpaint_size = int(self.vid.split('_')[-1]) // 2

    def forward(self, img):
        inpaint_mask = torch.ones_like(img)
        if 'inpaint' in self.vid:
            gt = img.clone()
            h,w = img.shape[-2:]
            inpaint_mask = torch.ones((h,w)).to(img.device)
            for ctr_x, ctr_y in [(1/2, 1/2), (1/4, 1/4), (1/4, 3/4), (3/4, 1/4), (3/4, 3/4)]:
                ctr_x, ctr_y = int(ctr_x * h), int(ctr_y * w)
                inpaint_mask[ctr_x - self.inpaint_size: ctr_x + self.inpaint_size, ctr_y - self.inpaint_size: ctr_y + self.inpaint_size] = 0
            input = (img * inpaint_mask).clamp(min=0,max=1)
        else:
            input, gt = img, img

        return input, gt, inpaint_mask.detach()


###################################  ConvNeXt  ###################################
class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, stage_blocks=0, strds=[2,2,2,2], dims=[96, 192, 384, 768], 
            in_chans=3, drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.stage_num = len(dims)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, stage_blocks*self.stage_num)] 
        cur = 0
        for i in range(self.stage_num):
            if i > 0:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i]),
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=strds[i], stride=strds[i]),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                )                
            self.downsample_layers.append(downsample_layer)

            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(stage_blocks)]
            )
            self.stages.append(stage)
            cur += stage_blocks

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_list = []
        for i in range(self.stage_num):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_list.append(x)
        return out_list[-1]


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class CLIPManager(nn.Module):
    def __init__(self, patch_size=448, stride=224, device='cuda'):
        super().__init__()
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.model = None
        self.preprocess = None

    def _ensure_model_loaded(self):
        """Lazy load CLIP model on the current device"""
        if self.model is None:
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                device = f'cuda:{current_device}'
            else:
                device = 'cpu'
            self.model, self.preprocess = clip.load("ViT-B/32", device=device)
            self.model.eval()

    def get_patch_embeddings_grid(self, image_path, num_patches_h, num_patches_w):
        """
        Extract CLIP embeddings for patches in a regular grid.
        Args:
            image_path: Path to image
            num_patches_h: Number of patches in vertical direction
            num_patches_w: Number of patches in horizontal direction
        Returns:
            List of embeddings, one per patch (row-major order)
        """
        self._ensure_model_loaded()
        
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        patch_h = height // num_patches_h
        patch_w = width // num_patches_w
        
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y = i * patch_h
                x = j * patch_w
                patch = image.crop((x, y, x + patch_w, y + patch_h))
                # Resize patch to CLIP input size (224x224)
                patch = patch.resize((224, 224), Image.BICUBIC)
                patches.append(patch)
        
        preprocessed_patches = torch.stack([self.preprocess(patch) for patch in patches])
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            preprocessed_patches = preprocessed_patches.to(f'cuda:{current_device}')
        
        with torch.no_grad():
            embeddings = self.model.encode_image(preprocessed_patches)
        
        # Return as list of individual embeddings
        return [embeddings[i].cpu() for i in range(len(patches))]

    def get_clip_embeddings(self, image_path):
        """Legacy method for compatibility"""
        self._ensure_model_loaded()
        
        image = Image.open(image_path).convert("RGB")
        patches, coords = self.extract_patches(image)
        if not patches:
            return None, None
        
        preprocessed_patches = torch.stack([self.preprocess(patch) for patch in patches])
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            preprocessed_patches = preprocessed_patches.to(f'cuda:{current_device}')
        
        with torch.no_grad():
            embeddings = self.model.encode_image(preprocessed_patches)
        
        return embeddings.cpu(), coords

    def extract_patches(self, image):
        patches = []
        coords = []
        width, height = image.size
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
                patches.append(patch)
                coords.append((x, y))
        return patches, coords
