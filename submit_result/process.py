#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import os
import time
from pathlib import Path
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import torch
import argparse
from efficientnet_pytorch import EfficientNet
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import (PreprocessingSettings, Sample,
                                      crop_or_pad, resample_img)
from report_guided_annotation import extract_lesion_candidates
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


#from segmentation.model import itunet_2d

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., num_patches = None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, patch_size = 16, depth = 2, heads = 8,  dropout = 0.5, attention = Attention):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.outsize = (image_height // patch_size, image_width// patch_size)
        h = image_height // patch_height
        w = image_width // patch_width
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        mlp_dim = out_channels * 2
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, out_channels))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(out_channels, attention(out_channels, heads = heads, dim_head = out_channels // heads, dropout = dropout, num_patches=(h,w))),
                PreNorm(out_channels, FeedForward(out_channels, mlp_dim, dropout = dropout))
            ]))
        self.re_patch_embedding = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = 1, p2 = 1, h = h)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, img):
        x = self.patch_embeddings(img)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        x = self.dropout(embeddings)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.re_patch_embedding(x)
        return F.interpolate(x, self.outsize)

class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv2D,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#-------------------------------------------

class Down2D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder):
        super(Down2D,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_builder(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

#-------------------------------------------

class Up2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder):
        super(Up2D,self).__init__()

        self.conv = conv_builder(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1,scale_factor=2, mode='bilinear', align_corners=False)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#-------------------------------------------

class Tail2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tail2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#-------------------------------------------

class UNet(nn.Module):
    def __init__(self, stem, down, up, tail, width, conv_builder, n_channels=1, n_classes=2, dropout_flag=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.width = width
        self.dropout_flag = dropout_flag
        factor = 2 

        self.inc = stem(n_channels, width[0])
        self.down1 = down(width[0], width[1], conv_builder)
        self.down2 = down(width[1], width[2], conv_builder)
        self.down3 = down(width[2], width[3], conv_builder)
        self.down4 = down(width[3], width[4] // factor, conv_builder)
        self.up1 = up(width[4], width[3] // factor, conv_builder)
        self.up2 = up(width[3], width[2]// factor, conv_builder)
        self.up3 = up(width[2], width[1] // factor, conv_builder)
        self.up4 = up(width[1], width[0], conv_builder)
        self.dropout = nn.Dropout(p=0.5)
        self.outc = tail(width[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.dropout_flag:
            x = self.dropout(x)
        logits = self.outc(x)
        return logits

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x


class ITUNet_2d(nn.Module):
    def __init__(self, stem, down, up, tail, width, conv_builder,image_size = 128, transformer_depth = 18, n_channels=1, n_classes=2, dropout_flag=True):
        super(ITUNet_2d, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.width = width
        self.dropout_flag = dropout_flag
        factor = 2 

        self.transblock = TransformerBlock(n_channels, width[-1] // factor, image_size, depth = transformer_depth)
        self.vision_0 = UpConv(width[1], width[0])
        self.vision_1 = UpConv(width[2], width[1])
        self.vision_2 = UpConv(width[3], width[2])
        self.vision_3 = UpConv(width[-1] // factor, width[3])

        self.inc = stem(n_channels, width[0])
        self.down1 = down(width[0], width[1], conv_builder)
        self.down2 = down(width[1], width[2], conv_builder)
        self.down3 = down(width[2], width[3], conv_builder)
        self.down4 = down(width[3], width[4] // factor, conv_builder)
        self.up1 = up(width[4], width[3] // factor, conv_builder)
        self.up2 = up(width[3], width[2]// factor, conv_builder)
        self.up3 = up(width[2], width[1] // factor, conv_builder)
        self.up4 = up(width[1], width[0], conv_builder)
        self.dropout = nn.Dropout(p=0.5)
        self.outc = tail(width[0], n_classes)

        self.conv1x1_1 = nn.Conv2d(width[1] // factor, n_classes, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(width[2] // factor, n_classes, kernel_size=1, stride=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(width[3] // factor, n_classes, kernel_size=1, stride=1, padding=0)
        self.conv1x1_4 = nn.Conv2d(width[4] // factor, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        trans = self.transblock(x)
        trans_4 = trans
        trans_3 = self.vision_3(trans_4)
        trans_2 = self.vision_2(trans_3)
        trans_1 = self.vision_1(trans_2)
        trans_0 = self.vision_0(trans_1)
        x1 = self.inc(x)   # 32
        x1 = x1 + trans_0
        x2 = self.down1(x1)   # 64
        x2 = x2 + trans_1
        x3 = self.down2(x2)  # 128
        x3 = x3 + trans_2
        x4 = self.down3(x3)  #256
        x4 = x4 + trans_3
        x5 = self.down4(x4)  #512
        x5 = x5 + trans_4

        out4 = self.conv1x1_4(x5) 
        x = self.up1(x5, x4)  # 128
        out3 = self.conv1x1_3(x)
        x = self.up2(x, x3)
        out2 = self.conv1x1_2(x)
        x = self.up3(x, x2)
        out1 = self.conv1x1_1(x)
        x = self.up4(x, x1)
        
        if self.dropout_flag:
            x = self.dropout(x)
        logits = self.outc(x)
        # return logits
        return [logits, out1, out2, out3, out4]


def itunet_2d(**kwargs):
    return ITUNet_2d(stem=DoubleConv2D,
                down=Down2D,
                up=Up2D,
                tail=Tail2D,
                width=[32,64,128,256,512],
                conv_builder=DoubleConv2D,
                **kwargs)

def unet(**kwargs):
    return UNet(stem=DoubleConv2D,
                down=Down2D,
                up=Up2D,
                tail=Tail2D,
                width=[32,64,128,256,512],
                conv_builder=DoubleConv2D,
                **kwargs)

class csPCaAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy trained baseline U-Net model from
    https://github.com/DIAGNijmegen/picai_baseline as a
    grand-challenge.org algorithm.
    """

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        parser = argparse.ArgumentParser()
        parser.add_argument('--caseid', type=str, default="10017")
        args, _ = parser.parse_known_args()
        self.caseid = args.caseid

        # set expected i/o paths in gc env (image i/p, algorithms, prediction o/p)
        # see grand-challenge.org/algorithms/interfaces/ for expected path per i/o interface
        # note: these are fixed paths that should not be modified
        self.start_time = time.time()

        # directory to model weights
        # change this to the weights of 4 folds model
        self.algorithm_weights_dir = Path("/model/weights/")
        # self.algorithm_weights_dir = Path("./weights/")

        #path to image files
        self.image_input_dirs = [
             f"/input/{self.caseid}/t2w/",
             f"/input/{self.caseid}/adc/",
             f"/input/{self.caseid}/hbv/"
            #"/input/images/transverse-t2-prostate-mri/",
            #"/input/images/transverse-adc-prostate-mri/",
            #"/input/images/transverse-hbv-prostate-mri/",
            # "/input/images/coronal-t2-prostate-mri/",  # not used in this algorithm
            # "/input/images/sagittal-t2-prostate-mri/"  # not used in this algorithm
        ]
        self.image_input_dir = f"/input/{self.caseid}/"
        self.pattern = [
            "*_t2w.mha",
            "*_adc.mha",
            "*_hbv.mha"
        ]
        # self.image_input_dirs = [
        #     "./test/images/transverse-t2-prostate-mri/",
        #     "./test/images/transverse-adc-prostate-mri/",
        #     "./test/images/transverse-hbv-prostate-mri/",
        #     # "/input/images/coronal-t2-prostate-mri/",  # not used in this algorithm
        #     # "/input/images/sagittal-t2-prostate-mri/"  # not used in this algorithm
        # ]
        #self.image_input_paths = [list(Path(x).glob("*.mha"))[0] for x in self.image_input_dirs]
        self.image_input_paths = [list(Path(self.image_input_dir).glob(x))[0] for x in self.pattern]
        print(self.image_input_paths)
        # load clinical information
        # with open("./test/clinical-information-prostate-mri.json") as fp:
        #     self.clinical_info = json.load(fp)

        # # path to output files
        # self.detection_map_output_path = Path("./test/cspca-detection-map/cspca_detection_map.mha")
        # self.case_level_likelihood_output_file = Path("./test/cspca-case-level-likelihood.json")

        # load clinical information
        #with open("/input/clinical-information-prostate-mri.json") as fp:
        #    self.clinical_info = json.load(fp)

        # path to output files
        self.detection_map_output_path = Path(f"/output/images/cspca-detection-map/{self.caseid}_cspca_detection_map.mha")
        self.case_level_likelihood_output_file = Path(f"/output/{self.caseid}_cspca-case-level-likelihood.json")

        # create output directory
        self.detection_map_output_path.parent.mkdir(parents=True, exist_ok=True)

        # define compute used for training/inference ('cpu' or 'cuda')
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # define input data specs [image shape, spatial res, num channels, num classes]
        self.img_spec = {
            'image_shape': [20, 256, 256],
            'spacing': [3.0, 0.5, 0.5],
            'num_channels': 3,
            'num_classes': 2,
        }

        # load trained algorithm architecture + weights
        self.models = []
        model_folds = [range(5)]

        # for each given architecture
        for folds in model_folds:

            # for each trained 5-fold instance of a given architecture
            for fold in folds:
                # path to trained weights for this architecture + fold (e.g. 'unet_F4.pt')
                weight_path = self.algorithm_weights_dir / f'F{fold}.pth'

                # skip if model was not trained for this fold
                if not os.path.exists(weight_path):
                    continue

                # define the model specifications used for initialization at train-time
                # note: if the default hyperparam listed in picai_baseline was used,
                # passing arguments 'image_shape', 'num_channels', 'num_classes' and
                # 'model_type' via function 'get_default_hyperparams' is enough.
                # otherwise arguments 'model_strides' and 'model_features' must also
                # be explicitly passed directly to function 'neural_network_for_run'

                model = itunet_2d(n_channels=3,n_classes=3, image_size= tuple([256,256]), transformer_depth = 20)

                # load trained weights for the fold
                checkpoint = torch.load(weight_path,map_location=self.device)
                model.load_state_dict(checkpoint['state_dict'])
                model.to(self.device)
                self.models += [model]
                print("Complete.")
                print("-"*100)

        # path to trained weights for this architecture + fold (e.g. 'unet_F4.pt')
        self.cls_models = []
        model_folds = [range(5)]
        # weight_path = self.algorithm_weights_dir / 'CLS_F0.pth'
        for folds in model_folds:
    
            # for each trained 5-fold instance of a given architecture
            for fold in folds:
                # path to trained weights for this architecture + fold (e.g. 'unet_F4.pt')
                weight_path = self.algorithm_weights_dir / f'CLS_F{fold}.pth'

                # skip if model was not trained for this fold
                if not os.path.exists(weight_path):
                    continue

                model = EfficientNet.from_name(model_name='efficientnet-b5')
                num_ftrs = model._fc.in_features
                model._fc = torch.nn.Linear(num_ftrs, 3)
                # load trained weights for the fold
                checkpoint = torch.load(weight_path,map_location=self.device)
                model.load_state_dict(checkpoint['state_dict'])
                model.to(self.device)
                self.cls_models += [model]
        print("Complete.")
        print("-"*100)

        # display error/success message
        if len(self.models) == 0:
            raise Exception("No models have been found/initialized.")
        else:
            print(f"Success! {len(self.models)} model(s) have been initialized.")
            print("-"*100)

    # generate + save predictions, given images
    def predict(self):

        print("Preprocessing Images ...")

        # read images (axial sequences used for this example only)
        sample = Sample(
            scans=[
                sitk.ReadImage(str(path))
                for path in self.image_input_paths
            ],
            settings=PreprocessingSettings(
                matrix_size=self.img_spec['image_shape'], 
                spacing=self.img_spec['spacing']
            )
        )

        # preprocess - align, center-crop, resample
        sample.preprocess()
        cropped_img = [
            sitk.GetArrayFromImage(x).astype(np.int16)
            for x in sample.scans
        ]
        image = np.stack(cropped_img,axis=0).astype(np.float32)

        zero_mask = np.ones((20,),dtype=np.float32)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if np.max(image[i,j]) != 0:
                    image[i,j] = image[i,j]/np.max(image[i,j]) 
                else:
                    zero_mask[i] = 0

        data = torch.from_numpy(image)
        data = data.transpose(1,0).to(self.device)

        cls_results = []

        for p in range(len(self.cls_models)):

            self.cls_models[p].eval()
            with torch.no_grad():
                with autocast(True):
                    cls_result = self.cls_models[p](data)
            cls_result = F.softmax(cls_result.float(), dim=1)
            cls_result = cls_result[:,1].detach().cpu().numpy()
            cls_results.append(cls_result)

        cls_result = np.mean(np.asarray(cls_results),axis=0)
        cls_result = cls_result * zero_mask

        cls_result.sort()
        # print(cls_result)
        cls_p = np.mean(cls_result[-7:])

        outputs = []
        print("Generating Predictions ...")

        # for each member model in ensemble
        for p in range(len(self.models)):

            # switch model to evaluation mode
            self.models[p].eval()

            # scope to disable gradient updates
            with torch.no_grad():
                rs = []
                for i in range(data.size()[0]):
                    with autocast(False):
                        output = self.models[p](data[i:i+1,...])
                    if isinstance(output,tuple) or isinstance(output,list):
                        output = output[0]
                    else:
                        output = output  
                    # print(seg_output.size())
                    output = output.float()
                    output = torch.softmax(output,dim=1).squeeze().detach().cpu().numpy()  #N*H*W 
                    output = output[0]
                    rs.append(output)

                output = np.stack(rs,axis=0)
                # print(output.shape)

                # gaussian blur to counteract checkerboard artifacts in
                # predictions from the use of transposed conv. in the U-Net
                outputs += [
                    output
                ]

        # ensemble softmax predictions
        ensemble_output = np.mean(outputs, axis=0).astype('float32')
        ensemble_output = 1 - ensemble_output

        print("ensemble_output OK!!!")

        # read and resample images (used for reverting predictions only)
        sitk_img = [
            sitk.ReadImage(str(path)) for path in self.image_input_paths
        ]
        resamp_img = [
            sitk.GetArrayFromImage(
                resample_img(x, out_spacing=self.img_spec['spacing'])
            )
            for x in sitk_img
        ]

        # revert softmax prediction to original t2w - reverse center crop
        cspca_det_map_sitk: sitk.Image = sitk.GetImageFromArray(crop_or_pad(
            ensemble_output, size=resamp_img[0].shape))
        cspca_det_map_sitk.SetSpacing(list(reversed(self.img_spec['spacing'])))

        # revert softmax prediction to original t2w - reverse resampling
        cspca_det_map_sitk = resample_img(cspca_det_map_sitk,
                                          out_spacing=list(reversed(sitk_img[0].GetSpacing())))

        # process softmax prediction to detection map
        cspca_det_map_npy = extract_lesion_candidates(
            sitk.GetArrayFromImage(cspca_det_map_sitk), threshold='dynamic')[0]

        # remove (some) secondary concentric/ring detections
        cspca_det_map_npy[cspca_det_map_npy<(np.max(cspca_det_map_npy)/2)] = 0

        # make sure that expected shape was matched after reverse resampling (can deviate due to rounding errors)
        cspca_det_map_npy = crop_or_pad(
            cspca_det_map_npy, size=sitk.GetArrayFromImage(sitk_img[0]).shape)
        cspca_det_map_sitk: sitk.Image = sitk.GetImageFromArray(cspca_det_map_npy)
        # print(cspca_det_map_sitk.GetSize())

        # works only if the expected shape matches
        cspca_det_map_sitk.CopyInformation(sitk_img[0])

        # save detection map
        atomic_image_write(cspca_det_map_sitk, self.detection_map_output_path)
        print('cspca_det_map_sitk write OK!!')

        # save case-level likelihood
        with open(str(self.case_level_likelihood_output_file), 'w') as f:
            # json.dump((float(np.max(cspca_det_map_npy))+float(cls_p))/2, f)
            json.dump(float(np.max(cspca_det_map_npy)), f)
        # print(np.max(cspca_det_map_npy))
        print('finished!!')

if __name__ == "__main__":
    csPCaAlgorithm().predict()
