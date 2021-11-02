import os
import sys

import click
from rich import traceback, print

import numpy as np
import torch
import mlflow.pytorch
import torch.nn.functional as F

# to import model module from liver_ct_segmentation_package
import liver_ct_segmentation_package
#sys.path.insert(1, liver_ct_segmentation_package.__path__[0]) # initial try
sys.path.append(liver_ct_segmentation_package.__path__[0]) #probably best to add it at the search end

# mrc for easy result visualization
import mrcfile as mrc
from liver_ct_segmentation_package.utils import save_vol, merge_vol, split_vol, monte_carlo_dropout_proc

###############################################
# dist settings
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group("gloo", rank=0, world_size=1)

###############################################

from captum.attr import visualization as viz
from captum.attr import LayerGradCam, FeatureAblation, LayerActivation, LayerAttribution
from captum.attr import GuidedGradCam

# Default device
device = "cpu"

import torch.nn as nn

###############################################
###############################################

#WD = os.path.dirname(__file__)

@click.command()
@click.option('-i', '--input', required=True, type=str, help='Path to input data file, on which to predict')
@click.option('-m', '--model', default='snapshots/lits_unet_3d/model/', type=str, help='ID/Path to an already trained model')
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to enable cuda or not')
@click.option('-o', '--output', type=str, help='Path to output folder')
@click.option('-f', '--feat', default='_feat_vol.mrc', type=str, help='Filename for ggcam features output')
@click.option('-t', '--target', default='0', type=str, help='Output indices for which gradients are computed (target class)')
def main(input: str, model: str, cuda: bool, output: str, feat: str, target: str):
    """Command-line interface for liver-ct-segmentation-package"""

    print(r"""[bold blue]
        liver-ct-segmentation-package: Package of 3D U-Nets reproducibly trained on the LiTS dataset.
        -->Segmentation prediction CLI
        """)

    print('[bold blue]Run [green]liver-ct-seg-feat-ggcam --help [blue]for an overview of all commands\n')

    out_filename = feat
    target_class = int(target)

    print('[bold blue] Loading model: ' + model)
    #if model == 'lits-unet':
    #    #model = get_pytorch_model(f'{WD}/models/snapshots/lits_unet_3d/model/')
    #    model = get_pytorch_model(f'model/snapshots/lits_unet_3d/model/')

    model_obj = get_pytorch_model(model)
    #if cuda:
    #    model.cuda()
    
    print('[bold blue] Reading input data: ' + input)
    volume_image = read_input_volume(input)
    print('[bold blue] Calculating Guided Grad-CAM features...')
    print('[bold blue] Target class: ' + target)
    features = features_ggcam(net=model_obj, img=volume_image, target_class=target_class)
    print('features shape: ' + str(features.shape))
    if output:
        print(f'[bold blue]Writing features to {output}' + out_filename)
        write_results(volume_image, features, output, out_filename)


def read_input_volume(input_path: str):
    """
    TODO
    """

    if input_path[len(input_path)-4:] == ".mrc":
        with mrc.open(input_path, permissive=True) as mrc_file:
            img = np.array(mrc_file.data) # TODO: find a more efficient way to make writeable
    else:
        img = torch.load(input_path)

    # save for debugging
    #save_vol('img_i.mrc', img)
    #save_vol('img_t.mrc', np.transpose(img, axes=[2, 1, 0]))

    return img

def features_ggcam(net, img, target_class):
    """
    TODO
    """
    
    net.eval()

    img_t = torch.tensor(img)
    img_t = img_t.unsqueeze(0)
    img_t = img_t.unsqueeze(0)
    img_t = img_t.float()

    wrapped_net = agg_segmentation_wrapper_module(net)
    guided_gc = GuidedGradCam(wrapped_net, wrapped_net._model.model.outc.conv_1)

    gc_attr = guided_gc.attribute(img_t, target=target_class)
    gc_attr = torch.abs(gc_attr)

    #print("ggcam out shape: " + str(gc_attr.shape))
    img_out = gc_attr.squeeze(0).squeeze(0).cpu().detach().numpy()

    return img_out

def write_results(input_image, output_volume, output_path, output_filename, transpose=True) -> None:
    """
    Writes the uncertainty values into a human readable file.
    :param output_volume: std volume as a numpy array
    :param output_path: Path to write the uncertainty values to
    """

    if transpose:
        output_filename = '_tsp_' + output_filename

        output_volume = np.transpose(output_volume, axes=[2, 1, 0])
        save_vol(output_path + '_tsp_input.mrc', np.transpose(input_image, axes=[2, 1, 0]))

    # mrc for easy visualization
    save_vol(output_path + output_filename, output_volume)

def get_pytorch_model(path_to_pytorch_model: str):
    """
    Fetches the model of choice and creates a booster from it.
    :param path_to_pytorch_model: Path to the xgboost model1
    """
    model = mlflow.pytorch.load_model(path_to_pytorch_model, map_location=torch.device('cpu')).module
    return model

class agg_segmentation_wrapper_module(nn.Module):
    
    def __init__(self, model):
        super(agg_segmentation_wrapper_module, self).__init__()
        
        self._model = model

    def forward(self, x):
        
        model_out = self._model(x)
        out_max = torch.argmax(model_out, dim=1, keepdim=True)
        
        selected_inds = torch.zeros_like(model_out[0:2]).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2, 3, 4))
        
        #out = self._model(x)
        #return out


if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover
