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

###############################################
###############################################

#WD = os.path.dirname(__file__)

@click.command()
@click.option('-i', '--input', required=True, type=str, help='Path to input data file, on which to predict')
@click.option('-m', '--model', default='snapshots/lits_unet_3d/model/', type=str, help='ID/Path to an already trained model')
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to enable cuda or not')
@click.option('-o', '--output', type=str, help='Path to output folder')
@click.option('-u', '--uncert', default='_std_vol.mrc', type=str, help='Filename for uncertainty output (std volume)')
@click.option('-t', '--tparam', default='3', type=str, help='t parameter for the monte-carlo dropout procedure (number of predictions)')
def main(input: str, model: str, cuda: bool, output: str, uncert: str, tparam: str):
    """Command-line interface for liver-ct-segmentation-package"""

    print(r"""[bold blue]
        liver-ct-segmentation-package: Package of 3D U-Nets reproducibly trained on the LiTS dataset.
        -->Segmentation prediction CLI
        """)

    print('[bold blue]Run [green]liver-ct-seg-uncert --help [blue]for an overview of all commands\n')

    out_filename = uncert
    t_param = int(tparam)

    print('[bold blue] Loading model: ' + model)
    #if model == 'lits-unet':
    #    #model = get_pytorch_model(f'{WD}/models/snapshots/lits_unet_3d/model/')
    #    model = get_pytorch_model(f'model/snapshots/lits_unet_3d/model/')

    model_obj = get_pytorch_model(model)
    #if cuda:
    #    model.cuda()
    
    print('[bold blue] Reading input data: ' + input)
    volume_image = read_input_volume(input)
    print('[bold blue] Calculating prediction uncertainty...')
    std_volume = prediction_std(net=model_obj, img=volume_image, t=t_param)
    print('prediction std shape: ' + str(std_volume.shape))
    if output:
        print(f'[bold blue]Writing STD volume to {output}' + out_filename)
        write_results(volume_image, std_volume, output, out_filename)


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

def prediction_std(net, img, t, use_gpu=False):
    """
    TODO
    """
    
    net.eval()

    img = np.expand_dims(img, axis=0)

    img = torch.tensor(img)
    img = img.unsqueeze(0)

    if use_gpu:
        img = img.cuda()

    pred_std = monte_carlo_dropout_proc(net, img, T=t)
    pred_std = pred_std.detach().cpu().numpy().astype(np.float32)

    return pred_std

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

if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover
