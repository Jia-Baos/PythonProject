import os
import cv2
import torch
from nanodet.util import cfg, load_config, Logger
from demo.demo import Predictor

from nanodet.util import overlay_bbox_cv

from IPython.display import display
from PIL import Image


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device('cuda')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def cv2_imshow(a, convert_bgr_to_rgb=True):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a: np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
            (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
            image.
        convert_bgr_to_rgb: switch to convert BGR to RGB channel.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if convert_bgr_to_rgb and a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(a))


if __name__ == "__main__":

    config_path = './config/nanodet-plus-m-1.5x_416.yml'
    model_path = './pth/nanodet-plus-m-1.5x_416.pth'
    image_path = './imgs/1.png'

    load_config(cfg, config_path)
    logger = Logger(-1, use_tensorboard=False)
    predictor = Predictor(cfg, model_path, logger, device=device)
    meta, res = predictor.inference(image_path)



    result = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.35)
    imshow_scale = 1.0
    cv2_imshow(cv2.resize(result, None, fx=imshow_scale, fy=imshow_scale))