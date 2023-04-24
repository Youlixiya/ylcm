import torch
import math
from PIL import Image
from torchvision.utils import make_grid
def get_grid(images : torch.Tensor,
              image_size : int,
              ) -> Image:
    grid = make_grid(
        images,
        nrow=math.ceil(math.sqrt(images.size(0))),
        padding=image_size // 16,
    )
    return grid