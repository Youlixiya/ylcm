import PIL
import math
import torch
import numpy as np
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from diffusers import DiffusionPipeline, UNet2DModel
from diffusers.utils import randn_tensor, BaseOutput
@dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]

class ConsistencyPipeline(DiffusionPipeline):
    unet: UNet2DModel

    def __init__(
        self,
        unet: UNet2DModel,
    ) -> None:
        super().__init__()
        self.register_modules(unet=unet)

    @torch.no_grad()
    def __call__(
        self,
        batch_size : int = 1,
        num_class: Optional[int] = None,
        label_index: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eps: float = 0.002,
        T: float = 80.0,
        data_std: float = 0.5,
        num_inference_steps: int = 1,
        output_type: Optional[str] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        model = self.unet
        device = model.device
        image_labels = None
        if label_index is not None:
            assert label_index + 1 <= num_class, 'label_index must <= num_class!'
            image_labels = torch.LongTensor([label_index]).repeat(batch_size).to(device)
        else:
            if num_class is not None:
                image_labels = torch.randint(low=0, high=num_class, size=[1])
                image_labels = image_labels.repeat(batch_size).to(device)
        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        time: float = T

        sample = randn_tensor(shape, generator=generator, device=device) * time

        for step in self.progress_bar(range(num_inference_steps)):
            if step > 0:
                time = self.search_previous_time(time)
                sigma = math.sqrt(time ** 2 - eps ** 2 + 1e-6)
                sample = sample + sigma * randn_tensor(
                    sample.shape, device=sample.device, generator=generator
                )

            out = model(sample, torch.tensor([time], device=sample.device), image_labels).sample

            skip_coef = data_std ** 2 / ((time - eps) ** 2 + data_std ** 2)
            out_coef = data_std * (time - eps) / (time ** 2 + data_std ** 2) ** (0.5)

            sample = (sample * skip_coef + out * out_coef).clamp(-1.0, 1.0)

        sample = (sample / 2 + 0.5).clamp(0, 1)
        # image = sample.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = sample.cpu().permute(0, 2, 3, 1).numpy()
            image = self.numpy_to_pil(image)
        else:
            image = sample

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

        # TODO: Implement greedy search on FID

    def search_previous_time(
            self, time, eps: float = 0.002, T: float = 80.0
    ):
        return (2 * time + eps) / 3

