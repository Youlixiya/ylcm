import math
from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput, UNet2DModel
from diffusers.utils import randn_tensor


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
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        image_labels = None
        if label_index is not None:
            assert label_index + 1 <= num_class, 'label_index must <= num_class!'
            image_labels = torch.LongTensor([label_index]).repeat(batch_size).to(self.device)
        else:
            if num_class is not None:
                image_labels = torch.randint(low=0, high=num_class, size=[1])
                image_labels = image_labels.repeat(batch_size).to(self.device)
        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet
        timesteps = torch.linspace(T, eps, num_inference_steps, device=self.device)
        time: float = timesteps[0]

        sample = randn_tensor(shape, generator=generator) * time.item()

        for step in self.progress_bar(range(num_inference_steps)):
            time = timesteps[step]
            if step > 0:
                time = timesteps[step]
                sigma = math.sqrt(time.item() ** 2 - eps ** 2 + 1e-6)
                sample = sample + sigma * randn_tensor(
                    sample.shape, device=sample.device, generator=generator
                )

            out = model(sample, time, image_labels).sample

            skip_coef = data_std ** 2 / ((time - eps) ** 2 + data_std ** 2)
            out_coef = data_std * time / (time ** 2 + data_std ** 2) ** (0.5)

            sample = (sample * skip_coef + out * out_coef).clamp(-1.0, 1.0)

        sample = (sample / 2 + 0.5).clamp(0, 1)
        image = sample.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    # TODO: Implement greedy search on FID
    def search_previous_time(
        self, time, time_min: float = 0.002, time_max: float = 80.0
    ):
        return (2 * time + time_min) / 3
