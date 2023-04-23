from typing import Union, List, Tuple
from PIL import Image
def make_grid(images: Union[List, Tuple],
              rows: int,
              cols: int
              ) -> Image:
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid