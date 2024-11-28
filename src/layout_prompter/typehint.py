from typing import Annotated, Any, Dict

from PIL import Image

JsonDict = Dict[str, Any]

PilImage = Annotated[Image.Image, "PIL Image"]
