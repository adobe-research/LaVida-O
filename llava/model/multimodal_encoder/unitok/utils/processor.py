from llava.model.multimodal_encoder.unitok.utils.data import normalize_01_into_pm1
from torchvision.transforms import transforms, InterpolationMode

class UnitokImageProcessor:

    def __init__(self):
        self._preprocess_fn = transforms.Compose([
            transforms.ToTensor(), normalize_01_into_pm1,
        ])

    def preprocess(self,image):
        return self._preprocess_fn(image).unsqueeze(0)

    