""" 
defines predictor class
"""

import string
from typing import Union, Optional
from numpy import ndarray
from torch import Tensor, load, from_numpy, unsqueeze, float32
from torch import device as _device
from torch.nn import Conv2d
from torchvision.models import resnet50
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os  
from skimage import io
from extract_char import extract_text

class CharacterPredictor:
    """Initialize a CharacterPredictor instance with a pretrained model

    Parameters
    ----------
    model_path : str
        Model file name, e.g. `model.pth`.
    device : str, torch.device, optional
        device where model will be ran on. default "cpu".

    """

    def __init__(self, model_path: str, device: Optional[Union[str, _device]] = "cpu"):
        self.device = device

        # initialize model
        self.model = resnet50()
        self.model.conv1 = Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.load_state_dict(load(model_path, map_location="cpu"))
        self.model.to(self.device)

        self.model.eval()

    def predict(self, char_image: Union[ndarray, Tensor]):
        """Predict a character from an image.

        Parameters
        ----------
        char_image : np.ndarray or torch.Tensor
            28x28 image of a character with 0 background (black)

        Returns
        -------
        predicted_char : str
            Predicted output of model as a capital character
            e.g. "A".

        """

        # check if char_image is of the right type and shape
        if isinstance(char_image, ndarray):
            char_image = from_numpy(char_image)
        elif not isinstance(char_image, Tensor):
            raise TypeError(
                f"expected image of type np.ndarray instead got {type(char_image)}"
            )
        if char_image.shape != (28, 28):
            raise ValueError(
                f"expected image of shape (28, 28), instead got {char_image.shape}"
            )

        # convert to 4D (batch, channels, x, y) and change type
        char_image = unsqueeze(unsqueeze(char_image, 0), 0)
        char_image = char_image.to(device=self.device, dtype=float32)

        # make prediction
        pred = self.model(char_image)
        pred = pred.argmax(1)
        predicted_char = list(string.ascii_uppercase)[pred]

        return predicted_char


if __name__ == "__main__":

    

    root = os.getcwd()
    if not os.path.exists(os.path.join(root,'extract_text')):
        os.makedirs(os.path.join(root,'extract_text'))
        os.makedirs(os.path.join(root,'extract_text_output'))
        os.makedirs(os.path.join(root,'test'))

    for i in os.listdir(os.path.join(root,'test_images')):

        extract_text(i)
        # create an instance and point to the pretrained model file
        predictor = CharacterPredictor(model_path=os.path.join(root,'model.pth'))
        # get example image filepaths
        example_images = os.listdir(os.path.join(root,'extract_text_output'))
        example_images = sorted(example_images)
        file_names = []
        for j in example_images:
            file_names.append(int(j.split(".")[0]))
        file_names.sort()
        example_images = []
        for j in file_names:
            example_images.append(str(j)+'.png')
        fin = []
        for example in example_images:
            image = io.imread(os.path.join("extract_text_output", example), as_gray=True)

            fin.append(predictor.predict(image))
        print("".join(fin))
        out_file = i.split(".")[0]+"_text"+".txt"
        text_file = open(os.path.join(root,'out_files',out_file), "w")
        n = text_file.write("".join(fin))
        text_file.close()
        print("\n")
