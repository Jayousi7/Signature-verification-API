import cv2 as cv
import random
import numpy as np 
import torch
from torchvision import transforms
def preprocessing(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.resize(img,(220,155),interpolation=cv.INTER_LINEAR)
    img = 255-img
    img = cv.GaussianBlur(img, (5, 5), 0)

    return img


def set_seed(seed=42):
    """Ensures completely reproducible results across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False