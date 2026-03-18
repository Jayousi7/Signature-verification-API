import cv2 as cv
import numpy as np 
import torch
from torchvision import transforms
def preprocessing(img) -> torch.Tensor:
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (220, 155), interpolation=cv.INTER_LINEAR)
    
    img = 255 - img
    img = cv.GaussianBlur(img, (5, 5), 0)
    
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor
