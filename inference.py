from preprocessing import preprocessing
import torch 
import torch.nn.functional as F

def euclidean_distance(output1:torch.tensor, output2:torch.tensor)->torch.tensor:
    distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-6)
    return distance

def verify_sig(img1,img2,model,device,threshold:float):

    img1 = preprocessing(img1)
    img2 = preprocessing(img2)

    with torch.no_grad():
        img1= img1.to(device)
        img2 = img2.to(device)
        
        image1_map = model(img1)
        image2_map = model(img2)

    image1_map = F.normalize(image1_map)
    image2_map = F.normalize(image2_map)

    d = euclidean_distance(image1_map,image2_map).item()

    label_d = 'Genuine' if d < threshold else  'Forged'

    return {
        "verification1": label_d,
        "distance": f'{d}',
    }


