import torch
import torch.nn as nn 
from model import SigNet

def load_signet_model(model_path: str ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # 1. NEW: Handle Tuples by searching inside them for the dictionary or model
    if isinstance(checkpoint, tuple):
        for item in checkpoint:
            # If we find a dictionary, assume it's the state_dict (or contains it)
            if isinstance(item, dict):
                checkpoint = item
                break
            # If we find a full model object, extract its state_dict
            elif isinstance(item, nn.Module):
                checkpoint = item.state_dict()
                break
        else:
            raise ValueError("The checkpoint is a tuple, but no dictionary or model was found inside it.")

    # 2. Handle full model objects (if it wasn't a tuple)
    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
        
    # 3. Handle standard dictionary checkpoints (your original robust logic)
    elif isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint  # fallback (legacy SigNet)
            
    # 4. If it is none of the above, it's truly invalid
    else:
        raise ValueError(f"Invalid checkpoint format. Got {type(checkpoint)}")

    # Initialize your model and load the weights
    model = SigNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device