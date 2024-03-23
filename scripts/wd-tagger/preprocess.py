import torch.nn.functional as F

def preprocess(tensor, target_size=448, padding_value=1, mode='bilinear'):
    tensor = tensor.permute(0, 3, 1, 2)
    _, _, h, w = tensor.size()
    
    max_side = max(h, w)
    padding_left = (max_side - w) // 2
    padding_right = max_side - w - padding_left
    padding_top = (max_side - h) // 2
    padding_bottom = max_side - h - padding_top
    
    tensor = F.pad(tensor, (padding_left, padding_right, padding_top, padding_bottom), value=padding_value)
    tensor = F.interpolate(tensor, size=(target_size, target_size), mode=mode, align_corners=False)
    tensor = tensor.flip(1) # RGB to BGR
    tensor = (tensor - 0.5) / 0.5
    return tensor