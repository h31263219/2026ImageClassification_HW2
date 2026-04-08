import torch
import json
ckpt = torch.load('./output/best_model.pth', map_location='cpu', weights_only=False)
args = ckpt.get('args', {})
print(json.dumps(args, indent=2))
