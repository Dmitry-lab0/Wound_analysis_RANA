# SAM-webui
<img src="https://user-images.githubusercontent.com/84118285/232520114-e737f6f7-55d5-465c-b7b7-8c15059e8384.gif" width="600"/>
<img src="https://user-images.githubusercontent.com/84118285/232520000-6606629d-f375-4fe7-b88f-b08f0eb64321.gif" width="600"/>
<img src="https://user-images.githubusercontent.com/84118285/232520088-47c8879a-2c0f-45cf-aa1e-acd5a6a8591a.gif" width="600"/>
<img src="https://user-images.githubusercontent.com/84118285/233614241-d43ad1cd-29c4-437d-86db-9e095710f44e.gif" width="600"/>

# News
- Release code

# TODO List
- [x] Perimeter calculation
- [x] Area calculation
- [x] 3D models
- [ ] Volume calculation - in processing

# Features
- SAM Box Segmentation
- Photo editing app
- Depth calculation
- Perimeter calculation
- Area calculation
- 3D models

# Install
The code requires `python>=3.10`,  and `torchvision>=0.15`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.`

## Model Checkpoints
You can download the model checkpoints [here](https://github.com/facebookresearch/segment-anything#model-checkpoints).  

# Run

MODEL_TYPE: `vit_h`, `vit_l`, `vit_b`
```bash!
python app.py --model_type vit_h --checkpoint ../models/sam_vit_h_4b8939.pth
```

If you want to run on cpu, 
```bash!
python app.py --model_type vit_h --checkpoint ../models/sam_vit_h_4b8939.pth --device cpu
```

# Credits

- Segment-Anything - https://github.com/facebookresearch/segment-anything
