# SAM-webui
![](https://github.com/Dmitry-lab0/Wound_analysis_RANA/blob/main/images/image1.png?raw=true)

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
