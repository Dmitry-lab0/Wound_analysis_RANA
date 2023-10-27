# RANA 0.1
The aim of our project is to create software for the analysis of wounds from photographs

![](https://github.com/Dmitry-lab0/Wound_analysis_RANA/blob/main/images/image_1.png?raw=true)
![](https://github.com/Dmitry-lab0/Wound_analysis_RANA/blob/main/images/mask_1.png?raw=true)
![](https://github.com/Dmitry-lab0/Wound_analysis_RANA/blob/main/images/image_2.png?raw=true)
![](https://github.com/Dmitry-lab0/Wound_analysis_RANA/blob/main/images/mask_2.png?raw=true)
![](https://github.com/Dmitry-lab0/Wound_analysis_RANA/blob/main/images/depth.png?raw=true)

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
You can download the model checkpoints [here](https://github.com/facebookresearch/segment-anything#model-checkpoints) for SAM and [here](https://drive.google.com/drive/folders/1vteIYoWKHEvI0jz8wWF5r-3W0wSvDgXt?usp=sharing) for depth estimation.  

# Run

MODEL_TYPE: `vit_h`, `vit_l`, `vit_b`
```bash!
python app.py --model_type vit_h --checkpoint ../models/sam_vit_h_4b8939.pth
```

If you want to run on cpu, 
```bash!
python app.py --model_type vit_h --checkpoint ../models/sam_vit_h_4b8939.pth --device cpu
```
# License

The app is licensed under the [Apache 2.0 license](https://github.com/Dmitry-lab0/Wound_analysis_RANA/blob/main/LICENSE).

# Contributing

Need a lot of help in making the UI better.
When creating an issue, please include as much detail as possible, including steps to reproduce the issue if applicable.
Create a pull request (PR) to the original repository. Please use black formatter when making code changes.

# Credits

- Segment-Anything - https://github.com/facebookresearch/segment-anything
- MiDas - https://github.com/isl-org/MiDaS
