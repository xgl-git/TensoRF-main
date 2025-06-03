
## Installation

Install environment:
```
conda create -n TensoRF python=3.8
conda activate TensoRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
```



## Quick Start
The training script is in `train.py`, to train a TensoRF and then evaluate the model:

```
python train.py --config configs/your_own_data.txt
```


## Rendering

```
python train.py --config configs/your_own_data.txt --ckpt ./log/tensorf_column_VM/tensorf_column_VM.th --render_only 1 --render_test 1 
```
