# Anytime Classification
Code for paper [Towards Anytime Classification in Early-Exit Architectures by Enforcing Conditional Monotonicity]().

Code for pretrained models is taken from the following repos:
- [MSDNet](https://arxiv.org/abs/1703.09844):  https://github.com/kalviny/MSDNet-PyTorch
- [IMTA](https://arxiv.org/pdf/1908.06294.pdf): https://github.com/kalviny/IMTA 

## Main Dependencies
* Python = 3.6
* Pytorch = 1.7

## Setup 
1. Clone or download this repo. `cd` yourself to it's root directory.
2. Create and activate python [conda](https://www.anaconda.com/) enviromnent: `conda create --name anytime-class python=3.8`
3. Activate conda environment:  `conda activate anytime-class`
4. Install dependencies, using `pip install -r requirements.txt`

## Code
- Main (anytime) functions are implemented in `anytime_predictors.py`
- Figures from the main paper are reproduced in `paper_plots.ipynb`. For code to run, the following steps need to be performed:
    - Download [ImageNet](https://image-net.org) validation dataset and store it in `data\ImageNet\val`
    - Download pretrained models (e.g., MSDNet) from [here](https://drive.google.com/drive/folders/1EV0qhNRCkZTLRPcdNU-PwGOKfjHBKnlF?usp=sharing) and store them in `pretrained_models` directory. Alternatively, use repos of pretrained models to train those from scratch

## Acknowledgements
The [Robert Bosch GmbH](https://www.bosch.com) is acknowledged for financial support.
