# Region-guided focal adversarial learning for CT-to-MRI translation: a proof-of-concept and validation study in hepatocellular carcinoma

### Installation
you can create a new Conda environment using `conda env create -f environment.yaml`.
 
Importantly, we have modified the `transforms.py` file based on `torchvision 0.14.1`, please replace this file.

### FocalGAN train/test
create a path `/path/to/data/trainA/img`

create a path `/path/to/data/trainA/seg`

create a path `/path/to/data/trainB/img`

create a path `/path/to/data/trainB/seg`

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script


#### Train a model:
```bash
python train.py --dataroot ./data --name arterial_seg_cycle_gan --model seg_cycle_gan --no_flip
```
To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.

#### Test the model:
```bash
python test.py --dataroot ./data --name arterial_seg_cycle_gan --model test --no_dropout
```
- The test results will be saved to a html file here: `./results/arterial_seg_cycle_gan/latest_test/index.html`.
