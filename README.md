# Motion Prediction
## Enviroment

```sh
git clone https://github.com/Obat2343/MotionPrediction.git
mkdir git
cd git
```

Install following app in git directory.

- CoppeliaSim (<https://www.coppeliarobotics.com/downloads.html>)
- Pyrep (<https://github.com/stepjam/PyRep>)
- RLBench (<https://github.com/stepjam/RLBench>)
- RotationConinuity (<https://github.com/papagina/RotationContinuity>)

Next, Install requirements

```sh
pip install -r requirements.txt
```

Then, Download the dataset

```sh
mkdir dataset
```

google drive link (<https://drive.google.com/file/d/1LbY_-rg1Mls_YLRUIgPTfk-sQkgVm43M/view?usp=sharing>)

Unzip and put it in "dataset" directory as "RLBench4".

## How to train model

Training mp with FTIP.

Set PRED_TRAJECTORY: True in yaml file.
```sh
cd main
python train_mp.py --config_file ../config/RLBench_MP.yaml
```

Training vp with FIGL.

Set VIDEO_HOUR:MODE: 'pcf'

```sh
cd main
python train_vp.py --config_file ../config/RLBench_VP.yaml
```

Finally recurrent training with FTIP and FIGL.

```sh
cd main
python train_mp.py --config_file ../config/RLBench_MPS.yaml --vp_path path/to/checkpoint_dir --hourglass_path path/to/pth_file
```

## How to eval model

Please change the configuration in evaluation.py before running the code.

```
cd eval
python evaluation.py
```