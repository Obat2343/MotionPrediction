# Motion Prediction
## Enviroment

```sh
git clone https://github.com/Obat2343/MotionPrediction.git
mkdir git
cd git
```

install following app in git directory.

- CoppeliaSim (<https://www.coppeliarobotics.com/downloads.html>)
- Pyrep (<https://github.com/stepjam/PyRep>)
- RLBench (<https://github.com/stepjam/RLBench>)
- RotationConinuity (<https://github.com/papagina/RotationContinuity>)

Next, Install requirement

```sh
pip install -r requirements.txt
```

Then, Download the dataset

```sh
mkdir dataset 
```

google drive link (<https://drive.google.com/file/d/1Nkl53xlV9m87Zm_l93F-T3K6Kd5p4Aca/view?usp=sharing>)

google drive link (<https://drive.google.com/file/d/1LbY_-rg1Mls_YLRUIgPTfk-sQkgVm43M/view?usp=sharing>)

Unzip these data and put them in "dataset" directory as "HMD" and "RLBench3", respectively.

## How to use

```sh
cd main
python train_mp.py 
```
