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

google drive link (<https://drive.google.com/file/d/1LbY_-rg1Mls_YLRUIgPTfk-sQkgVm43M/view?usp=sharing>)

Unzip the data and put it in "dataset" directory as "RLBench3".

## How to use

```sh
cd main
python train_mp.py 
```
