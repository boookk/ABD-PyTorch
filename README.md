# Deep Learning based anomaly behavior detection system using CCTV


## Summary

본 레퍼지토리는 대학교 졸업작품으로 진행한 프로젝트입니다.

![demo](https://user-images.githubusercontent.com/76933244/144737136-c668c095-44c0-4b45-a57f-755f608aa142.gif)


## Environment
- Ubuntu 20.04 LTS
- Python 3.8
- PyTorch 1.9


## Dataset
- [UCF-Crime](https://webpages.uncc.edu/cchen62/dataset.html)  


## model
- [3D ResNets](https://github.com/kenshohara/3D-ResNets-PyTorch)

사람의 행동을 인식하기 위해서는 공간 정보와 시간 정보를 고려해야 한다. 그렇기 때문에 일반 Convolutional 3D model이 아닌 심층의 3D ResNets 모델을 활용하였다.


## Process

### 1. Conda
```shell
conda create -n ABD python=3.8
```
```shell
conda activate ABD
```

### 2. Git clone
```shell
git clone https://github.com/boookk/ABD-PyTorch.git
```
```shell
cd ABD-PyTorch
```

### 3. Data pre-processing
```shell
python data/video2image.py
```

### 4. Training
```shell
python main.py
```

### 5. Demo (Webcam)
```shell
python demo/demo.py
```

