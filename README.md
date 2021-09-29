# Abnormal behavior detection system using CCTV based on deep learning

## Summary

본 레퍼지토리는 대학교 졸업작품으로 진행하고 있는 프로젝트

## Environment
- Ubuntu 20.04 LTS
- CUDA 10.2
- Python 3.8
- PyTorch 1.9
- cuDNN

## Dataset
- [이상행동 CCTV 영상](https://aihub.or.kr/aidata/139)  

본 데이터는 과학기술정보통신부가 주관하고 한국지능정보사회진흥원이 지원하는 '인공지능 학습용 데이터 구축사업'으로 구축된 데이터입니다.  

- 클래스 선정

12개의 class 중 4개(assault. burglary, swoon, datefight)의 class를 이용

<img src="https://user-images.githubusercontent.com/76933244/134613782-b04d2890-7b1a-4c3d-9dbc-41f272b813b6.png" width="500" height="400"> 


## model
- [3D ResNets](https://github.com/kenshohara/3D-ResNets-PyTorch)

사람의 행동을 인식하기 위해서는 공간적인 요소뿐만 아니라, 시간적인 요소도 고려해야 한다. 그렇기 때문에 구현된 3D ResNets 모델을 활용하려고 한다.



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

