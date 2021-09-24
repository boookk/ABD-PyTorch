# Abnormal-behavior-detection-system-using-CCTV-based-on-deep-learning

## Summary

본 레퍼지토리는 대학교 '캡스톤디자인' 과목에서 수행하고 있는 프로젝트

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



## Process

### 1. Conda
```shell
conda create -n abd python=3.8
```

### 2. Git clone
```shell
git clone https://github.com/boookk/ABD-PyTorch.git
```

