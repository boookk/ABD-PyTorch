## 📝 Summary

### Description.
- 범죄 예방을 목표로 CCTV를 이용하여 인공지능이 실시간으로 이상 행동을 감지하는 시스템 개발
- 본 레퍼지토리는 대학교 졸업작품으로 진행한 프로젝트입니다.

### 문제 의식.
- CCTV 설치율의 증가
- CCTV 수에 비해 적은 관제센터의 인력
- CCTV를 이용하여 사건을 예방하기보다 사건 발생 후 증빙 자료로 활용

### 기대 효과.
- CCTV 통합관제센터의 효율적인 운영
- 실시간으로 사건 발생 여부 파악 및 대응
- 범죄 예방 효과

<br>

### 전체 구성도.
<img src="https://user-images.githubusercontent.com/76933244/150387942-1a7517a0-8359-48b5-9956-b8024acb1eb1.png" width="380" height="600">


<br>

### 결과.
![demo](https://user-images.githubusercontent.com/76933244/144737136-c668c095-44c0-4b45-a57f-755f608aa142.gif)

<br>


## Environment
- Ubuntu 20.04 LTS
- Python 3.8
- PyTorch 1.9


<br>


## Dataset
- [UCF-Crime](https://webpages.uncc.edu/cchen62/dataset.html)  


<br>


## model
- [3D ResNets](https://github.com/kenshohara/3D-ResNets-PyTorch)

사람의 행동을 인식하기 위해서는 공간 정보와 시간 정보를 고려해야 한다. 그렇기 때문에 일반 Convolutional 3D model이 아닌 심층의 3D ResNets 모델을 활용하였다.


<br>


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

