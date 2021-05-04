# Abnormal-behavior-detection-system-using-CCTV-based-on-deep-learning

Environment
- Ubuntu 20.04 LTS
- CUDA 10.1
- Python 3.
- PyTorch 1.6
- cuDNN 

## 1. Docker Container 생성
```
sudo docker run -d -it -e DISPLAY=$DISPLAY -v /tmp/.x11-unix:/tmp/.x11-unix --device=/dev/video0:/dev/video0 -v /home/bobo/share:/root/share --gpus all --name Abnormal_behavior pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
```

### 컨테이너 접속
```
sudo docker attach Abnormal_behavior
```

