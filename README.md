## ๐ Summary

### Description.
- ๋ฒ์ฃ ์๋ฐฉ์ ๋ชฉํ๋ก CCTV๋ฅผ ์ด์ฉํ์ฌ ์ธ๊ณต์ง๋ฅ์ด ์ค์๊ฐ์ผ๋ก ์ด์ ํ๋์ ๊ฐ์งํ๋ ์์คํ ๊ฐ๋ฐ
- ๋ณธ ๋ ํผ์งํ ๋ฆฌ๋ ๋ํ๊ต ์กธ์์ํ์ผ๋ก ์งํํ ํ๋ก์ ํธ์๋๋ค.

### ๋ฌธ์  ์์.
- CCTV ์ค์น์จ์ ์ฆ๊ฐ
- CCTV ์์ ๋นํด ์ ์ ๊ด์ ์ผํฐ์ ์ธ๋ ฅ
- CCTV๋ฅผ ์ด์ฉํ์ฌ ์ฌ๊ฑด์ ์๋ฐฉํ๊ธฐ๋ณด๋ค ์ฌ๊ฑด ๋ฐ์ ํ ์ฆ๋น ์๋ฃ๋ก ํ์ฉ

### ๊ธฐ๋ ํจ๊ณผ.
- CCTV ํตํฉ๊ด์ ์ผํฐ์ ํจ์จ์ ์ธ ์ด์
- ์ค์๊ฐ์ผ๋ก ์ฌ๊ฑด ๋ฐ์ ์ฌ๋ถ ํ์ ๋ฐ ๋์
- ๋ฒ์ฃ ์๋ฐฉ ํจ๊ณผ

<br>

### ์ ์ฒด ๊ตฌ์ฑ๋.
<img src="https://user-images.githubusercontent.com/76933244/150387942-1a7517a0-8359-48b5-9956-b8024acb1eb1.png" width="380" height="600">


<br>

### ๊ฒฐ๊ณผ.
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


## Model
- [3D ResNets](https://github.com/kenshohara/3D-ResNets-PyTorch)

์ฌ๋์ ํ๋์ ์ธ์ํ๊ธฐ ์ํด์๋ ๊ณต๊ฐ ์ ๋ณด์ ์๊ฐ ์ ๋ณด๋ฅผ ๊ณ ๋ คํด์ผ ํ๋ค. ๊ทธ๋ ๊ธฐ ๋๋ฌธ์ ์ผ๋ฐ Convolutional 3D model์ด ์๋ ์ฌ์ธต์ 3D ResNets ๋ชจ๋ธ์ ํ์ฉํ์๋ค.


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

