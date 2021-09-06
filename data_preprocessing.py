# 이 스크립트로 실신데이터 전처리 완료
import os
import argparse
import xml.etree.ElementTree as elemTree
import subprocess
from tqdm import tqdm
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/bobo/ABD-PyTorch/datas/', type=Path, help='path of video')
    # parser.add_argument('--save_dir', default='/home/bobo/ABD-PyTorch/datas', type=Path, help='Path to save the image')
    parser.add_argument('--size', default=224, type=int, help='image size')
    parser.add_argument('--class', default=2, type=int, help='num of class')

    return parser.parse_args()


def get_list(dir_path):
    xmls = []
    videos = []
    for path, dir, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith('.xml'):
                # 아래는 파일명만 리스트에 추가
                # xmls.append(filename)
                # 경로를 추가하려면 아래와 같이
                xmls.append(os.path.join(path, filename))
            elif filename.endswith('.mp4'):
                videos.append(os.path.join(path, filename))

    return sorted(xmls), sorted(videos)


def save_frame(c, save_dir, s, d, size):
    save_ = save_dir / c / str(i + 1)

    if not os.path.exists(save_):
        os.makedirs(save_)

    if s == 0:
        cmd = ['ffmpeg', '-t', d, '-i', video, '-s', f'{size}*{size}',
               f'{save_}/%06d.png']
    else:
        cmd = ['ffmpeg', '-ss', s, '-t', d, '-i', video, '-s', f'{size}*{size}',
               f'{save_}/%06d.png']

    subprocess.run(cmd, capture_output=True)


if __name__ == '__main__':
    args = get_args()

    classes = ['normal', 'swoon']

    xmls, videos = get_list(args.data)

    for i, video in enumerate(tqdm(videos)):
        if i >= 640:
            xml = video.replace('.mp4', '.xml')

            try:
                tree = elemTree.parse(xml)
                event = tree.find('./event')
                start = event.find('./starttime').text
                duration = event.find('./duration').text

                # 이미지 저장 경로
                save_dir = args.data.parent / 'png'

                for idx in range(len(classes)):
                    if classes[idx] in video:
                        save_frame('normal', save_dir, 0, start, args.size)
                        save_frame(classes[idx], save_dir, start, duration, args.size)
                        break
            except Exception as e:
                print(e)
