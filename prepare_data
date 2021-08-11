# local version
import os
import cv2
import argparse
import xml.etree.ElementTree as elemTree
import subprocess
from glob import glob
from pathlib import Path
# from moviepy.video.io.ffmpeg_tools import ffm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', default='/home/bobo/ABD-PyTorch/test/', type=Path, help='path of video')
    parser.add_argument('--save_dir', default='/home/bobo/ABD-PyTorch/datas', type=Path, help='Path to save the image')
    parser.add_argument('--size', default=224, type=int, help='image size')
    parser.add_argument('--class', default=2, type=int, help='num of class')

    return parser.parse_args()


def get_list(dir_path):
    xmls = sorted(glob(os.path.join(dir_path, '*.xml')))
    videos = sorted(glob(os.path.join(dir_path, '*.mp4')))

    # 개수가 동일한지, 비어있지 않은지 확인
    assert len(xmls) == len(videos)
    assert xmls or videos

    return xmls, videos


def get_xml():
    pass


if __name__ == '__main__':
    args = get_args()

    xmls, videos = get_list(args.dir_path)

    for i in range(len(videos)):
        xml = xmls[i]
        video = videos[i]

        # xml 파일과 video 파일 이름의 일치 여부 확인
        assert xml.split('/')[-1].split('.')[0] == video.split('/')[-1].split('.')[0]

        tree = elemTree.parse(xml)
        event = tree.find('./event')
        start = event.find('./starttime').text
        duration = event.find('./duration').text

        out_dir = os.path.join(args.save_dir, str(i+1))

        # if not os.path.exists(out_dir):
        #     os.makedirs(out_dir)

        cmd = f'ffmpeg -ss {start} -t {duration} -i {video} -s {args.size}*{args.size}'
        cmd = cmd.split()
        cmd.append(os.path.join(out_dir, '%05d.png'))
        print(cmd)
        subprocess.run(cmd, capture_output=True)
