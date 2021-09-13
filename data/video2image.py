import os
import argparse
import xml.etree.ElementTree as elemTree
import subprocess
from tqdm import tqdm
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/bobo/datas/', type=Path, help='path of video')
    parser.add_argument('--save_dir', default='/home/bobo/png', type=Path, help='Directory Path to save the image')
    parser.add_argument('--size', default=224, type=int, help='image size')
    parser.add_argument('--class', default=2, type=int, help='num of class')

    return parser.parse_args()


def get_list(dir_path):
    videos = []
    for path, dir, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith('.mp4'):
                videos.append(os.path.join(path, filename))

    return sorted(videos)


def save_frame(c, videoname, save_dir, s, d, size):
    save_ = save_dir / c / videoname
    if not os.path.exists(save_):
        os.makedirs(save_)

    if s == 0:
        cmd = ['ffmpeg', '-t', d, '-i', video, '-s', f'{size}*{size}',
               f'{save_}/img_%06d.png']
    else:
        cmd = ['ffmpeg', '-ss', s, '-t', d, '-i', video, '-s', f'{size}*{size}',
               f'{save_}/img_%06d.png']

    subprocess.run(cmd, capture_output=True)


if __name__ == '__main__':
    classes = ['swoon']
    args = get_args()
    videos = get_list(args.data)
    fail = 0

    for i, video in enumerate(tqdm(videos)):
        xml = video.replace('.mp4', '.xml')

        try:
            tree = elemTree.parse(xml)
            videoname = tree.find('./filename').text[:-4]
            event = tree.find('./event')
            start = event.find('./starttime').text
            duration = event.find('./duration').text

            for idx in range(len(classes)):
                if classes[idx] in videoname:
                    save_frame('normal', videoname, args.save_dir, 0, start, args.size)
                    save_frame(classes[idx], videoname, args.save_dir, start, duration, args.size)
                    break
        except Exception as e:
            fail += 1
            print(e)

    print(f'Data preprocessing completed.\t{len(videos)-fail} \\ {len(videos)}')
