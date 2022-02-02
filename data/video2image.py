import argparse
from pathlib import Path
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/bobo/ucf_crime', type=Path, help='path of video')
    parser.add_argument('--save_dir', default='/home/bobo/output', type=Path, help='path of save image')
    parser.add_argument('--size', default=224, type=int, help='image size')
    parser.add_argument('--n_classes', default=2, type=int, help='num of class')
    parser.add_argument('--clip_len', default=16, type=int)

    return parser.parse_args()


def process_video(video, action_name, save_dir):
    video_name = video.name.split('.')[0]
    resize_height = 320
    resize_width = 240

    # Create a class directory to store the converted data.
    if 'Normal' in action_name:
        save_dir = save_dir / 'Normal'
    else:
        save_dir = save_dir / 'Anomaly'
    save_dir = save_dir / video_name
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    capture = cv2.VideoCapture(str(video))

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 해당 비디오가 16프레임 미만일 경우
    EXTRACT_FREQUENCY = 4
    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1

    count = 0
    i = 0
    retaining = True

    while count < frame_count and retaining:
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            if (frame_height != resize_height) or (frame_width != resize_width):
                frame = cv2.resize(frame, (resize_width, resize_height))
            cv2.imwrite(filename=str(save_dir / '0000{}.jpg'.format(str(i))), img=frame)
            i += 1
        count += 1

    capture.release()


if __name__ == '__main__':
    args = get_args()

    train_dir = args.save_dir / 'train'
    test_dir = args.save_dir / 'test'

    if not train_dir.exists():
        train_dir.mkdir(parents=True)
    if not test_dir.exists():
        test_dir.mkdir(parents=True)

    # Annotation file provided by the dataset.
    with open('/ANNOTATION/PATH/Anomaly_Train.txt', 'r') as f:
        train_list = f.read().split('\n')
    with open('/ANNOTATION/PATH/Anomaly_Test.txt', 'r') as f:
        test_list = f.read().split('\n')

    # Based on the training data..
    for train_ in train_list:
        # Verify that the file type is mp4.
        if not train_.endswith('.mp4'):
            continue
        video = args.data / train_
        if video.exists():
            process_video(video, video.parent.name, train_dir)

    # Based on the testing data..
    for test_ in test_list:
        # Verify that the file type is mp4.
        if not test_.endswith('.mp4'):
            continue
        video = args.data / test_
        if video.exists():
            process_video(video, video.parent.name, test_dir)
