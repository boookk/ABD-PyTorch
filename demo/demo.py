import cv2

from abd_model import abd_model


def text_in_frame(img, relation):
    x, y = [10, 20]
    img = cv2.putText(img, f"{relation}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return img


if __name__ == '__main__':
    # we create the video capture object cap
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("We cannot open webcam.")

    ############################
    # 모델 불러오기
    model = abd_model()
    ############################

    relation = None

    while True:
        ret, frame = cap.read()
        img = frame.copy()

        model.save_clip(img)
        scr, cls = model.relation_recognition()
        if cls:
            relation = cls

        img = text_in_frame(img, relation)
        cv2.imshow("CCTV", img)
        
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
