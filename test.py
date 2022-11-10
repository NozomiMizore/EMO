from emo_rec import Emo_Rec
import cv2

if __name__ == '__main__':
    emotion_model = Emo_Rec('model/CNN.98-0.62.hdf5')
    image_1 = cv2.imread('img/smile.jpg')
    result_1 = emotion_model.run(image_1)
    print("smile_pic:"+result_1)

    image_2 = cv2.imread('img/anger.jpg')
    result_2 = emotion_model.run(image_2)
    print("anger_pic:"+result_2)

    image_3 = cv2.imread('img/sad.jpg')
    result_3 = emotion_model.run(image_3)
    print("sad_pic:" + result_3)

