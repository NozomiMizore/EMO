import cv2
import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from load_process import process_data
from PyQt5 import QtGui, QtWidgets


class Emo_Rec:
    def __init__(self, model_path=None):
        detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
        if model_path is None:
            emotion_model_path = 'model/CNN.98-0.62.hdf5'
        else:
            emotion_model_path = model_path

        self.face_detection = cv2.CascadeClassifier(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "normal"]

    def run(self, frame_in, canvas, label_face, label_result):

        frame = imutils.resize(frame_in, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                                     minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        predicts = []
        label = None
        (fX, fY, fW, fH) = None, None, None, None  # 人脸位置
        frame_clone = frame.copy()  # 复制画面
        if len(faces) > 0:
            # 根据ROI大小将检测到的人脸排序
            faces = sorted(faces, reverse=False, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))  # 按面积从小到大排序

            for i in range(len(faces)):  # 遍历每张检测到的人脸，默认识别全部人脸
                # 如果只希望识别和显示最大的那张人脸，可取消注释此处if...else的代码段
                # if i == 0:
                #     i = -1
                # else:
                #     break

                (fX, fY, fW, fH) = faces[i]

                # 从灰度图中提取感兴趣区域（ROI），将其大小转换为与模型输入相同的尺寸，并为通过CNN的分类器准备ROI
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, self.emotion_classifier.input_shape[1:3])
                roi = process_data(roi)
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # 用模型预测各分类的概率
                predicts = self.emotion_classifier.predict(roi)[0]
                label = self.EMOTIONS[predicts.argmax()]  # 选取最大概率的表情类
                # 圈出人脸区域并显示识别结果
                cv2.putText(frame_clone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0), 1)
                cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH), (255, 255, 0), 1)

        for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, predicts)):
            # 用于显示各类别概率
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # 绘制表情类和对应概率的条形图
            w = int(prob * 300) + 7
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (224, 200, 130), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        # 调整画面大小与界面相适应
        frame_clone = cv2.resize(frame_clone, (420, 280))

        # 在Qt界面中显示人脸
        show = cv2.cvtColor(frame_clone, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
        QtWidgets.QApplication.processEvents()

        # 在显示结果的label中显示结果
        show = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_result.setPixmap(QtGui.QPixmap.fromImage(showImage))

        return label