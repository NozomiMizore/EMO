import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')
from gui import MyWindow
from sys import argv, exit
from PyQt5.QtWidgets import QApplication,QMainWindow

if __name__ == '__main__':
    app = QApplication(argv)

    window = QMainWindow()
    ui = MyWindow(window)

    window.show()
    exit(app.exec_())
