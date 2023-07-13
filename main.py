import sys

import torch
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtGui import QPixmap

import arthrosis_trainer
import common
from hand_view import Ui_MainWindow
import hand_bone_detect


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.bind_slots()
        # 加载yolov5模型（本地训练好的）
        self.mode = torch.hub.load('ultralytics/yolov5', 'custom', path='params/best.pt')
        self.mode.eval()
        # self.mode.conf = 0.4  # 置信度
        print("yolov5模型加载完成")

    # 信号槽函数
    def btn_open_img(self):
        print("点击按钮")
        file_path, _ = QFileDialog.getOpenFileName(self, directory="./img", filter="Image Files (*.png *.jpeg *.jpg)")
        if file_path:
            # 选择图片
            print(file_path)
            # 回显手骨x光片
            self.label_2.setPixmap(QPixmap(file_path))

            # 获取性别
            sex = 'boy' if self.radioButton.isChecked() else 'girl'
            print(sex)

            # 侦测
            result = hand_bone_detect.detect(self.mode, sex, file_path)


            # 显示检测结果
            self.label_3.setText(result)

    # 绑定槽
    def bind_slots(self):
        self.pushButton.clicked.connect(self.btn_open_img)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
