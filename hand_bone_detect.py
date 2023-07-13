from PIL import Image
from torchvision.transforms import ToTensor

import arthrosis_trainer
import common
import cv2
import os
import torch
from torchvision import models
from torch import nn


def detect(yolo_model, sex, path_img):
    results = yolo_model(path_img)
    # 获取侦测结果
    box = results.xyxy[0]
    print(box)

    if box.shape[0] != 21:
        return "侦测的到的关节数量不正确，请重新放一张x光片！"
    # 左右手判断
    # 如果是右手  return "检测到右手，请放入左手x光片！"
    # 关节筛选
    box = common.bone_filter(box)
    print(box)
    img = cv2.imread(path_img)
    for i, name in enumerate(common.arthrosis_new):
        x1 = int(box[i][0])
        y1 = int(box[i][1])
        x2 = int(box[i][2])
        y2 = int(box[i][3])

        print(x1, y1, x2, y2)
        # 裁切图片的坐标    开始的y坐标:结束的y坐标,开始x:结束的x
        img_roi = img[y1:y2, x1:x2]
        if not os.path.exists("cut_img"):
            os.mkdir("cut_img")
        cv2.imwrite(f'cut_img/{name}.png', img_roi)

    result = []
    res = {}
    result_score = 0
    for item in common.arthrosis:
        idex = inference_single_image(common.arthrosis[item][0], item)
        result.append(common.SCORE[sex][item][idex])
        result_score += common.SCORE[sex][item][idex]
        res[item] = [idex, common.SCORE[sex][item][idex]]

    age = common.calcBoneAge(result_score, sex)

    # result_str = ''
    # for i, name in enumerate(common.arthrosis):
    #     result_str = result_str+(name+' :  '+str(result[i])+'\n')

    return common.export(res, result_score, age)


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def inference_single_image(category, name):
    # 加载模型
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, arthrosis_trainer.arthrosis[category][1])
    model = model.to(device)

    # 加载保存的模型参数
    model_path = f"params/{category}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载和预处理图片

    image = common.trans_square(Image.open(f'cut_img/{name}.png'))
    image = image.convert("L")  # 将图片转为灰度图像
    transform = ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)  # 将图片转为张量并添加一个维度

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    print(prediction)
    return prediction
