import glob
import random
import os
import numpy as np
import math

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import sys


def KIS_make_datapath_list(opt):

    #フォルダの名前の数を調べる
    str_len = len('pix_system_ver.1/input/'.format(opt.dataset_name))
    #画像リスト(***.jpg形式)
    filenames =  glob.glob('pix_system_ver.1/input/*.jpg'.format(opt.dataset_name))
    img_list = [filename[str_len:] for filename in filenames]
    #imgのリストをシャフルする
    # img_list = random.sample(img_list, len(img_list))

    img_path = "pix_system_ver.1/input/%s".format(opt.dataset_name)

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    data = list()

    for img_l in img_list:
        # file_id = line.strip()  # 空白スペースと改行を除去
        path = (img_path % img_l)  # 画像のパス
        data.append(path)

    return data



class KIS_ImageDataset(Dataset):

    def __init__(self, data, opt, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.data = data
        self.opt = opt

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.data)

    def __getitem__(self, index):
        #画像Aの読み込み(入力画像)
        input_image = self.data[index]
        #convert("L").convert("RGB")としているのはモノクロ画像を3chにするため。
        #カラー画像を使う場合はconvert("L").convert("RGB")を削除する。
        if self.opt.mode == "gray2color":
            input_image = Image.open(input_image).convert("L").convert("RGB")
        else:
            input_image = Image.open(input_image)


        #transformはPILファイルで入れる
        input_image = self.transform(input_image)

        return input_image

