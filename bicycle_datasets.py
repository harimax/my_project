import glob
import random
import os
import numpy as np
import math

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import sys

def make_datapath_list(opt, phase="train"):

    #フォルダの名前の数を調べる
    str_len = len('data/{}/{}_A/'.format(opt.dataset_name, phase))
    #画像リスト(***.jpg形式)
    filenames =  glob.glob('data/{}/{}_A/*.jpg'.format(opt.dataset_name, phase))
    img_list = [filename[str_len:] for filename in filenames]
    #imgのリストをシャフルする
    #img_list = random.sample(img_list, len(img_list))

    img_path_A = "data/{}/{}_A/%s".format(opt.dataset_name, phase)
    img_path_B = "data/{}/{}_B/%s".format(opt.dataset_name, phase)

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    data_A = list()
    data_B = list()

    for img_l in img_list:
        # file_id = line.strip()  # 空白スペースと改行を除去
        path_A = (img_path_A % img_l)  # 画像のパス
        path_B = (img_path_B % img_l)  # アノテーションのパス
        data_A.append(path_A)
        data_B.append(path_B)

    return data_A, data_B



class ImageDataset(Dataset):

    def __init__(self, data_A, data_B, opt, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.data_A = data_A
        self.data_B = data_B
        self.opt = opt

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.data_A)

    def __getitem__(self, index):
        #画像Aの読み込み(入力画像)
        data_A = self.data_A[index]
        #convert("L").convert("RGB")としているのはモノクロ画像を3chにするため。
        #カラー画像を使う場合はconvert("L").convert("RGB")を削除する。
        if self.opt.mode == "gray2color":
            img_A = Image.open(data_A).convert("L").convert("RGB")
        else:
            img_A = Image.open(data_A)

        #画像Bの読み込み(正解画像)
        data_B = self.data_B[index]
        img_B = Image.open(data_B).convert('RGB')   # [高さ][幅][色RGB]

        #transformはPILファイルで入れる
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}


