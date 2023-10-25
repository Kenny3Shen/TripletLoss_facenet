import json
import os

import numpy as np
import torch
import umap
from PIL import Image
from torch.backends import cudnn
from tqdm import tqdm

from nets.facenet import Facenet
from utils.utils import resize_image, preprocess_input

config = json.load(open('eval_config.json', 'r'))
# ---------------------#
#   训练集所在的路径
# ---------------------#
datasets_path = config["datasets_path"]
# -------------------------------------------#
#   是否进行不失真的resize
# -------------------------------------------#
letterbox_image = True

input_shape = config["input_shape"]
backbone = config["backbone"]
model_path = config["model_path"]
half_face = config["half_face"]

cuda = True


def get_embedding(img):
    img = resize_image(img, [input_shape[1], input_shape[0]], letterbox_image=letterbox_image)
    photo_1 = torch.from_numpy(
        np.expand_dims(np.transpose(preprocess_input(np.array(img, np.float32)), (2, 0, 1)), 0))
    if cuda:
        photo_1 = photo_1.cuda()
    output1 = model(photo_1).cpu().detach().numpy()
    return output1


def calc_recognition(person_list, database):
    true_recognition = 0
    total = len(person_list)
    for name in tqdm(person_list):
        img_nums = len(os.listdir(f"{datasets_path}/{name}"))
        i = np.random.randint(1, img_nums)
        input_img = f"{datasets_path}/{name}/{name}_{i:04d}.jpg"
        if half_face:
            img = Image.open(input_img).convert('1').convert('RGB')
        else:
            img = Image.open(input_img)
        img_embedding = get_embedding(img)

        dist = 10000
        min_dist_person = None
        for k, v in database.items():
            cur_dist = np.linalg.norm(v - img_embedding)
            if cur_dist < dist:
                dist = cur_dist
                min_dist_person = k
        if min_dist_person == name:
            true_recognition += 1
    file_name = "lfw_recognition.txt" if not half_face else "lfw_recognition_half.txt"
    with open(file_name, "w") as f:
        f.write(f"true_recongnition: {true_recognition}, total: {total}, accuracy: {true_recognition / total}")
    print(f"true_recongnition: {true_recognition}, total: {total}, accuracy: {true_recognition / total}")


if __name__ == "__main__":

    # -------------------------------------------#
    #   load model
    # -------------------------------------------#

    model = Facenet(backbone=backbone, mode="predict")
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.eval()

    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    person_list = os.listdir(datasets_path)
    # -------------------------------------------#
    #   load database
    # -------------------------------------------#
    if not os.path.exists("database.json"):
        database = {}
        for name in person_list:
            database[name] = f"{datasets_path}/{name}/{name}_{1:04d}.jpg"
        with open("database.json", "w") as f:
            json.dump(database, f)
    else:
        with open("database.json", "r") as f:
            database = json.load(f)

    for (k, v) in tqdm(database.items()):
        database[k] = get_embedding(Image.open(v))

    calc_recognition(person_list, database)
