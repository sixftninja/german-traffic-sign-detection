from __future__ import print_function
from tqdm import tqdm
import os
import PIL.Image as Image
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from model import Net1, Net2, Net3, Net4, Net5


parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--outfile', type=str, default='gtsrb_kaggle.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()


test_dir = args.data + '/test_images'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

model_files = [
    ('best_models/Net1_epoch6_val0.058281_acc99.198.pth', Net1, True),
    ('best_models/Net2_epoch7_val0.020730_acc99.638.pth', Net2, True),
    ('best_models/Net3_epoch47_val0.012727_acc99.689.pth', Net3, True),
    ('best_models/Net4_epoch51_val0.012395_acc99.793.pth', Net4, True),
    ('best_models/Net5_epoch22_val0.020809_acc99.715.pth', Net5, True),
]
models = []; clahes = [True, True, True, True, True]
for modelf, cls, clahe in model_files:
    model = cls()
    model.load_state_dict(torch.load(modelf))
    model.eval()
    models.append(model)
    clahes.append(clahe)


output_file = open(args.outfile, "w")
output_file.write("Filename,ClassId\n")
with torch.no_grad():
    for f in tqdm(os.listdir(test_dir)):
        if 'ppm' not in f: continue
        output = torch.zeros([1, 43], dtype=torch.float32).to(device)
        for model, clahe in zip(models, clahes):
            for t in random_transforms:
                tdt = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.RandomChoice(random_transforms),
                    transforms.Resize((48, 48)),
                    CLAHE(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ]) if clahe else transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.RandomChoice(random_transforms),
                    transforms.Resize((48, 48)),
                    # CLAHE(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ])
                data = tdt(pil_loader(test_dir + '/' + f))
                data = data.view(1, data.size(0), data.size(1), data.size(2))
                output = output.add(model(data))
        pred = output.data.max(1, keepdim=True)[1]
        file_id = f[0:5]
        output_file.write("%s,%d\n" % (file_id, pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle '
      'competition at https://www.kaggle.com/c/nyu-cv-fall-2018/')
