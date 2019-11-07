from __future__ import print_function
from tqdm import tqdm
import os
import PIL.Image as Image

test_dir = args.data + '/test_images'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
from model import Net1, Net2

model_files = [
    ('best_models/GNet2_epoch22_val0.02080939228840332_acc99.7157622739018.pth', GNet2, False),
    ('best_models/GNet2_epoch7_val0.02073047148779981_acc99.63824289405684.pth', GNet2, False),
    ('best_models/GNet2Aug_epoch3_val0.07817342521608338_acc99.7416000366211.pth', GNet2, True),
    # ('best_models/GNet3NoDrop_epoch47_val0.012727167224390238_acc99.68992248062015.pth', GNet3, True),
    ('best_models/GNet3Im_epoch51_val0.01239507444759332_acc99.79328165374677.pth', GNet3, True),
]
models = []; clahes = [False, False, True, True]
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
'''
