import numpy as np
import os
import matplotlib.image as mping
from PIL import Image
import torch

def clip_img(img):
    w = img.shape[0]
    h = img.shape[1]
    if w < h:
        s = int((h-w)/2)
        return Image.fromarray(img[:,s:s+w]).convert('RGB')
    elif w > h:
        s = int((w - h) / 2)
        return Image.fromarray(img[s:s+h,:]).convert('RGB')
    else:
        return Image.fromarray(img).convert('RGB')


def read_imgs(dirs):
    imgs_list = [[],[]]
    num = 0
    for directory in dirs:
        num += 1
        imgs = []
        for file in os.listdir(directory):
            img = mping.imread(os.path.join(directory, file))
            imgs.append((np.array(clip_img(img).resize((224, 224), Image.ANTIALIAS)).swapaxes(0, 2)/255).astype('float32'))
        catergory = np.zeros(len(imgs)) + num
        imgs_list[0].extend(imgs)
        imgs_list[1].extend(catergory.astype('uint').tolist())
    return imgs_list


if __name__ == '__main__':
    root = 'flowers'
    dirs = [os.path.join(root, i) for i in os.listdir(root)]
    imgs = read_imgs(dirs)
    print()