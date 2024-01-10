#
# Imports
#
from tqdm import tqdm
import pathlib
import os

import numpy as np
from PIL import Image
import cv2

from fastai.vision.all import *


def cut_tiles_NEW2(new2path, tile_size:int):
   gt_files = []
   gt_files = get_image_files(pathlib.Path(new2path+r'images'))
   for fn in tqdm(gt_files):
      img = np.array(PILImage.create(fn))
      msk_fn = str(fn).replace('images', 'gt')
      msk = np.array(PILMask.create(msk_fn))
      x, y, _ = img.shape

      for i in range(x//tile_size):
        for j in range(y//tile_size):
           img_tile = img[i*tile_size:(i+1)*tile_size,j*tile_size:(j+1)*tile_size]
           msk_tile = msk[i*tile_size:(i+1)*tile_size,j*tile_size:(j+1)*tile_size]
           Image.fromarray(img_tile).save(f'{new2path}/im_tiles/{fn.name[:-4]}_{i}_{j}.png')
           Image.fromarray(msk_tile).save(f'{new2path}/gt_tiles/{fn.name[:-4]}_{i}_{j}.png')

#
# Downscaling images
#            
def downscale_NEW2(ipath, size:int, format="PNG"):
   arr = os.listdir(ipath)
   for ip in arr:
      pathread = pathlib.Path(ipath+'/'+ip)
      pathsave = pathlib.Path(ipath+'_ds2/'+ip)
      im = Image.open(pathread)
      im.thumbnail((size, size), Image.Resampling.LANCZOS)
      im.save(pathsave, format=format)

#
# Data Loading Functions
#
def get_image_tiles_NEW2(path:Path) -> L:
   files = L()
   files = get_image_files(path=path, folders='im_tiles_ds2')
   return files

def get_y_fn_NEW2(fn:Path) -> str:
  #print(str(fn).replace('im_tiles', 'gt_tiles'))
  return str(fn).replace('im_tiles', 'gt_tiles')

def get_y_NEW2(fn:Path) -> PILMask:
   fn = get_y_fn_NEW2(fn)
   msk = np.array(PILMask.create(fn))
   #print(msk)
   #msk[msk==255] = 1
   msk = msk / 255
   #print(msk)
   return PILMask.create(msk)


if __name__ == "__main__":
  print('NEW2')
  pathNEW2 = pathlib.Path(r"C:\Users\Timbo\Documents\Projet\multi\data\NEW2\AerialImageDataset\train")
