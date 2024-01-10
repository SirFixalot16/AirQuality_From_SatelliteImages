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

#
# Downscaling images
#          
def downscale_SN3(ipath, size:int, format="PNG"):
   for ct in tqdm(pathlib.Path(ipath).ls().sorted()):
      for img in (ct/'images').ls():
         msk = ct/'mask'/(img.name[:-4]+'.png')
         pathread_i = pathlib.Path(img)
         pathread_m = pathlib.Path(msk)
         pathsave_i = pathlib.Path(r'C:\Users\Timbo\Documents\Projet\multi\data\spacenet3/ids/'+ img.name[:-4]+'.png'
                                   )
         pathsave_m = pathlib.Path(r'C:\Users\Timbo\Documents\Projet\multi\data\spacenet3/mds/'+ img.name[:-4]+'.png'
                                   )

         #print(pathread_i)
         #print(pathread_m)
         #print(pathsave_i)
         #print(pathsave_m)

         im = Image.open(pathread_i)
         mk = Image.open(pathread_m)
         im.thumbnail((size, size), Image.Resampling.LANCZOS)
         im.save(pathsave_i, format=format)
         mk.thumbnail((size, size), Image.Resampling.LANCZOS)
         mk.save(pathsave_m, format=format)

#
# Data Loading Functions
#
def get_image_tiles_SN3(path:Path) -> L:
   files = L()
   files = get_image_files(path=path, folders='ids')
   return files

def get_y_fn_SN3(fn:Path) -> str:
  #print(str(fn).replace('im_tiles', 'gt_tiles'))
  return str(fn).replace('ids', 'mds')

def get_y_SN3(fn:Path) -> PILMask:
   fn = get_y_fn_SN3(fn)
   msk = np.array(PILMask.create(fn))
   #print(msk)
   #msk[msk==255] = 1
   msk = msk / 255
   #print(msk)
   return PILMask.create(msk)

if __name__ == "__main__":
  print('SN3')
  pathSN3 = pathlib.Path(r"C:\Users\Timbo\Documents\Projet\multi\data\spacenet3")