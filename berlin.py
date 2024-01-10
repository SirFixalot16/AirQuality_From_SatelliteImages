# imports
import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize

import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union

from fastai.vision.all import *
from tqdm import tqdm
import pathlib

# Parametres
BATCH_SIZE = 12 
BATCH_SIZE_NEW2 = 1
TILES_PER_SCENE = 16
ARCHITECTURE = xresnet34
EPOCHS = 40
CLASS_WEIGHTS = [0.25,0.75]
LR_MAX = 3e-4
ENCODER_FACTOR = 10
CODES = ['Land','Building']
path = pathlib.Path(r"C:\Users\Timbo\Documents\Projet\multi\data\SN7_buildings_train\train")
pathNEW2 = pathlib.Path(r"C:\Users\Timbo\Documents\Projet\multi\data\NEW2\AerialImageDataset\train")

#
# SOURCE:  https://lpsmlgeo.github.io/2019-09-22-binary_mask/
#
def generate_mask(raster_path, shape_path, output_path=None, file_name=None):

    """Function that generates a binary mask from a vector file (shp or geojson)
    raster_path = path to the .tif;
    shape_path = path to the shapefile or GeoJson.
    output_path = Path to save the binary mask.
    file_name = Name of the file.
    """
    
    #load raster
    
    with rasterio.open(raster_path, "r") as src:
        raster_img = src.read()
        raster_meta = src.meta
    
    #load o shapefile ou GeoJson
    train_df = gpd.read_file(shape_path)
    #print(train_df)
    
    #Verify crs
    if train_df.crs != src.crs:
        print(" Raster crs : {}, Vector crs : {}.\n Convert vector and raster to the same CRS.".format(src.crs,train_df.crs))
        
        
    #Function that generates the mask
    def poly_from_utm(polygon, transform):
        poly_pts = []

        poly = cascaded_union(polygon)
        for i in np.array(poly.exterior.coords):

            poly_pts.append(~transform * tuple(i))

        new_poly = Polygon(poly_pts)
        return new_poly
    
    
    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in train_df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            #poly = cascaded_union(row['geometry'])
            #polyl = []
            #for i in np.array(poly.exterior.coords): polyl.append(tuple(i))
            #poly = Polygon(polyl)
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, src.meta['transform'])
                #poly = p
                #polyl = []
                #for i in np.array(poly.exterior.coords): polyl.append(tuple(i))
                #poly = Polygon(polyl)
                poly_shp.append(poly)

    #set_trace()
    
    if len(poly_shp) > 0:
      mask = rasterize(shapes=poly_shp,
                      out_shape=im_size)
    else:
      mask = np.zeros(im_size)
    
    # Save or show mask
    mask = mask.astype("uint8")    
    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})
    if (output_path != None and file_name != None):
      os.chdir(output_path)
      with rasterio.open(file_name, 'w', **bin_mask_meta) as dst:
          dst.write(mask * 255, 1)
    else: 
      return mask
        
#
# Creating binary masks
#
def save_masks():
  for scene in tqdm(path.ls().sorted()):
    for img in (scene/'images_masked').ls():
      shapes = scene/'labels_match'/(img.name[:-4]+'_Buildings.geojson')
      if not os.path.exists(scene/'binary_mask'/img.name):
        if not os.path.exists(scene/'binary_mask'):
          os.makedirs(scene/'binary_mask')
        generate_mask(img, shapes, scene/'binary_mask', img.name)

#SN6_Train_AOI_11_Rotterdam_PS-RGB_20190804111224_20190804111453_tile_8679
#SN6_Train_AOI_11_Rotterdam_Buildings_20190804111224_20190804111453_tile_8679
def save_masks_sn6(ipath):
  for img in (ipath/'PS-RGB').ls():
    shapes = ipath/'geojson_buildings'/((img.name[:-4]).replace('PS-RGB', 'Buildings')+'.geojson')
    if not os.path.exists(ipath/'binary_mask'/img.name):
      if not os.path.exists(ipath/'binary_mask'):
        os.makedirs(ipath/'binary_mask')
      generate_mask(img, shapes, ipath/'binary_mask', img.name)
   
def get_masked_images(path:Path, n=1)->list:
  "Returns the first `n` pictures from every scene"
  files = []
  for folder in path.ls():
    files.extend(get_image_files(path=folder, folders='images_masked')[:n])
  return files


#
# Cutting images in tiles
#
def cut_tiles(tile_size:int):
  "Cuts the large images and masks into equal tiles and saves them to disk"
  masked_images = get_masked_images(path, 5)
  for fn in tqdm(masked_images):
    scene = fn.parent.parent

    # Create directories
    if not os.path.exists(scene/'img_tiles'):
      os.makedirs(scene/'img_tiles')
    if not os.path.exists(scene/'mask_tiles'):
      os.makedirs(scene/'mask_tiles')

    # Create mask for current image
    img = np.array(PILImage.create(fn))
    msk_fn = str(fn).replace('images_masked', 'binary_mask')
    msk = np.array(PILMask.create(msk_fn))
    x, y, _ = img.shape

    # Cut tiles and save them
    for i in range(x//tile_size):
      for j in range(y//tile_size):
        img_tile = img[i*tile_size:(i+1)*tile_size,j*tile_size:(j+1)*tile_size]
        msk_tile = msk[i*tile_size:(i+1)*tile_size,j*tile_size:(j+1)*tile_size]
        Image.fromarray(img_tile).save(f'{scene}/img_tiles/{fn.name[:-4]}_{i}_{j}.png')
        Image.fromarray(msk_tile).save(f'{scene}/mask_tiles/{fn.name[:-4]}_{i}_{j}.png')

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
def downscale_NEW2(ipath, size:int):
   arr = os.listdir(ipath)
   for ip in arr:
      pathread = pathlib.Path(ipath+'/'+ip)
      pathsave = pathlib.Path(ipath+'_ds2/'+ip)
      im = Image.open(pathread)
      im.thumbnail((size, size), Image.Resampling.LANCZOS)
      im.save(pathsave, "PNG")
      
#
# Data Loading Functions
#
def get_image_tiles(path:Path, n_tiles=TILES_PER_SCENE) -> L:
  "Returns a list of the first `n` image tile filenames in `path`"
  files = L()
  for folder in path.ls():
    files.extend(get_image_files(path=folder, folders='img_tiles')[:n_tiles])
  return files

def get_y_fn(fn:Path) -> str:
  "Returns filename of the associated mask tile for a given image tile"
  return str(fn).replace('img_tiles', 'mask_tiles')

def get_y(fn:Path) -> PILMask:
  "Returns a PILMask object of 0s and 1s for a given tile"
  fn = get_y_fn(fn)
  msk = np.array(PILMask.create(fn))
  msk[msk==255] = 1
  return PILMask.create(msk)

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


#
# Visualizing the Data
#
def show_tiles(n):
  all_tiles = get_image_tiles(path)
  subset = random.sample(all_tiles, n)
  fig, ax = plt.subplots(n//2, 4, figsize=(14,14))
  for i in range(n):
    y = i//2
    x = 2*i%4
    PILImage.create(subset[i]).show(ctx=ax[y, x])
    get_y(subset[i]).show(ctx=ax[y, x+1], cmap='cividis')
  fig.tight_layout()
  plt.show()

#
# Validation Strategy
#
VALID_SCENES = ['L15-0571E-1075N_2287_3888_13',
 'L15-1615E-1205N_6460_3370_13',
 'L15-1210E-1025N_4840_4088_13',
 'L15-1185E-0935N_4742_4450_13',
 'L15-1481E-1119N_5927_3715_13',
 'L15-0632E-0892N_2528_4620_13',
 'L15-1438E-1134N_5753_3655_13',
 'L15-0924E-1108N_3699_3757_13',
 'L15-0457E-1135N_1831_3648_13']

def valid_split(item):
  scene = item.parent.parent.name
  return scene in VALID_SCENES

#
# Undersampling
#
def has_buildings(fn:Path) -> bool:
  """Returns whether the mask of a given image tile
  contains at least one pixel of a building"""
  fn = get_y_fn(fn)
  msk = tensor(PILMask.create(fn))
  count = torch.count_nonzero(msk)
  return count>0.

def get_undersampled_tiles(path:Path) -> L:
  """Returns a list of image tile filenames in `path`.
  For tiles in the training set, empty tiles are ignored.
  All tiles in the validation set are included."""

  files = get_image_tiles(path)
  train_idxs, valid_idxs = FuncSplitter(valid_split)(files)
  train_files = L(filter(has_buildings, files[train_idxs]))
  valid_files = files[valid_idxs]

  return train_files + valid_files


def show_single_pred(dls, preds, probs, index:int):
  fig, ax = plt.subplots(1, 4, figsize=(20,5))
  dls.valid_ds[index][0].show(ctx=ax[0]);
  ax[0].set_title("Input")
  show_at(dls.valid_ds, index, cmap='Blues', ctx=ax[1]);
  ax[1].set_title("Target")
  preds[index].show(cmap='Blues', ctx=ax[2]);
  ax[2].set_title("Prediction Mask")
  probs[index][1].show(cmap='viridis', ctx=ax[3]);
  ax[3].set_title("Building class probability")


#
# Show complete scenes
#
def save_predictions(learn, scene, path=path) -> None:
  "Predicts all 16 tiles of one scene and saves them to disk"
  output_folder = path/scene/'predicted_tiles'
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  tiles = get_image_files(path/scene/'img_tiles').sorted()
  for i in range(16):
    tile_preds = learn.predict(tiles[i])
    to_image(tile_preds[2][1].repeat(3,1,1)).save(output_folder/f'{i:02d}.png')


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.

    Source: https://stackoverflow.com/a/16873755
    """
    try: # with color channel
      n, nrows, ncols, c = arr.shape
      return (arr.reshape(h//nrows, -1, nrows, ncols, c)
                .swapaxes(1,2)
                .reshape(h, w, c))
    except ValueError: # without color channel
      n, nrows, ncols = arr.shape
      return (arr.reshape(h//nrows, -1, nrows, ncols)
                .swapaxes(1,2)
                .reshape(h, w))
    
def get_saved_preds(scene, path=path):
  "Load saved prediction mask tiles for a scene and return image + assembled mask"
  image_file = (path/scene/'images_masked').ls()[0]
  image = load_image(image_file)

  mask_tiles = get_image_files(path/scene/'predicted_tiles').sorted()
  mask_arrs = np.array(list(maps(partial(load_image, mode="L"), np.asarray, mask_tiles)))
  mask_array = unblockshaped(np.array(mask_arrs), 1020, 1020)

  return (image, mask_array)

def show_complete_preds(image, mask_array, scene):
  figsize = (25, 16)
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
  _ = ax0.imshow(image)
  ax0.set_xticks([])
  ax0.set_yticks([])
  ax0.set_title('Image')
  _ = ax1.imshow(mask_array, cmap='viridis')
  ax1.set_xticks([])
  ax1.set_yticks([])
  ax1.set_title('Prediction Mask')
  plt.suptitle(scene)
  plt.tight_layout()
  plt.savefig(os.path.join(path, scene + '_im0+mask0+dice575.png'))
  plt.show()

# Main

def preprocess():
   #save_masks()
   TILE_SIZE = 255
   cut_tiles(TILE_SIZE)

def define_model():
   #
   # Distribution of building density
   #
   tiles = DataBlock(
      blocks = (ImageBlock(),MaskBlock(codes=CODES)),
      get_items = get_image_tiles,
      get_y = get_y
      )              
   dls = tiles.dataloaders(path, bs=BATCH_SIZE)
   dls.vocab = CODES 

   #
   # Creating Dataloaders
   #
   tfms = [Dihedral(0.5),              # Horizontal and vertical flip
        Rotate(max_deg=180, p=0.9), # Rotation in any direction possible
        Brightness(0.2, p=0.75),
        Contrast(0.2),
        Saturation(0.2),
        (Normalize.from_stats(*imagenet_stats))
        ]
   tiles = DataBlock(
      blocks = (ImageBlock(),MaskBlock(codes=CODES)), # Independent variable is Image, dependent variable is Mask
      get_items = get_undersampled_tiles,             # Collect undersampled tiles
      get_y = get_y,                                  # Get dependent variable: mask
      splitter = FuncSplitter(valid_split),           # Split into training and validation set
      batch_tfms = tfms                               # Transforms on GPU: augmentation, normalization
    )   
   dls = tiles.dataloaders(path, bs=BATCH_SIZE, num_workers=0)
   dls.vocab = CODES
   print(len(dls.train_ds), len(dls.valid_ds))
   inputs, targets = dls.one_batch()
   print(inputs.shape, targets.shape)
   print(targets[0].unique())
   #
   # Defining the Model
   #
   weights = Tensor(CLASS_WEIGHTS).cuda()
   loss_func = CrossEntropyLossFlat(axis=1, weight=weights)
   learn = unet_learner(dls,                                 # DataLoaders
                     ARCHITECTURE,                        # xResNet34
                     loss_func = loss_func,               # Weighted cross entropy loss
                     opt_func = Adam,                     # Adam optimizer
                     metrics = [Dice(), foreground_acc],  # Custom metrics
                     self_attention = False,
                     cbs = [SaveModelCallback(
                              monitor='dice',
                              comp=np.greater,
                              fname='best-model'
                            )]
                     )
   print(learn.summary())
   print(learn.model)
   learn.export(fname='berlin.pkl')
   return learn, dls

def define_model_NEW2():
   #
   # Creating Dataloaders
   #
   tfms = [Dihedral(0.5),              # Horizontal and vertical flip
        Rotate(max_deg=180, p=0.9), # Rotation in any direction possible
        Brightness(0.2, p=0.75),
        Contrast(0.2),
        Saturation(0.2),
        (Normalize.from_stats(*imagenet_stats))
        ]
   tiles = DataBlock(
      blocks = (ImageBlock(),MaskBlock(codes=CODES)), 
      get_items = get_image_tiles_NEW2,
      get_y = get_y_NEW2,                                           
      batch_tfms = tfms                               
    )   
   dls = tiles.dataloaders(pathNEW2, bs=BATCH_SIZE_NEW2, num_workers=0)
   dls.vocab = CODES
   print(len(dls.train_ds), len(dls.valid_ds))
   inputs, targets = dls.one_batch()
   print(inputs.shape, targets.shape)
   print(targets[0].unique())
   #
   # Defining the Model
   #
   weights = Tensor(CLASS_WEIGHTS).cuda()
   loss_func = CrossEntropyLossFlat(axis=1, weight=weights)
   learn = unet_learner(dls,                                 # DataLoaders
                     ARCHITECTURE,                        # xResNet34
                     loss_func = loss_func,               # Weighted cross entropy loss
                     opt_func = Adam,                     # Adam optimizer
                     metrics = [Dice(), foreground_acc],  # Custom metrics
                     self_attention = False,
                     cbs = [SaveModelCallback(
                              monitor='dice',
                              comp=np.greater,
                              fname='best-model'
                            )]
                     )
   #print(learn.summary())
   #print(learn.model)
   learn.export(fname='berlin3.pkl')
   return learn, dls
   
def train(learn, dls):
   lr_max = LR_MAX # 3e-4
   learn.unfreeze()
   learn.fit_one_cycle(
        EPOCHS,
        lr_max=slice(lr_max/ENCODER_FACTOR, lr_max) 
   )
   learn.recorder.plot_loss()
   learn.load('best-model')
   #
   #Visualizing the Results
   #
   probs,targets,preds,losses = learn.get_preds(dl=dls.valid,
                                             with_loss=True,
                                             with_decoded=True,
                                             act=None)
   loss_sorted = torch.argsort(losses, descending=True)
   n = len(loss_sorted)

def import_model(path):
   learn = load_learner(path+r'berlin.pkl')
   return learn
   
def predict(learn):
   learn.load('best-model')
   
   scene = VALID_SCENES[0] # 'L15-0571E-1075N_2287_3888_13'
   save_predictions(learn, scene)
   show_complete_preds(*get_saved_preds(scene), scene)
   
def main(model_path, img):
   torch.cuda.empty_cache()
   model = load_learner(model_path+r'berlin3.pkl', cpu=False)
   model.load('best-model')
   im = Image.open(img)
   im.thumbnail((3000, 3000), Image.Resampling.LANCZOS)
   m = model.predict(im)
   m = to_image(m[2][1].repeat(3,1,1))
   pathsave = r'C:\Users\Timbo\Documents\Projet\multi\data\mask.jpg'
   m.save(pathsave, "JPEG")
   print(m)
   m.show()

def ghnm(model_path, img_path):
   torch.cuda.empty_cache()
   model = load_learner(model_path+r'berlin.pkl', cpu=False)
   model.load('best-model')
   img = get_image_files(pathlib.Path(img_path))
   #print(img)
   npf = np.asarray(os.listdir(img_path))
   for i in range(len(img)):
      m = model.predict(img[i])
      m = to_image(m[2][1].repeat(3,1,1))
      pathsave = r'C:\Users\Timbo\Documents\Projet\multi\data\earth_mask/' + npf[i]
      m.save(pathsave, "JPEG")


if __name__ == "__main__":
    #test()
    print('dick pachinko')
    print(torch.cuda.is_available())
    #preprocess()
    #model, loader = define_model()
    #model_path = r'C:/Users/Timbo/Documents/Projet/multi/models/'
    model_path = r'./'
    #import_model(model_path)
    #predict(model)
    i = r'C:/Users/Timbo/Documents/Projet/multi/data/test/phoco.jpg'
    #ghnm(model_path, i)
    main(model_path=model_path, img=i)

    new2path = r'C:/Users/Timbo/Documents/Projet/multi/data/NEW2/AerialImageDataset/train/'
    #cut_tiles_NEW2(new2path=new2path, tile_size=2500)
    #downscale_NEW2(r'C:/Users/Timbo/Documents/Projet/multi/data/NEW2/AerialImageDataset/train/im_tiles', 1250)
    #model, loader = define_model_NEW2()
    #train(model, loader)
