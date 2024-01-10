#
# Imports
#
from fastai.vision.all import *
from tqdm import tqdm
import pathlib

from data_SN7 import *
from data_NEW2 import *
from data_SN3 import *


# Parametres
ARCHITECTURE = xresnet34
CLASS_WEIGHTS = [0.25,0.75]
LR_MAX = 3e-4
ENCODER_FACTOR = 10
CODES = ['Land','Building']


#SN6_Train_AOI_11_Rotterdam_PS-RGB_20190804111224_20190804111453_tile_8679
#SN6_Train_AOI_11_Rotterdam_Buildings_20190804111224_20190804111453_tile_8679
def save_masks_sn6(ipath):
  for img in (ipath/'PS-RGB').ls():
    shapes = ipath/'geojson_buildings'/((img.name[:-4]).replace('PS-RGB', 'Buildings')+'.geojson')
    if not os.path.exists(ipath/'binary_mask'/img.name):
      if not os.path.exists(ipath/'binary_mask'):
        os.makedirs(ipath/'binary_mask')
      generate_mask(img, shapes, ipath/'binary_mask', img.name)
   

def define_model(get_x, get_y, model_fname, weight_fname, data_path, batch_size=1, vocabs=CODES, trans=False, custom_splitter=False):
   #
   # Distribution of building density
   #
   tiles = DataBlock(
      blocks = (ImageBlock(),MaskBlock(codes=CODES)),
      get_items = get_x,
      get_y = get_y
      )              
   dls = tiles.dataloaders(data_path, bs=batch_size)
   dls.vocab = vocabs

   #
   # Creating Dataloaders
   #
   if (trans==True):
     tfms = [
        Dihedral(0.5),              # Horizontal and vertical flip
        Rotate(max_deg=180, p=0.9), # Rotation in any direction possible
        Brightness(0.2, p=0.75),
        Contrast(0.2),
        Saturation(0.2),
        (Normalize.from_stats(*imagenet_stats))
        ]
     if (custom_splitter==True):
        tiles = DataBlock(
           blocks = (ImageBlock(),MaskBlock(codes=CODES)), 
           get_items = get_x,             
           get_y = get_y,                                  
           splitter = FuncSplitter(valid_split),           
           batch_tfms = tfms                               
           )
     else:
        tiles = DataBlock(
           blocks = (ImageBlock(),MaskBlock(codes=CODES)), 
           get_items = get_x,             
           get_y = get_y,                                           
           batch_tfms = tfms                               
           )    
        
   dls = tiles.dataloaders(data_path, bs=batch_size, num_workers=0)
   dls.vocab = vocabs

   # Print data split
   print(len(dls.train_ds), len(dls.valid_ds))
   inputs, targets = dls.one_batch()
   # Print data shape
   print(inputs.shape, targets.shape)
   # Print data feature
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
                              fname=weight_fname
                            )]
                     )
   print(learn.summary())
   print(learn.model)
   learn.export(fname=model_fname+'.pkl')
   return learn, dls
   

def train(learn, dls, weight_fname, lr=1e-4, epochs=5):
   learn.unfreeze()
   learn.fit_one_cycle(
        epochs,
        lr_max=slice(lr/ENCODER_FACTOR, lr) 
   )
   learn.recorder.plot_loss()
   learn.load(weight_fname)
   #
   #Visualizing the Results
   #
   probs,targets,preds,losses = learn.get_preds(dl=dls.valid,
                                             with_loss=True,
                                             with_decoded=True,
                                             act=None)
   loss_sorted = torch.argsort(losses, descending=True)
   n = len(loss_sorted)
   
   
def predict(model_path, weight_fname, img, resize=(3000, 3000)):
   torch.cuda.empty_cache()
   model = load_learner(model_path, cpu=False)
   model.load(weight_fname)
   im = Image.open(img)
   im.thumbnail(resize, Image.Resampling.LANCZOS)
   m = model.predict(im)
   m = to_image(m[2][1].repeat(3,1,1))
   pathsave = r'C:\Users\Timbo\Documents\Projet\multi\data\mask.jpg'
   m.save(pathsave, "JPEG")
   print(m)
   m.show()


if __name__ == "__main__":
    print('main')
    print(torch.cuda.is_available())