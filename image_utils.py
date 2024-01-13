#
# Imports
#
import numpy as np
from PIL import Image
import cv2

import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize

import pandas
import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union

import os


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
# Function to convert black-and-white mask to transparent mask
#
def greyscale_to_transparent(img:Image)->Image:
    data = img.getdata()
    
    newData = []
    for item in data:
        if item[0] > 100:
            newData.append((255, 255))
        else:
            newData.append((0, 0))

    img.putdata(newData)
    return img

#
# Convert white footprints to one of the three colors: red, green, blue 
#
def white_to_color(img:Image, color, pixel_shape=3000, background=False)->Image:
    red, green, blue = 0, 0, 0
    if (color == "red"): red = 255
    elif (color == "green"): green = 255
    elif (color == "blue"): blue = 255

    res = Image.new('RGBA', (pixel_shape, pixel_shape))
    data = img.getdata()
    newData = []

    for item in data:
        if item[0] == 255 and item[1] == 255:
            newData.append((red, green, blue, 255))
        elif background==True:
            newData.append((0, 0, 0, 255))
        else:
            newData.append((0, 0, 0, 0))
    
    res.putdata(newData)
    return res

#
# Merge 3 transparent masks to create RGB mask
#
def merge_masks(r, g:Image, b)->Image:
    # Open as PIL greyscale
    r = Image.open(r)
    r = r.convert('LA')
    #g = Image.open(g)
    #g = g.convert('LA')
    b = Image.open(b)
    b = b.convert('LA')

    # Convert to numpy array
    npr = np.array(r)
    #npg = np.array(g)
    npb = np.array(b)
    npr[npr <= 100] = 0
    #npg[npg <= 100] = 0
    npb[npb <= 100] = 0

    # Convert to channels
    r = Image.fromarray(npr, 'LA')
    #g = Image.fromarray(npg, 'LA')
    b = Image.fromarray(npb, 'LA')
    r = greyscale_to_transparent(r)
    #g = greyscale_to_transparent(g)
    b = greyscale_to_transparent(b)
    r = white_to_color(img=r, color='red', background=True)
    #g = white_to_color(img=b, color='green')
    b = white_to_color(img=b, color='blue')
    r.paste(g, g)
    r.paste(b, b)
    print(r)

#
# Segment all green elements in image and covert them to full green pixels
#
def generate_greens(img)->Image:
    im = Image.open(img)
    im.thumbnail((3000, 3000), Image.Resampling.LANCZOS)
    HSVim = im.convert('HSV')
    #im.show()
    im_og = np.array(im)
    im_arr = np.array(HSVim)
    H = im_arr[:,:,0]
    lo, hi = 100, 140
    lo = int((lo * 255) / 360)
    hi = int((hi * 255) / 360)
    green = np.where((H>lo) & (H<hi))
    im_og[green] = [0,0,0]
    im=Image.fromarray(im_og)
    data = im.getdata()
    newData = []
    for item in data:
        if item[0]==0 and item[1]==0 and item[2]==0:
            newData.append((0, 255, 0, 255))
        else:
            newData.append((0, 0, 0, 0))
        #print(item)
    res = Image.new('RGBA', (3000, 3000))
    res.putdata(newData)
    
    return res