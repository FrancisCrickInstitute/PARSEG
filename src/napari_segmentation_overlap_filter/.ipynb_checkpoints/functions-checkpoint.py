
import numpy as np
import pandas as pd
from collections import Counter
from skimage.segmentation import relabel_sequential
import napari
from timeit import default_timer as time
import dask.array as da

def timer_func(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f'{func.__name__}() executed in {(end-start):.6f} s')
        # print(f'{(end - start):.6f}')
        return result
    return wrapper

def return_overlap_segmentation(Img_A, Img_B):
    """
    function to return the overlapping segmentation in two images
    :param Img_A:
    :param Img_B:
    :return: 3D image where overlapping pixels have Img_A labels
    """
    Img_temp = np.where([Img_B > 0], Img_A, Img_B)

    return Img_temp[0, :, :, :]

def return_non_zero_pixels_from_Image(Img):
    """
    return the coordinates of non-zero pixels in an image
    :param Img:
    :return: Array of coordinates ((x1, y1, z1), (x2, y2, z2), (x3, y3, z3), ...)
    """
    coords = np.nonzero(Img)
    overlap_pixels = np.transpose(coords)

    return overlap_pixels

def overlapping_objects(Img_A, Img_B):
    """
    finds coordinates of overlapping pixels then returns a list of corresponding labels from Img_A and Img_B paired
    up with one another at each overlapping pixel
    :param Img_A:
    :param Img_B:
    :return:
    """
    Img_compare = return_overlap_segmentation(Img_A, Img_B)
    overlap_pixels = return_non_zero_pixels_from_Image(Img_compare)
    
    A_objects = []
    B_objects = []
    
    for n in np.arange(len(overlap_pixels)):
        X, Y, Z = overlap_pixels[n]
        
        A_objects.append(Img_A[X, Y, Z])
        B_objects.append(Img_B[X, Y, Z])
    
    objs = list((zip(A_objects, B_objects)))

    return objs

def get_object_sizes(Img, Object_list):
    """
    finds size of object by getting label of each overlapping pixel and counting their occurrences
    :param Img:
    :param Object_list:
    :return: dict_object of label: number of occurrences
    """
    Img_Non_Zero_pixels = return_non_zero_pixels_from_Image(Img)

    Obj_count = []
    for n in np.arange(len(Img_Non_Zero_pixels)):
        X, Y, Z = Img_Non_Zero_pixels[n]
        Obj_count.append(Img[X, Y, Z])

    Obj_dict = Counter(Obj_count)
    Obj_dict = {k: Obj_dict[k] for k in Object_list}

    return Obj_dict

@timer_func
def create_overlap_df(Img_A, Img_B):
    """
    creates a dataframe of all overlapping objects in each image, with object size and percentage of overlap
    :param Img_A:
    :param Img_B:
    :return: pd.dataframe
    """
    objects_zip = overlapping_objects(Img_A, Img_B) #this is ordered so the objects will correspond
    
    objects_overlap_count = Counter(objects_zip)
    objects_set = objects_overlap_count.keys()
    objects_overlap_size = objects_overlap_count.values()
    
    A_objects = [ls[0] for ls in objects_set]
    B_objects = [ls[1] for ls in objects_set]
    
    A_overlap_size = list(objects_overlap_size)
    B_overlap_size = list(objects_overlap_size)
    
    A_obj_dict = get_object_sizes(Img_A, A_objects)
    B_obj_dict = get_object_sizes(Img_B, B_objects)
    
    Obj_A_size = []
    for obj in A_objects:
        Obj_A_size.append(A_obj_dict[obj])
    
    Obj_B_size = []
    for obj in B_objects:
        Obj_B_size.append(B_obj_dict[obj])    
    
    Overlap_Index = [n for n in np.arange(len(A_objects))]  # create overlap index
    
    data = {'Overlap_Index': Overlap_Index, 
       'Image_A_Object_ID': A_objects,
       'Image_B_Object_ID': B_objects,
       'Image_A_Object_Size': Obj_A_size,
       'Image_B_Object_Size': Obj_B_size,
       'Image_A_Overlap_Size': A_overlap_size,
       'Image_B_Overlap_Size': B_overlap_size}

    df = pd.DataFrame(data)
    assert(list(df['Image_A_Overlap_Size'])==list(df['Image_B_Overlap_Size'])) ## these should be identical
    df['Image_A_Overlap_PCT'] = df['Image_A_Overlap_Size']/df['Image_A_Object_Size']
    df['Image_B_Overlap_PCT'] = df['Image_B_Overlap_Size']/df['Image_B_Object_Size']
    
    return df

@timer_func
def dask_remove_objects(image, objects):
    """
    chunks image into a dask array and removes listed elements from the chunks parallely
    :param Img:
    :param objects:
    :return:
    """
    Img = da.from_array(image, chunks=(1, 'auto', 'auto'))

    mask = np.isin(Img, objects).astype(int)
    Img_removed = np.where(mask, 0, Img)
    
    return Img_removed

@timer_func
def remove_objects(image, objects):
    """
    removes listed elements from image
    :param Img:
    :param objects:
    :return:
    """

    mask = np.isin(image, objects).astype(int)
    Img_removed = np.where(mask, 0, image)

    return Img_removed

def combine_images(Img_A, Img_B):
    """
    combines two labelled images into one with continuity in label IDs
    :param Img_A:
    :param Img_B:
    :return: image with labels for segmentations from both images
    """
    Img_A[Img_A != 0] += np.max(Img_B)
    combined = Img_A + Img_B
    
    return combined

def pct_overlap_filter(df, pct):
    """
    removes objects with overlap less than the threshold percent from the overlapping objects dataframe
    :param Img_A:
    :param Img_B:
    :return: image with labels for segmentations from both images
    """
    df_filtered = df[df['Image_A_Overlap_PCT'] >= pct]
    return df_filtered


