from skimage import exposure
import numpy as np
import torch


def extract_water(img, band):
    p_min, p_max = np.percentile(img, (5, 95))
    img = exposure.rescale_intensity(img, in_range=(p_min, p_max), out_range=(0, 1))
    img = exposure.adjust_gamma(img, gamma=0.001)
    return img


def compute_NDVI(img, c=1, percentile=5):
    nir = img[:, :, :, 3]
    p_min, p_max = np.percentile(nir, (percentile, 100.0 - percentile))
    nir = exposure.rescale_intensity(nir, in_range=(p_min, p_max), out_range=(0, 1))

    red = img[:, :, :, 2]
    p_min, p_max = np.percentile(red, (percentile, 100.0 - percentile))
    red = exposure.rescale_intensity(red, in_range=(p_min, p_max), out_range=(0, 1))

    out = (nir - red) / (nir + red)
    out[np.isnan(out)] = -1.0
    return out[:, :, :, None]


def compute_NDBI(img, percentile=5):
    nir = img[:, :, :, 3]
    p_min, p_max = np.percentile(nir, (percentile, 100.0 - percentile))
    nir = exposure.rescale_intensity(nir, in_range=(p_min, p_max), out_range=(0, 1))

    swir = img[:, :, :, 4]
    p_min, p_max = np.percentile(swir, (percentile, 100.0 - percentile))
    swir = exposure.rescale_intensity(swir, in_range=(p_min, p_max), out_range=(0, 1))

    out = ((swir - nir) / (swir + nir))
    out[np.isnan(out)] = -1.0
    return out[:, :, :, None]


def compute_BU(ndvi, ndbi, percentile=5):
    out = ndbi - ndvi
    p_min, p_max = np.percentile(out, (percentile, 100.0 - percentile))
    out = exposure.rescale_intensity(out, in_range=(p_min, p_max), out_range=(0, 1))
    return out


def segment_satelite_image(satelite_image, sub_size=32):
    data_out = []
    height = int(satelite_image.shape[1]/float(sub_size))
    width = int(satelite_image.shape[2]/float(sub_size))
    for i in range(0, satelite_image.shape[1]-sub_size, sub_size):
        for j in range(0, satelite_image.shape[2]-sub_size, sub_size):
            min_w, min_h = int(i), int(j)
            max_w, max_h = int(i+sub_size), int(j+sub_size)
            data_out.append(satelite_image[:, min_w:max_w, min_h:max_h])

    return torch.tensor(np.asarray(data_out)), (width, height)

def recombine_image(images, out_size, sub_size):
    combined_image = torch.zeros((images.shape[1], out_size[0]*sub_size, out_size[1]*sub_size))
    for i in range(0, out_size[0]):
        for j in range(0, out_size[1]):
            combined_image[:, i*sub_size:i*sub_size+sub_size,
            j*sub_size:j*sub_size+sub_size] = images[i*out_size[0]+j]
    return combined_image

