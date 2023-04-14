"""
The modules provides for utility functions with respect to data from the prepared datasets
"""

# Import libraries
import math
import matplotlib.pyplot as plt



def plotArray(arr, n_channels= None, 
band_names= [
        'B1: Aerosols',
        'B2: Blue',
        'B3: Green',
        'B4: Red',
        'B5: Red Edge 1',
        'B6: Red Edge 2',
        'B7: Red Edge 3',
        'B8: NIR',
        'B8A: Red Edge 4',
        'B9: Water Vapor',
        'B11: SWIR 1',
        'B12: SWIR 2'
    ]):
    """
    Plot the all the channels in the 3D array (C x H x W)

    # Parameters\n
    - arr: np array\n
    - n_channels: The first n channels will be plotted, if None then plots all
    - band_names: Used for labelling the plots, if any
    
    """
    # Visualise all
    if n_channels is None or n_channels > arr.shape[0]:
        n_channels = arr.shape[0]

    band_count = 0

    fig, axs = plt.subplots(math.ceil(n_channels/2), 2, figsize= (8, math.ceil(n_channels/2)*4))
    for col in range(math.ceil(n_channels/2)):
        for row in range(2):
            f = plt.subplot(math.ceil(n_channels/2), 2, band_count+1)
            f_img = plt.imshow(arr[band_count, :, :])

            plt.title(band_names[band_count])
            band_count += 1
            # fig.colorbar(f_img, ax= axs[col, row])
            plt.colorbar()
    plt.show()