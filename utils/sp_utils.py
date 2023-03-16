import math
import pyproj
import numpy as np
import cv2
import geopandas
from shapely import Point
import matplotlib.pyplot as plt


def getCoordinates(geo_transform, col_index, row_index):
    """
    Returns the coordinates given the pixel position in the array\n
    Designed for EPSG:3857, gets wonky with other projections\n
    """
    # print(geo_transform)
    px = geo_transform[0]
    py = geo_transform[3]
    rx = geo_transform[1]
    ry = geo_transform[5]
    x = px + col_index*rx
    y = py + row_index*ry # ry is negative
    return x, y

def getPixel(geo_transform, x,  y):
    """
    Returns the indices of the pixels given the coordinates in the array\n
    Designed for EPSG:3857, gets wonky with other projections\n
    """
    # print(geo_transform)
    px = geo_transform[0]
    py = geo_transform[3]
    rx = geo_transform[1]
    ry = geo_transform[5]
    col_index = (x - px) / rx
    row_index = (y - py) / ry
    # Compared to int(), using math.ceil() keeps the deviation per iteration asymptopic
    # For other datasets/projects would be good to re-evaluate
    return math.ceil(row_index), math.ceil(col_index)

def convertCoords(x, y, src, target):
    """
    Converts the coordinates from the source projection to the target projection\n
    # Parameters \n
    x: Usually the latitude or the x value\n
    y: Usually the longitude or the y value\n
    src: The source projection eg. "epsg:3857"\n
    target: The target projection eg. "epsg:4326"\n
    """
    transformer = pyproj.Transformer.from_crs(src, target)
    return transformer.transform(x, y)

def getBounds(data):
    """
    Gets the bounds of the gdal dataset object\n
    # Parameters:\n
    data: gdal.Open() object \n
    # Returns:\n
    min_x, min_y, max_x, max_y \n
    """
    ulx, xres, xskew, uly, yskew, yres  = data.GetGeoTransform()
    lrx = ulx + (data.RasterXSize * xres)
    lry = uly + (data.RasterYSize * yres)
    print("Upper Left Corner:", ulx, uly)
    print("Lower Right Corner:", lrx, lry)
    print("Width:", abs(lrx - ulx))
    print("Height:", abs(uly- lry))
    print("Area:", abs(lrx - ulx) * abs(uly- lry))
    print("Pixel Width:", abs(data.GetGeoTransform()[1]))
    min_x = min(lrx, ulx)
    min_y = min(lry, uly)
    max_x = max(lrx, ulx)
    max_y = max(lry, uly)
    return min_x, min_y, max_x, max_y

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

    
# Function to calculate UMP directly from shp file
# Self-note: X and Y are col and row, they are swapped in array indexing 
def calculateUMP(shp_df, min_x, min_y, max_x, max_y, percentile= 98, verbose= True):
    """
    # Parameters\n
    shp_df: A geopandas dataframe with columns "height" and "geometry"\n
    min_x, min_y, max_x, max_y: The minimum/maximum x/y coordinates respectively of the area to calculate the UMP for\n
    percentile: <int, [0, 100]> The percentile for calculation of the PercentileHeight\n
    verbose: <bool> To print the results of the UMP calculation\n
    
    # Returns\n
    A dictionary with the keys below:\n
    - AverageHeightArea\n
    - AverageHeightBuilding\n
    - AverageHeightTotalArea\n
    - MaximumHeight\n
    - MinimumHeight\n
    - PercentileHeight\n
    - Percentile\n
    - StandardDeviation\n
    - PlanarArea\n
    - PlanarAreaIndex\n
    - FrontalArea\n
    - FrontalAreaIndex\n
    - TotalArea\n
    """
    # Clip the shapedf to desired area
    df_clipped = geopandas.clip(shp_df, [min_x, min_y, max_x, max_y])

    # Some entries contain multipolygons which needs to be converted to single polygons first
    df_clipped_exploded = df_clipped.explode(ignore_index= True)

    # Calculate the area of each polygon
    df_clipped_exploded["area"] = df_clipped_exploded["geometry"].apply(lambda x: x.area)

    r = {}
    total_area = (max_x - min_x) * (max_y - min_y)
    assert total_area >= 0 # Ensure that no funny business is going on
    frontal_area = 0
    weighted_height = None
        # print("Height:", row[1])
        # print("Coordinates:", *row[3].exterior.coords)

    r["TotalArea"] = total_area

    # If no buildings in the area
    if len(df_clipped_exploded["area"]) == 0:
        r["PlanarArea"] = 0
        r["PlanarAreaIndex"] = 0
        r["MaximumHeight"] = 0
        r["MinimumHeight"] = 0
        r["AverageHeightBuilding"] = 0
        r["AverageHeightArea"] = 0
        r["AverageHeightTotalArea"] = 0
        r["PercentileHeight"] = 0
        r["StandardDeviation"] = 0

    else:
        r["PlanarArea"] = df_clipped_exploded["area"].sum()

        r["PlanarAreaIndex"] = r["PlanarArea"] / total_area

        r["MaximumHeight"] = df_clipped_exploded["height"].max()
        
        r["MinimumHeight"] = df_clipped_exploded["height"].min()

        r["AverageHeightBuilding"] = df_clipped_exploded["height"].mean()

        weighted_height = df_clipped_exploded.apply(lambda x: x["height"] * x["area"], axis= 1).sum()

        r["AverageHeightArea"] = weighted_height / r["PlanarArea"]

        r["AverageHeightTotalArea"] = weighted_height / total_area


        r["PercentileHeight"] = np.percentile(df_clipped_exploded["height"], percentile)

        r["StandardDeviation"] = df_clipped_exploded["height"].std()

    r["Percentile"] = percentile

    debug_frontal = 0

    # Frontal Area
    for poly in df_clipped_exploded.itertuples():
        # Get the height
        height = poly[1]

        # List containing all points + one more (initial)
        points_lst = list(poly[3].exterior.coords)
        points_lst.append(points_lst[0])
       
        
        # Points that are invalid cause they are on the border
        invalid_lst = []
        
        frontal_span = 0
        span_minx = None
        span_maxx = None

        # Returns back to the original point to ensure that all edges are checked 
        for p in points_lst:
            if span_minx is None or span_minx > p[0]:
                span_minx = p[0]
            if span_maxx is None or span_maxx < p[0]:
                span_maxx = p[0]
            # If point is at top edge (max y), consider it invalid (top is "front" for calc of frontal area hence only top considered)
            
            invalid_lst.append(p)
            # This assumes the points are cyclical
            if len(invalid_lst) == 2:
                for subp in invalid_lst:
                    if subp[1] == max_y:
                        frontal_span -= abs(invalid_lst[0][0] - invalid_lst[1][0])
                invalid_lst = []
        debug_frontal += frontal_span
        frontal_span += span_maxx - span_minx
        # To account for computational shennanigans
        if frontal_span < 0: frontal_span = 0
        frontal_area += frontal_span * height

    r["FrontalArea"] = frontal_area

    r["FrontalAreaIndex"] = frontal_area / total_area

    if verbose:
        print("Frontal edge substracted:", debug_frontal)
        print(*r.items(), sep= "\n")

    return r

def generateCorners(ll_coord, step= 1000, return_int= False):
    """
    Takes in a coordinates as the Lower Left corner, and returns a list of four shapely.Point objects, each a designated step apart\n
    Order: [LL, LR, UR, UL] ie. anti-clockwise\n
    # Parameters:\n
    - return_int: Returns list of tuples instead of shapely Points if True, default= False
    """
    x = ll_coord[0]
    y = ll_coord[1]

    if return_int:
        return [(x, y), (x+step, y), (x+step, y+step), (x, y+step)]
    return [Point(x, y), Point(x+step, y), Point(x+step, y+step), Point(x, y+step)]

def inPoly(poly, p_list):
    """
    Checks if all of the points in p_list are in polygon, and returns the indices of the points NOT in the polygon\n

    # Paramaters:\n
    - polygon: shapely Polygon object\n
    - p_list: List of shapely Point objects\n

    # Returns:\n
    - outlist: list of indices of the points not in the polygon. Each index corresponds to a specific corner:\n
    Order: [0=LL, 1=LR, 2=UR, 3=UL] ie. anti-clockwise

    """
    count = 0
    out_list = []

    for p in p_list:
        if not poly.contains(p):
            out_list.append(count)
        count += 1
    
    return out_list

def boundingRectangle(center, width, height):
    """
    Takes in the center coordinate, width, and height of the desired rectangle and returns the coordinates of the LL and UR, 
    assuming Up and Right as positive axis directions
    """
    min_x = center[0] - width
    min_y = center[1] - height
    max_x = center[0] - width
    max_y = center[1] - height
    return [min_x, min_y, max_x, max_y]

# Function to calculate UMP directly from shp file
# Self-note: X and Y are col and row, they are swapped in array indexing 
def calculateUMP(shp_df, min_x, min_y, max_x, max_y, percentile= 98, verbose= True):
    """
    # Parameters\n
    shp_df: A geopandas dataframe with columns "height" and "geometry"\n
    min_x, min_y, max_x, max_y: The minimum/maximum x/y coordinates respectively of the area to calculate the UMP for\n
    percentile: <int, [0, 100]> The percentile for calculation of the PercentileHeight\n
    verbose: <bool> To print the results of the UMP calculation\n
    
    # Returns\n
    A dictionary with the keys below:\n
    - AverageHeightArea\n
    - AverageHeightBuilding\n
    - AverageHeightTotalArea\n
    - MaximumHeight\n
    - MinimumHeight\n
    - PercentileHeight\n
    - Percentile\n
    - StandardDeviation\n
    - PlanarArea\n
    - PlanarAreaIndex\n
    - FrontalArea\n
    - FrontalAreaIndex\n
    - TotalArea\n
    """
    # Clip the shapedf to desired area
    df_clipped = geopandas.clip(shp_df, [min_x, min_y, max_x, max_y])

    # Some entries contain multipolygons which needs to be converted to single polygons first
    df_clipped_exploded = df_clipped.explode(ignore_index= True)

    # Calculate the area of each polygon
    df_clipped_exploded["area"] = df_clipped_exploded["geometry"].apply(lambda x: x.area)

    r = {}
    total_area = (max_x - min_x) * (max_y - min_y)
    assert total_area >= 0 # Ensure that no funny business is going on
    frontal_area = 0
    weighted_height = None
        # print("Height:", row[1])
        # print("Coordinates:", *row[3].exterior.coords)

    r["TotalArea"] = total_area

    # If no buildings in the area
    if len(df_clipped_exploded["area"]) == 0:
        r["PlanarArea"] = 0
        r["PlanarAreaIndex"] = 0
        r["MaximumHeight"] = 0
        r["MinimumHeight"] = 0
        r["AverageHeightBuilding"] = 0
        r["AverageHeightArea"] = 0
        r["AverageHeightTotalArea"] = 0
        r["PercentileHeight"] = 0
        r["StandardDeviation"] = 0

    else:
        r["PlanarArea"] = df_clipped_exploded["area"].sum()

        r["PlanarAreaIndex"] = r["PlanarArea"] / total_area

        r["MaximumHeight"] = df_clipped_exploded["height"].max()
        
        r["MinimumHeight"] = df_clipped_exploded["height"].min()

        r["AverageHeightBuilding"] = df_clipped_exploded["height"].mean()

        weighted_height = df_clipped_exploded.apply(lambda x: x["height"] * x["area"], axis= 1).sum()

        r["AverageHeightArea"] = weighted_height / r["PlanarArea"]

        r["AverageHeightTotalArea"] = weighted_height / total_area


        r["PercentileHeight"] = np.percentile(df_clipped_exploded["height"], percentile)

        r["StandardDeviation"] = df_clipped_exploded["height"].std()

    r["Percentile"] = percentile

    debug_frontal = 0

    # Frontal Area
    for poly in df_clipped_exploded.itertuples():
        # Get the height
        height = poly[1]

        # List containing all points + one more (initial)
        points_lst = list(poly[3].exterior.coords)
        points_lst.append(points_lst[0])
       
        
        # Points that are invalid cause they are on the border
        invalid_lst = []
        
        frontal_span = 0
        span_minx = None
        span_maxx = None

        # Returns back to the original point to ensure that all edges are checked 
        for p in points_lst:
            if span_minx is None or span_minx > p[0]:
                span_minx = p[0]
            if span_maxx is None or span_maxx < p[0]:
                span_maxx = p[0]
            # If point is at top edge (max y), consider it invalid (top is "front" for calc of frontal area hence only top considered)
            
            invalid_lst.append(p)
            # This assumes the points are cyclical
            if len(invalid_lst) == 2:
                for subp in invalid_lst:
                    if subp[1] == max_y:
                        frontal_span -= abs(invalid_lst[0][0] - invalid_lst[1][0])
                invalid_lst = []
        debug_frontal += frontal_span
        frontal_span += span_maxx - span_minx
        # To account for computational shennanigans
        if frontal_span < 0: frontal_span = 0
        frontal_area += frontal_span * height

    r["FrontalArea"] = frontal_area

    r["FrontalAreaIndex"] = frontal_area / total_area

    if verbose:
        print("Frontal edge substracted:", debug_frontal)
        print(*r.items(), sep= "\n")

    return r
def generateCorners(ll_coord, step= 1000, return_int= False):
    """
    Takes in a coordinates as the Lower Left corner, and returns a list of four shapely.Point objects, each a designated step apart\n
    Order: [LL, LR, UR, UL] ie. anti-clockwise\n
    # Parameters:\n
    - return_int: Returns list of tuples instead of shapely Points if True, default= False
    """
    x = ll_coord[0]
    y = ll_coord[1]

    if return_int:
        return [(x, y), (x+step, y), (x+step, y+step), (x, y+step)]
    return [Point(x, y), Point(x+step, y), Point(x+step, y+step), Point(x, y+step)]

def inPoly(poly, p_list):
    """
    Checks if all of the points in p_list are in polygon, and returns the indices of the points NOT in the polygon\n

    # Paramaters:\n
    - polygon: shapely Polygon object\n
    - p_list: List of shapely Point objects\n

    # Returns:\n
    - outlist: list of indices of the points not in the polygon. Each index corresponds to a specific corner:\n
    Order: [0=LL, 1=LR, 2=UR, 3=UL] ie. anti-clockwise

    """
    count = 0
    out_list = []

    for p in p_list:
        if not poly.contains(p):
            out_list.append(count)
        count += 1
    
    return out_list

def boundingRectangle(center, width, height):
    """
    Takes in the center coordinate, width, and height of the desired rectangle and returns the coordinates of the LL and UR, 
    assuming Up and Right as positive axis directions
    """
    min_x = center[0] - width
    min_y = center[1] - height
    max_x = center[0] - width
    max_y = center[1] - height
    return [min_x, min_y, max_x, max_y]


def visualiseBands(data, bands= None, img_plot= True):
    """
    Returns a figure with the selected bands plotted

    # Parameters\n
    data: gdal Dataset object\n
    bands: list of indices of desired bands\n
        If None, all bands will be plotted

    Note that if len(bands) == 3, it will default to RGB plot, else they will be plotted separately, unless img_plot is set to False\n
    """

    if bands is None:
        bands = list(range(12))

    if len(bands) == 3 and img_plot:
        # RGB Image Visualisation
        band1 = data.GetRasterBand(1)
        band2 = data.GetRasterBand(2)
        band3 = data.GetRasterBand(3)

        b1 = band1.ReadAsArray()
        b2 = band2.ReadAsArray()
        b3 = band3.ReadAsArray()
        b1.shape
        img = np.dstack((b1, b2, b3))
        img *= (255.0/img.max())
        f = plt.figure()
        img = np.array(img, dtype= "uint8")
        img = adjust_gamma(img, 2)
        plt.imshow(img, interpolation= "nearest")
        plt.show()
        return
    
    # Visualise all
    
    band_count = 0

    bands_lst = [
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
    ]

    fig, axs = plt.subplots(4, 3, figsize= (15, 15))
    data_arr = data.ReadAsArray()
    for col in range(4):
        for row in range(3):
            band = bands[band_count]
            f = plt.subplot(4, 3, band_count+1)
            f_img = plt.imshow(data_arr[band, :, :])

            plt.title(bands_lst[band])
            band_count += 1
            # fig.colorbar(f_img, ax= axs[col, row])
    plt.show()
    return

