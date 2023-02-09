# This module contains utility functions for processing sentinel data

# Library Imports
from datetime import datetime

import ee
import geemap
import gdown

CLOUD_FILTER = 60 # This value is used because in some areas are permanently masked due to false positive
# CLD_PRB_THRESH = 40
# CLD_PRB_THRESH = 50
CLD_PRB_THRESH = 60
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 100


### Function to get and merge the collections for cloudless processing
def get_s2SR(aoi= ee.Geometry.Point(103.851959, 1.290270)):
    """
    Returns image collection of COPERNICUS/S2_SR, filtered by the AOI\n
    Params:\n
        aoi: Area of Interest <ee.Geometry.Point>\n
    """
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    # s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(aoi)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


### Definition of functions for cloud masking

def maskS2clouds(image):
    """
    Returns a sentinel-2 image with the clouds masked 
    """
    qa60 = image.select("QA60")

    # The bit mask for QA60 as specified in the documentation
    cloudBitMask = 1 << 10 # Indicates opaque clouds present
    cirrusBitMask = 1 << 11 # Indicates cirrus clouds present

    # mask = ((qa60 & cloudBitMask) == 0) and ((qa60 & cirrusBitMask) == 0)
    mask = qa60.bitwiseAnd(cloudBitMask).eq(0).And(qa60.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask).divide(10000)


def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)

def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)

def remove_clouds(collection):
    """
    Takes in a Sentinel-2 image collection and returns a collection with clouds masked/removed\n
    For additional configuration of parameters, change the constants used for masking in definition script\n

    ### WIP ###
    Allow for proper passing of cloud masking paramters
    """
    return collection.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)




### Dataset filtering functions

def filterCollection(collection, params, filter_type):
    """
    Filters a collection by the date, can be individual or range, see filter types below:\n
    Y_S: Filter by custom years, pass in list of years in "YYYY" format\n
    M_S: Filter by custom months, pass in list of months in "MM" format\n
    Y_R: Filter by custom years, pass in list of tuples, [(start_incl, end_excl)] of years in "YYYY" format\n
    M_R: Filter by custom months, pass in list of tuples, [(start_incl, end_excl)] of months in "MM" format\n
    """

    col = None
    
    # Filtering based on params
    if (filter_type == "Y_S"):
        for year in params:
            if col is None:
                col = collection.filter(ee.Filter.calendarRange(year, year, "year"))
            else:
                col = col.merge(col.filter(ee.Filter.calendarRange(year, year, "year")))

    elif filter_type == "M_S":
        for month in params:
            if col is None:
                col = collection.filter(ee.Filter.calendarRange(month, month, "month"))
            else:
                col = col.merge(col.filter(ee.Filter.calendarRange(month, month, "month")))
            
    elif filter_type == "Y_R":
        for years in params:
            if col is None:
                col = collection.filter(ee.Filter.calendarRange(years[0], years[1], "year"))
            else:
                col = col.merge(col.filter(ee.Filter.calendarRange(years[0], years[1], "year")))
    elif filter_type == "M_R":
        for months in params:
            if col is None:
                col = collection.filter(ee.Filter.calendarRange(months[0], months[1], "month"))
            else:
                col = col.merge(col.filter(ee.Filter.calendarRange(months[0], months[1], "month")))
    else:
        print(f"Wrong Filter Type: {filter_type}")
        raise ValueError

    return col

# Date retrieval function
def getDates(collection, fmt= "%Y/%m/%d %H:%M"):
    """
    Gets the dates of images of an ImageCollection in a list, format is specified by the datetime strftime format
    """
    # Retrieve the dates 
    date_lst = collection.aggregate_array("system:time_start")

    # Download to local
    date_lst = date_lst.getInfo()

    # Parse into datetime
    date_lst = list(map(lambda x: datetime.fromtimestamp(x / 1e3).strftime(fmt), date_lst))
    
    return date_lst

# Image General Information Retrieval
def getInfo(image):
    """
    Gets some general information of an image\n

    """
    # Get information about the bands as a list.
    bandNames = image.bandNames().getInfo()
    print('Band names:', bandNames)  # ee.List of band names

    # Get projection information from band 1.
    for band in bandNames:
        print("Band:", band)
        b1proj = image.select(band).projection().getInfo()
        print(f'{band} projection:', b1proj)  # ee.Projection object

        # Get scale (in meters) information from band 1.
        b1scale = image.select(band).projection().nominalScale().getInfo()
        print(f'{band} scale:', b1scale)  # ee.Number

    # Get a list of all metadata properties.
    properties = image.propertyNames().getInfo()
    print('Metadata properties:', properties)  # ee.List of metadata properties

    # Get the timestamp and convert it to a date.
    try:
        date = ee.Date(image.get('system:time_start')).getInfo()["value"]
        date = datetime.fromtimestamp(date / 1e3).strftime("%Y/%m/%d %H:%M")
    except:
        date = "Unknown (Possibly due to mean/median operation)"
    print('DateTime:', date)  # ee.Date

### Function Definitions for exporting images

def boundingRectangle(aoi, width, height, projection):
    """
    Returns ee.Geometry.Rectangle centered on AOI with width and height in metres\n
    aoi: ee.Geometry.Point\n
    width, height: Dimensions in metres\n
    projection: Type of project used, eg.'EPSG:32648'
    """

    # Create bounding rectangle
    # https://gis.stackexchange.com/questions/363706/area-and-dimensions-of-ee-geometry-rectangle
    xRadius = width/2
    yRadius = height/2
    pointLatLon = aoi
    pointMeters = pointLatLon.transform(projection, 0.001)
    # coords = pointLatLon.coordinates()
    coords = pointMeters.coordinates()
    minX = ee.Number(coords.get(0)).subtract(xRadius)
    minY = ee.Number(coords.get(1)).subtract(yRadius)
    maxX = ee.Number(coords.get(0)).add(xRadius)
    maxY = ee.Number(coords.get(1)).add(yRadius)
    # rect = ee.Geometry.Rectangle([minX, minY, maxX, maxY])
    rect = ee.Geometry.Rectangle([minX, minY, maxX, maxY], projection, False)
    return rect

def create_bounding_box(point: ee.Geometry.Point, diameter: float):
    intermediate_circle = point.buffer(diameter / 2.0, 0.0)
    return intermediate_circle.bounds()

def exportImageToDrive(image, aoi, bound, dir, projection, dt= None, prefix= "S2HR"):
    """
    Takes in an image and exports it to the specific directory in drive\n
    Files are named as <prefix>_<YYYYMMDD_HHMM> by default\n
    image: ee.Image, assumed filtered before input\n
    aoi: Geometry.Point(Long, Lat) of the area of interest\n
    bound: The (width, height) of the bound centered on the AOI in metres\n
    If ee.Geometry.Rectangle is provided, bound will not be re-calculated within the function\n
    dir: Drive directory\n
    projection: The projection information (dict) of the image, should contain "crs" and "transform"\n
    dt: Date string corresponding to image in <YYYY/MM/DD HH:MM> format. \n
    Will be automatically generated if blank.\n
    prefix: The prefix of the files\n

    Returns:\n
        task: task object for the uploading of the file

    ### WIP ###
    Remove need to recalulate bounding box for each image\n
    Auto selection of highest resolution band for projection\n
    Get the bands first \n
    Allow for hard user preference \n

    """

    # Retrieve data of image
    # if projection is None:
    #     band_sample_id = img_dict["bands"][0]["id"]
    #     # id_sample_name = img_dict["id"]
    #     # The projection will also determine the scale/resolution of the output image, so selection of bands is significant
    #     for band in ["B2", "B3", "B4"]: # These bands are the usually the RGB bands and are of the highest resolution
    #         try:  
    #             projection = image.select(band).projection().getInfo()
    #             break
    #         except:
    #             continue
    #     if projection is None: # If none of the bands specified above works, use the first band
    #         projection = image.select(band_sample_id).projection().getInfo()
                 
    if dt is None:
        img_dict = image.getInfo()
        try:
            dt = datetime.fromtimestamp(img_dict["properties"]["system:time_start"] / 1e3).strftime("%Y%m%d_%H%M")
        except:
            dt = "UNKNOWN_DT"

    # Create bounding rectangle if not passed
    if not isinstance(bound, ee.Geometry):
        bounding_box = boundingRectangle(aoi, bound[0], bound[1], projection["crs"])
    else:
        bounding_box = bound
    # bounding_box = create_bounding_box(aoi, bound[0])

    # geemap.ee_export_image(
    #     image, filename= dir, region=bounding_box, file_per_band=False, scale= 90
    #     )

    task = ee.batch.Export.image.toDrive(image,
                                    description=f"{prefix}_{dt}",
                                    folder=dir,
                                    crs=projection['crs'],
                                    crsTransform=projection['transform'],
                                    region=bounding_box,
                                    fileFormat="GeoTIFF",
                                    maxPixels=10000000000000,
                                    )

    try:                         
        task.start()
    except:
        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
        finally:
            task.start()
        

    return task
        
    
def exportCollectionToDrive(collection, aoi, bound, dir, date_lst= None, projection= None, projection_bands= None, prefix= "S2HR"):
    """
    Takes in an image collection and exports it to the specific directory in drive\n
    collection: ee.ImageCollection, assumed filtered before input\n
    aoi: Geometry.Point(Long, Lat) of the area of interest\n
    bound: The (width, height) of the bound centered on the AOI in metres\n
    dir: Drive directory\n
    date_lst: A list of dates corresponding to each image in the collection. Will be automatically generated if blank.
    projection: The projection information of the image, should contain crs and transform\n
    projection_bands: A list of bands from which to choose the projection information from\n
    prefix: The prefix of the files

    """
    ### Iterate through the collection and export each image individually

    # Convert image collection to list of images
    col_size = collection.size().getInfo()
    col_lst = collection.toList(col_size)

    # Converts each element from ee.ComputedObject to ee.Image
    img_lst = []
    for i in range(col_size):
        img_lst.append(ee.Image(col_lst.get(i)))

    # Create lst to store tasks
    task_lst = []

    # Retrieve necessary data

    if date_lst is None:
        date_lst = getDates(collection, "%Y%m%d_%H%M")
    if projection is None:
        if projection_bands is None:
            raise ValueError("Projection bands must be provided if projection is not provided. Please refer to documentation for available bands.")
        col_bands = collection.first().bandNames().getInfo()
        sel_band = None
        for band in projection_bands:
            if band in col_bands:
                sel_band = band
                break
        if sel_band is None:
            raise KeyError("Band not found, check for case and typos.")

        projection = collection.first().select(sel_band).projection().getInfo()
    
    # Calculate the bounds to prevent repeated calculation
    bounding_box = boundingRectangle(aoi, bound[0], bound[1], projection["crs"])

    # Iterate and export
    for img_i in range(col_size):
        task_lst.append(exportImageToDrive(
                                            image= img_lst[img_i], 
                                            aoi= aoi, 
                                            bound= bounding_box, 
                                            dir= dir, 
                                            dt= date_lst[img_i], 
                                            projection= projection, 
                                            prefix= prefix,
                                            )
                                            )

    return task_lst


def download_drive_folder(folder_id, output_dir= "data/"):
    """
    Wrapper for the gdown folder download function\n
    """
    folder_id = 'https://drive.google.com/drive/folders/' + folder_id
    print('Folder ID: ', folder_id)

    gdown.download_folder(url=folder_id, quiet=True, use_cookies=True, output= output_dir)

def download_drive_file(file_id, output_dir= "data/"):
    gdown.download(id= file_id, output= output_dir, quiet= True, use_cookies= True)
