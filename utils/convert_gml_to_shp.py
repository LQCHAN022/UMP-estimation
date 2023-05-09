import os.path
from multiprocessing import Pool

from itertools import repeat
from glob import glob
import xml.dom.minidom

import geopandas as gpd
import pandas as pd
import numpy as np

import fiona
import shapely
import pyproj


def xml_extract_gdf(in_path, 
    building_tag= "bldg:Building", 
    lod_tag= "bldg:lod1Solid", 
    polygon_tag= "gml:Polygon", 
    coord_tag= "gml:posList", 
    height_tag= "bldg:measuredHeight", 
    src_crs= "EPSG:6668", 
    tgt_crs= "EPSG:3857"):
    """
    Extracts the building geometry (first polygon of each building only) and height, and assemble them into a GeoDataFrame\n
    # Parameters:\n
    - in_path: File path to the input gml file\n
    - building_tag: The tag for each building, number of tags should be equivalent to the number of buildings\n
    - lod_tag: The tag specifying which lod to extract from\n
    - polygon_tag: The tag for polygons making up each building, only the first polygon will be extracted\n
    - coord_tag: The tag for the list of coordinates\n
    - height_tag: The tag for the height, should have same number of instances as buildings\n
    """
    # Manually read the gml as xml and extract the geometry
    doc = xml.dom.minidom.parse(in_path)
    buildings = doc.getElementsByTagName("bldg:lod1Solid") # This is Plateau dataset specific
    buildings_coords = [building\
                        .getElementsByTagName("gml:Polygon")[0]\
                        .getElementsByTagName("gml:posList")[0]\
                        .childNodes[0]\
                        .nodeValue.split(" ") for building in buildings]

    # Abort if no geometry found
    if len(buildings_coords) == 0:
        return None

    # Conversion is done here to preserve the precision due to python rounding shennanigans
    transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs)
    buildings_coords = [
                        [transformer.transform(buildings_coords[bld][i], 
                            buildings_coords[bld][i+1]) for i in range(0, len(buildings_coords[bld]), 3)] 
                        for bld in range(len(buildings_coords))
                        ]
    geometry = gpd.GeoSeries([shapely.Polygon(coords) for coords in buildings_coords])
    
    # Manually extract the height
    buildings_height = doc.getElementsByTagName("bldg:measuredHeight")

    # Abort if no height data found
    if len(buildings_height) == 0:
        return None

    buildings_height = [float(height.childNodes[0].nodeValue) for height in buildings_height]
    height_series = pd.DataFrame(buildings_height, columns= ["height"])

    gdf = gpd.GeoDataFrame(data= height_series, geometry= geometry)
    return gdf

def gml_to_feather(in_path, out_path, mode= None, log_name= "gml_convert", src_crs= "EPSG:6668", tgt_crs= "EPSG:3857", force_manual= False):
    """
    Takes in a gml file and outputs it as a feather file\n
    W/R with feather files is much faster and takes up much less space than using shp files\n
    # Parameters:\n
    - in_path: The path for the gml file\n
    - out_path: The output path for the shape file, must end with a .shp\n
    - mode: 
        - 'o' = overwrites any file at output path, \n
        - None = raises error if file already exists\n
    - src_crs: Source projection\n
    - tgt_crs: Target projection\n
    """
    # Extracts features
    with fiona.open(in_path, 'r') as src:
        features = list(src)

    # Converts and places it in geopandas format
    # There seems to be some gml files without the measured height column, will try to log those files in
    gdf = gpd.GeoDataFrame.from_features(features)
    try:
        # Goes to manual extraction if flag is True
        # Hacky but it works
        if force_manual:
            raise
        gdf = gdf[['measuredHeight', 'geometry']]
        gdf.rename(columns={'measuredHeight':'height'}, inplace= True)

        # Remove the NaN values
        gdf = gdf.dropna().reset_index(drop= True)

        # Covert it to correct projection and strip to polygon instead from multi polygon
        gdf = gdf.explode(index_parts= True).set_crs(src_crs).to_crs(tgt_crs).loc[(slice(None), slice(0)), :].reset_index(drop= True)

        # Convert coordinates from 2D to 3D
        gdf_geometry = gpd.GeoSeries.from_wkb(gdf.to_wkb(output_dimension= 2)["geometry"])
        gdf.drop(["geometry"], axis= 1, inplace= True)
        gdf = gpd.GeoDataFrame(gdf, geometry= gdf_geometry)
    except Exception as e:
        # If exception occurs try to extract manually
        gdf_manual = xml_extract_gdf(in_path)

        if gdf_manual is None:
            print(f"{e}: {os.path.basename(in_path)}")
            if not log_name is None:
                if not os.path.exists("logs"):
                    os.makedirs("logs")
                with open(f"logs/{log_name}.txt", "a") as f:
                    f.write(in_path + "\n")
            return len(gdf)
        else:
            gdf = gdf_manual

    # Check if parent directory exists
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
        
    # Outputs to the desired path
    if os.path.exists(out_path):
        if mode == "a":
            gdf.to_feather(out_path, mode= "a")
        elif mode == "o":
            gdf.to_feather(out_path)
        else:
            raise FileExistsError("Output path already exists")
    else:
        gdf.to_feather(out_path)
    
    return 0

def batch_gml_to_feather(in_dir, out_path, n_processes= 12, log_name= None, mode= None, src_crs= "EPSG:6668", tgt_crs= "EPSG:3857", force_manual= False):

    # Get all the paths of the gml files
    in_paths = glob(f"{in_dir}/*.gml")
    print("Total input files:", len(in_paths))

    # Reads the gml file and extract features
    with Pool(processes= n_processes) as pool:
        r = pool.starmap(
            gml_to_feather, 
            zip(in_paths, 
                [f'{in_dir}/temp/{os.path.basename(path).replace(".gml", ".feather")}' for path in in_paths], 
                repeat(mode), 
                repeat(log_name),
                repeat(src_crs),
                repeat(tgt_crs),
                repeat(force_manual)))

    # Check for invalid buildings
    print(f"There are {sum(r)} invalid buildings from {len(list(filter(lambda x: x > 0, r)))} files")

    # Get all the paths of the shp files
    in_paths = glob(f"{in_dir}/temp/*.feather")
    print("Total files to merge:", len(in_paths))

    gdfs = [gpd.read_feather(in_path) for in_path in in_paths]
    gdf = gpd.GeoDataFrame(pd.concat(gdfs)).reset_index(drop= True)
    gdf.to_feather(out_path)

    for temp_file in in_paths:
        os.remove(temp_file)

    return gdf


in_dir = "data/osaka/udx/bldg"
out_path = "data/osaka/osaka_full_manual.feather"

batch_gml_to_feather(in_dir, out_path, mode= "o", log_name= "osaka_manual", force_manual= True)