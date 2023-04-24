"""
This is a collection of functions and classes that are used in the processing of CityGML data, 
from initial loading in to the final output as Urban Morphological Parameters (UMPs)
"""
import os.path
import utils.istarmap as istarmap # Patches mp with this to enable tqdm
from multiprocessing import Pool

from itertools import repeat
from glob import glob
import xml.dom.minidom

import pandas as pd
import numpy as np
import math
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import dask_geopandas as dgpd

import fiona
import shapely
import pyproj

import bisect
import matplotlib.tri as tri
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.ops import linemerge

import tqdm

### Convert GML to feather ###

def multi_to_poly(geometry):
    if isinstance(geometry, shapely.Polygon):
        return geometry
    elif isinstance(geometry, shapely.MultiPolygon):
        return shapely.ops.unary_union(list(geometry.geoms))

def xml_extract_gdf(in_path, 
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
    buildings = doc.getElementsByTagName(lod_tag) # This is Plateau dataset specific
    buildings_coords = [building\
                        .getElementsByTagName(polygon_tag)[0]\
                        .getElementsByTagName(coord_tag)[0]\
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
    buildings_height = doc.getElementsByTagName(height_tag)

    # Abort if no height data found
    if len(buildings_height) == 0:
        return None

    buildings_height = [float(height.childNodes[0].nodeValue) for height in buildings_height]
    height_series = pd.DataFrame(buildings_height, columns= ["height"])

    gdf = gpd.GeoDataFrame(data= height_series, geometry= geometry)

    # Remove the NaN values
    gdf = gdf.dropna().reset_index(drop= True)
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

        # Reproject
        gdf = gdf.set_crs(src_crs).to_crs(tgt_crs).reset_index(drop= True)
        
        # Convert coordinates from 3D to 2D
        gdf_geometry = gpd.GeoSeries.from_wkb(gdf.to_wkb(output_dimension= 2)["geometry"])
        gdf.drop(["geometry"], axis= 1, inplace= True)
        gdf = gpd.GeoDataFrame(gdf, geometry= gdf_geometry)

        # Covert it to correct projection and merge to polygon from multi polygon
        # This is because multipolygon has stacking polygons which affects the area calculation
        # gdf = gdf.explode(index_parts= True).set_crs(src_crs).to_crs(tgt_crs).loc[(slice(None), slice(0)), :].reset_index(drop= True)
        gdf["geometry"] = gdf.apply(lambda x: multi_to_poly(x["geometry"]), axis= 1).reset_index(drop= True)

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

### Divide into Grids ###

class ConcaveHull:
    
    def __init__(self):
        self.triangles = {}
        self.crs = {}
        
    
    def loadpoints(self, points):
        #self.points = np.array(points)
        self.points = points
        
        
    def edge(self, key, triangle):
        '''Calculate the length of the triangle's outside edge
        and returns the [length, key]'''
        pos = triangle[1].index(-1)
        if pos==0:
            x1, y1 = self.points[triangle[0][0]]
            x2, y2 = self.points[triangle[0][1]]
        elif pos==1:
            x1, y1 = self.points[triangle[0][1]]
            x2, y2 = self.points[triangle[0][2]]
        elif pos==2:
            x1, y1 = self.points[triangle[0][0]]
            x2, y2 = self.points[triangle[0][2]]
        length = ((x1-x2)**2+(y1-y2)**2)**0.5
        rec = [length, key]
        return rec
        
    
    def triangulate(self):
        
        if len(self.points) < 2:
            raise Exception('CountError: You need at least 3 points to Triangulate')
        
        temp = list(zip(*self.points))
        x, y = list(temp[0]), list(temp[1])
        del(temp)
        
        triang = tri.Triangulation(x, y)
        
        self.triangles = {}
        
        for i, triangle in enumerate(triang.triangles):
            self.triangles[i] = [list(triangle), list(triang.neighbors[i])]
        

    def calculatehull(self, tol=50):
        
        self.tol = tol
        
        if len(self.triangles) == 0:
            self.triangulate()
        
        # All triangles with one boundary longer than the tolerance (self.tol)
        # is added to a sorted deletion list.
        # The list is kept sorted from according to the boundary edge's length
        # using bisect        
        deletion = []    
        self.boundary_vertices = set()
        for i, triangle in self.triangles.items():
            if -1 in triangle[1]:
                for pos, neigh in enumerate(triangle[1]):
                    if neigh == -1:
                        if pos == 0:
                            self.boundary_vertices.add(triangle[0][0])
                            self.boundary_vertices.add(triangle[0][1])
                        elif pos == 1:
                            self.boundary_vertices.add(triangle[0][1])
                            self.boundary_vertices.add(triangle[0][2])
                        elif pos == 2:
                            self.boundary_vertices.add(triangle[0][0])
                            self.boundary_vertices.add(triangle[0][2])
            if -1 in triangle[1] and triangle[1].count(-1) == 1:
                rec = self.edge(i, triangle)
                if rec[0] > self.tol and triangle[1].count(-1) == 1:
                    bisect.insort(deletion, rec)
                    
        while len(deletion) != 0:
            # The triangles with the longest boundary edges will be 
            # deleted first
            item = deletion.pop()
            ref = item[1]
            flag = 0
            
            # Triangle will not be deleted if it already has two boundary edges            
            if self.triangles[ref][1].count(-1) > 1:
                continue
                
            # Triangle will not be deleted if the inside node which is not
            # on this triangle's boundary is already on the boundary of 
            # another triangle
            adjust = {0: 2, 1: 0, 2: 1}            
            for i, neigh in enumerate(self.triangles[ref][1]):
                j = adjust[i]
                if neigh == -1 and self.triangles[ref][0][j] in self.boundary_vertices:
                    flag = 1
                    break
            if flag == 1:
                continue
           
            for i, neigh in enumerate(self.triangles[ref][1]):
                if neigh == -1:
                    continue
                pos = self.triangles[neigh][1].index(ref)
                self.triangles[neigh][1][pos] = -1
                rec = self.edge(neigh, self.triangles[neigh])
                if rec[0] > self.tol and self.triangles[rec[1]][1].count(-1) == 1:
                    bisect.insort(deletion, rec)
                    
            for pt in self.triangles[ref][0]:
                self.boundary_vertices.add(pt)
                                        
            del self.triangles[ref]
            
        self.polygon()
            
                    

    def polygon(self):
        
        edgelines = []
        for i, triangle in self.triangles.items():
            if -1 in triangle[1]:
                for pos, value in enumerate(triangle[1]):
                    if value == -1:
                        if pos==0:
                            x1, y1 = self.points[triangle[0][0]]
                            x2, y2 = self.points[triangle[0][1]]
                        elif pos==1:
                            x1, y1 = self.points[triangle[0][1]]
                            x2, y2 = self.points[triangle[0][2]]
                        elif pos==2:
                            x1, y1 = self.points[triangle[0][0]]
                            x2, y2 = self.points[triangle[0][2]]
                        line = LineString([(x1, y1), (x2, y2)])
                        edgelines.append(line)

        bound = linemerge(edgelines)
    
        self.boundary = Polygon(bound.coords)

class GridGenerator():
    """
    A class to generate a series of grids in GeoDataFrame format that can be used for futher processing
    """
    def __init__(self, polygon: shapely.Polygon):
        """
        Initialises the class\n
        # Parameters:\n
        - polygon: The polygon that the grid\n
        """
        self.polygon = polygon
        self.min_x = min(self.polygon.exterior.coords.xy[0])
        self.max_x = max(self.polygon.exterior.coords.xy[0])
        self.min_y = min(self.polygon.exterior.coords.xy[1])
        self.max_y = max(self.polygon.exterior.coords.xy[1])
        self.grids = None

    def generateCell(self, x_coord, y_coord, width):
        """
        Generates a Polygon with x_coord and y_coord as the LL corner
        """
        x0 = x_coord
        x1 = x_coord + width
        y0 = y_coord
        y1 = y_coord + width

        return shapely.Polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    def generateGrid(self, step, width, round_f= None, n_processes= 12) -> gpd.GeoDataFrame:
        """
        Generates the grid from the polygon used to instantiate the class
        # Parameters:\n
        - step: The distance* between each grid\n
        - width: The dimensions of each grid (square)\n  
        - crop: If True (Default), constraints the grids to be within the polyon only\n
        - round: The function to round the initial coordinates, defaults to None\n
        * distance/dimensions will be in the same reference system as the polygon\n
        """

        if not round_f is None:
            min_x = round_f(self.min_x)
            max_x = round_f(self.max_x)
            min_y = round_f(self.min_y)
            max_y = round_f(self.max_y)    
        else:
            min_x = self.min_x
            max_x = self.max_x
            min_y = self.min_y
            max_y = self.max_y  
            
        grids = [[x, y] for x in np.arange(min_x, max_x, step) for y in np.arange(min_y, max_y, step)]

        grids = [self.generateCell(*grid, width= width) for grid in grids]

        grids = gpd.GeoDataFrame(geometry= grids)
        
        # Use this to speed up processing
        grids = dgpd.from_geopandas(grids, n_processes)
        
        # grids = list(filter(lambda x: self.polygon.contains(x), tqdm.tqdm(grids, desc= "Filtering Cells")))
        filter_df = grids.apply(self.polygon.contains, axis= 1, meta={'geometry': 'bool'}).compute()

        grids = grids[filter_df["geometry"]].compute().reset_index(drop= True)
        
        self.grids = gpd.GeoDataFrame(geometry= grids["geometry"])
        return self.grids

### Calculate UMP ###

def calculateFrontalArea(df_row):
    geom = df_row["geometry"]
    # If multipolygon then convert to polygon first by merging them
    if isinstance(geom, shapely.MultiPolygon):
        geom = shapely.ops.unary_union(list(geom.geoms))
        # If multipolygon cannot be merged
        if isinstance(geom, shapely.MultiPolygon):
            geom = geom.convex_hull
    return (max(geom.exterior.coords.xy[0]) - min(geom.exterior.coords.xy[0])) * df_row["height"]

def calculateUMP(shp_df, clip_poly, percentile= 98):
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
    - Displacement
    - FrontalAreaIndex\n
    - MaximumHeight\n
    - PercentileHeight\n
    - Percentile\n
    - PlanarAreaIndex\n
    - Rotation\n
    - RoughnessLength
    - StandardDeviation\n
    """
    # Clip the shapedf to desired area
    df_clipped = gpd.clip(shp_df, clip_poly)

    # Calculate the area of each polygon
    # Optimise using dgpd?
    df_clipped["area"] = df_clipped["geometry"].apply(lambda x: x.area)

    df_clipped["weighted_height"] = df_clipped.apply(lambda x: x["height"] * x["area"], axis= 1)

    r = {}
    total_area = clip_poly.area
    assert total_area >= 0 # Ensure that no funny business is going on
    frontal_area = 0

    total_weighted_height = df_clipped["weighted_height"].sum()

    # If no buildings in the area
    if len(df_clipped["area"]) == 0:
        r["PlanarAreaIndex"] = 0
        r["MaximumHeight"] = 0
        r["AverageHeightBuilding"] = 0
        r["AverageHeightArea"] = 0
        r["AverageHeightTotalArea"] = 0
        r["PercentileHeight"] = 0
        r["StandardDeviation"] = 0
        r["FrontalAreaIndex"] = [0] * 4
        r["Displacement"] = 0
        r["RoughnessLength"] = [0] * 4

    else:
        planar_area = df_clipped["area"].sum()

        r["PlanarAreaIndex"] = planar_area / total_area

        r["MaximumHeight"] = df_clipped["height"].max()
        
        r["AverageHeightBuilding"] = df_clipped["height"].mean()

        r["AverageHeightArea"] = total_weighted_height / planar_area

        r["AverageHeightTotalArea"] = total_weighted_height / total_area


        r["PercentileHeight"] = np.percentile(df_clipped["height"], percentile)

        # Standard deviation
        # Just treat it as a probability distribution
        # Get a GeoSeries of the difference to mean squared, times the "probability" aka area
        r["StandardDeviation"] = math.sqrt(df_clipped.apply(lambda x:((x["height"] - r["AverageHeightTotalArea"])**2) * x["geometry"].area, axis= 1).sum() / total_area)    

        # Do a frontal area for each rotation (just 90 is enough)
        
        # Lazy Frontal Area
        frontal_area_lst = []
        # 0/180 degrees rotation
        frontal_area_lst.append(df_clipped.apply(calculateFrontalArea, axis= 1).sum())
        # 90/270 rotation
        frontal_area_lst.append(gpd.GeoDataFrame(geometry= df_clipped.rotate(90, origin= df_clipped.unary_union.centroid), data= df_clipped["height"]).apply(calculateFrontalArea, axis= 1).sum())
        # df_clipped["frontal_area"] = frontal_area_lst * 2
        r["FrontalAreaIndex"] = [frontal_area / total_area for frontal_area in frontal_area_lst] * 2

        # Zero-plane displacement
        # https://link.springer.com/article/10.1007/s10546-014-9985-4#:~:text=calculated%20from%20MD1998.-,KA2013,-expands%20the%20parametrization
        # d_mac dimensionless constants
        alpha = 4.43

        d_mac = (1+4.43**(-r["PlanarAreaIndex"])*(r["PlanarAreaIndex"] - 1))*r["AverageHeightTotalArea"]
        # Regressed constants
        a_0, b_0, c_0 = 1.29, 0.36, -0.17
        X = (r["StandardDeviation"] + r["AverageHeightTotalArea"]) / r["MaximumHeight"]
        d = (c_0*(X**2) + (a_0*(r["PlanarAreaIndex"]**b_0) - c_0) * X) / r["MaximumHeight"]

        # For cases where there are no buildings/height is 0
        if math.isnan(d):
            d = 0
        r["Displacement"] = d

        # Aerodynamic Roughness Length
        # Regressed constants
        a_1, b_1, c_1 = 0.7076, 20.2067, -0.7711
        beta = 1
        c_lb = 1.2 # Drag coefficient
        k = 0.4 # von Karman constant

        z_mac_0 = (1 - d_mac / r["AverageHeightTotalArea"]) * math.exp(-((0.5 * beta * c_lb/(k**2) * (1-d_mac/r["AverageHeightTotalArea"]) * r["FrontalAreaIndex"][0])**(-0.5))) * r["AverageHeightTotalArea"]
        z_mac_90 = (1 - d_mac / r["AverageHeightTotalArea"]) * math.exp(-((0.5 * beta * c_lb/(k**2) * (1-d_mac/r["AverageHeightTotalArea"]) * r["FrontalAreaIndex"][1])**(-0.5))) * r["AverageHeightTotalArea"]
        Y = (r["PlanarAreaIndex"] * r["StandardDeviation"]) / r["AverageHeightTotalArea"]
        z_kanda_0 = (b_1 * Y**2 + c_1 * Y + a_1) * z_mac_0
        z_kanda_90 = (b_1 * Y**2 + c_1 * Y + a_1) * z_mac_90

        # For cases where there are no buildings/height is 0
        if math.isnan(z_kanda_0):
            z_kanda_0 = 0
        if math.isnan(z_kanda_90):
            z_kanda_90 = 0
        r["RoughnessLength"] = [z_kanda_0, z_kanda_90] * 2
    r["Percentile"] = percentile
    r["Rotation"] = [0, 90, 180, 270]
    
    # Add the key in form of the geometry
    # r["geometry"] = clip_poly

    # Sort the dictionary
    myKeys = list(r.keys())
    myKeys.sort()
    r = {i: r[i] for i in myKeys}

    # return {"AverageHeightArea":r["AverageHeightArea"]}
    return r

def calculateUMP_batch(shp_df, clip_poly_list, percentile= 98):
    """
    Calculates UMP in batches of cells, wraps around the calculateUMP function\n
    Function is defined here instead of utils as multiprocessing only can pickle top-level defined functions\n
    # Parameters:\n
    - shp_df: The `GeoDataFrame` that contains the building data
    - clip_poly_list: A `List` of `Polygon`s that each acts as a cell for which the UMP will be calculated
    - percentile: The percentile to get for the one of the UMPs 'PercentileHeight'
    """
    # print("Start Batch:", batch_no)
    r = []

    for cell in clip_poly_list:
        d = {"geometry":cell}
        d.update(calculateUMP(shp_df, cell, percentile))
        r.append(d)
    
    r_gdf = gpd.GeoDataFrame.from_dict(r)
    return r_gdf

# Divide into n portions, with n overlap

def split_polygon(bounding_polygon: shapely.Polygon, n_splits: int, buffer: float):
    """
    Takes a `Polygon` and returns an evenly split `List` of `Polygon`s based on its envelope
    """  
    min_x, min_y, max_x, max_y = bounding_polygon.envelope.bounds
    width = max_x - min_x
    height = max_y - min_y
    width_step = width / n_splits
    height_step = height / n_splits

    # len should be == n_splits
    segments = [shapely.Polygon([
        (x - buffer, y - buffer), # LL
        (x + width_step + buffer, y - buffer), # LR
        (x + width_step + buffer, y + height_step + buffer), # UR
        (x - buffer, y + height_step + buffer), # UL
        ]) 
        for x in np.linspace(min_x, max_x, num= n_splits, endpoint= False) 
        for y in np.linspace(min_y, max_y, num= n_splits, endpoint= False)]
    
    return segments

def parallel_UMP_calc(building_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame, segments, n_processes):
    """
    Takes in a GeoDataFrame of building footprints and height, and a `List` of `Polygons`, for each of which to calculate the UMP for buildings within
    """
    # Divide the grids
    segmented_grid = []
    for segment in segments:
        # Only includes cells that are fully within the segment
        segmented_grid.append(grid_gdf["geometry"][grid_gdf["geometry"].within(segment)])

    # Divide the gdf
    segmented_gdf = []
    for segment in segments:
        # Buildings are just clipped
        segmented_gdf.append(building_gdf.clip(segment))

    # Runs in parallel, but no point in more processes than segments
    result = []
    with Pool(min(n_processes, len(segments))) as pool:
        for r in tqdm.tqdm(pool.istarmap(calculateUMP_batch, zip(segmented_gdf, segmented_grid)), total= len(segmented_gdf)):
            result.append(r)
    
    return result