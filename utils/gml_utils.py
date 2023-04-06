"""
This is a collection of functions and classes that are used in the processing of CityGML data, 
from initial loading in to the final output as Urban Morphological Parameters (UMPs)
"""

import geopandas as gpd
import math
import numpy as np
import shapely

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
    - MaximumHeight\n
    - PercentileHeight\n
    - Percentile\n
    - StandardDeviation\n
    - PlanarAreaIndex\n
    - FrontalAreaIndex\n
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
        r["FrontalAreaIndex"] = 0

    else:
        planar_area = df_clipped["area"].sum()

        r["PlanarAreaIndex"] = planar_area / total_area

        r["MaximumHeight"] = df_clipped["height"].max()
        
        r["AverageHeightBuilding"] = df_clipped["height"].mean()

        r["AverageHeightArea"] = total_weighted_height / planar_area

        r["AverageHeightTotalArea"] = total_weighted_height / total_area


        r["PercentileHeight"] = np.percentile(df_clipped["height"], percentile)

        # REDO standard deviation
        r["StandardDeviation"] = 0
            
        # Lazy Frontal Area
        df_clipped["frontal_area"] = df_clipped.apply(calculateFrontalArea, axis= 1)
        frontal_area = df_clipped["frontal_area"].sum()
        r["FrontalAreaIndex"] = frontal_area / total_area
    
    r["Percentile"] = percentile

    # df_clipped["frontal_area"] = df_clipped.apply(lambda x: x["height"] * max(x["geometry"].exterior.coords.xy[0]) - min(x["geometry"].exterior.coords.xy[0])), axis= 1)
    

    # Frontal Area
    # for poly in df_clipped.itertuples():
    #     # Get the height
    #     height = poly[1]

    #     # List containing all points + one more (initial)
    #     points_lst = list(poly[3].exterior.coords)
    #     points_lst.append(points_lst[0])
       
        
    #     # Points that are invalid cause they are on the border
    #     invalid_lst = []
        
    #     frontal_span = 0
    #     span_minx = None
    #     span_maxx = None

    #     # Returns back to the original point to ensure that all edges are checked 
    #     for p in points_lst:
    #         if span_minx is None or span_minx > p[0]:
    #             span_minx = p[0]
    #         if span_maxx is None or span_maxx < p[0]:
    #             span_maxx = p[0]
    #         # If point is at top edge (max y), consider it invalid (top is "front" for calc of frontal area hence only top considered)
            
    #         invalid_lst.append(p)
    #         # This assumes the points are cyclical
    #         if len(invalid_lst) == 2:
    #             for subp in invalid_lst:
    #                 if subp[1] == max_y:
    #                     frontal_span -= abs(invalid_lst[0][0] - invalid_lst[1][0])
    #             invalid_lst = []
    #     debug_frontal += frontal_span
    #     frontal_span += span_maxx - span_minx
    #     # To account for computational shennanigans
    #     if frontal_span < 0: frontal_span = 0
    #     frontal_area += frontal_span * height
    
    # Sort the dictionary
    myKeys = list(r.keys())
    myKeys.sort()
    r = {i: r[i] for i in myKeys}

    # return {"AverageHeightArea":r["AverageHeightArea"]}
    return r