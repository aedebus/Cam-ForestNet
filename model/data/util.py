import numpy as np
from math import sin, cos, sqrt, atan2, radians

from util.merge_scheme import *

def add_labels_and_label_names(args):
    merge_scheme = args['merge_scheme']
    if merge_scheme is None:
        labels = DATASET_LABELS_NAMES[args['dataset']]['labels']
        label_names = DATASET_LABELS_NAMES[args['dataset']]['label_names']
    else:
        dataset = args['dataset']
        assert(merge_scheme in DATASET_MERGE_SCHEMES)
        #assert(dataset in DATASET_MERGE_SCHEMES[merge_scheme]['datasets']) #AD
        labels = DATASET_MERGE_SCHEMES[merge_scheme]['labels']
        label_names = DATASET_MERGE_SCHEMES[merge_scheme]['label_names']

    args['labels'] = labels
    args['label_names'] = label_names

def draw_img_roi(draw, shape, label):
    shape_type = shape.geom_type
    if label is not None:
        label = int(label)
    #Need to convert lon/lat shape in coordinates matrix (AD)
    lon = shape.centroid.xy[0][0]
    lat = shape.centroid.xy[1][0]
    m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat) - 0.0023 * np.cos(6 * lat)
    m_per_deg_lon = 111132.954 * np.cos(lat) - 93.5 * np.cos(3 * lat) + 0.118 * np.cos(5 * lat)
    res = 4.77 #TO CHANGE WITH SENSOR (15 FOR LANDSAT AND 4.77 FOR PLANETSCOPE)
    deg_lat = (332 * res * 0.5) / m_per_deg_lat #AD
    deg_lon = (332 * res * 0.5) / m_per_deg_lon #AD

    div_lon = 2 * deg_lon / 332 #AD
    div_lat = 2 * deg_lat / 332 #AD

    if shape_type == POLYGON_SHAPE:
        coords = np.array(shape.exterior.coords)
        for coord in coords: #AD
            coord[0] = int((coord[0] - lon + deg_lon) / div_lon) #AD
            coord[1] = int((coord[1] - lat + deg_lat) / div_lat) #AD

        draw.polygon([tuple(coord) for coord in coords],
                     outline=label, fill=label)
    else:
        for poly in shape:
            coords = np.array(poly.exterior.coords)
            for coord in coords: #AD
                coord[0] = int((coord[0] - lon + deg_lon) / div_lon) #AD
                coord[1] = int((coord[1] - lat + deg_lat) / div_lat) #AD

            draw.polygon([tuple(coord) for coord in coords],
                         outline=label, fill=label)

def get_area(polygon):
    """Get area in pixels scaled by max area."""
    shape_type = polygon.geom_type
    if shape_type == POLYGON_SHAPE:
        area = polygon.area
    else:
        area = 0.
        for poly in polygon:
            area += poly.area

    return area / MAX_POLYGON_AREA_IN_PIXELS


def get_num_channels(model_args):
    num_channels = len(model_args['bands'])
    return num_channels
