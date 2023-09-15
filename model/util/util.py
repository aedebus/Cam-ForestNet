"""Define Logger class for logging information to stdout and disk."""
import os
import json
import numpy as np
from os.path import join

from .constants import *

LIGHTNING_CKPT_PATH = lambda v_num: f'lightning_logs/version_{v_num}/checkpoints/'
LIGHTNING_TB_PATH = lambda v_num: f'lightning_logs/version_{v_num}/'
LIGHTNING_METRICS_PATH = lambda v_num: f'lightning_logs/version_{v_num}/metrics.csv'


class Args(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(args[0])

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            AttributeError("No such attribute: " + name)

def get_version_num():
    try:
        job_id = os.environ['SLURM_JOB_ID']
        job_id = int(job_id)
    except Exception:
        job_id = 0
    return job_id

def init_exp_folder(args):
    version_num = get_version_num()
    save_path = os.path.abspath(args.get("save_path"))
    exp_name = args.get("exp_name")
    exp_path = join(save_path, exp_name)
    exp_metrics_path = join(exp_path, "metrics.csv")
    exp_tb_path = join(exp_path, "tb")
    global_tb_path = args.get("tb_path")
    global_tb_exp_path = join(global_tb_path, exp_name)

    # init exp path
    # if os.path.exists(exp_path):
    #     raise FileExistsError(f"Experiment path [{exp_path}] already exists!")
    # os.makedirs(exp_path, exist_ok=True)
    #
    # os.makedirs(global_tb_path, exist_ok=True)
    # if os.path.exists(global_tb_exp_path):
    #     raise FileExistsError(f"Experiment exists in the global "
    #                           f"Tensorboard path [{global_tb_path}]!")
    # os.makedirs(global_tb_path, exist_ok=True)

    if os.path.exists(exp_path) == False: #AD
        os.makedirs(exp_path, exist_ok=True)

    if os.path.exists(global_tb_exp_path) == False: #AD
        os.makedirs(global_tb_path, exist_ok=True)

    # dump hyper-parameters/arguments
    if os.path.exists(join(save_path, exp_name, "args.json")) == False: #AD
        json.dump(locals(), open(join(save_path, exp_name, "args.json"), "w+"))

    if os.path.islink(exp_metrics_path) == False:
        os.symlink(join(exp_path, LIGHTNING_METRICS_PATH(version_num)), exp_metrics_path)

    if os.path.islink(exp_tb_path) == False:
        os.symlink(join(exp_path, LIGHTNING_TB_PATH(version_num)), exp_tb_path)

    if os.path.islink(global_tb_exp_path) == False:
        os.symlink(exp_tb_path, global_tb_exp_path)



def draw_img_roi(draw, shape, label=None):
    shape_type = shape.geom_type
    if shape_type == POLYGON_SHAPE:
        coords = np.array(shape.exterior.coords)
        draw.polygon([tuple(coord) for coord in coords],
                     outline=label, fill=label)
    else:
        for poly in shape:
            coords = np.array(poly.exterior.coords)
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
