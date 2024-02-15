from .constants import *
from matplotlib.colors import ListedColormap
import torch

DATASET_MERGE_SCHEMES = {
    "four-class": {
        'datasets': [INDONESIA_DATASET_NAME, CAMEROON_DATASET_NAME], #AD
        'label_names': [
            'Plantation',
            'Grassland shrubland',
            'Smallholder agriculture',
            'Other'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c']),
        'original_label_names': INDONESIA_ALL_LABELS,
        'scheme': {
            'Oil palm plantation': 'Plantation',
            'Timber plantation': 'Plantation',
            'Other large-scale plantations': 'Plantation',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale agriculture': 'Smallholder agriculture',
            'Small-scale mixed plantation': 'Smallholder agriculture',
            'Small-scale oil palm plantation': 'Smallholder agriculture',
            'Small-scale cassava plantation': 'Smallholder agriculture', #AD
            'Mining': 'Other',
            'Fish pond': 'Other',
            'Logging': 'Other',
            'Secondary forest': 'Other',
            'Other': 'Other',
            'Rubber plantation': 'Plantation', # AD
            'Timber sales': 'Other', # AD
            'Fruit plantation': 'Plantation', # AD
            'Infrastructure' : 'Other', #AD
            'Wildfire': 'Other' }}, #AD
    "four-class2": {
        'datasets': [INDONESIA_DATASET_NAME, CAMEROON_DATASET_NAME], #AD
        'label_names': [
            'Plantation',
            'Grassland shrubland',
            'Smallholder agriculture',
            'Other'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c']),
        'original_label_names': INDONESIA_ALL_LABELS,
        'scheme': {
            'Oil palm plantation': 'Plantation',
            'Timber plantation': 'Plantation',
            'Other large-scale plantations': 'Plantation',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale agriculture': 'Smallholder agriculture',
            'Small-scale oil palm plantation': 'Smallholder agriculture',
            'Small-scale maize plantation': 'Smallholder agriculture', #AD
            'Small-scale other plantation': 'Smallholder agriculture', #AD
            'Selective logging': 'Other', #AD
            'Mining': 'Other',
            'Fish pond': 'Other',
            'Logging': 'Other',
            'Secondary forest': 'Other',
            'Other': 'Other',
            'Other large-scale plantations': 'Plantation', # AD
            'Rubber plantation': 'Plantation', # AD
            'Fruit plantation' : 'Plantation', # AD
            'Infrastructure' : 'Other', #AD
            'Hunting': 'Other', #AD
            'Wildfire' : 'Other'}}, #AD
    #AD below
    "detailed-temp": {
        'datasets': [INDONESIA_DATASET_NAME, CAMEROON_DATASET_NAME], 
        'label_names': [
            'Oil palm plantation',
            'Timber plantation',
            'Fruit plantation',
            'Rubber plantation',
            'Other large-scale plantations',
            'Grassland shrubland',
            'Small-scale mixed plantation',
            'Small-scale oil palm plantation',
            'Small-scale cassava plantation',
            'Mining',
            'Logging',
            'Timber sales',
            'Infrastructure',
            'Wildfire'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c']),
        'original_label_names': INDONESIA_ALL_LABELS,
        'scheme': {
            'Oil palm plantation': 'Oil palm plantation',
            'Timber plantation': 'Timber plantation',
            'Fruit plantation': 'Fruit plantation',
            'Rubber plantation': 'Rubber plantation',
            'Other large-scale plantations': 'Other large-scale plantations',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale mixed plantation': 'Small-scale mixed plantation',
            'Small-scale oil palm plantation': 'Small-scale oil palm plantation',
            'Small-scale cassava plantation': 'Small-scale cassava plantation',
            'Mining': 'Mining',
            'Logging': 'Logging',
            'Timber sales': 'Timber sales',
            'Infrastructure' : 'Infrastructure',
            'Wildfire': 'Wildfire' }},
    "detailed-temp2": {
        'datasets': [INDONESIA_DATASET_NAME, CAMEROON_DATASET_NAME], 
        'label_names': [
            'Oil palm plantation',
            'Timber plantation',
            'Fruit plantation',
            'Rubber plantation',
            'Other large-scale plantations',
            'Grassland shrubland',
            'Small-scale mixed plantation',
            'Small-scale oil palm plantation',
            'Small-scale cassava plantation',
            'Mining',
            'Selective logging',
            'Infrastructure',
            'Wildfire'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c']),
        'original_label_names': INDONESIA_ALL_LABELS,
        'scheme': {
            'Oil palm plantation': 'Oil palm plantation',
            'Timber plantation': 'Timber plantation',
            'Fruit plantation': 'Fruit plantation',
            'Rubber plantation': 'Rubber plantation',
            'Other large-scale plantations': 'Other large-scale plantations',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale mixed plantation': 'Small-scale mixed plantation',
            'Small-scale oil palm plantation': 'Small-scale oil palm plantation',
            'Small-scale cassava plantation': 'Small-scale cassava plantation',
            'Mining': 'Mining',
            'Selective logging': 'Selective logging',
            'Infrastructure' : 'Infrastructure',
            'Wildfire': 'Wildfire'}},
    "detailed-temp3": {
        'datasets': [INDONESIA_DATASET_NAME, CAMEROON_DATASET_NAME], 
        'label_names': [
            'Oil palm plantation',
            'Timber plantation',
            'Fruit plantation',
            'Rubber plantation',
            'Other large-scale plantations',
            'Grassland shrubland',
            'Small-scale oil palm plantation',
            'Small-scale other plantation',
            'Small-scale maize plantation',
            'Mining',
            'Selective logging',
            'Infrastructure',
            'Wildfire',
            'Hunting', 
            'Other',],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c']),
        'original_label_names': CAMEROON_ALL_LABELS,
        'scheme': {
            'Oil palm plantation': 'Oil palm plantation',
            'Timber plantation': 'Timber plantation',
            'Fruit plantation': 'Fruit plantation',
            'Rubber plantation': 'Rubber plantation',
            'Other large-scale plantations': 'Other large-scale plantations',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale oil palm plantation': 'Small-scale oil palm plantation',
            'Small-scale maize plantation': 'Small-scale maize plantation',
            'Small-scale other plantation': 'Small-scale other plantation',
            'Mining': 'Mining',
            'Selective logging': 'Selective logging',
            'Infrastructure' : 'Infrastructure',
            'Wildfire': 'Wildfire',
            'Hunting' : 'Hunting', 
            'Other':'Other'}}
    
}

"""
For the given schemes above, generate a label_mapping, which is a dict from
original label (int) to mapped label (int).
For example, in `aggressive-merge`, the label_mapping will look like:
    {0: 0, ...}
This is because `Oil palm plantation` (indexed by 0 in original labels) will be
mapped to `Smallholder Plantation` (indexed by 0 in the new labels).
"""
for k, merge_scheme in DATASET_MERGE_SCHEMES.items():
    merge_scheme['labels'] = list(range(len(merge_scheme['label_names'])))
    name_mapping = merge_scheme['scheme']
    label_mapping = {}
    reverse_label_mapping = {}
    for label_name_origin, label_name_dst in name_mapping.items():
        label_origin = merge_scheme['original_label_names'].index(
            label_name_origin)
        label_dst = merge_scheme['label_names'].index(label_name_dst)
        label_mapping[label_origin] = label_dst
        reverse_label_mapping[label_dst] = label_origin
    merge_scheme['label_mapping'] = label_mapping
    merge_scheme['reverse_label_mapping']= reverse_label_mapping

for k, merge_scheme in DATASET_MERGE_SCHEMES.items():
    if "landcover-merge" in k:
        continue
    merge_scheme['original_label_index'] = [INDONESIA_ALL_LABELS.index(l) for l in merge_scheme['original_label_names']]

# Default mapping
DATASET_MERGE_SCHEMES[None] = {'label_mapping': {i: i for i in range(20)}}

# Map tensor from src merge_scheme to dst merge_scheme by first returning
# to original labels. 
def map_scheme_to_scheme(src_scheme_name, dst_scheme_name, tensor):
    to_original = DATASET_MERGE_SCHEMES[src_scheme_name]['reverse_label_mapping']
    to_dst = DATASET_MERGE_SCHEMES[dst_scheme_name]['label_mapping']

    mapped_tensor = torch.zeros_like(tensor).to(tensor.device)
    for src_scheme_label, original_label in to_original.items():
        mapped_tensor[tensor == src_scheme_label] = to_dst[original_label]
    
    return mapped_tensor
