from .constants import *
from matplotlib.colors import ListedColormap
import torch

DATASET_MERGE_SCHEMES = {
    "none": {
        'datasets': [INDONESIA_DATASET_NAME],
        'label_names': INDONESIA_LABELS,
        'original_label_names': INDONESIA_LABELS,
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c', 'm', '#ffa500', '#964b00', 'g', 'k', 'tab:pink', 'tab:olive']),
        'scheme': {
            'Oil palm plantation': 'Oil palm plantation',
            'Timber plantation': 'Timber plantation',
            'Other large-scale plantations': 'Other large-scale plantations',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale agriculture': 'Small-scale agriculture',
            'Small-scale mixed plantation': 'Small-scale mixed plantation',
            'Small-scale oil palm plantation': 'Small-scale oil palm plantation',
            'Mining': 'Mining',
            'Fish pond': 'Fish pond',
            'Logging': 'Logging',
            'Secondary forest': 'Secondary forest'}},
    "plantation-merge": {
        'datasets': [INDONESIA_DATASET_NAME],
        'label_names': [
            'Plantation',
            'Grassland shrubland',
            'Small-scale agriculture',
            'Mining',
            'Fish pond',
            'Logging',
            'Secondary forest'],
        'original_label_names': INDONESIA_LABELS,
        'scheme': {
            'Oil palm plantation': 'Plantation',
            'Timber plantation': 'Plantation',
            'Other large-scale plantations': 'Plantation',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale agriculture': 'Small-scale agriculture',
            'Small-scale mixed plantation': 'Plantation',
            'Small-scale oil palm plantation': 'Plantation',
            'Mining': 'Mining',
            'Fish pond': 'Fish pond',
            'Logging': 'Logging',
            'Secondary forest': 'Secondary forest'}},
    "eight-class": {
        'datasets': [INDONESIA_DATASET_NAME],
        'label_names': [
            'Plantation',
            'Grassland shrubland',
            'Smallholder agriculture',
            'Mining',
            'Fish pond',
            'Logging',
            'Secondary forest',
            'Other'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c', 'm', '#ffa500', '#964b00', 'g']),
        'original_label_names': INDONESIA_ALL_LABELS,
        'scheme': {
            'Oil palm plantation': 'Plantation',
            'Timber plantation': 'Plantation',
            'Other large-scale plantations': 'Plantation',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale agriculture': 'Smallholder agriculture',
            'Small-scale mixed plantation': 'Smallholder agriculture',
            'Small-scale oil palm plantation': 'Smallholder agriculture',
            'Mining': 'Mining',
            'Fish pond': 'Fish pond',
            'Logging': 'Logging',
            'Secondary forest': 'Secondary forest',
            'Other': 'Other'}},
    "small-scale-plantation-merge": {
        'datasets': [INDONESIA_DATASET_NAME],
        'label_names': [
            'Plantation',
            'Grassland shrubland',
            'Smallholder agriculture',
            'Mining',
            'Fish pond',
            'Logging',
            'Secondary forest'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c', 'm', '#ffa500', '#964b00']),
        'original_label_names': INDONESIA_LABELS,
        'scheme': {
            'Oil palm plantation': 'Plantation',
            'Timber plantation': 'Plantation',
            'Other large-scale plantations': 'Plantation',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale agriculture': 'Smallholder agriculture',
            'Small-scale mixed plantation': 'Smallholder agriculture',
            'Small-scale oil palm plantation': 'Smallholder agriculture',
            'Mining': 'Mining',
            'Fish pond': 'Fish pond',
            'Logging': 'Logging',
            'Secondary forest': 'Secondary forest'}},
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
    "aggressive-merge": {
        'datasets': [INDONESIA_DATASET_NAME],
        'label_names': [
            'Smallholder Plantation',
            'Grassland shrubland',
            'Mining',
            'Fish pond',
            'Logging',
            'Secondary forest'],
        'original_label_names': INDONESIA_LABELS,
        'scheme': {
            'Oil palm plantation': 'Smallholder Plantation',
            'Timber plantation': 'Smallholder Plantation',
            'Other large-scale plantations': 'Smallholder Plantation',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale agriculture': 'Smallholder Plantation',
            'Small-scale mixed plantation': 'Smallholder Plantation',
            'Small-scale oil palm plantation': 'Smallholder Plantation',
            'Mining': 'Mining',
            'Fish pond': 'Fish pond',
            'Logging': 'Logging',
            'Secondary forest': 'Secondary forest'}},
    "landcover-merge-eight-class": {
        'datasets': [INDONESIA_DATASET_NAME],
        'label_names': [
            'Plantation',
            'Grassland shrubland',
            'Smallholder agriculture',
            'Mining',
            'Fish pond',
            'Logging',
            'Secondary forest',
            'Other'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c', 'm',
                                        '#ffa500', '#964b00', 'g']),
        'original_label_names': LANDCOVER_LABELS,
        'scheme': {
            'Unknown': 'Other',
            'Primary Dryland Forest': 'Other',
            'Secondary Dryland Forest': 'Secondary forest',
            'Primary Swamp Forest': 'Other',
            'Secondary Swamp Forest': 'Secondary forest',
            'Primary Mangrove Forest': 'Other',
            'Secondary Mangrove Forest': 'Secondary forest',
            'Bush/Shrub': 'Grassland shrubland',
            'Swamp Shrub': 'Grassland shrubland',
            'Grass Land': 'Grassland shrubland',
            'Plantation Forest': 'Plantation',
            'Estate Cropplantation': 'Smallholder agriculture',
            'Dryland Agriculture': 'Smallholder agriculture',
            'Shrub-Mixed Dryland Farm': 'Smallholder agriculture',
            'Transmigration Area': 'Other',
            'Rice Field': 'Smallholder agriculture',
            'Fish Pond': 'Fish pond',
            'Barren Land': 'Grassland shrubland',
            'Mining Area': 'Mining',
            'Settlement Area': 'Other',
            'Airport': 'Other',
            'Swamp': 'Other',
            'Cloud Covered': 'Other',
            'Bodies of Water': 'Other'}},
    "landcover-merge-four-class": {
        'datasets': [INDONESIA_DATASET_NAME],
        'label_names': [
            'Plantation',
            'Grassland shrubland',
            'Smallholder agriculture',
            'Other'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c']),
        'original_label_names': LANDCOVER_LABELS,
        'scheme': {
            'Unknown': 'Other',
            'Primary Dryland Forest': 'Other',
            'Secondary Dryland Forest': 'Other',
            'Primary Swamp Forest': 'Other',
            'Secondary Swamp Forest': 'Other',
            'Primary Mangrove Forest': 'Other',
            'Secondary Mangrove Forest': 'Other',
            'Bush/Shrub': 'Grassland shrubland',
            'Swamp Shrub': 'Grassland shrubland',
            'Grass Land': 'Grassland shrubland',
            'Plantation Forest': 'Plantation',
            'Estate Cropplantation': 'Smallholder agriculture',
            'Dryland Agriculture': 'Smallholder agriculture',
            'Shrub-Mixed Dryland Farm': 'Smallholder agriculture',
            'Transmigration Area': 'Other',
            'Rice Field': 'Smallholder agriculture',
            'Fish Pond': 'Other',
            'Barren Land': 'Grassland shrubland',
            'Mining Area': 'Other',
            'Settlement Area': 'Other',
            'Airport': 'Other',
            'Swamp': 'Other',
            'Cloud Covered': 'Other',
            'Bodies of Water': 'Other'}},
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