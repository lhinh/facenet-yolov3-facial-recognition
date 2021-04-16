# Small Python script to copy matching IDs from the celebrity face dataset

import os
from shutil import copyfile
DATA_PARENT_DIR = './Data/'
SRC_DIR = DATA_PARENT_DIR + 'img_celeba/'
DST_DIR = DATA_PARENT_DIR + 'subset/'
CELEB_ID_DIR = DATA_PARENT_DIR + 'selected_ids.txt'
CELEB_IMG_ID_DIR = DATA_PARENT_DIR + 'identity_CelebA.txt'

# Load all celebrity IDs into Python as list
with open(CELEB_ID_DIR) as f:
    celeb_id_list = f.read().splitlines()

celeb_id_list = [int(x) for x in celeb_id_list]

# Load all filename/ID mappings into Python as dict
celeb_img_id_dict = {}
with open(CELEB_IMG_ID_DIR) as f:
    celeb_img_id_list = f.read().splitlines()

# Make keys : ID and values : file_names
for line in celeb_img_id_list:
    split_kv = line.split(' ')
    if int(split_kv[1]) in celeb_img_id_dict.keys():
        celeb_img_id_dict[int(split_kv[1])].append(split_kv[0])
    else:
        celeb_img_id_dict[int(split_kv[1])] = [split_kv[0]]

# Copy all file names associated to selected IDs into /Data/subset/
for c_id in celeb_id_list:
    file_names = celeb_img_id_dict.get(c_id, [])
    for file_name in file_names:
        target_file = SRC_DIR + file_name
        dst_file = DST_DIR + file_name
        copyfile(target_file, dst_file)

print('Processing complete! There should be 1,200 images in /Data/subset/')