"""
Check is alignment? detection effect
"""

import os
import glob
frames = glob.glob("/data1/gaoxiang_cong/TalkNet-ASD-main/Chem_buwan/Chem_frame/*")
faces = glob.glob("/data1/gaoxiang_cong/TalkNet-ASD-main/Chem_buwan/Chem_face/*")
print("frames: ", len(frames))
print("faces: ", len(frames))

import os
import glob
import json
def get_subfolders(main_folder):
    return [f.path for f in os.scandir(main_folder) if f.is_dir()]

def count_jpg_files(folder_path):
    return len(glob.glob(os.path.join(folder_path, '*.jpg')))

def count_jpg_files_face(folder_path):
    return len(glob.glob(os.path.join(folder_path, "pyframes", '*.jpg')))

frames_main_folder = "/data1/gaoxiang_cong/TalkNet-ASD-main/Chem_buwan/Chem_frame"
faces_main_folder = "/data1/gaoxiang_cong/TalkNet-ASD-main/Chem_buwan/Chem_face"

a = 0 
json_all = []
frames_subfolders = get_subfolders(frames_main_folder)
faces_subfolders = get_subfolders(faces_main_folder)
all_counts_match = True
for frames_subfolder in frames_subfolders:
    subfolder_name = os.path.basename(frames_subfolder)
    faces_subfolder = os.path.join(faces_main_folder, subfolder_name)
    if os.path.exists(faces_subfolder):
        frames_count = count_jpg_files(frames_subfolder)
        faces_count = count_jpg_files_face(faces_subfolder)
        if frames_count != faces_count:
            a = a+1
            json_all.append(subfolder_name)
            all_counts_match = False
            print(f"Discrepancy found: {frames_subfolder} has {frames_count} images, while {faces_subfolder} has {faces_count} images.")
    else:
        all_counts_match = False
        print(f"Matching subfolder does not exist for {frames_subfolder}")

if all_counts_match:
    print("All corresponding subfolders have the same number of images.")
else:
    print("Some corresponding subfolders do not have the same number of images.")

print("The number of not alignment", a)

with open('/data1/gaoxiang_cong/TalkNet-ASD-main/chem_output_face/all_chem.json', 'w') as json_file:
    json.dump(json_all, json_file)
