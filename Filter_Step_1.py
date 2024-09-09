import numpy as np
import nibabel as nib
import os
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from nibabel.streamlines.array_sequence import ArraySequence
from dipy.io.streamline import save_trk


source_trk_file = 'saved_trk_9_tracks'
saved_trk_filter = 'saved_trk_9_tracks_wm'

os.makedirs(saved_trk_filter, exist_ok=True)

def process_streamlines(subject, trk_path, mask_path,nii):
    trk_file = nib.streamlines.load(trk_path)
    streamlines = trk_file.streamlines
    mask_affine = nib.load(nii).affine
    mask_img = nib.load(mask_path)
    mask_data =  mask_img.get_fdata()

    processed_streamlines = []

    for sl in streamlines:
        save_line = True
        processed_sl = []
        step = 0
        sl_len =len(sl)
        for i in range(sl_len):
            point = sl[i]
            if step < 6:
                processed_sl.append(point)
                step += 1
            else:
                x, y, z = np.clip(np.round(point).astype(int), 0, np.array(mask_data.shape[:3]) - 1)
                if mask_data[x, y, z, 3] == 1:
                    save_line = False
                    break
                elif mask_data[x, y, z, 0] == 1 or mask_data[x, y, z, 1] == 1:
                    try:
                        processed_sl.append(point)
                        processed_sl.append(sl[i+1])
                        break
                    except:
                        processed_sl.append(point)
                        break

                elif mask_data[x, y, z, 2] == 0:
                    save_line = False
                    break
                else:
                    processed_sl.append(point)

        if save_line and processed_sl:
            processed_sl = nib.affines.apply_affine(mask_affine, processed_sl)
            processed_streamlines.append(processed_sl)

    save_address_folder = os.path.join(saved_trk_filter)
    os.makedirs(save_address_folder, exist_ok=True)

    saved_addresss = os.path.join(save_address_folder, 'processed_streamlines' + str(subject) + '.trk')
    sft = StatefulTractogram(ArraySequence(processed_streamlines), nii, Space.RASMM)
    save_trk(sft, saved_addresss)


def read_paths(paths_file):
    with open(paths_file, 'r') as file:
        lines = file.readlines()
        data_list = [line.strip() for line in lines]
    return data_list


data_list = read_paths('../Track_Subjects/filenames_test_1.txt')

# Demo usage
for subject in data_list:
    for i in range(1):
        if os.path.exists(os.path.join(source_trk_file, 'predict-subject-' + str(subject) + '-'  +  '.trk')):
            trk_path = os.path.join(source_trk_file, 'predict-subject-' + str(subject) + '-'  + '.trk')
            mask_path = os.path.join('../Track_Subjects/All_segmentations_final', subject + '_5tt.nii.gz')
            nii = os.path.join('../Track_Subjects/All_segmentations_final', subject + '.nii.gz')
            process_streamlines(subject, trk_path, mask_path,nii)
