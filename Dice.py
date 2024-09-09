from dipy.io.streamline import load_trk
from dipy.tracking import utils
import nibabel as nib
import numpy as np
import os
import pandas as pd
import fnmatch

def dice(x1, x2, eps=1e-3):
    if np.mean(x1 == 0) + np.mean(x1 == 1) < 1 or np.mean(x2 == 0) + np.mean(x2 == 1) < 1:
        print('The arrays should include ones and zeros only')
        val = None

    else:
        dice_num = 2 * np.sum((x1 == 1) * (x2 == 1))
        dice_den = np.sum(x1 == 1) + np.sum(x2 == 1)
        # den_zero= dice_den==0
        val = (dice_num + eps) / (dice_den + eps)

    return val


def calculate_segmentation_metrics(pred, gt):
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    # Intersection and Union
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    TP = intersection
    FP = pred_bool.sum() - TP
    FN = gt_bool.sum() - TP

    # Calculate IoU, Recall, and Precision
    iou = TP / union if union != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    return iou, recall, precision

DENSITY_THR_PERC = 0

def calculate_dice(ref_image_address,trk_address_1,trk_address_2):
    ref_img = nib.load(ref_image_address)

    trk_data_1 = load_trk(trk_address_1, ref_image_address)
    streamlines_data_1 = trk_data_1.streamlines

    trk_density_1 = utils.density_map(streamlines_data_1, ref_img.affine, ref_img.shape[:3])
    density_v_1 = trk_density_1[trk_density_1 > 0]
    density_threshold_1 = np.percentile(density_v_1, DENSITY_THR_PERC)
    trk_mask_1 = (trk_density_1 > density_threshold_1).astype(np.int8)

    trk_data_2 = load_trk(trk_address_2, ref_image_address)
    streamlines_data_2 = trk_data_2.streamlines

    trk_density_2 = utils.density_map(streamlines_data_2, ref_img.affine, ref_img.shape[:3])
    density_v_2 = trk_density_2[trk_density_2 > 0]
    density_threshold_2 = np.percentile(density_v_2, DENSITY_THR_PERC)
    trk_mask_2 = (trk_density_2 > density_threshold_2).astype(np.int8)

    iou, recall, precision = calculate_segmentation_metrics(trk_mask_2, trk_mask_1)
    dice_v = dice(trk_mask_1, trk_mask_2)
    return dice_v, iou, recall, precision


def read_paths(paths_file):
    with open(paths_file, 'r') as file:
        lines = file.readlines()
        data_list = [line.strip() for line in lines]
    return data_list

subject_list = read_paths('../Track_Subjects/filenames_test_1.txt')
track_list = ['or.right.trk', 'or.left.trk', 'atr.right.trk', 'atr.left.trk', 'cst.right.trk', 'cst.left.trk',
              'ilf.right.trk','ilf.left.trk', 'ifo.right.trk','ifo.left.trk','cc_7.trk','cc_1.trk','cc_2.trk','uf.right.trk','uf.left.trk']

df_dice = pd.DataFrame(columns=subject_list)
for row_name in track_list:
    df_dice.loc[row_name] = [pd.NA] * len(subject_list)

df_iou = pd.DataFrame(columns=subject_list)
for row_name in track_list:
    df_iou.loc[row_name] = [pd.NA] * len(subject_list)

df_recall = pd.DataFrame(columns=subject_list)
for row_name in track_list:
    df_recall.loc[row_name] = [pd.NA] * len(subject_list)

df_precision = pd.DataFrame(columns=subject_list)
for row_name in track_list:
    df_precision.loc[row_name] = [pd.NA] * len(subject_list)

excel_root = './dice_final_v0'
saved_trk_filter = ''
source_trk_filter = '../evaluation_files/iFOD2_2'

for subject in subject_list:
    for track in track_list:

        trk_path = os.path.join(source_trk_filter, str(subject), str(subject) + '_' + str(track) )

        nii = os.path.join('../Track_Subjects/All_segmentations_final', subject + '.nii.gz')
        pre_trk_path = os.path.join(saved_trk_filter, str(subject), str(subject) + '_' + str(track) )

        if os.path.exists(pre_trk_path) and os.path.exists(trk_path):
            # print('now calculate', pre_trk_path)
            try:
                dice_value, iou, recall, precision = calculate_dice(nii, trk_path, pre_trk_path)
                df_dice.at[track, subject] = dice_value
                df_iou.at[track, subject] = iou
                df_recall.at[track, subject] = recall
                df_precision.at[track, subject] = precision
            except:
                print('something wrong with trk of,', pre_trk_path)

        elif os.path.exists(pre_trk_path) and (os.path.exists(trk_path) is False):
            print('The file exist in pre but not in iFOD2:', pre_trk_path)

        elif (os.path.exists(pre_trk_path) is False) and (os.path.exists(trk_path)):
            print('The file exist in iFOD2 but not in pre:', pre_trk_path)


os.makedirs(excel_root,exist_ok=True)
excel_file_path_dice = os.path.join(excel_root,'./excel_dice.xlsx')
excel_file_path_iou = os.path.join(excel_root,'./excel_iou.xlsx')
excel_file_path_precision = os.path.join(excel_root,'./excel_precision.xlsx')
excel_file_path_recall = os.path.join(excel_root,'./excel_recall.xlsx')

df_dice.to_excel(excel_file_path_dice, index=True, engine='openpyxl')
df_iou.to_excel(excel_file_path_iou, index=True, engine='openpyxl')
df_precision.to_excel(excel_file_path_precision, index=True, engine='openpyxl')
df_recall.to_excel(excel_file_path_recall, index=True, engine='openpyxl')
