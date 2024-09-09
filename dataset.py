import os.path

from scipy.interpolate import CubicSpline
import torch.nn.functional as F
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import random
from dipy.io.streamline import load_trk
import pandas as pd
default_dim = 128.0
step_length = 0.5

class DWI_dataset(Dataset):
    def __init__(self, subject_name,  max_len=20):

        self.subject_name = subject_name
        self.trk_path = os.path.join('../Track_Subjects/Tracts_use/tracks_all_merged', 'fused_all_tracks_' + self.subject_name + '.trk')

        self.fod_path = os.path.join('../Track_Subjects/Fods', subject_name + '_fod.nii.gz')
        self.dwi_path = os.path.join('../Track_Subjects/All_segmentations_final', subject_name + '.nii.gz')
        self.mask_path = os.path.join('../Track_Subjects/All_segmentations_final', subject_name + '_5tt.nii.gz')
        self.position_path = os.path.join('../Track_Subjects/parcel_5', subject_name + '.csv')

        self.fod_img = torch.tensor(np.squeeze(nib.load(self.fod_path).get_fdata())).float()
        self.trk = load_trk(self.trk_path, self.dwi_path).streamlines
        self.dwi_feature = torch.tensor(np.squeeze(nib.load(self.dwi_path).get_fdata())).float()
        self.mask = torch.tensor(np.squeeze(nib.load(self.mask_path).get_fdata())).float()
        self.max_len = max_len

    def __getitem__(self, index):
        path_gt = self.trk[index]
        mask_affine_inv = np.linalg.inv(nib.load(self.dwi_path).affine)
        trk_list = nib.affines.apply_affine(mask_affine_inv, path_gt)
        length = trk_list.shape[0]

        mean_positions = np.genfromtxt(self.position_path, delimiter=',', skip_header=1)[:, 1:]


        if random.random() > 0.5:
            differences = trk_list[:-1, :][:, np.newaxis, :] - mean_positions[np.newaxis, :, :]
            distances = np.sqrt(np.sum(differences ** 2, axis=2))
            sum_distances = distances.sum(axis=1, keepdims=True)
            normalized_distances = distances / np.where(sum_distances != 0, sum_distances, np.finfo(float).eps)

            r, t, p = cartesian_to_spherical(trk_list[:-1, :], trk_list[1:, :])
            start = random.randint(0, length - self.max_len - 1)
            theta = t[start:start + self.max_len]
            phi = p[start:start + self.max_len]
            radius = r[start:start + self.max_len]
            gt_point = trk_list[start:start + self.max_len, :]
            distance_vector = normalized_distances[start:start + self.max_len, :]

        else:
            trk_list = trk_list[::-1, :].copy()

            differences = trk_list[:-1, :][:, np.newaxis, :] - mean_positions[np.newaxis, :, :]
            distances = np.sqrt(np.sum(differences ** 2, axis=2))
            sum_distances = distances.sum(axis=1, keepdims=True)
            normalized_distances = distances / np.where(sum_distances != 0, sum_distances, np.finfo(float).eps)

            r, t, p = cartesian_to_spherical(trk_list[:-1, :], trk_list[1:, :])
            start = random.randint(0, length - self.max_len - 1)
            theta = t[start:start + self.max_len]
            phi = p[start:start + self.max_len]
            radius = r[start:start + self.max_len]
            gt_point = trk_list[start:start + self.max_len, :]
            distance_vector = normalized_distances[start:start + self.max_len, :]

        return gt_point, theta, phi, radius, distance_vector

    def __len__(self):
        return len(self.trk)

def cartesian_to_spherical(p1, p2):
    # Ensure p1 and p2 are numpy arrays
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    # Calculate the vectors from each point in p1 to the corresponding point in p2
    vectors = p2 - p1

    # Calculate the radial distances
    r = np.linalg.norm(vectors, axis=1)

    # Calculate the polar angles (theta)
    theta = np.arccos(vectors[:, 2] / r)

    # Calculate the azimuthal angles (phi)
    phi = np.arctan2(vectors[:, 1], vectors[:, 0])

    # Handle cases where r is zero to avoid division by zero
    theta[r == 0] = 0

    # return np.vstack((r, theta, phi)).T
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    # Extracting r, theta, and phi
    # theta, phi = spherical_coords[:, 0], spherical_coords[:, 1]
    # Convert spherical to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.column_stack((x, y, z))

def spherical_to_cartesian_torch(r, theta, phi):
    # Extracting theta and phi from the spherical coordinates
    # theta, phi = spherical_coords[:, 0], spherical_coords[:, 1]
    # Convert spherical to Cartesian coordinates using PyTorch operations
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    # Stacking the Cartesian coordinates together
    cartesian_coords = torch.stack((x, y, z), dim=1)
    return cartesian_coords

def interpolate_curve_cubic_spline(samples, step_size=0.5):
    # Convert the list of samples to a NumPy array
    samples = np.array(samples)

    # Calculate the cumulative distance along the curve to use as the parameter
    distances = np.cumsum(np.sqrt(np.sum(np.diff(samples, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Include the starting point

    # Create cubic spline interpolation for each dimension
    cubic_spline_funcs = [CubicSpline(distances, samples[:, i]) for i in range(3)]

    # Interpolate points
    max_distance = distances[-1]
    interpolated_points = [func(np.arange(0, max_distance, step_size)) for func in cubic_spline_funcs]

    return np.array(interpolated_points).T  # Transpose to get points in the original format

def calculate_intersection_distance(spherical_coords, k=step_length):
    # Convert spherical to Cartesian direction
    # r, theta, phi = spherical_coords
    theta, phi = spherical_coords[:,0], spherical_coords[:,1]
    direction = np.array([  #[B,3]
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ]).T

    # Normalize the direction vector    这里似乎不需要正则化？因为上一行里面每个向量的长度一定为1
    # direction = direction / np.linalg.norm(direction)

    # Calculate intersection distances for each face of the cube
    distances = []
    for i in range(3):
        if np.all(direction[i] != 0):   #需要np.all()
            # Positive and negative faces
            d_pos = (k - 0) / direction[i]
            d_neg = (0 - 0) / direction[i]
            distances.extend([d_pos, d_neg])

    # Get the smallest positive distance
    r1 = min([d for d in distances if d > 0])

    return r1


def calculate_position(start_point, theta, phi, alpha, r):
    direction = torch.cat([theta, phi, alpha],dim=1)
    next_point = start_point + direction * r
    return next_point


def generate_3d_grid(center, range=1, step=1):
    x_center, y_center, z_center = center
    x_range = np.arange(x_center - range, x_center + range + 1, step)
    y_range = np.arange(y_center - range, y_center + range + 1, step)
    z_range = np.arange(z_center - range, z_center + range + 1, step)

    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    return grid_points


def get_features_at_grid_points(features, grid_points):
    feature_dim = features.shape[-1]  # Last dimension size
    extracted_features = []
    for point in grid_points:
        x, y, z = point
        # Use zero-padding for out-of-bound points
        feature = features[x, y, z, :] if 0 <= x < features.shape[0] and 0 <= y < features.shape[1] and 0 <= z < features.shape[2] else np.zeros(feature_dim)
        extracted_features.append(feature)
    return np.array(extracted_features)


# def extract_features_with_padding(features, center_points, grid_range=1, step=1, grid_dim=3):
#     batch_size, _ = center_points.size()
#     feature_dim = features.shape[-1]  # The feature dimension
#
#     # Dimensions of the grid
#     grid_dim = grid_dim
#
#     # Prepare a tensor to hold all extracted features (on GPU)
#     all_extracted_features = torch.zeros(batch_size, grid_dim, grid_dim, grid_dim, feature_dim)
#
#     for i, center in enumerate(torch.floor(center_points).int()):
#         x_center, y_center, z_center = center.tolist()
#
#         x_range = torch.arange(x_center - grid_range, x_center + grid_range + 1, step, device=features.device)
#         y_range = torch.arange(y_center - grid_range, y_center + grid_range + 1, step, device=features.device)
#         z_range = torch.arange(z_center - grid_range, z_center + grid_range + 1, step, device=features.device)
#
#         # Extract features with zero-padding for out-of-bound points
#         x_mask = (x_range >= 0) & (x_range < features.shape[0])
#         y_mask = (y_range >= 0) & (y_range < features.shape[1])
#         z_mask = (z_range >= 0) & (z_range < features.shape[2])
#
#         x_range = x_range[x_mask]
#         y_range = y_range[y_mask]
#         z_range = z_range[z_mask]
#
#         sub_features = torch.index_select(features, 0, x_range)
#         sub_features = torch.index_select(sub_features, 1, y_range)
#         sub_features = torch.index_select(sub_features, 2, z_range)
#
#         if all_extracted_features[i].shape != sub_features.shape:
#             return extract_features_with_padding_gpu(features, center_points)
#         all_extracted_features[i] = sub_features
#     return all_extracted_features

def extract_features_with_padding(features, center_points, grid_range=1, step=1, grid_dim=3):
    batch_size, _ = center_points.size()
    feature_dim = features.shape[-1]  # The feature dimension

    # Create a meshgrid of relative indices
    rel_idx = torch.arange(-grid_range, grid_range + 1, step, device=features.device)
    rel_grid = torch.stack(torch.meshgrid(rel_idx, rel_idx, rel_idx, indexing='ij'), dim=-1).view(-1, 3)

    # Floor the center_points and adjust them to be used as indices
    center_points_idx = (torch.round(center_points).to(torch.int64).unsqueeze(1) + rel_grid).view(-1, 3)

    # Clip the indices to be within the range of the features tensor
    max_idx = torch.tensor(features.shape[:3], device=features.device) - 1
    clipped_idx = torch.max(torch.min(center_points_idx, max_idx), torch.tensor(0, device=features.device))

    # Extract features using advanced indexing
    all_extracted_features = features[clipped_idx[:, 0], clipped_idx[:, 1], clipped_idx[:, 2]].view(batch_size,
                                                                                                    *([grid_dim] * 3),
                                                                                                    feature_dim)
    return all_extracted_features


def extract_features_with_padding_gpu(features, center_points, grid_range=1, step=1, grid_dim=3):
    center_points = torch.round(center_points).to(torch.int64)
    batch_size, _ = center_points.size()
    feature_dim = features.shape[-1]  # The feature dimension

    # Dimensions of the grid
    grid_dim = grid_dim

    # Prepare a tensor to hold all extracted features (on GPU)
    all_extracted_features = torch.zeros(batch_size, grid_dim, grid_dim, grid_dim, feature_dim, device='cuda')

    for i, center in enumerate(center_points):
        x_center, y_center, z_center = center.tolist()

        # Generate the grid for the current center point
        x_range = torch.arange(x_center - grid_range, x_center + grid_range + 1, step, device='cuda')
        y_range = torch.arange(y_center - grid_range, y_center + grid_range + 1, step, device='cuda')
        z_range = torch.arange(z_center - grid_range, z_center + grid_range + 1, step, device='cuda')
        X, Y, Z = torch.meshgrid(x_range, y_range, z_range, indexing='ij')

        # Extract features with zero-padding for out-of-bound points
        for x in range(grid_dim):
            for y in range(grid_dim):
                for z in range(grid_dim):
                    x_idx, y_idx, z_idx = X[x, y, z], Y[x, y, z], Z[x, y, z]
                    if 0 <= x_idx < features.shape[0] and 0 <= y_idx < features.shape[1] and 0 <= z_idx < features.shape[2]:
                        all_extracted_features[i, x, y, z] = features[x_idx, y_idx, z_idx, :]

    return all_extracted_features


def get_metric(output, target):
    mae = F.l1_loss(output, target)
    mse = F.mse_loss(output, target)
    rmse = torch.sqrt(mse)
    mape = torch.mean(torch.abs((output - target) / target))
    return mae.sum().item(), mse.sum().item(), rmse.sum().item(), mape.sum().item()

def convert_seconds(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"



if __name__ == '__main__':
    train_set = DWI_dataset('m-atlas_tensor_f1204s1-CWLLS1', max_len=30)
