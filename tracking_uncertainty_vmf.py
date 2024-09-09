import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
from scipy.stats import vonmises
from dataset_tracking import DWI_dataset, spherical_to_cartesian_torch

import torch.backends.cudnn as cudnn
import random
from Model import Model
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from nibabel.streamlines.array_sequence import ArraySequence
import torch.nn as nn
from torch.utils.data import DataLoader
import gc
gc.collect()

device = 'cuda'

def read_paths(paths_file):
    with open(paths_file, 'r') as file:
        lines = file.readlines()
        data_list = [line.strip() for line in lines]
    return data_list

saved_trk_folder = './saved_trk_vmf'

seed = 123
val_bs = 128

# Keep the same seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
cudnn.benchmark = True #Benchmark
cudnn.deterministic = True#

os.makedirs(saved_trk_folder, exist_ok=True)


saved_model  = './saved_model_merge_1/final.pth'

model = Model(in_channels=28, out_channels=16, img_size=[128, 128, 128])
model.load_state_dict(torch.load(saved_model), strict=True)
model.eval()


cuda_num = torch.cuda.device_count()
model= nn.DataParallel(model).to(device)


def remove_after_threshold_in_list(arr_list, threshold=128):
    result_list = []
    for arr in arr_list:
        indices_above_threshold = np.where(arr > threshold)[0]
        indices_below_zero = np.where(arr < 0)[0]

        indices = np.concatenate((indices_below_zero, indices_above_threshold))
        first_index = indices.min() if indices.size > 0 else None

        if first_index is not None:
            result_list.append(arr[:first_index])
        else:
            result_list.append(arr)
    return result_list


def _random_VMF_cos(d, kappa, n, device):
    b = (d - 1) / (2 * kappa + torch.sqrt(4 * kappa ** 2 + (d - 1) ** 2))
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * torch.log(1 - x0 ** 2)

    out = torch.empty(0, device=device)
    while out.size(0) < n:
        m = int((n - out.size(0)) * 1.5)
        z = torch.distributions.Beta((d - 1) / 2, (d - 1) / 2).sample((m,)).to(device)
        t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        u = torch.distributions.Exponential(torch.tensor([1.0], device=device)).sample((m,)).squeeze().to(device)
        test = kappa * t + (d - 1) * torch.log(1 - x0 * t) - c
        accept = test >= -u
        if out.size(0) == 0:
            out = t[accept]
        else:
            out = torch.cat((out, t[accept]), 0)

    return out[:n]

def random_VMF(mu_or, kappa, size=None, device='cuda'):
    mu_or = mu_or.to(device)
    bs, d = mu_or.shape
    n = torch.prod(torch.tensor(size)).item() if size is not None else 1

    # Normalize the batch of mean directions
    mu_or = mu_or / torch.norm(mu_or, dim=1, keepdim=True)

    # Generate z for the entire batch
    z = torch.randn(bs, n, d, device=device)
    z /= torch.norm(z, dim=2, keepdim=True)

    # Compute dot product and adjust z to be orthogonal to mu
    mu_z = torch.bmm(z, mu_or.unsqueeze(-1)).squeeze(-1)
    z = z - mu_z.unsqueeze(-1) * mu_or.unsqueeze(1)
    z /= torch.norm(z, dim=2, keepdim=True)

    cos = _random_VMF_cos(d, kappa, n * bs, device).view(bs, n, 1)
    sin = torch.sqrt(1 - cos ** 2)

    mu_or_expanded = mu_or.view(bs, 1, d).expand(-1, n, -1)

    x = z * sin + cos * mu_or_expanded

    return x.view(bs, n, d)


def rotate_batch_vectors(a, b, m):
    a_unit = a / torch.linalg.norm(a, dim=1, keepdim=True)
    b_unit = b / torch.linalg.norm(b, dim=1, keepdim=True)

    axis = torch.cross(a_unit, b_unit)
    axis_norm = torch.linalg.norm(axis, dim=1, keepdim=True)

    valid = axis_norm.squeeze() > 0
    axis[valid] = axis[valid] / axis_norm[valid]

    cos_angle = torch.sum(a_unit * b_unit, dim=1, keepdim=True)
    angle = torch.arccos(torch.clamp(cos_angle, -1.0, 1.0))

    scaled_angle = angle * m

    a_rot = (a_unit * torch.cos(scaled_angle) +
             torch.cross(axis, a_unit) * torch.sin(scaled_angle) +
             axis * (torch.sum(axis * a_unit, dim=1, keepdim=True)) * (1 - torch.cos(scaled_angle)))

    # print(torch.linalg.norm(a_rot, dim=1, keepdim=True))
    return a_rot


def custom_sigmoid(a, a_min=0.15, a_max=0.25, b_min=0.1, b_max=0.9):
    # Parameters to adjust the function
    steepness = 15
    midpoint = (a_min + a_max) / 2

    a = torch.tensor(a, dtype=torch.float32)

    b = (b_max - b_min) / (1 + torch.exp(steepness * (a - midpoint))) + b_min

    return b

data_list = read_paths('../Track_Subjects/filenames_test.txt')

repeat_number = 5

with torch.no_grad():
    for subject in data_list:
        print('Now let us train the subject of', subject)

        for i in range(1):
            index = 0
            saved_lines = []
            train_set = DWI_dataset(subject)
            train_loader = DataLoader(train_set, batch_size=val_bs, shuffle=False, num_workers=64, pin_memory=True)

            fod_img = train_set.fod_img.to(device)
            features_5d = fod_img.permute(3, 0, 1, 2).unsqueeze(0)

            fa_img = train_set.fa.to(device)

            img_nii = train_set.dwi_path

            mean_positions = train_set.mean_positions

            for _, data in enumerate(train_loader):
                path_point, theta, phi = data
                path_point = path_point[:,0]
                bs = path_point.shape[0]
                path_point, theta, phi = path_point.repeat(repeat_number,1).float(), theta.repeat(repeat_number,1), phi.repeat(repeat_number,1)
                print(path_point.shape,theta.shape,phi.shape)
                random_tensor = (torch.rand(bs * repeat_number, 3) - 0.5)
                path_point = path_point + random_tensor

                random_tensor_theta = (torch.rand(bs * repeat_number, 1) * 0.2 - 0.1)   # Scale to 0.2 and shift by -0.1
                random_tensor_phi = (torch.rand(bs * repeat_number, 1) * 0.2 - 0.1)
                theta = theta + random_tensor_theta
                phi = phi + random_tensor_phi

                path_point, theta, phi = path_point.to(device), theta.to(device), phi.to(device)

                theta = torch.clamp(theta, min=0, max=torch.pi)
                phi = torch.clamp(phi, min=-torch.pi, max=torch.pi)

                current_point = path_point

                pre_theta = torch.zeros(path_point.shape[0], 12).to(device)
                pre_phi = torch.zeros(path_point.shape[0], 12).to(device)

                cols_to_update = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Columns you want to update
                pre_theta[:, cols_to_update] = theta[:, 0].detach().unsqueeze(1).expand(-1, len(cols_to_update)).float()
                pre_phi[:, cols_to_update] = phi[:, 0].detach().unsqueeze(1).expand(-1, len(cols_to_update)).float()


                saved_points = [[] for _ in range(current_point.shape[0])]

                current_point_np = current_point.detach().cpu().numpy()
                for b in range(current_point.shape[0]):  # Save points for each item in the batch
                    saved_points[b].append(current_point_np[b])

                for step in range(240):
                    differences = current_point_np[:, np.newaxis, :] - mean_positions[np.newaxis, :, :]
                    distances = np.sqrt(np.sum(differences ** 2, axis=2))
                    sum_distances = distances.sum(axis=1, keepdims=True)
                    normalized_distances = distances / np.where(sum_distances != 0, sum_distances, np.finfo(float).eps)
                    distance_vector = torch.from_numpy(normalized_distances).to(device).float()

                    channels_to_extract = [0, 2, 4, 6, 8, 10]
                    extracted_theta = pre_theta[:, channels_to_extract]
                    extracted_phi = pre_phi[:, channels_to_extract]

                    next_xyz_pre_define = spherical_to_cartesian_torch(0.5, pre_theta[:, 0], pre_phi[:, 0])
                    next_point_pre_define = current_point + next_xyz_pre_define
                    out_theta, out_phi = model(features_5d, current_point, extracted_theta, extracted_phi, next_point_pre_define,distance_vector)

                    out_theta = torch.clamp(out_theta, min=0, max=torch.pi)
                    out_phi = torch.clamp(out_phi, min=-torch.pi, max=torch.pi)

                    pre_theta[:, 1:] = pre_theta[:, :-1].detach()
                    pre_phi[:, 1:] = pre_phi[:, :-1].detach()
                    pre_theta[:, 0] = out_theta[:, 0].detach()
                    pre_phi[:, 0] = out_phi[:, 0].detach()

                    delta_xyz_pre = spherical_to_cartesian_torch(0.5, out_theta, out_phi)
                    bs, d, _ = delta_xyz_pre.shape

                    delta_xyz = random_VMF(delta_xyz_pre.squeeze(), kappa=torch.tensor(64).to(device), size=1, device=device).reshape(bs, d, 1)

                    ## Fa part
                    # point_round = torch.round(current_point).to(torch.int32)
                    # z, y, x = point_round[:, 2], point_round[:, 1], point_round[:, 0]
                    # try:
                    #     fa_value = custom_sigmoid(fa_img[x, y, z])
                    #     new_xyz = rotate_batch_vectors(delta_xyz_pre, delta_xyz, fa_value) * 0.5
                    # except:
                    #     new_xyz = delta_xyz
                    ## End Fa part

                    try:
                        # next_point = new_xyz[:,:,0].detach().cpu().numpy()
                        next_point = delta_xyz[:, :, 0].detach().cpu().numpy()
                    except:
                        print('testing with wrong, break')
                        break

                    current_point_np = current_point.detach().cpu().numpy()

                    current_point_np += next_point

                    for b in range(current_point.shape[0]):  # Save points for each item in the batch
                        saved_points[b].append(current_point_np[b])

                    current_point = torch.from_numpy(current_point_np).to(device)

                for b in range(current_point.shape[0]):  # Save final points for each item in the batch
                    saved_lines.append(np.array(saved_points[b]).squeeze())  # Append the saved points as a line

                index += path_point.shape[0]
                print('Finished subject', subject, 'The subject has lines', index )

            sft = StatefulTractogram(ArraySequence(remove_after_threshold_in_list(saved_lines)), img_nii, Space.RASMM)
            save_address_folder = os.path.join(saved_trk_folder)
            save_address= os.path.join(save_address_folder, 'predict-subject-' + str(subject) + '-'  +  '.trk')
            save_trk(sft, save_address)