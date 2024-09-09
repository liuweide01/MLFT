from typing import Tuple, Union
import torch
import torch.nn as nn

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets import ViT

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



class Model(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )

        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.grid_dim = 3
        # self.fc_theta_pre = nn.Conv2d(feature_size* 21 *(self.grid_dim**3), feature_size* 21 *(self.grid_dim**3), kernel_size=1, bias=False)  # type: ignore
        self.fc_theta_pre_1_next = nn.Linear(8864, 1024)
        self.fc_theta = nn.Linear( 1024 , 1)
        self.fc_phi_pre_1_next = nn.Linear(8864, 1024)
        self.fc_phi = nn.Linear(1024,1)  # type: ignore
        self.encoder_direction_theta_pre = nn.Linear(6, 128)
        self.encoder_direction_phi_pre = nn.Linear(6, 128)
        self.activation = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.real_feature_3 = nn.Linear(64 + 64, 64)
        self.real_feature_2 = nn.Linear(32 + 64, 64)
        self.real_feature_1 = nn.Linear(16 + 32, 128)

        self.real_feature_3_next = nn.Linear(64 + 64, 64)
        self.real_feature_2_next = nn.Linear(32 + 64, 64)
        self.real_feature_1_next = nn.Linear(16 + 32, 128)

        self.position_encoding = nn.Linear(5, 64)

        self.fix_encoder_0 = nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1)
        self.fix_encoder_1 = nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1)
        self.fix_encoder_2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)


        self.fix_encoder_0_tom = nn.Conv3d(6, 8, kernel_size=3, stride=2, padding=1)
        self.fix_encoder_1_tom = nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1)
        self.fix_encoder_2_tom = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)


    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def calculate_batch_value_at_points(self, extracted_feature, target_points):
        bs, _, _, _, features = extracted_feature.shape
        device = extracted_feature.device

        # Create grid indices for the 3x3x3 surrounding points
        x = torch.tensor([0, 1, 2], device=device)
        y = torch.tensor([0, 1, 2], device=device)
        z = torch.tensor([0, 1, 2], device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

        # Stack and repeat the grid indices to match the batch size
        grid_points = torch.stack((grid_x, grid_y, grid_z), dim=-1).float()
        grid_points = grid_points.reshape(1, 27, 3).repeat(bs, 1, 1)  # Shape: [bs, 27, 3]

        # Expand target points to match the grid points shape
        target_points = target_points.view(bs, 1, 3).repeat(1, 27, 1)  # Shape: [bs, 27, 3]

        # Calculate distances from each surrounding point to the target point
        distances = torch.sqrt(((grid_points - target_points) ** 2).sum(dim=2))

        # Compute weights as the inverse of distances, add epsilon to avoid division by zero
        weights = 1.0 / (distances + 1e-6)
        weights /= weights.sum(dim=1, keepdim=True)  # Normalize weights

        values_reshaped = extracted_feature.view(bs, 27, features)
        weighted_values = torch.sum(values_reshaped * weights.unsqueeze(2), dim=1)

        return weighted_values  # [bs, 64]


    def forward(self, x_in, point,pre_theta,pre_phi, next_point, position_vector, seg, fix_0, fix_1):
        tom_cat  = torch.cat([fix_0, fix_1], dim= 1)

        tom_feature = self.relu(self.fix_encoder_1_tom(self.relu(self.fix_encoder_0_tom(tom_cat))))
        tom_feature_real = self.relu(self.fix_encoder_2_tom(tom_feature) )

        fix_feature = self.relu(self.fix_encoder_1(self.relu(self.fix_encoder_0(seg))))
        seg_feature = self.relu(self.fix_encoder_2(fix_feature))

        fix_feature = torch.cat([fix_feature, tom_feature], dim= 1)
        seg_feature = torch.cat([seg_feature, tom_feature_real], dim= 1)


        bs = point.shape[0]

        x, hidden_states_out = self.vit(x_in)

        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))

        x4 = hidden_states_out[9]

        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        point_32 = point/4

        interpolated_feature_2 = F.interpolate(seg_feature, size=(dec2.shape[2], dec2.shape[3], dec2.shape[4]),
                                             mode='trilinear', align_corners=False)
        dec2_fix = torch.cat([dec2, interpolated_feature_2], dim=1)

        feat_32 = extract_features_with_padding(dec2_fix.permute(0,2,3,4,1).squeeze(),point_32)  # torch.Size([2, 3, 3, 3, 64])
        real_point32 = point_32 - torch.floor(point_32).int() + 1.0
        real_point32_feature = self.calculate_batch_value_at_points(feat_32, real_point32)
        feat_32 = feat_32[:,1,1,1,:].view(bs, -1)
        real_point32_feature = self.real_feature_3(real_point32_feature.to(torch.float)).view(bs, -1)

        dec1 = self.decoder3(dec2, enc2)

        interpolated_feature_1 = F.interpolate(seg_feature, size=(dec1.shape[2], dec1.shape[3], dec1.shape[4]),
                                               mode='trilinear', align_corners=False)
        dec1_fix = torch.cat([dec1, interpolated_feature_1], dim=1)

        point_31 = point/2
        feat_31 = extract_features_with_padding(dec1_fix.permute(0,2,3,4,1).squeeze(),point_31)
        real_point31 = point_31 - torch.floor(point_31).int() + 1.0
        real_point31_feature = self.calculate_batch_value_at_points(feat_31, real_point31)
        feat_31 = feat_31.view(bs, -1)
        real_point31_feature = self.real_feature_2(real_point31_feature.to(torch.float)).view(bs, -1)

        out = self.decoder2(dec1, enc1)

        interpolated_feature_out = F.interpolate(fix_feature, size=(out.shape[2], out.shape[3], out.shape[4]),
                                               mode='trilinear', align_corners=False)
        out_fix = torch.cat([out, interpolated_feature_out], dim=1)

        feat_30 = extract_features_with_padding(out_fix.permute(0,2,3,4,1).squeeze(),point)
        real_point30 = point - torch.floor(point).int() + 1.0
        real_point30_feature = self.calculate_batch_value_at_points(feat_30, real_point30)
        feat_30 = feat_30.view(bs, -1)
        real_point30_feature = self.real_feature_1(real_point30_feature.to(torch.float)).view(bs, -1)

        ####################################################
        next_point_4 = next_point / 4
        feat_32_next = extract_features_with_padding(dec2_fix.permute(0, 2, 3, 4, 1).squeeze(),
                                                next_point_4)
        real_point32_next = next_point_4 - torch.floor(next_point_4).int() + 1.0
        real_point32_feature_next = self.calculate_batch_value_at_points(feat_32_next, real_point32_next)
        feat_32_next = feat_32_next[:, 1, 1, 1, :].view(bs, -1)
        real_point32_feature_next = self.real_feature_3_next(real_point32_feature_next.to(torch.float)).view(bs, -1)


        point_31_next = next_point / 2
        feat_31_next = extract_features_with_padding(dec1_fix.permute(0, 2, 3, 4, 1).squeeze(), point_31_next)
        real_point31_next = point_31_next - torch.floor(point_31_next).int() + 1.0
        real_point31_feature_next = self.calculate_batch_value_at_points(feat_31_next, real_point31_next)
        feat_31_next = feat_31_next.view(bs, -1)
        real_point31_feature_next = self.real_feature_2_next(real_point31_feature_next.to(torch.float)).view(bs, -1)


        feat_30_next = extract_features_with_padding(out_fix.permute(0, 2, 3, 4, 1).squeeze(), next_point)
        real_point30_next = next_point - torch.floor(next_point).int() + 1.0
        real_point30_feature_next = self.calculate_batch_value_at_points(feat_30_next, real_point30_next)
        feat_30_next = feat_30_next.view(bs, -1)
        real_point30_feature_next = self.real_feature_1_next(real_point30_feature_next.to(torch.float)).view(bs, -1)

        ###################################################


        direct_feats_theta = self.relu(self.encoder_direction_theta_pre(pre_theta))
        direct_feats_phi = self.relu(self.encoder_direction_phi_pre(pre_phi))
        position_feature = self.position_encoding(position_vector)

        # print(feat_32.shape,feat_31.shape,feat_30.shape)
        feats_thetaphi = torch.cat([feat_32, feat_31,feat_30,real_point32_feature, real_point31_feature, real_point30_feature,
                                    feat_32_next, feat_31_next,feat_30_next,real_point32_feature_next, real_point31_feature_next, real_point30_feature_next,
                                    direct_feats_theta,direct_feats_phi,position_feature],dim=1)

        theta = self.fc_theta(self.activation(self.fc_theta_pre_1_next(feats_thetaphi)))
        phi = self.fc_phi(self.activation(self.fc_phi_pre_1_next(feats_thetaphi)))

        return theta, phi

if __name__ == '__main__':

    import torch.nn.functional as F
    from dataset import spherical_to_cartesian_torch

    model = Model(in_channels= 28 , out_channels=16, img_size=[128, 128, 128]).to('cuda')
    x = torch.randn([1, 28,128,128,128]).to('cuda')
    point = torch.tensor([[40.6, 60.8, 112.4],[43.1, 62.8, 64.9]]).to('cuda')
    pre_theta = torch.zeros(point.shape[0],6).to('cuda')
    pre_phi = torch.zeros(point.shape[0], 6).to('cuda')

    distance_vector = torch.zeros(point.shape[0], 5).to('cuda')

    x1 = torch.randn([1, 1, 128, 128, 128]).to('cuda')
    x2 = torch.randn([1, 3, 128, 128, 128]).to('cuda')

    theta, phi = model(x, point,pre_theta,pre_phi,point,distance_vector, x1, x2, x2)

    delta_xyz = spherical_to_cartesian_torch(1, theta, phi)

    cos_sim = F.cosine_similarity(delta_xyz, delta_xyz)
    cos_sim_loss = 1 - cos_sim

    print(theta, phi, cos_sim_loss)