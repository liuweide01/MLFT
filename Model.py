from typing import Tuple, Union
import torch
import torch.nn as nn

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets import ViT

from dataset import extract_features_with_padding

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
        self.fc_theta_pre_1 = nn.Linear(1808 + 64 + 1616, 1024)
        self.fc_theta = nn.Linear( 1024 , 1)
        self.fc_phi_pre_1 = nn.Linear(1808 + 64 + 1616, 1024)
        self.fc_phi = nn.Linear(1024,1)  # type: ignore
        self.encoder_direction_theta = nn.Linear(2, 128)
        self.encoder_direction_phi = nn.Linear(2, 128)
        self.activation = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.real_feature_3 = nn.Linear(64, 64)
        self.real_feature_2 = nn.Linear(32, 64)
        self.real_feature_1 = nn.Linear(16, 128)

        self.real_feature_enc_3 = nn.Linear(64, 64)
        self.real_feature_enc_2 = nn.Linear(32, 64)
        self.real_feature_enc_1 = nn.Linear(16, 128)

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

        # Apply weights to values
        # Reshape values to [bs, 27, 64] to match the weights shape
        values_reshaped = extracted_feature.view(bs, 27, features)
        weighted_values = torch.sum(values_reshaped * weights.unsqueeze(2), dim=1)

        return weighted_values  # [bs, 64]


    def forward(self, x_in, point,pre_theta,pre_phi):
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
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)

        point_4 = point/4
        point_2 = point / 2

        feat_dec_4 = extract_features_with_padding(dec2.permute(0,2,3,4,1).squeeze(),point_4)  # torch.Size([2, 3, 3, 3, 64])
        real_point4 = point_4 - torch.floor(point_4).int() + 1.0
        print(feat_dec_4.shape, real_point4.shape)
        real_point4_dec_feature = self.calculate_batch_value_at_points(feat_dec_4, real_point4)
        feat_dec_4 = feat_dec_4[:,1,1,1,:].view(bs, -1)
        real_point4_dec_feature = self.real_feature_3(real_point4_dec_feature).view(bs, -1)


        feat_dec_2 = extract_features_with_padding(dec1.permute(0,2,3,4,1).squeeze(),point_2)
        real_point2 = point_2 - torch.floor(point_2).int() + 1.0
        real_point2_dec_feature = self.calculate_batch_value_at_points(feat_dec_2, real_point2)
        feat_dec_2 = feat_dec_2.view(bs, -1)
        real_point2_dec_feature = self.real_feature_2(real_point2_dec_feature).view(bs, -1)


        feat_dec_1 = extract_features_with_padding(out.permute(0,2,3,4,1).squeeze(),point)
        real_point1 = point - torch.floor(point).int() + 1.0
        real_point1_dec_feature = self.calculate_batch_value_at_points(feat_dec_1, real_point1)
        feat_dec_1 = feat_dec_1.view(bs, -1)
        real_point1_dec_feature = self.real_feature_1(real_point1_dec_feature).view(bs, -1)

        feat_enc_4 = extract_features_with_padding(enc3.permute(0, 2, 3, 4, 1).squeeze(),
                                                   point_4)  # torch.Size([2, 3, 3, 3, 64])
        real_point4_enc_feature = self.calculate_batch_value_at_points(feat_enc_4, real_point4)
        feat_enc_4 = feat_enc_4[:, 1, 1, 1, :].view(bs, -1)
        real_point4_enc_feature = self.real_feature_enc_3(real_point4_enc_feature).view(bs, -1)

        feat_enc_2 = extract_features_with_padding(enc2.permute(0, 2, 3, 4, 1).squeeze(), point_2)
        real_point2_enc_feature = self.calculate_batch_value_at_points(feat_enc_2, real_point2)
        feat_enc_2 = feat_enc_2.view(bs, -1)
        real_point2_enc_feature = self.real_feature_enc_2(real_point2_enc_feature).view(bs, -1)

        feat_enc_1 = extract_features_with_padding(enc1.permute(0, 2, 3, 4, 1).squeeze(), point)
        real_point1_enc_feature = self.calculate_batch_value_at_points(feat_enc_1, real_point1)
        feat_enc_1 = feat_enc_1.view(bs, -1)
        real_point1_enc_feature = self.real_feature_enc_1(real_point1_enc_feature).view(bs, -1)


        direct_feats_theta = self.relu(self.encoder_direction_theta(pre_theta))
        direct_feats_phi = self.relu(self.encoder_direction_phi(pre_phi))

        feats_thetaphi = torch.cat([feat_dec_4, feat_dec_2,feat_dec_1,feat_enc_4,feat_enc_2,feat_enc_1,
                                    real_point4_dec_feature, real_point2_dec_feature, real_point1_dec_feature,
                                    real_point4_enc_feature, real_point2_enc_feature, real_point1_enc_feature,
                                    direct_feats_theta,direct_feats_phi],dim=1)
        theta = self.fc_theta(self.activation(self.fc_theta_pre_1(feats_thetaphi)))
        phi = self.fc_phi(self.activation(self.fc_phi_pre_1(feats_thetaphi)))
        # print(theta,phi)

        return theta, phi

if __name__ == '__main__':
    model = Model(in_channels= 28 , out_channels=16, img_size=[128, 128, 128]).to('cuda')
    x = torch.randn([1, 28,128,128,128]).to('cuda')
    point = torch.tensor([[40.6, 60.8, 112.4],[43.1, 62.8, 64.9]]).to('cuda')
    pre_theta = torch.zeros(point.shape[0],2).to('cuda')
    pre_phi = torch.zeros(point.shape[0], 2).to('cuda')
    theta, phi = model(x, point,pre_theta,pre_phi)
    print(theta, phi)