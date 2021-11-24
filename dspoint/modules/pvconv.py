import torch.nn as nn
import modules.functional as F
from modules.voxelization import Voxelization

class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)

    def forward(self, inputs):
        # 体素化+降采样
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        # print(voxel_features.shape),torch.Size([4, 10, 10, 10, 10]),batch, channel, resolution**3
        
        # 中间做global feature attention
        voxel_features = self.voxel_layers(voxel_features)
        # print(voxel_features.shape),torch.Size([4, 10, 5, 5, 5])

        # 去体素化+上采样
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)

        return voxel_features, coords


