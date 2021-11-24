"""
Embed PAConv into PointNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from paconv_util.PAConv_util import get_scorenet_input, knn, feat_trans_pointnet, ScoreNet
from paconv_cuda_lib.functional import assign_score_withk_halfkernel as assemble_pointnet




class PAConv(nn.Module):
    def __init__(self,in_channel,out_channel,m=8):
        super(PAConv, self).__init__()
        self.calc_scores = 'softmax'

        self.m2 = m # m controls number of weight matrix, higher, more capacity to fit, 8 by default
        self.scorenet2 = ScoreNet(66, self.m2, hidden_unit=[64,32,16])


        i2 = in_channel  # channel dim of output_1st and input_2nd
        o2 = out_channel  # channel dim of output_2st and input_3rd

        tensor2 = nn.init.kaiming_normal_(torch.empty(self.m2, i2, o2), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i2, self.m2 * o2)

        # convolutional weight matrices in Weight Bank:
        self.matrice2 = nn.Parameter(tensor2, requires_grad=True)


        self.bn1 = nn.BatchNorm1d(in_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)


        self.conv1 = nn.Conv1d(in_channel, in_channel, kernel_size=1, bias=False)



    def forward(self, x,idx,xyz_score):

        x = self.conv1(x)
        x = F.relu(self.bn1(x),inplace=True)
        ##################
        # replace the intermediate 3 MLP layers with PAConv:
        """CUDA implementation of PAConv: (presented in the supplementary material of the paper)"""
        """feature transformation:"""
        # do conv with m kernel, project feature to m space
        x = feat_trans_pointnet(point_input=x, kernel=self.matrice2, m=self.m2)  # b,n,m1,o1
        
        score2 = self.scorenet2(xyz_score, calc_scores=self.calc_scores, bias=0)
        """assemble with scores:"""
        # point-wise conv
        x = assemble_pointnet(score=score2, point_input=x, knn_idx=idx, aggregate='sum')   # b,o1,n
        x = F.relu(self.bn2(x),inplace=True)



        return x

if __name__== "__main__":
    conv=PAConv(in_channel=64,out_channel=128).to("cuda:0")
    x=torch.ones(32,64,100).to("cuda:0")
    xyz=torch.ones(32,3,100).to("cuda:0")
    x=conv(x,xyz)
    print(x.shape)
