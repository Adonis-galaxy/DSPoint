from pickle import TRUE
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_
import modules.functional as F
from modules.voxelization import Voxelization
from PointNet_PAConv import PAConv
from paconv_util.PAConv_util import get_scorenet_input, knn

# Hypaparameter, both to be 1 by default
CO_LAMBDA_POINT=1
CO_LAMBDA_VOXEL=1

# used for data prepossing, useless now
class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = nn.functional.relu(self.bn1(self.conv1(x))) # B, D, N
        # x = nn.functional.relu(self.bn2(self.conv2(x))) # B, D, N
        x = nn.functional.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

##################################################
# Positional encoding (section 5.1), 
# From NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
##################################################

# IF batchsize>=32, change 32 to largest batchsize here
# pose embed for voxelized self-attention
pose_global=torch.zeros(32,3,12,12,12)
for a in range(32):
    for c in range(12):
        for d in range(12):
            for e in range(12):
                pose_global[a,:,c,d,e]=torch.tensor([c,d,e])

embed_fn_global, input_ch_global =get_embedder(10) # L=10
pose_global_hf=embed_fn_global(pose_global.transpose(1,4)).transpose(1,4)

# Global Attention Branch
class PVAtt(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, num_heads,normalize=True, eps=0,):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)

        self.dim_sa=in_channels
        self.norm_sa1 = nn.LayerNorm(self.dim_sa)
        self.channel_up = nn.Linear(self.dim_sa, 3 * self.dim_sa) # 这个是attention用的把input生成qkv的linear
        self.attn = Attention_pure(self.dim_sa, num_heads=num_heads)

        # # pos embedpos embed
        # self.pos_embed=nn.Conv3d(self.dim_sa, self.dim_sa, kernel_size=3, stride=1, padding=1, groups=self.dim_sa ,padding_mode='zeros')
        
        # feature fusion
        self.pos_forward=nn.Conv3d(self.dim_sa, self.dim_sa, kernel_size=1, stride=1)

        self.pos_embed=nn.Linear(input_ch_global,self.dim_sa)

        # hf fusion
        self.embed_fuc,self.fusion_ch=get_embedder(10)
        self.fusion_block=nn.Linear(self.fusion_ch,in_channels)
        

    def forward(self, inputs,coords):
        sa, voxel_coords = self.voxelization(inputs, coords)
        # print(voxel_coords.shape)torch.Size([8, 3, 1024])
        # print(sa.shape),torch.Size([4, 10, 10, 10, 10]),batch, channel, resolution**3
        # print(inputs.shape),torch.Size([32, 16, 1024])
        # print(coords.shape),torch.Size([32, 3, 1024])

        # feature fusion, engineer trick, not mentioned in paper due to lack of space
        sa=self.pos_forward(sa)

        # pose embed using voxel coord
        pose=pose_global_hf[0:sa.shape[0],:,0:sa.shape[2],0:sa.shape[3],0:sa.shape[4]].clone()
        pose=pose.to("cuda")
        pose=pose.transpose(1,4)
        # print(pose.shape),torch.Size([32, 12, 12, 12, 63])
        pose=self.pos_embed(pose)
        pose=pose.transpose(1,4)
        sa=sa+pose

        # attention init
        B, _, H_down, W_down, L_down = sa.shape
        sa = sa.flatten(2).transpose(1, 2)

        # residual init
        res_sa=sa
        
        # do attention
        sa = self.norm_sa1(sa)
        sa = self.channel_up(sa)
        sa = self.attn(sa)

        # residual
        sa+=res_sa

        sa=sa.reshape(B, -1, H_down, W_down,L_down)
        # print(voxel_features.shape),torch.Size([4, 10, 5, 5, 5])

        # devoxel
        voxel_features = F.trilinear_devoxelize(sa, voxel_coords, self.resolution, self.training)

        # print(voxel_features.shape),torch.Size([32, 16, 1024]),最后把fusion之后的coord变成这个shape就行

        # hf fusion for attention
        hf=self.fusion_block(self.embed_fuc(coords.transpose(1,2))).transpose(1,2)
        voxel_features=voxel_features+CO_LAMBDA_VOXEL*hf

        return voxel_features, coords

# MLP implemented with conv
class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Atteniton Module
class Attention_pure(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = int(C // 3)
        qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x

# DSBlock
class DSBlock(nn.Module):
    def __init__(self, dim, num_heads, resolution=8, mlp_ratio=4., drop=0., act_layer=nn.GELU, paconv_m=8):
        super().__init__()
        # block init
        self.pos_embed = nn.Conv1d(3, dim, 1)
        self.dim = dim
        self.norm1 = nn.BatchNorm1d(dim)
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.dim_conv = int(dim // 4 * 3)
        self.num_heads = num_heads
        self.dim_sa = dim - self.dim_conv
        self.norm_conv1 = nn.BatchNorm1d(self.dim_conv)
        
        # local and global branch init
        self.pvatt=PVAtt(in_channels=self.dim_sa, out_channels=self.dim_sa, resolution=resolution,num_heads=num_heads)
        self.conv=PAConv(self.dim_conv,self.dim_conv,m=paconv_m)

        # local hf fusion
        self.resolution=resolution
        self.embed_fuc,self.fusion_ch=get_embedder(10)
        self.fusion_block_1=nn.Linear(self.fusion_ch,self.dim_conv)
        self.voxelization = Voxelization(resolution)
        self.fusion_block_2=nn.Linear(self.dim_conv,self.dim_conv)

        # feature post-possesing, not mentioned in paper due to lack of space
        self.norm2 = nn.BatchNorm1d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,xyz,idx,xyz_score):

        # pose embed | high frequency pose embed
        x = x + self.pos_embed(xyz)

        # split feature channel-wise
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        qkv = x[:, :self.dim_sa, :]
        conv = x[:, self.dim_sa:, :]
        residual_conv = conv

        # local conv
        conv=self.norm_conv1(conv)
        conv = residual_conv + self.conv(conv,idx,xyz_score)

        # voxel high frequcy fusion to conv feature
        hf_conv, hf_voxel_coords = self.voxelization(xyz, xyz)   
        pose=pose_global_hf[0:hf_conv.shape[0],:,0:hf_conv.shape[2],0:hf_conv.shape[3],0:hf_conv.shape[4]].clone()
        pose=pose.to("cuda").transpose(1,4)
        hf_conv=self.embed_fuc(hf_conv.transpose(1,4))
        hf_conv=hf_conv+pose
        hf_conv=self.fusion_block_1(hf_conv).transpose(1,4)
        hf_conv = F.trilinear_devoxelize(hf_conv, hf_voxel_coords, self.resolution, self.training)
        hf_conv=self.fusion_block_2(hf_conv.transpose(1,2)).transpose(1,2)
        conv=conv+CO_LAMBDA_POINT*hf_conv

        # attention
        sa,xyz=self.pvatt(qkv,xyz)

        # cat local and global feature
        x = torch.cat([conv, sa], dim=1)

        # feature post-possesing, only tricks, not mentioned in paper due to lack of space
        x = residual + self.conv2(x)
        x = x + self.mlp(self.norm2(x))
        return x,xyz

class PatchEmbed(nn.Module):
    """ Patch Embedding, dimension tranformation
    """
    def __init__(self, in_chans=3, embed_dim=64):
        super().__init__()

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.proj(x)
        return x
    

class DSPoint(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    # 最后一个embed到384就极限了，不能到512，不然会卡住，不知道为什么
    def __init__(self, args,img_size=32, in_chans=63, num_classes=40, embed_dim=[64,64,128],
                 depth=[1, 1, 1], resolutions=[8,6,4],num_heads=[1, 2, 4], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,k_paconv=8,m_paconv=8):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.img_size=img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer=partial(nn.LayerNorm, eps=1e-6)

        self.resolutions=resolutions
        self.k=k_paconv
        self.patch_embed1 = PatchEmbed(
            in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            in_chans=embed_dim[1], embed_dim=embed_dim[2])

        
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks1 = nn.ModuleList([
            DSBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio,
                drop=drop_rate, resolution=resolutions[0],paconv_m=m_paconv)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            DSBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, 
                drop=drop_rate, resolution=resolutions[1],paconv_m=m_paconv)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            DSBlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio,
                drop=drop_rate, resolution=resolutions[2],paconv_m=m_paconv)
            for i in range(depth[2])])

        self.norm = nn.BatchNorm1d(embed_dim[-1])

        # pos encoding nerf
        self.embed_fn, self.input_ch =get_embedder(10) # L=10

        # used for data prepossing, useless now
        self.gather_local = Local_op(in_channels=in_chans*2, out_channels=in_chans)

        # Classifier head
        self.conv5 = nn.Conv1d(embed_dim[-1], 1024, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm1d(1024)
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, num_classes)
    

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x,xyz):
        idx, _ = knn(xyz, k=self.k)  # get the idx of knn in 3D space : b,n,k
        xyz_score = get_scorenet_input(xyz, k=self.k, idx=idx)  # ScoreNet input: 3D coord difference : b,6,n,k
        
        x = self.patch_embed1(x)
        x = self.pos_drop(x)

        for blk in self.blocks1:
            x,xyz = blk(x,xyz,idx,xyz_score)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x,xyz = blk(x,xyz,idx,xyz_score)

        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x,xyz = blk(x,xyz,idx,xyz_score)

        
        x = self.norm(x)
        return x

    def forward(self, x):
        # print(x.shape),torch.Size([32, 3, 1024])
        xyz=x.clone().detach()
        x=self.embed_fn(x.transpose(1,2)).transpose(1,2)

        x = self.forward_features(x,xyz)

        # cls head
        x = self.conv5(x)
        x = nn.functional.relu(self.bn5(x))
        x = nn.functional.adaptive_max_pool1d(x, 1).view(x.shape[0], -1)
        x = nn.functional.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)


        return x





