import torch
from torch import nn
class Autoregression(nn.Module):
    def __init__(self,device='cuda'):
        super(Autoregression,self).__init__()
        self.device = device

        mlp_depth=2
        self.num_joints = 23
        embedding_size = 69
        mlp_width = 128
        block_mlps = [nn.Linear(embedding_size, mlp_width), nn.ReLU()]
        
         # TODO: change heavy network
        for _ in range(0, mlp_depth-1):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        block_mlps += [nn.Linear(mlp_width, 9 * self.num_joints)] 

        self.block_mlps = nn.Sequential(*block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope the rotation matrix can be identity 
        
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()
    def forward(self,feature):
        joint_F = self.block_mlps(feature).view(-1, 3, 3)  # (bsize, 3, 3)

        joint_U, joint_S, joint_V = torch.svd(joint_F.cpu())  # (bsize, 3, 3), (bsize, 3), (bsize, 3, 3)

        return {
            "Rs": joint_F,
            "pose_U":joint_U,
            "pose_S":joint_S,
            "pose_V":joint_V,
        }
        
        
        
data = torch.rand(1,69)

net = Autoregression()

a = net(data)
print(a)