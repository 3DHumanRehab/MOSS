import torch
import torch.nn as nn
import math
from collections import defaultdict 

class Autoregression(nn.Module):
    def __init__(self,device='cuda'):
        super(Autoregression,self).__init__()
        self.device = device
        mlp_depth=2
        self.num_joints = 23
        embedding_size = 69
        # mlp_width = 128+9 * self.num_joints
        mlp_width = 128
        out_dim = 3
        block_mlps = [nn.Linear(embedding_size, mlp_width), nn.ReLU()]
        for _ in range(0, mlp_depth-1):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        block_mlps += [nn.Linear(mlp_width, 3 * self.num_joints)] 

        self.block_mlps = nn.Sequential(*block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope the rotation matrix can be identity 
        init_val = 1e-5

        self.rodriguez = RodriguesModule()
        self.fc_pose = []
        self.parents_dict = self.immediate_parent_to_all_ancestors()
        for joint in range(self.num_joints):
            num_parents = len(self.parents_dict[joint])
            input_dim = 3 + num_parents * out_dim #(9 + 3 + 9) 

            fc = nn.Sequential(nn.Linear(input_dim,out_dim))
            fc[-1].weight.data.uniform_(-init_val, init_val)
            fc[-1].bias.data.zero_()
            self.fc_pose.append(fc)
        self.fc_pose = nn.Sequential(*self.fc_pose)
                    
    def immediate_parent_to_all_ancestors(self,immediate_parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]):
        """
        :param immediate_parents: list with len = num joints, contains index of each joint's parent.
                - includes root joint, but its parent index is -1.
        :return: ancestors_dict: dict of lists, dict[joint] is ordered list of parent joints.
                - DOES NOT INCLUDE ROOT JOINT! Joint 0 here is actually joint 1 in SMPL.
        """
        ancestors_dict = defaultdict(list)
        for i in range(1, len(immediate_parents)):  # Excluding root joint
            joint = i - 1
            immediate_parent = immediate_parents[i] - 1
            if immediate_parent >= 0:
                ancestors_dict[joint] += [immediate_parent] + ancestors_dict[immediate_parent]
        return ancestors_dict
        
    def forward(self,feature):
        joint_F = self.block_mlps(feature[:, 3:])
        
        joint_F = joint_F.reshape(1,23,3)
        auto_joint_F = torch.zeros_like(joint_F,device=joint_F.device)
        
        for joint in range(self.num_joints):
            parents = self.parents_dict[joint]
            fc_joint = self.fc_pose[joint]
            embed = joint_F[:, joint]
            if len(parents) > 0:
                embed = embed.unsqueeze(1)
                parents_embed = joint_F[:, parents]
                auto_joint_F[:, joint] = fc_joint(torch.cat([embed, parents_embed], dim=1).reshape(1,-1))
            else:
                auto_joint_F[:, joint] = fc_joint(embed)

        auto_joint_F = self.rodriguez(auto_joint_F[0])

        joint_U, joint_S, joint_V = torch.svd(auto_joint_F)  # (Joints, 3, 3), (Joints, 3), (Joints, 3, 3)

        return {
            "Rs": auto_joint_F,
            "pose_U":joint_U,
            "pose_S":joint_S,
            "pose_V":joint_V,
        }

class BodyPoseRefiner(nn.Module):
    def __init__(self,
                 total_bones=24,
                 embedding_size=69,
                 mlp_width=256,
                 mlp_depth=4,
                 **_):
        super(BodyPoseRefiner, self).__init__()
        
        block_mlps = [nn.Linear(embedding_size, mlp_width), nn.ReLU()]
        
        for _ in range(0, mlp_depth-1):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        self.total_bones = total_bones - 1
        block_mlps += [nn.Linear(mlp_width, 3 * self.total_bones)]

        self.block_mlps = nn.Sequential(*block_mlps)
        initseq(self.block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope the rotation matrix can be identity 
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

        self.rodriguez = RodriguesModule()
 

    def forward_origin(self, pose_input):
        rvec = self.block_mlps(pose_input).view(-1, 3)
        Rs = self.rodriguez(rvec)

        return {
            "Rs": Rs
        }

 
    def forward(self, pose_input,residual=None):
        # if residual!=None:
        #     return self.forward_residual(pose_input,residual)
        return self.forward_origin(pose_input)

###############################################################################
## Init Functions
###############################################################################

def xaviermultiplier(m, gain):
    """ 
        Args:
            m (torch.nn.Module)
            gain (float)

        Returns:
            std (float): adjusted standard deviation
    """ 
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] \
                // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] \
                // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std

def xavier_uniform_(m, gain):
    """ Set module weight values with a uniform distribution.

        Args:
            m (torch.nn.Module)
            gain (float)
    """ 
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-(std * math.sqrt(3.0)), std * math.sqrt(3.0))

def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    """ Initialized module weights.

        Args:
            m (torch.nn.Module)
            gain (float)
            weightinitfunc (function)
    """ 
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]

def initseq(s):
    """ Initialized weights of all modules in a module sequence.

        Args:
            s (torch.nn.Sequential)
    """ 
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])

class RodriguesModule(nn.Module):
    def forward(self, rvec):
        r''' Apply Rodriguez formula on a batch of rotation vectors.

            Args:
                rvec: Tensor (B, 3)
            
            Returns
                rmtx: Tensor (B, 3, 3)
        '''
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh), 
        dim=1).view(-1, 3, 3)
    