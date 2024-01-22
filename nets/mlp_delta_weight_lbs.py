import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention_lbs(nn.Module):
    def __init__(self, feature_dim=24,rot_dim = 9, num_heads=3):
        super(CrossAttention_lbs, self).__init__()
        
        self.actvn = nn.ReLU()
        input_ch = 63
        D = 4
        W = 128
        self.skips = [2]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, feature_dim, 1)
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(rot_dim, rot_dim)
        self.value = nn.Linear(rot_dim, rot_dim)
        self.out_layer = nn.Linear(feature_dim,feature_dim)

    def forward(self, query, key):
        features = xyz_embedder(query)
        features = features.permute(0, 2, 1)
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        query = self.bw_fc(net).permute(0,2,1)
        
        key = torch.cat([torch.zeros(1,1,3,3).cuda(),key],dim=1).reshape(1,24,9)
 
        value = key
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        attention_scores = torch.matmul(Q, K) / (self.feature_dim ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention,V.transpose(-2, -1))
        
        # output = output*query
        # output = self.out_layer(output)

        return output

# class CrossAttention_lbs_without_bias(nn.Module):
#     def __init__(self, feature_dim=24,mesh_dim = 3,rot_dim = 3, num_heads=3,):
#         super(CrossAttention_lbs_without_bias, self).__init__()
        
#         self.feature_dim = feature_dim
#         self.query = nn.Linear(mesh_dim, feature_dim)
#         self.key = nn.Linear(rot_dim, feature_dim)
#         self.value = nn.Linear(rot_dim, feature_dim)
#         self.out_layer = nn.Linear(feature_dim,feature_dim)
#         init_val =1e-5
#         self.out_layer.weight.data.uniform_(-init_val, init_val)
#         self.out_layer.bias.data.zero_()

#     def forward(self, query, key):

#         value = key
#         Q = self.query(query) 
#         K = self.key(key)     
#         V = self.value(value) 

#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)
#         attention = F.softmax(attention_scores, dim=-1)

#         output = torch.matmul(attention, V)
#         output = self.out_layer(output)

#         return output


class LBSOffsetDecoder(nn.Module):
    def __init__(self, total_bones=24):
        super(LBSOffsetDecoder, self).__init__()

        self.total_bones = total_bones
        self.actvn = nn.ReLU()

        input_ch = 63
        D = 4
        W = 128
        self.skips = [2]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, total_bones, 1)

    def forward(self, pts):
        features = xyz_embedder(pts)
        features = features.permute(0, 2, 1)
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)
        return bw

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

xyz_embedder, xyz_dim = get_embedder(10)

