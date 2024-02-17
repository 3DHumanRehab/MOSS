#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.exp_name = ""
        self.smpl_type = "smplx"
        self.actor_gender = "neutral"
        self.motion_offset_flag = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = True
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.auto_regression = 0.00025
        self.cross_attention_lbs = 0.0001
        
        self.pose_refine_lr = 0.00025
        self.lbs_offset_lr = 0.0001
        
        #15.0
        # self.feature_lr= 0.0038212822691792176
        # self.opacity_lr= 0.015362017645255875
        # self.scaling_lr= 0.020453828953933734
        # self.rotation_lr= 0.006612742651448469
        # self.auto_regression= 0.0015225789466584093
        # self.cross_attention_lbs= 0.001381957467559466
        
        
        # 14.6
        # self.feature_lr= 0.0032537659603376658
        # self.opacity_lr= 0.008552910292707034
        # self.scaling_lr= 0.008537885113723305
        # self.rotation_lr= 0.003018524659263358
        # self.auto_regression= 0.0001703332628273477
        # self.cross_attention_lbs= 0.0012131787988640796
        
        # 14.734868
        # self.feature_lr= 0.002843694972004891
        # self.opacity_lr= 0.03982697745412786
        # self.scaling_lr= 0.005954196884938798
        # self.rotation_lr= 0.0038634481309237187
        # self.auto_regression= 0.002125081736151533
        # self.cross_attention_lbs= 0.0012654279445404807
        
        # 32.2
        # self.feature_lr= 0.002125050130441241
        # self.opacity_lr= 0.007280346815960768
        # self.scaling_lr= 0.0018549932329049916
        # self.rotation_lr= 0.007827731713126531
        # self.auto_regression= 0.0000802258576846021
        # self.cross_attention_lbs= 0.0011792443165401848
        
        # self.pose_refine_lr = 0.00005
        # self.lbs_offset_lr = 0.00005
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 4000
        self.densify_from_iter = 400 #500
        self.densify_until_iter = 2000 #15_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

