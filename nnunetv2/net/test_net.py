import torch.nn as nn

from nnunetv2.net.MiFDeU.network_architecture.MiFDeU import MiFDeU
from nnunetv2.net.MiFDeU.network_architecture.initialization import InitWeights_He
#from nnunetv2.training.nnUNetTrainer.SwinMM.MiFDeU.network_architecture import NexToU
#from nnunetv2.training.nnUNetTrainer.acdc.unetr_pp_acdc import UNETR_PP
#from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# x =nnUNetTrainer(plans={"1":1}, configuration="aaa", dataset_json={"2":2})
# plans = x.plans_manager
# config = x.configuration_manager
# data = x.dataset_json
class OptInit:
    def __init__(self, drop_path_rate=0., pool_op_kernel_sizes_len=4):
        self.k = [4, 8, 16] + [32] * (pool_op_kernel_sizes_len - 3)
        self.conv = 'mr'
        self.act = 'leakyrelu'
        self.norm = 'instance'
        self.bias = True
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = True
        self.drop_path = drop_path_rate
        # number of basic blocks in the backbone
        self.blocks = [1] * (pool_op_kernel_sizes_len - 2) + [1, 1]
        # number of reduce ratios in the backbone
        self.reduce_ratios = [4, 2, 1, 1] + [1] * (pool_op_kernel_sizes_len - 4)


class test_net(nn.Module):

    def __init__(self,):
        super(test_net, self).__init__()
        self.deep_supervision = True

        #self.num_input_channels = determine_num_input_channels(plans_manager=plans,configuration_or_config_manager=config,dataset_json=data)
        # 使用你自己的网络
        #self.model = UNETR_PP(in_channels=self.num_input_channels,
        #                     out_channels=self.num_input_channels,
        #                     feature_size=16,
        #                     num_heads=4,
        #                     depths=[3, 3, 3, 3],
        #                     dims=[32, 64, 128, 256],
        #                     do_ds=True)
        #self.model = ConvSwinTransformerSys(img_size=256*224,
        #                                    patch_size=4,
        #                                    in_chans=3,
        #                                    num_classes=4,
        #                                    embed_dim=96,
        #                                    depths=[2, 2, 6, 2],
        #                                    num_heads=[3, 6, 12, 24],
        #                                    window_size=7,
        #                                    mlp_ratio=4.,
        #                                   qkv_bias=True,
        #                                   qk_scale=None,
        #                                    drop_rate=0.0,
        #                                    drop_path_rate=0.1,
        #                                    ape=False,
        #                                   patch_norm=True,
        #                                   use_checkpoint=False)
        #self.model = ConvNeXt()
        # model = SwinUNETR(img_size=(128, 128, 128),
        #                   in_channels=3,
        #                   out_channels=4,
        #                   feature_size=24,
        #                   fusion_depths=(1, 1, 1, 1, 1, 1),
        #                   drop_rate=0.0,
        #                   attn_drop_rate=0.0,
        #                   dropout_path_rate=0.2,
        #                   use_checkpoint=None,
        #                   cross_attention_in_origin_view=True, )
        # model.cuda()
        #self.model = None
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        # self.embedding_dim = 24
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        opt = OptInit(pool_op_kernel_sizes_len=4)
        # x = torch.rand((1, 3, 512, 512, 250), device=cuda0)
        model =MiFDeU(input_channels=3, base_num_features=24, num_classes=4, num_pool=4, patch_size=(160, 96, 160),
                       num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=conv_op, norm_op=norm_op,
                       norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                       dropout_op_kwargs=dropout_op_kwargs,
                       nonlin=net_nonlin, nonlin_kwargs=net_nonlin_kwargs,
                       deep_supervision=True, dropout_in_localization=False, final_nonlin=lambda x: x,
                       weightInitializer=InitWeights_He(1e-2),
                       upscale_logits=False, convolutional_pooling=True, convolutional_upsampling=True, opt=opt)
        #model.cuda()
        #self.model = None

    def forward(self, x):
        output = self.model(x)
        if self.deep_supervision:
            return [output, ]
        else:
            return output
