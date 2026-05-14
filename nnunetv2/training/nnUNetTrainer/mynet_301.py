import torch.nn as nn

#from nnunetv2.training.nnUNetTrainer.acdc.unetr_pp_acdc import UNETR_PP
from nnunetv2.net.CSUnet.conv_swin_transformer_unet_skip_expand_decoder_sys import ConvSwinTransformerSys


# x =nnUNetTrainer(plans={"1":1}, configuration="aaa", dataset_json={"2":2})
# plans = x.plans_manager
# config = x.configuration_manager
# data = x.dataset_json

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
        self.model = ConvSwinTransformerSys(img_size=512,
                                            patch_size=4,
                                            in_chans=3,
                                            num_classes=3,
                                            embed_dim=96,
                                            depths=[2, 2, 6, 2],
                                            num_heads=[6, 12, 24, 48],
                                            window_size=7,
                                            mlp_ratio=4.,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0.0,
                                            drop_path_rate=0.1,
                                            ape=False,
                                            patch_norm=True,
                                            use_checkpoint=False)


    def forward(self, x):
        output = self.model(x)
        if self.deep_supervision:
            return [output ]
        else:
            return output
