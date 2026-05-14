import torch.nn as nn
import torch
from monai.networks.blocks import UnetResBlock

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        # ¼ì²édropout_rateÊÇ·ñÔÚ0ºÍ1Ö®¼ä
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        # ¼ì²éhidden_sizeÊÇ·ñ¿ÉÒÔ±»num_headsÕû³ý
        if hidden_size % 1 != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", 1)
            raise ValueError("hidden_size should be divisible by num_heads.")

        # ¶¨Òå²ã¹éÒ»»¯
        self.norm = nn.LayerNorm(hidden_size)
        # ¶¨Òå¿ÉÑ§Ï°µÄ²ÎÊýgamma
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        # ¶¨ÒåEPAÄ£¿é
        self.epa_block = EPA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size,
                             channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        # ¶¨ÒåÁ½¸öUnetResBlockÄ£¿é
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv52 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        # ¶¨ÒåÒ»¸öÐòÁÐÄ£¿é£¬°üº¬Ò»¸ö3D Dropout²ãºÍÒ»¸ö3D¾í»ý²ã
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        # Èç¹ûÊ¹ÓÃÎ»ÖÃÇ¶Èë£¬Ôò¶¨ÒåÎ»ÖÃÇ¶Èë
        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        # »ñÈ¡ÊäÈëÊý¾ÝxµÄÐÎ×´
        B, C, H, W, D = x.shape

        # ½«ÊäÈëÊý¾ÝxµÄÐÎ×´½øÐÐ±ä»»£¬È»ºó¸Ä±äÎ¬¶ÈµÄË³Ðò
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        # Èç¹ûÊ¹ÓÃÎ»ÖÃÇ¶Èë£¬Ôò½«Î»ÖÃÇ¶Èë¼Óµ½ÊäÈëÊý¾ÝxÉÏ
        if self.pos_embed is not None:
            x = x + self.pos_embed
        # ¶ÔÊäÈëÊý¾Ýx½øÐÐ²ã¹éÒ»»¯£¬È»ºóÍ¨¹ýEPAÄ£¿é½øÐÐ´¦Àí
        epa1 = self.epa_block(self.norm(x))
        # ¼ÆËã×¢ÒâÁ¦µÄÊä³ö
        attn = x + self.gamma * epa1

        # ½«×¢ÒâÁ¦µÄÊä³ö½øÐÐÐÎ×´±ä»»ºÍÎ¬¶ÈË³ÐòµÄ¸Ä±ä
        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        # ½«×¢ÒâÁ¦µÄÊä³öÍ¨¹ýÁ½¸öUnetResBlockÄ£¿é½øÐÐ´¦Àí
        attn = self.conv51(attn_skip)
        attn = self.conv52(attn)
        # ½«×¢ÒâÁ¦µÄÊä³öºÍÔ­Ê¼µÄ×¢ÒâÁ¦Êä³ö½øÐÐÏà¼Ó
        x = attn_skip + self.conv8(attn)
        # ·µ»Ø×îÖÕµÄÊä³ö
        return x

class EPA(nn.Module):
    """
    EPAÄ£¿é£¬ÓÃÓÚÊµÏÖEfficient Paired Attention
    """
    def __init__(self, input_size, hidden_size, proj_size, qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1):
        """
        ³õÊ¼»¯º¯Êý
        Args:
            input_size: ÊäÈëµÄ´óÐ¡
            hidden_size: Òþ²Ø²ãµÄ´óÐ¡
            proj_size: Í¶Ó°µÄ´óÐ¡
            num_heads: ×¢ÒâÁ¦Í·µÄÊýÁ¿
            qkv_bias: ÊÇ·ñ¶ÔqkvÌí¼ÓÆ«ÖÃ
            channel_attn_drop: Í¨µÀ×¢ÒâÁ¦µÄdropoutÂÊ
            spatial_attn_drop: ¿Õ¼ä×¢ÒâÁ¦µÄdropoutÂÊ
        """
        super().__init__()

        # ¶¨Òå¿ÉÑ§Ï°µÄ²ÎÊýtemperature
        self.num_heads = 1
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(1, 1, 1))

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.Wq = nn.Parameter(torch.randn(input_size, input_size))
        self.Wo = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wa = nn.Parameter(torch.randn(hidden_size, hidden_size))

        # ¶¨ÒåDropout²ã£¬ÓÃÓÚÔÚÑµÁ·¹ý³ÌÖÐËæ»ú¶ªÆúÒ»²¿·ÖÉñ¾­Ôª£¬ÒÔ·ÀÖ¹¹ýÄâºÏ
        self.attn_drop = nn.Dropout(channel_attn_drop)

        # ¶¨ÒåDropout²ã£¬ÓÃÓÚÔÚÑµÁ·¹ý³ÌÖÐËæ»ú¶ªÆúÒ»²¿·ÖÉñ¾­Ôª£¬ÒÔ·ÀÖ¹¹ýÄâºÏ
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)


    def forward(self, x):
        """
        Ç°Ïò´«²¥º¯Êý
        Args:
            x: ÊäÈëÊý¾Ý
        """
        # »ñÈ¡ÊäÈëÊý¾ÝxµÄÐÎ×´ N600 C 384

        B, N, C = x.shape

        # Í¨¹ýÏßÐÔ²ã¼ÆËãquery¡¢keyºÍvalue£¬È»ºó½«½á¹ûµÄÐÎ×´½øÐÐ±ä»»
        # qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkv = self.qkv(x).reshape(B, N, 3, C)
        # ¸Ä±äqkvvµÄÎ¬¶ÈµÄË³Ðò
        # (4,b,n,c)
        qkv = qkv.permute(2, 0, 1, 3)
        # (b,n,c)
        # ½«qkvv·Ö½âÎªËÄ¸ö²¿·Ö£º¹²ÏíµÄquery¡¢¹²ÏíµÄkey¡¢Í¨µÀ×¢ÒâÁ¦µÄvalueºÍ¿Õ¼ä×¢ÒâÁ¦µÄvalue
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ¸Ä±ä¹²ÏíµÄqueryµÄÎ¬¶ÈµÄË³Ðò #(b,c,n)
        q = q.transpose(-2, -1)

        q_projected = q @ self.Wq

        # ¸Ä±ä¹²ÏíµÄkeyµÄÎ¬¶ÈµÄË³Ðò (b,c,n)
        k = k.transpose(-2, -1)
        # ¶Ô¹²ÏíµÄkey½øÐÐ¹éÒ»»¯´¦Àí
        k = torch.nn.functional.normalize(k, dim=-1)

        attn_A = (q_projected @ k.transpose(-2, -1)) * self.temperature
        attn_A = attn_A.softmax(dim=-1)
        attn_A = self.attn_drop(attn_A)

        # ¼ÆËãÍ¨µÀ×¢ÒâÁ¦µÄ×îÖÕÊä³ö
        x_A = ((attn_A @ self.Wa @ v.transpose(-2, -1)).permute(0, 2, 1).reshape(B, N, C)) @ self.Wo
        return x_A
    #
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'temperature', 'temperature2'}


def custom_downsampling(input):
    B, C, H, W, D = input.shape

    downsampling_layers = nn.Sequential()
    # condition = (H > 10).item() and (W > 6).item() and (D > 10).item()
    # condition = torch.logical_and(H > 10, torch.logical_and(W > 6, D > 10))
    while (H > 10) and (W > 6) and (D > 10):
        C *= 2
        downsampling_layers.add_module(f'conv{C}',
                                       nn.Conv3d(in_channels=C // 2, out_channels=C, kernel_size=3, stride=2,
                                                 padding=1))
        downsampling_layers.add_module(f'maxpool{C}', nn.MaxPool3d(kernel_size=1))
        H //= 2
        W //= 2
        D //= 2

    input = input.to(torch.float32)
    output_tensor = downsampling_layers(input)

    return output_tensor


def custom_upsampling(input_tensor):
    B, C, H, W, D = input_tensor.shape
    upsampling_layers = nn.Sequential()
    while C > 24:
        # condition = (C > 24).item()
        # while condition:
        C //= 2
        H *= 2
        W *= 2
        D *= 2
        # upsampling_layers.add_module(f'conv{C}',
        #                              nn.ConvTranspose3d(in_channels=C * 2, out_channels=C, kernel_size=3, stride=2,
        #                                                 padding=1, output_padding=1).to(device))
        upsampling_layers.add_module(f'conv1{C}',nn.Conv3d(C * 2, C , kernel_size=1))
        upsampling_layers.add_module(f'upsampling{C}',nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
    input_tensor = input_tensor.to(torch.float32)
    restored_tensor = upsampling_layers(input_tensor)
    return restored_tensor


# [1,24,5,6,6]
# input_size = 150
# hidden_size = 24
# proj_size = 64
# num_heads = 4
# dropout_rate = 0.1

#[1,24,10,6,10]
# input_size = 600
# hidden_size = 24
# proj_size = 64
# num_heads = 4
# dropout_rate = 0.1

# ([1, 24, 20, 12, 20])
# input_size = 600
# hidden_size = 48
# proj_size = 64
# num_heads = 4
# dropout_rate = 0.1

# ([1, 24, 40, 24, 40])
# input_size = 600
# hidden_size = 96
# proj_size = 64
# num_heads = 4
# dropout_rate = 0.1

# ([1, 24, 80, 48, 80])
# input_size = 307200
# hidden_size = 24
# proj_size = 64
# num_heads = 4
# dropout_rate = 0.1

# ([1, 24, 160, 96, 160])
# input_size = 600
# hidden_size = 384
# proj_size = 64
# num_heads = 4
# dropout_rate = 0.1
#
# transformer_block = TransformerBlock(input_size, hidden_size, proj_size, num_heads, dropout_rate)
#
#
# batch_size = 1
# channels = 384
# height =10
# width = 6
# depth = 10
# input_tensor = torch.randn(batch_size, channels, height, width, depth)
#
#
# output = transformer_block(input_tensor)
#
#
# print("Output Shape:", output.shape)

sizes = [
    (1, 24, 160, 96, 160),
    (1, 24, 80, 48, 80),
    (1, 24, 40, 24, 40),
    (1, 24, 20, 12, 20),
    (1, 24, 10, 6, 10),
    (1, 24, 5, 6, 5)
]
# sizes = [
#     (1, 24, 5, 6, 5),
#     (1, 24, 10, 6, 10),
#     (1, 48, 10, 6, 10),
#     (1, 96, 10, 6, 10),
#     (1, 192, 10, 6, 10),
#     (1, 384, 10, 6, 10)
# ]

tensor_list1 = [torch.randn(*size) for size in sizes]
# tensor_list2 = [torch.randn(*size) for size in sizes]
input_size_list11 = [150, 600, 600, 600, 600, 600]
input_size_list = input_size_list11[::-1]
input_size_list21 = [24, 24, 48, 96, 192, 384]
input_size_list2 = input_size_list21[::-1]
input_size_list31 = [384, 192, 96, 48, 24, 24]
# final_tensor_list = []
final_tensor_list1 = []
# final_tensor_list2 = []
#
# for i, tensor in enumerate(tensor_list1):
#     tensor = custom_downsampling(tensor)
#     custom_input_size = input_size_list[i]
#     cush = input_size_list2[i]
#     pro = input_size_list31[i]
#     transformer = TransformerBlock(input_size=custom_input_size, hidden_size=cush, proj_size=pro, dropout_rate=0.2)
#     processed_tensor = transformer(tensor)
#     processed_tensor = custom_upsampling(processed_tensor)
#     final_tensor_list1.append(processed_tensor)
#
# for i, tensor in enumerate(tensor_list2):
#     tensor = custom_downsampling(tensor)
#     custom_input_size = input_size_list[i]
#     cush = input_size_list2[i]
#     transformer = TransformerBlock(input_size=custom_input_size, hidden_size=cush, proj_size=64, num_heads=4, dropout_rate=0.2)
#     processed_tensor = transformer(tensor)
#     processed_tensor = custom_upsampling(processed_tensor)
#     final_tensor_list2.append(processed_tensor)
#
# final_tensor_list = [torch.add(z1,z2) for z1,z2 in zip(final_tensor_list1, final_tensor_list2)]
#
# for i in final_tensor_list1:
#     print(i.shape)

