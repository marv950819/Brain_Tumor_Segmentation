import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange
from einops.layers.torch import Rearrange



def initial_layer(in_dim, out_dim_pre, out_dim):
    return nn.Sequential(nn.Conv3d(in_dim, out_dim_pre, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm3d(out_dim_pre), nn.ReLU(inplace=True),
                         nn.Conv3d(out_dim_pre, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm3d(out_dim), nn.ReLU(inplace=True))


def conv_block_layer_en(in_dim, out_dim):
    return nn.Sequential(nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm3d(in_dim), nn.ReLU(inplace=True),
                         nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm3d(out_dim), nn.ReLU(inplace=True))


def max_pool_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_trans_block_3d(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.ReLU(inplace=True))


class Unet3D(nn.Module):
    def __init__(self, in_dim=4, out_dim=4, num_filters=64):
        super(Unet3D, self).__init__()
        self.in_dim = in_dim
        self.num_filters = num_filters
        self.out_dim = out_dim

        self.conv1 = initial_layer(self.in_dim, 32, self.num_filters)
        self.pool1 = max_pool_3d()

        self.conv2 = conv_block_layer_en(self.num_filters, self.num_filters * 2)
        self.pool2 = max_pool_3d()

        self.conv3 = conv_block_layer_en(self.num_filters * 2, self.num_filters * 4)
        self.pool3 = max_pool_3d()

        self.bridge = conv_block_layer_en(self.num_filters * 4, self.num_filters * 8)

        self.upconv2 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8)  # 512
        self.dconv3 = conv_block_layer_en(self.num_filters * 12, self.num_filters * 4)  # 512 + 256 | 256

        self.upconv3 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4)  # 256
        self.dconv2 = conv_block_layer_en(self.num_filters * 6, self.num_filters * 2)  # 256 + 128 | 128

        self.upconv4 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2)  # 128
        self.dconv1 = conv_block_layer_en(self.num_filters * 3, self.num_filters * 1)  # 128 + 64 | 64

        self.final_conv = nn.Sequential(nn.Conv3d(self.num_filters, self.out_dim, kernel_size=3, padding=1))

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        bridge = self.bridge(pool3)

        trans_2 = self.upconv2(bridge)
        concat_2 = torch.cat([trans_2, conv3], dim=1)
        dconv3 = self.dconv3(concat_2)

        trans_3 = self.upconv3(dconv3)
        concat_3 = torch.cat([trans_3, conv2], dim=1)
        dconv2 = self.dconv2(concat_3)

        trans_4 = self.upconv4(dconv2)
        concat_2 = torch.cat([trans_4, conv1], dim=1)
        dconv1 = self.dconv1(concat_2)

        x = self.final_conv(dconv1)
        x = F.softmax(x, dim=1)
        return x



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)



class SimpleViT(nn.Module):
    def __init__(self, image_size, image_patch_size, slice_depth, slice_depth_patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert slice_depth % slice_depth_patch_size == 0, 'Frames must be divisible by the frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (slice_depth // slice_depth_patch_size)
        patch_dim = channels * patch_height * patch_width * slice_depth_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = slice_depth_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        #self.to_latent = nn.Identity()
        #self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, video):
        *_, h, w, dtype = *video.shape, video.dtype

        x = self.to_patch_embedding(video)
        pe = posemb_sincos_3d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        #x = x.mean(dim = 1)

        #x = self.to_latent(x)
        #return self.linear_head(x)
        return x



class ProposedYnet(nn.Module):
    def __init__(self, image_size, slice_depth, image_patch_size, slice_depth_patch_size, dim, depth, heads, mlp_dim, channels, dim_head, num_classes):
        super().__init__()
        self.image_size = image_size
        self.slice_depth = slice_depth
        self.image_patch_size = image_patch_size
        self.slice_depth_patch_size = slice_depth_patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.dim_head = dim_head
        self.num_classes = num_classes

        self.vit3d = SimpleViT(image_size=image_size, image_patch_size=image_patch_size, slice_depth=slice_depth, slice_depth_patch_size=slice_depth_patch_size,
                               dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels, dim_head=dim_head)
        self.downconv = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm3d(dim),
                                      nn.ReLU(inplace=True))
        self.upconv = nn.Sequential(nn.ConvTranspose3d(dim, int(dim/2), kernel_size=4, stride=2, padding=1, output_padding=0),
                                    nn.BatchNorm3d(int(dim/2)),
                                    nn.ReLU(inplace=True))
        self.lastconv = nn.Sequential(nn.ConvTranspose3d(dim, 2, kernel_size=8, stride=4, padding=2, output_padding=0))
        self.onedownconv = nn.Sequential(nn.Conv3d(dim, int(dim/2), kernel_size=1, stride=1, padding=0),
                                         nn.BatchNorm3d(int(dim/2)),
                                         nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.mlpclassification = nn.Linear(dim, num_classes)
        self.mlpregression = nn.Linear(dim, 1)

    def forward(self, x):
        # t1 = self.vit3d(x)
        # t1d = rearrange(t1, 'b (d h w) k -> b k d h w', d=int(self.slice_depth/self.slice_depth_patch_size),
        #                 h=int(self.image_size/self.image_patch_size))
        # t1dc = self.downconv(t1d)
        # t1dc = rearrange(t1dc, 'b k d h w -> b (d h w) k')
        #
        # t2 = self.vit3d.transformer(t1dc)
        # t2d = rearrange(t2, 'b (d h w) k -> b k d h w', d=int(self.slice_depth / (2 * self.slice_depth_patch_size)),
        #                 h=int(self.image_size / (2 * self.image_patch_size)))
        # t2dc = self.downconv(t2d)
        # t2dc = rearrange(t2dc, 'b k d h w -> b (d h w) k')
        #
        # t3 = self.vit3d.transformer(t2dc)
        # t3d = rearrange(t3, 'b (d h w) k -> b k d h w', d=int(self.slice_depth / (4 * self.slice_depth_patch_size)),
        #                 h=int(self.image_size / (4 * self.image_patch_size)))
        # t3dc = self.downconv(t3d)
        # t3dc = rearrange(t3dc, 'b k d h w -> b (d h w) k')
        #
        # t4 = self.vit3d.transformer(t3dc)
        # t4u = rearrange(t4, 'b (d h w) k -> b k d h w', d=int(self.slice_depth / (8 * self.slice_depth_patch_size)),
        #                 h=int(self.image_size / (8 * self.image_patch_size)))
        # t4uc = self.upconv(t4u)
        # t4ucat = torch.cat((self.onedownconv(t3d), t4uc), dim=1)
        # t4ucat = rearrange(t4ucat, 'b k d h w -> b (d h w) k')
        #
        # t5 = self.vit3d.transformer(t4ucat)
        # t5u = rearrange(t5, 'b (d h w) k -> b k d h w', d=int(self.slice_depth / (4 * self.slice_depth_patch_size)),
        #                 h=int(self.image_size / (4 * self.image_patch_size)))
        # t5uc = self.upconv(t5u)
        # t5ucat = torch.cat((self.onedownconv(t2d), t5uc), dim=1)
        # t5ucat = rearrange(t5ucat, 'b k d h w -> b (d h w) k')
        #
        # t6 = self.vit3d.transformer(t5ucat)
        # t6u = rearrange(t6, 'b (d h w) k -> b k d h w', d=int(self.slice_depth / (2 * self.slice_depth_patch_size)),
        #                 h=int(self.image_size / (2 * self.image_patch_size)))
        # t6uc = self.upconv(t6u)
        # t6ucat = torch.cat((self.onedownconv(t1d), t6uc), dim=1)
        # t6ucat = rearrange(t6ucat, 'b k d h w -> b (d h w) k')
        #
        # t7 = self.vit3d.transformer(t6ucat)
        # t7u = rearrange(t7, 'b (d h w) k -> b k d h w', d=int(self.slice_depth / (self.slice_depth_patch_size)),
        #                 h=int(self.image_size / (self.image_patch_size)))
        # out = self.lastconv(t7u)
        # x = F.softmax(out, dim=1)

        t1 = self.vit3d(x)
        t1d = rearrange(t1, 'b (d h w) k -> b k d h w', d=int(self.slice_depth / self.slice_depth_patch_size),
                        h=int(self.image_size / self.image_patch_size))
        t1dc = self.downconv(t1d)
        t1dc = rearrange(t1dc, 'b k d h w -> b (d h w) k')

        t2 = self.vit3d.transformer(t1dc)
        t2u = rearrange(t2, 'b (d h w) k -> b k d h w', d=int(self.slice_depth / (2 * self.slice_depth_patch_size)),
                        h=int(self.image_size / (2 * self.image_patch_size)))
        t2uc = self.upconv(t2u)
        t2ucat = torch.cat((self.onedownconv(t1d), t2uc), dim=1)
        t2ucat = rearrange(t2ucat, 'b k d h w -> b (d h w) k')

        t3 = self.vit3d.transformer(t2ucat)
        t3u = rearrange(t3, 'b (d h w) k -> b k d h w', d=int(self.slice_depth / (self.slice_depth_patch_size)),
                        h=int(self.image_size / (self.image_patch_size)))
        out = self.lastconv(t3u)
        # out = F.softmax(out, dim=1)
        return out






def Dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num / den


class softmax_dice(nn.Module):
    def __init__(self):
        super(softmax_dice, self).__init__()

    def forward(self, output, target):
        output = output.to(self.device)
        target = target.to(self.device)
        output = F.softmax(output, dim=1)
        loss0 = Dice(output[:, 0, ...], (target == 0).float())
        loss1 = Dice(output[:, 1, ...], (target == 1).float())
        # loss2 = Dice(output[:, 2, ...], (target == 2).float())
        # loss3 = Dice(output[:, 3, ...], (target == 3).float())

        return loss0 + loss1
               # + loss2 + loss3, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data