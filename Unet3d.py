import torch
import torch.nn as nn
import numpy as np



def output_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ELU())


def conv_trans_block_3d(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ELU())


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_3d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ELU(),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ELU())


# The key:
# 1.Not to change the size but the channels through conv
# 2.Use pooling to change the size
# 3.ConvT change the size and cat with down-sample feature
# 4.Up-sample zip the channels
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        
        # Down sampling
        self.down_1 = conv_block_3d(self.in_channels, self.num_filters)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_3d(self.num_filters, self.num_filters * 2)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_3d(self.num_filters * 2, self.num_filters * 4)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_3d(self.num_filters * 4, self.num_filters * 8)
        self.pool_4 = max_pooling_3d()
        #self.down_5 = conv_block_3d(self.num_filters * 8, self.num_filters * 16)
        #self.pool_5 = max_pooling_3d()
        
        # Bridge
        #self.bridge = conv_block_3d(self.num_filters * 16, self.num_filters * 32)
        self.bridge = conv_block_3d(self.num_filters * 8, self.num_filters * 16)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16)
        self.up_1 = conv_block_3d(self.num_filters * 24, self.num_filters * 8)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8)
        self.up_2 = conv_block_3d(self.num_filters * 12, self.num_filters * 4)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4)
        self.up_3 = conv_block_3d(self.num_filters * 6, self.num_filters * 2)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2)
        self.up_4 = conv_block_3d(self.num_filters * 3, self.num_filters * 1)
        # self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2)
        # self.up_5 = conv_block_3d(self.num_filters * 3, self.num_filters * 1)
        
        # Output
        self.out = output_block(self.num_filters, out_channels)
    
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) # -> [1, 4, 240, 240, 240]
        pool_1 = self.pool_1(down_1) # -> [1, 4, 120, 120, 120]
        
        down_2 = self.down_2(pool_1) # -> [1, 8, 120, 120, 120]
        pool_2 = self.pool_2(down_2) # -> [1, 8, 60, 60, 60]
        
        down_3 = self.down_3(pool_2) # -> [1, 16, 60, 60, 60]
        pool_3 = self.pool_3(down_3) # -> [1, 16, 30, 30, 30]
        
        down_4 = self.down_4(pool_3) # -> [1, 32, 30, 30, 30]
        pool_4 = self.pool_4(down_4) # -> [1, 32, 15, 15, 15]
        #print(f'pool_4_size: {pool_4.shape}')
        
        #down_5 = self.down_5(pool_4) # -> [1, 64, 16, 16, 16]
        #pool_5 = self.pool_5(down_5) # -> [1, 64, 8, 8, 8]
        
        # Bridge
        #bridge = self.bridge(pool_5) # -> [1, 128, 8, 8, 8]
        bridge = self.bridge(pool_4) # -> [1, 64, 15, 15, 15]

        # Up sampling
        trans_1 = self.trans_1(bridge) # -> [1, 64, 30, 30, 30]
        concat_1 = torch.cat([trans_1, down_4], dim=1) # -> [1, 96, 30, 30, 30]
        up_1 = self.up_1(concat_1) # -> [1, 32, 30, 30, 30]
        
        trans_2 = self.trans_2(up_1) # -> [1, 32, 60, 60, 60]
        concat_2 = torch.cat([trans_2, down_3], dim=1) # -> [1, 48, 60, 60, 60]
        up_2 = self.up_2(concat_2) # -> [1, 16, 60, 60, 60]
        
        trans_3 = self.trans_3(up_2) # -> [1, 16, 120, 120, 120]
        concat_3 = torch.cat([trans_3, down_2], dim=1) # -> [1, 24, 120, 120, 120]
        up_3 = self.up_3(concat_3) # -> [1, 8, 120, 120, 120]
        
        trans_4 = self.trans_4(up_3) # -> [1, 8, 240, 240, 240]
        concat_4 = torch.cat([trans_4, down_1], dim=1) # -> [1, 12, 240, 240, 240]
        up_4 = self.up_4(concat_4) # -> [1, 4, 240, 240, 240]
        #print(f'up_4_size: {up_4.shape}')
        
        #trans_5 = self.trans_5(up_4) # -> [1, 8, 256, 256, 256]
        #concat_5 = torch.cat([trans_5, down_1], dim=1) # -> [1, 12, 256, 256, 256]
        #up_5 = self.up_5(concat_5) # -> [1, 4, 256, 256, 256]
        
        # Output
        out = self.out(up_4) # -> [1, 1, 240, 240, 240]
        out = torch.clamp(out, min=0) # (-inf, inf)->[0, inf)
        out = torch.sign(out) # [0, inf)-> 0 or 1
        return out






class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_n_filter=4):
        super(UNet3D, self).__init__()

        # Down sampling
        self.down_1 = self.conv_block(in_channels, base_n_filter)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down_2 = self.conv_block(base_n_filter, base_n_filter * 2)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down_3 = self.conv_block(base_n_filter * 2, base_n_filter * 4)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down_4 = self.conv_block(base_n_filter * 4, base_n_filter * 8)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bridge
        self.bridge = self.conv_block(base_n_filter * 8, base_n_filter * 16)

        # Up sampling
        self.trans_1 = self.up_conv(base_n_filter * 16, base_n_filter * 8)
        self.up_1 = self.conv_block(base_n_filter * 16, base_n_filter * 8)
        self.trans_2 = self.up_conv(base_n_filter * 8, base_n_filter * 4)
        self.up_2 = self.conv_block(base_n_filter * 8, base_n_filter * 4)
        self.trans_3 = self.up_conv(base_n_filter * 4, base_n_filter * 2)
        self.up_3 = self.conv_block(base_n_filter * 4, base_n_filter * 2)
        self.trans_4 = self.up_conv(base_n_filter * 2, base_n_filter)
        self.up_4 = self.conv_block(base_n_filter * 2, base_n_filter)

        # Final output
        self.out = nn.Conv3d(base_n_filter, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ELU()
        )
        return block

    def up_conv(self, in_channels, out_channels):
        up_conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ELU()
        )
        return up_conv

    def forward(self, x):

        # Down sampling
        d1 = self.down_1(x)
        p1 = self.pool_1(d1)
        d2 = self.down_2(p1)
        p2 = self.pool_2(d2)
        d3 = self.down_3(p2)
        p3 = self.pool_3(d3)
        d4 = self.down_4(p3)
        p4 = self.pool_4(d4)

        # Bridge
        b = self.bridge(p4)

        # Up sampling + Concatenation
        t1 = self.trans_1(b)
        c1 = torch.cat((t1, d4), dim=1)
        u1 = self.up_1(c1)

        t2 = self.trans_2(u1)
        c2 = torch.cat((t2, d3), dim=1)
        u2 = self.up_2(c2)

        t3 = self.trans_3(u2)
        c3 = torch.cat((t3, d2), dim=1)
        u3 = self.up_3(c3)

        t4 = self.trans_4(u3)
        c4 = torch.cat((t4, d1), dim=1)
        u4 = self.up_4(c4)

        # Final output
        out = self.out(u4)
        return out





class Modified3DUNet(nn.Module):
	def __init__(self, in_channels, n_classes, base_n_filter = 8):
		super(Modified3DUNet, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.base_n_filter = base_n_filter

		self.lrelu = nn.LeakyReLU()
		self.dropout3d = nn.Dropout3d(p=0.6)
		self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
		self.softmax = nn.Softmax(dim=1)

		# Level 1 context pathway
		self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
		self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
		self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

		# Level 2 context pathway
		self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
		self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter*2)

		# Level 3 context pathway
		self.conv3d_c3 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
		self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter*4)

		# Level 4 context pathway
		self.conv3d_c4 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
		self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter*8)

		# Level 5 context pathway, level 0 localization pathway
		self.conv3d_c5 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
		self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
		self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*8)

		self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
		self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

		# Level 1 localization pathway
		self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*16)
		self.conv3d_l1 = nn.Conv3d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*4)

		# Level 2 localization pathway
		self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*8)
		self.conv3d_l2 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*2)

		# Level 3 localization pathway
		self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*4)
		self.conv3d_l3 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
		self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter)

		# Level 4 localization pathway
		self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter*2)
		self.conv3d_l4 = nn.Conv3d(self.base_n_filter*2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

		self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter*8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
		self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter*4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)




	def conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def norm_lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2, mode='nearest'),
			# should be feat_in*2 or feat_in
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def forward(self, x):
		#  Level 1 context pathway
		out = self.conv3d_c1_1(x)
		residual_1 = out
		out = self.lrelu(out)
		out = self.conv3d_c1_2(out)
		out = self.dropout3d(out)
		out = self.lrelu_conv_c1(out)
		# Element Wise Summation
		out += residual_1
		context_1 = self.lrelu(out)
		out = self.inorm3d_c1(out)
		out = self.lrelu(out)

		# Level 2 context pathway
		out = self.conv3d_c2(out)
		residual_2 = out
		out = self.norm_lrelu_conv_c2(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c2(out)
		out += residual_2
		out = self.inorm3d_c2(out)
		out = self.lrelu(out)
		context_2 = out

		# Level 3 context pathway
		out = self.conv3d_c3(out)
		residual_3 = out
		out = self.norm_lrelu_conv_c3(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c3(out)
		out += residual_3
		out = self.inorm3d_c3(out)
		out = self.lrelu(out)
		context_3 = out

		# Level 4 context pathway
		out = self.conv3d_c4(out)
		residual_4 = out
		out = self.norm_lrelu_conv_c4(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c4(out)
		out += residual_4
		out = self.inorm3d_c4(out)
		out = self.lrelu(out)
		context_4 = out

		# Level 5
		out = self.conv3d_c5(out)
		residual_5 = out
		out = self.norm_lrelu_conv_c5(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c5(out)
		out += residual_5
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

		out = self.conv3d_l0(out)
		out = self.inorm3d_l0(out)
		out = self.lrelu(out)

		# Level 1 localization pathway
		out = torch.cat([out, context_4], dim=1)
		out = self.conv_norm_lrelu_l1(out)
		out = self.conv3d_l1(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

		# Level 2 localization pathway
		out = torch.cat([out, context_3], dim=1)
		out = self.conv_norm_lrelu_l2(out)
		ds2 = out
		out = self.conv3d_l2(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

		# Level 3 localization pathway
		out = torch.cat([out, context_2], dim=1)
		out = self.conv_norm_lrelu_l3(out)
		ds3 = out
		out = self.conv3d_l3(out)
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

		# Level 4 localization pathway
		out = torch.cat([out, context_1], dim=1)
		out = self.conv_norm_lrelu_l4(out)
		out_pred = self.conv3d_l4(out)

		ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
		ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
		ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
		ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
		ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

		out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
		seg_layer = out
		#out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
		#out = out.view(-1, self.n_classes)
		#out = self.softmax(out)
		return seg_layer




if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = np.ones([1,1,240,240,240])
    x = torch.from_numpy(x).type(torch.float32)
    x.to(device)
    # #   print("x size: {}".format(x.size()))
    
    model = Modified3DUNet(in_channels=1, n_classes=1, base_n_filter=4)
    model.to(device)
    model.train()
    # #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    out = model(x)
    print(out.shape)
    #print("out size: {}".format(out.shape))
    # #   print(out)
    # model(x).mean().backward()
    # #optimizer.step()
    
    # for name, parms in model.named_parameters():
    #     if parms.requires_grad:
    #         print('name:', name)
    #         print('grad_value:',torch.norm(parms.grad).item())
    # print(torch.norm(x))
    # x = torch.nn.functional.normalize(x, p=2, dim=(2,3,4))
    # print(x[0][0])
