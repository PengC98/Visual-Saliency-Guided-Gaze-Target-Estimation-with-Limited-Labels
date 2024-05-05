from layer import *

import torch
import torch.nn as nn
from torch.nn import init

import resnet_scene

import torchvision


class Decoder(nn.Module):
	def __init__(self, outlayer=1):
		super(Decoder, self).__init__()

		self.c5_conv = nn.Conv2d(512, 256, (1, 1))
		self.c4_conv = nn.Conv2d(256, 256, (1, 1))
		self.c3_conv = nn.Conv2d(128, 256, (1, 1))
		self.c2_conv = nn.Conv2d(64, 256, (1, 1))
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
		self.p2_conv = nn.Conv2d(256, 256, (3, 3), padding=1)

		# predict heatmap
		self.sigmoid = nn.Sigmoid()
		self.conv7 = nn.Conv2d(256, 1, (3, 3), padding=1)



		self.relu = nn.ReLU(inplace=True)


	def forward(self, x,l3,l2,l1):
		p5 = self.c5_conv(x)
		p4 = self.upsample(p5) + self.c4_conv(l3)
		p3 = self.upsample(p4) + self.c3_conv(l2)
		p2 = self.upsample(p3) + self.c2_conv(l1)
		heat = self.conv7(p2)

		return heat

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()

		self.conv1 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(1024)
		self.conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(512)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		return out


class Sence_B(nn.Module):
	def __init__(self):
		super(Sence_B, self).__init__()
		self.raw_backbone = resnet_scene.resnet34(pretrained=True)
		self.raw_backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

		self.face_backbone = resnet_scene.resnet34(pretrained=True)


		self.enc = Encoder()
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.avgpool = nn.AvgPool2d(7)
		self.attn = nn.Linear(512+784, 1 * 7 * 7)
		self.patch_decoder = Decoder()

	def forward(self, raw_image,depth,head,face):

		rh_i = torch.concat((raw_image, depth), dim=1)


		scene_feat,l1,l2,l3 = self.raw_backbone(rh_i,True)

		face_feat = self.face_backbone(face)

		head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
		gaze_reduced = self.avgpool(face_feat).view(-1, 512)

		attn_weights = self.attn(torch.cat((head_reduced, gaze_reduced), 1))
		attn_weights = attn_weights.view(-1, 1, 49)
		attn_weights = F.softmax(attn_weights, dim=2)
		attn_weights = attn_weights.view(-1, 1, 7, 7)

		# Scene feature map * attention
		attn_applied_scene_feat = torch.mul(attn_weights, scene_feat)
		fs = torch.cat((attn_applied_scene_feat,face_feat),1)
		fs = self.enc(fs)

		patch_pred = self.patch_decoder(scene_feat,l3,l2,l1)
		patch_pred = self.sigmoid(patch_pred)


		heat = self.patch_decoder(fs,l3,l2,l1)
		heat = self.sigmoid(heat)
		return heat,patch_pred



def init_weights(net, init_type='normal', init_gain=0.02):
	"""Initialize network weights.
	Parameters:
		net (network)   -- network to be initialized
		init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
		init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
	We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
	work better for some applications. Feel free to try yourself.
	"""
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
	"""Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
	Parameters:
		net (network)      -- the network to be initialized
		init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
		gain (float)       -- scaling factor for normal, xavier and orthogonal.
		gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
	Return an initialized network.
	"""
	#if gpu_ids:
	#	assert(torch.cuda.is_available())
	#	net.to(gpu_ids[0])
	#	net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	init_weights(net, init_type, init_gain=init_gain)
	return net

def build_grid(source_size,target_size):
	k = float(target_size)/float(source_size)
	direct = torch.linspace(-k,k,target_size).unsqueeze(0).repeat(target_size,1).unsqueeze(-1)
	full = torch.cat([direct,direct.transpose(1,0)],dim=2).unsqueeze(0)
	return full.cuda()

def random_crop_grid(x,grid):
	delta = x.size(2)-grid.size(1)
	grid = grid.repeat(x.size(0),1,1,1).cuda()
	#Add random shifts by x
	grid[:,:,:,0] = grid[:,:,:,0]+ torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(2)
	#Add random shifts by y
	grid[:,:,:,1] = grid[:,:,:,1]+ torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(2)
	return grid


