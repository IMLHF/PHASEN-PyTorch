import torch
import numpy as np
import torch.nn as nn

from .dual_transf import Dual_Transformer


class SPConvTranspose2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, r=1):
    # upconvolution only along second dimension of image
    # Upsampling using sub pixel layers
    super(SPConvTranspose2d, self).__init__()
    self.out_channels = out_channels
    self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
    self.r = r

  def forward(self, x):
    out = self.conv(x)
    batch_size, nchannels, H, W = out.shape
    out = out.view((batch_size, self.r, nchannels // self.r, H, W))
    out = out.permute(0, 2, 3, 4, 1)
    out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
    return out


class DenseBlock(nn.Module):
  def __init__(self, input_size, depth=5, in_channels=64):
    super(DenseBlock, self).__init__()
    self.depth = depth
    self.in_channels = in_channels
    self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
    self.twidth = 2
    self.kernel_size = (self.twidth, 3)
    for i in range(self.depth):
      dil = 2 ** i
      pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
      setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
      setattr(self, 'conv{}'.format(i + 1),
              nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                        dilation=(dil, 1)))
      setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
      setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

  def forward(self, x):
    skip = x
    for i in range(self.depth):
      out = getattr(self, 'pad{}'.format(i + 1))(skip)
      out = getattr(self, 'conv{}'.format(i + 1))(out)
      out = getattr(self, 'norm{}'.format(i + 1))(out)
      out = getattr(self, 'prelu{}'.format(i + 1))(out)
      skip = torch.cat([out, skip], dim=1)
    return out



class Net(nn.Module):
  def __init__(self, frequency_dim, width):
    super(Net, self).__init__()
    self.frequency_dim = frequency_dim
    # self.device = device
    self.in_channels = 2     #2  for stft
    self.out_channels = 2    #2
    self.kernel_size = (2, 3)
    # self.elu = nn.SELU(inplace=True)
    self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
    self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
    self.width = width

    self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 256]
    self.inp_norm = nn.LayerNorm(self.frequency_dim)   #256 for stft
    self.inp_prelu = nn.PReLU(self.width)

    self.enc_dense1 = DenseBlock(self.frequency_dim, 4, self.width) #256 for stft
    self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 1))  # [b, 64, nframes, 256], stride=(1, 1) for stft
    self.enc_norm1 = nn.LayerNorm(self.frequency_dim)
    self.enc_prelu1 = nn.PReLU(self.width)

    self.dual_transformer = Dual_Transformer(64, 64, num_layers=4)  # # [b, 64, nframes, 8]

    # gated output layer
    self.output1 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
      nn.Tanh()
    )
    self.output2 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
      nn.Sigmoid()
    )

    self.maskconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
    self.maskrelu = nn.ReLU(inplace=True)

    self.dec_dense1 = DenseBlock(self.frequency_dim, 4, self.width)
    self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=1)   # r=1 for stft
    self.dec_norm1 = nn.LayerNorm(self.frequency_dim)     #256 for stft
    self.dec_prelu1 = nn.PReLU(self.width)

    self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))

    #show_model(self)
    #show_params(self)
  def forward(self, x):
    '''
    x: [b, 2, F, T]
    '''
    # print(x.shape)
    x = x.permute(0, 1, 3, 2) # [b, 2, T, F]
    out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, num_frames, frame_size]

    out = self.enc_dense1(out)   # [b, 64, num_frames, frame_size]
    x1 = self.enc_prelu1(self.enc_norm1(self.enc_conv1(self.pad1(out))))  # [b, 64, num_frames, 256]

    #print(x1.shape)
    out = self.dual_transformer(x1)  # [b, 64, num_frames, 256]

    out = self.output1(out) * self.output2(out)  # mask [b, 64, num_frames, 256]

    out = self.maskrelu(self.maskconv(out))  # mask
    out = x1 * out
    out = self.dec_dense1(out)
    out = self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad1(out))))
    out = self.out_conv(out)
    out = out.permute(0, 1, 3, 2) # [b, 2, F, T]
    return out


if __name__ == "__main__":
  x = torch.ones((1, 2, 257, 30)).cuda()
  model = Net(257, 64).cuda()
  out = model(x)
  print(out.shape)


  def numParams(net):
      num = 0
      for param in net.parameters():
          if param.requires_grad:
              num += int(np.prod(param.size()))
      return num


  print(numParams(model))

