import tensorflow as tf
import numpy as np
import scipy.stats as st
import numpy as np
import scipy.stats as st


# def gauss_kernel(kernlen=21, nsig=3, channels=1):
#     interval = (2 * nsig + 1.) / (kernlen)
#     x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
#     kern1d = np.diff(st.norm.cdf(x))
#     kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
#     kernel = kernel_raw / kernel_raw.sum()
#     out_filter = np.array(kernel, dtype=np.float32)
#     out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
#     out_filter = np.repeat(out_filter, channels, axis=2)
#     return out_filter
#
#
# class Blur(nn.Module):
#     def __init__(self, nc):
#         super(Blur, self).__init__()
#         self.nc = nc
#         kernel = gauss_kernel(kernlen=21, nsig=3, channels=self.nc)
#         kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1)
#         self.weight = nn.Parameter(data=kernel, requires_grad=False)
#
#     def forward(self, x):
#         if x.size(1) != self.nc:
#             raise RuntimeError(
#                 "The channel of input [%d] does not match the preset channel [%d]" % (x.size(1), self.nc))
#         x = F.conv2d(x, self.weight, stride=1, padding=10, groups=self.nc)
#         return x
#
#
# class ColorLoss(nn.Module):
#     def __init__(self):
#         super(ColorLoss, self).__init__()
#
#     def forward(self, x1, x2):
#         return torch.sum(torch.pow((x1 - x2), 2)).div(2 * x1.size()[0])
#
#