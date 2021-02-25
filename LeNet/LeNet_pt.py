import torch
import torch.nn as nn
import torch.optim as optim
# Create Model
class LeNet_pt(nn.Module):
	def __init__(self):
		super(LeNet_pt , self).__init__()
		self.ConvModel = nn.Sequential( 
					nn.Conv2d(in_channels = 1 , out_channels  = 6 , kernel_size   = (3 , 3) , padding = (0 , 0) , stride = (1 , 1)),
					nn.ReLU() ,
					nn.AvgPool2d(kernel_size = (2 , 2) , stride = (2 , 2)) ,
					nn.Conv2d(in_channels = 6 , out_channels  = 16 , kernel_size   = (3 , 3) , padding = (0 , 0) , stride = (1 , 1)),
					nn.ReLU() ,
					nn.AvgPool2d(kernel_size = (2 , 2) , stride = (2 , 2)) ,
					nn.Conv2d(in_channels = 16 , out_channels = 120 , kernel_size = (3 , 3) , padding = (0 , 0) , stride = (1 , 1)) )

		self.DenseModel = nn.Sequential(
					nn.Linear(120 , 84),
					nn.Linear(84  , 10))

	def forward(self , x):
		y = self.ConvModel(x)
		y = y.reshape(y.shape[0] , -1)
		y = self.DenseModel(y)
		return y
