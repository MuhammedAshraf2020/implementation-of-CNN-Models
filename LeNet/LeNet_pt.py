import torch
import torch.nn as nn
import torch.optim as optim
# Create Model
class LeNet(nn.Module):
	def __init__(self):
		super(LeNet , self).__init__()
		self.relu     = nn.ReLU()
		self.AvgPool  = nn.AvgPool2d(kernel_size = (2 , 2) , stride = (2 , 2))
		self.Conv2D_1 = nn.Conv2d(in_channels = 1 , out_channels  = 6 , kernel_size   = (5 , 5) , padding = (0 , 0) , stride = (1 , 1))
		self.Conv2D_2 = nn.Conv2d(in_channels = 6 , out_channels  = 16 , kernel_size  = (5 , 5) , padding = (0 , 0) , stride = (1 , 1))
		self.Conv2D_3 = nn.Conv2d(in_channels = 16 , out_channels = 120 , kernel_size = (5 , 5) , padding = (0 , 0) , stride = (1 , 1))
		self.Dense_1  = nn.Linear(120 , 84)
		self.Dense_2  = nn.Linear(84  , 10)

	def forward(self , x):
		x = self.Conv2D_1(x)
		x = self.relu(x)
		x = self.AvgPool(x)
		x = self.Conv2D_2(x)
		x = self.relu(x)
		x = self.AvgPool(x)
		x = self.Conv2D_3(x)
		x = x.reshape(x.shape[0] , -1)
		x = self.relu(self.Dense_1(x))
		x = self.Dense_2(x)
		return x


epochs = 10
batch_size = 150
lr = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = LeNet().to(device = device)

loss  = nn.CrossEntropyLoss()
adam  = optim.Adam(model.parameters() , lr = lr)
