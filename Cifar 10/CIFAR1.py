# 1. Define the network

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
	#relu1
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
	#relu2	
        self.conv3 = nn.Conv2d(32,32,3,padding=1)
        self.bn3 = nn.BatchNorm2d(32)	
	#relu3	
        self.conv4 = nn.Conv2d(32,64,3,padding=1)	
        self.bn4 = nn.BatchNorm2d(64)	
        self.short1 = nn.Conv2d(32,64,1)
        self.bn5 = nn.BatchNorm2d(64)        
	#relu4
        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn6 = nn.BatchNorm2d(64)	
	#relu5
        self.conv6 = nn.Conv2d(64,128,3,padding=1)	
        self.bn7 = nn.BatchNorm2d(128)	
        self.short2 = nn.Conv2d(64,128,1)
        self.bn8 = nn.BatchNorm2d(128)        
	#relu6
        self.conv7 = nn.Conv2d(128,128,3,padding=1)
        self.bn9 = nn.BatchNorm2d(128)	
	#relu7
        self.conv8 = nn.Conv2d(128,256,3,padding=1)	
        self.bn10 = nn.BatchNorm2d(256)	
        self.short3 = nn.Conv2d(128,256,1)
        self.bn11 = nn.BatchNorm2d(256)        
        #relu8
        self.conv9 = nn.Conv2d(256,256,3,padding=1)
        self.bn12 = nn.BatchNorm2d(256)	
	#relu9
        self.conv10 = nn.Conv2d(256,512,3,padding=1)	
        self.bn13 = nn.BatchNorm2d(512)	
        self.short4 = nn.Conv2d(256,512,1)
        self.bn14 = nn.BatchNorm2d(512)
    #relu10
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(512*16*16, 10)
        
    def forward(self, x):
        x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        x = F.relu(self.bn5(self.short1(x) + self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = F.relu(self.bn8(self.short2(x) + self.bn7(self.conv6(F.relu(self.bn6(self.conv5(x)))))))
        x = F.relu(self.bn11(self.short3(x) + self.bn10(self.conv8(F.relu(self.bn9(self.conv7(x)))))))
        x = F.relu(self.bn14(self.short4(x) + self.bn13(self.conv10(F.relu(self.bn12(self.conv9(x)))))))
        x = self.pool(x)
        x = x.view(-1,512*16*16)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

# 2. Defining the training process    
def train(args, model, device, train_loader, optimizer,criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Changing to the Device
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad() 
        # forward 
        output = model(data)
        # Generating Loss values
        loss = criterion(output, target)
        # backward
        loss.backward()
        # optimize
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))

# 2. Defining the testing process 
def test(args, model, device, test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.optim as optim

def main():
    # 2. Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default= 4, metavar='N',help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    
    # 3. Load Clean Data and Normalize it
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,shuffle=True, num_workers=2,pin_memory= True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,shuffle=False, num_workers=2, pin_memory = True)
    #classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 4. Create a Model & Define a optimizer and Loss function
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Train the network and then test it
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer,criterion, epoch)
        test(args, model, device, test_loader,criterion)

if __name__ == '__main__':
    main()

