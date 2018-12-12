import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
import torch
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################
# Loading and normalizing

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.EMNIST(root='./data', split='byclass', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.EMNIST(root='./data',split='byclass', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = list(string.ascii_letters + string.ascii_uppercase + string.ascii_lowercase)


########################################################################
# Define a Convolution Neural Network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "ModelFromPytorchTutorial"

########################################################################
# Test the network on the segmented text

net = Net()
net = net.to(device)
net.load_state_dict(torch.load('./ModelFromPytorchTutorial'))

test_path = './Character_Segmentation-master/segmented_img'
test_path = './Character_Segmentation-master/segmented'
custom_test_dataset = datasets.ImageFolder(test_path, transform)
custom_test_loader = torch.utils.data.DataLoader(custom_test_dataset, batch_size=1, shuffle=False, num_workers=1)

for (images, labels) in custom_test_loader:
    images = images.to(device)
    labels = labels.to(device)
    inputSize = images.size(0)
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)
    i = predicted.cpu().numpy().tolist()[0]
    print(i)
#    import pdb; pdb.set_trace()
    #print(trainloader.dataset.classes[i])
    #print(testloader.dataset.classes[i])
    print(classes[i])

print('Done')





dataiter = iter(testloader)
for i in range(10):

    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))