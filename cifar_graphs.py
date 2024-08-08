import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import torch 
available_devices = torch.cuda.device_count()
for i in range(torch.cuda.device_count()):
    print (f"Device{i}: {torch.cuda.get_device_name(i)}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def load_data(batch_size):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return trainloader, testloader

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DynamicNet(nn.Module):
    def __init__(self, layer_config):
        super(DynamicNet, self).__init__()
        self.layers = self._make_layers(layer_config)
        self.fc_layers = self._make_fc_layers(layer_config)

    def _make_layers(self, layer_config):
        layers = []
        in_channels = 3
        for config in layer_config:
            if config['type'] == 'conv':
                layers.append(nn.Conv2d(in_channels, config['out_channels'], config['kernel_size']))
                in_channels = config['out_channels']
            elif config['type'] == 'pool':
                layers.append(nn.MaxPool2d(config['kernel_size'], config['stride']))
        return nn.Sequential(*layers)

    def _make_fc_layers(self, layer_config):
        layers = []
        in_features = self._get_conv_output_size(layer_config)
        for config in layer_config:
            if config['type'] == 'fc':
                layers.append(nn.Linear(in_features, config['out_features']))
                in_features = config['out_features']
        return nn.Sequential(*layers)

    def _get_conv_output_size(self, layer_config):
        # Create a dummy input tensor with the same dimensions as a CIFAR-10 image
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            for config in layer_config:
                if config['type'] == 'conv':
                    x = F.relu(nn.Conv2d(x.size(1), config['out_channels'], config['kernel_size'])(x))
                elif config['type'] == 'pool':
                    x = nn.MaxPool2d(config['kernel_size'], config['stride'])(x)
        return x.numel()

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

def train_net(net, criterion, optimizer, trainloader, epochs):
    net.to(device)
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return net

def evaluate_network(net, testloader):
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

import matplotlib.pyplot as plt
import torch.optim as optim


def run_experiments():
    results = {'batch_size': [], 'num_epochs': [], 'learning_rate': [], 'momentum': [], 'scheduler': [], 'architectures_optimizers': [], 'layers':[]}
    
    batch_sizes = [16, 32, 40, 50, 64]
    
    for batch_size in batch_sizes:
        print(f'Running experiment with batch size: {batch_size}')
        trainloader, testloader = load_data(batch_size)
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        net = train_net(net, criterion, optimizer, trainloader, epochs=10)
        accuracy = evaluate_network(net, testloader)
        results['batch_size'].append((batch_size, accuracy))
    

    

    learning_rates = [0.001, 0.003, 0.005, 0.007, 0.009]
    for lr in learning_rates:
        print(f'Running experiment with learning rate: {lr}')
        trainloader, testloader = load_data(32)  # fixed batch size
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        net = train_net(net, criterion, optimizer, trainloader, epochs=10)
        accuracy = evaluate_network(net, testloader)
        results['learning_rate'].append((lr, accuracy))
    
    epoch_counts = [1,2,3,4,6,8,10]
    for num_epochs in epoch_counts:
        print(f'Running experiment with number of epochs: {num_epochs}')
        trainloader, testloader = load_data(32)  # fixed batch size
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        net = train_net(net, criterion, optimizer, trainloader, epochs=num_epochs)
        accuracy = evaluate_network(net, testloader)
        results['num_epochs'].append((num_epochs, accuracy))

    

    momentums = [0.1, 0.3, 0.5, 0.7, 0.9]
    for momentum in momentums:
        print(f'Running experiment with momentum: {momentum}')
        trainloader, testloader = load_data(32)  # fixed batch size
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=momentum)
        net = train_net(net, criterion, optimizer, trainloader, epochs=10)
        accuracy = evaluate_network(net, testloader)
        results['momentum'].append((momentum, accuracy))
    
    scheduler_types = ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau']
    for scheduler_type in scheduler_types:
        print(f'Running experiment with scheduler: {scheduler_type}')
        trainloader, testloader = load_data(32)  # fixed batch size
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        
        if scheduler_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_type == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45], gamma=0.1)
        elif scheduler_type == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        net = train_net(net, criterion, optimizer, trainloader, epochs=10)
        val_loss = evaluate_network(net, testloader)  # Use validation loss or accuracy as the metric
        scheduler.step(val_loss)
        accuracy = evaluate_network(net, testloader)
        results['scheduler'].append((scheduler_type, accuracy))
    

    architectures_optimizers = [('ResNet18', 'Adam'), ('ResNet18', 'SGD'), ('VGG16', 'Adam'), ('VGG16', 'SGD'), ('SimpleNet', 'Adam')]
    for arch, opt in architectures_optimizers:
        print(f'Running experiment with architecture: {arch}, optimizer: {opt}')
        trainloader, testloader = load_data(64)  # fixed batch size
        if arch == 'ResNet18':
            net = torchvision.models.resnet18(num_classes=10)
        elif arch == 'VGG16':
            net = torchvision.models.vgg16(num_classes=10)
        elif arch == 'SimpleNet':
            net = Net()
        
        criterion = nn.CrossEntropyLoss()
        if opt == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=0.001)
        elif opt == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        
        net = train_net(net, criterion, optimizer, trainloader, epochs=10)
        accuracy = evaluate_network(net, testloader)
        results['architectures_optimizers'].append((arch, opt, accuracy))

    layer_configs = [
        [
            {'type': 'conv', 'out_channels': 6, 'kernel_size': 5},
            {'type': 'pool', 'kernel_size': 2, 'stride': 2},
            {'type': 'conv', 'out_channels': 16, 'kernel_size': 5},
            {'type': 'pool', 'kernel_size': 2, 'stride': 2},
            {'type': 'fc', 'out_features': 120},
            {'type': 'fc', 'out_features': 84}
        ],
        [
            {'type': 'conv', 'out_channels': 32, 'kernel_size': 3},
            {'type': 'conv', 'out_channels': 64, 'kernel_size': 3},
            {'type': 'pool', 'kernel_size': 2, 'stride': 2},
            {'type': 'conv', 'out_channels': 128, 'kernel_size': 3},
            {'type': 'pool', 'kernel_size': 2, 'stride': 2},
            {'type': 'fc', 'out_features': 256},
            {'type': 'fc', 'out_features': 128}
        ]
    ]
    
    for layer_config in layer_configs:
        print(f'Running experiment with layer config: {layer_config}')
        trainloader, testloader = load_data(32)  # fixed batch size
        net = DynamicNet(layer_config)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        net = train_net(net, criterion, optimizer, trainloader, epochs=10)
        accuracy = evaluate_network(net, testloader)
        results['layers'].append((str(layer_config), accuracy))

    
    return results

results = run_experiments()

plt.figure(figsize=(18, 12))

    # Plot Batch Size vs Accuracy
plt.subplot(2, 3, 1)
batch_sizes, accuracies = zip(*results['batch_size'])
plt.plot(batch_sizes, accuracies, marker='o')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy (%)')
plt.title('Batch Size vs Accuracy')
plt.ylim(0, 100)
plt.grid(True)

    # Plot Number of Epochs vs Accuracy
plt.subplot(2, 3, 2)
epoch_counts, accuracies = zip(*results['num_epochs'])
plt.plot(epoch_counts, accuracies, marker='o')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Number of Epochs vs Accuracy')
plt.ylim(0, 100)
plt.grid(True)

    # Plot Learning Rate vs Accuracy
plt.subplot(2, 3, 3)
learning_rates, accuracies = zip(*results['learning_rate'])
plt.plot(learning_rates, accuracies, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy (%)')
plt.title('Learning Rate vs Accuracy')
plt.ylim(0, 100)
plt.grid(True)

    # Plot Momentum vs Accuracy
plt.subplot(2, 3, 4)
momentums, accuracies = zip(*results['momentum'])
plt.plot(momentums, accuracies, marker='o')
plt.xlabel('Momentum')
plt.ylabel('Accuracy (%)')
plt.title('Momentum vs Accuracy')
plt.ylim(0, 100)
plt.grid(True)

# Plot Scheduler vs Accuracy
plt.subplot(2, 3, 5)
scheduler_types = [sched[0] for sched in results['scheduler']]
accuracies = [sched[1] for sched in results['scheduler']]
plt.bar(scheduler_types, accuracies)
plt.xlabel('Scheduler')
plt.ylabel('Accuracy (%)')
plt.title('Scheduler vs Accuracy')
plt.ylim(60, 70)
plt.grid(True)
plt.xticks(rotation=45)

plt.subplot(2, 3, 6)
layer_configs = [config[0] for config in results['layers']]
accuracies = [config[1] for config in results['layers']]
plt.barh(layer_configs, accuracies)
plt.xlabel('Accuracy (%)')
plt.ylabel('Layer Configurations')
plt.title('Layer Configurations vs Accuracy')
plt.xlim(0, 100)
plt.grid(True)

    # Adjust layout and save plot
plt.tight_layout()
plt.savefig('cifar_results.png')
plt.show()

