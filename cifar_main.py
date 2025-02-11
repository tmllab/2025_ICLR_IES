import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import _pickle as cPickle
import torch.optim.lr_scheduler as lr_scheduler
import csv
from resnet import ResNet18, ResNet34, ResNet50, ResNet101
import torchvision.models as models
import random
import argparse
import time
from torch.optim import Adam, AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="CIFAR Training Script")

parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100"], help="Dataset to use")

parser.add_argument("--root_dir", type=str, default=None, help="Root directory for dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for SGD optimizer")
parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
parser.add_argument("--moving_average_rate", type=int, default=3, help="Moving average rate")
parser.add_argument("--k", type=int, default=1, help="Window size for identifying mastered instances")
parser.add_argument("--threshold", type=float, default=1e-3, help="Threshold for identifying mastered instances")
parser.add_argument("--num_iterations", type=int, default=1, help="Number of iterations to run the experiment")
parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "vgg16", "densenet121"], help="ResNet model to use")
parser.add_argument("--optimizer", type=str, default="SGD_exponential_learning_rate",
                    choices=["Adam", "SGD_fixed_learning_rate", "SGD_exponential_learning_rate", "Adam_W", "SGD_linear_learning_rate"],
                    help="Optimizer to use")
parser.add_argument("--removal_criteria", type=str, default="second_derivative",
                    choices=["second_derivative"],
                    help="Criteria for sample removal")

def setup_seed(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='bytes')
    return dict

class CifarDataset(Dataset):
    def __init__(self, data, labels, index_list, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        true_idx = self.index_list[idx]
        img, target = self.data[true_idx], self.labels[true_idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, target, true_idx

class VGG16_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_CIFAR, self).__init__()
        self.features = nn.Sequential(
            self._make_layer(3, 64, 2),
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 3),
            self._make_layer(256, 512, 3),
            self._make_layer(512, 512, 3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layer(self, in_channels, out_channels, num_convs):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# DenseNet121 for CIFAR
class DenseNet121_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet121_CIFAR, self).__init__()
        self.model = models.densenet121(pretrained=False)

        # Modify the first convolution layer
        self.model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove the max pooling layer
        self.model.features.pool0 = nn.Identity()

        # Modify the final classifier
        self.model.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.model(x)


class cifar_dataset(Dataset):
    def __init__(self, data, labels, index_list, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        true_idx = self.index_list[idx]
        img, target = self.data[true_idx], self.labels[true_idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, target, true_idx

def get_cifar10_dataset(root_dir, transform):
    # Load all data and labels
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch = unpickle(os.path.join(root_dir, f'data_batch_{i}'))
        train_data.append(batch[b'data'])
        train_labels += batch[b'labels']
    train_data = np.concatenate(train_data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    train_labels = np.array(train_labels)

    # Load test data and labels
    test_batch = unpickle(os.path.join(root_dir, 'test_batch'))
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    test_labels = np.array(test_batch[b'labels'])

    # Create datasets
    train_dataset = cifar_dataset(train_data, train_labels, list(range(len(train_data))), transform)
    test_dataset = cifar_dataset(test_data, test_labels, list(range(len(test_data))), transform)

    return train_dataset, test_dataset

def get_cifar100_dataset(root_dir, transform):
    # Load all data and labels for training
    train_data = []
    train_labels = []
    batch = unpickle(os.path.join(root_dir, 'train'))
    train_data.append(batch[b'data'])
    train_labels += batch[b'fine_labels']
    train_data = np.concatenate(train_data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    train_labels = np.array(train_labels)

    # Load test data and labels
    test_batch = unpickle(os.path.join(root_dir, 'test'))
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    test_labels = np.array(test_batch[b'fine_labels'])

    # Create datasets
    train_dataset = cifar_dataset(train_data, train_labels, list(range(len(train_data))), transform)
    test_dataset = cifar_dataset(test_data, test_labels, list(range(len(test_data))), transform)

    return train_dataset, test_dataset

def create_dataloaders(train_dataset, test_dataset, batch_size, num_workers):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, test_loader

def correct_moving_average_new_new(data, window_size):
    abs_data = np.abs(data)
    window = sum(abs_data[:window_size])
    result = [window / window_size]
    for i in range(window_size, len(data)):
        window += abs_data[i] - abs_data[i - window_size]
        result.append(window / window_size)
    return result

def get_model(model_name, num_classes):
    if model_name == "resnet18":
        return ResNet18(num_classes)
    elif model_name == "resnet34":
        return ResNet34(num_classes)
    elif model_name == "resnet50":
        return ResNet50(num_classes)
    elif model_name == "resnet101":
        return ResNet101(num_classes)
    elif model_name == "vgg16":
        return VGG16_CIFAR(num_classes)
    elif model_name == "densenet121":
        return DenseNet121_CIFAR(num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_optimizer_and_scheduler(optimizer_name, model_parameters, args):
    if optimizer_name == "Adam":
        learning_rate = 0.001
        optimizer = Adam(model_parameters, lr=learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)
    elif optimizer_name == "Adam_W":
        learning_rate = 0.001
        optimizer = AdamW(model_parameters, lr=learning_rate, weight_decay=0.01)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)
    elif optimizer_name == "SGD_fixed_learning_rate":
        learning_rate = 0.001
        optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)
    elif optimizer_name == "SGD_linear_learning_rate":
        learning_rate = 0.1
        optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=150)
    elif optimizer_name == "SGD_exponential_learning_rate":
        learning_rate = 0.1
        optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer, scheduler, args

def calculate_derivative(x, order):
    if len(x) < order + 1:
        return 0
    if order == 2:
        return x[-1] - 2*x[-2] + x[-3]
    else:
        raise ValueError(f"Unsupported derivative order: {order}")

def main(args):
    start_time = time.time()
    global_file_name = f'global_{args.dataset}_{args.model}_{args.optimizer}_k{args.k}_{args.removal_criteria}_ma{args.moving_average_rate}'

    all_best_accuracies = []
    all_saved_ratios = []

    if args.dataset == "cifar10":
        num_classes = 10
        threshold = args.threshold
        get_dataset = get_cifar10_dataset
        transform = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.root_dir is None:
            args.root_dir = "./cifar-10"
    elif args.dataset == "cifar100":
        num_classes = 100
        threshold = args.threshold * 2
        get_dataset = get_cifar100_dataset
        transform = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
        if args.root_dir is None:
            args.root_dir = "./cifar-100"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    derivative_order = {"zero_derivative": 0, "first_derivative": 1, "second_derivative": 2, "third_derivative": 3}[
        args.removal_criteria]

    for iternum in range(args.num_iterations):

        setup_seed(iternum)

        # Initialize model, optimizer, and scheduler
        model = get_model(args.model, num_classes).to(device)
        optimizer, scheduler, args = get_optimizer_and_scheduler(args.optimizer, model.parameters(), args)

        file_name = f'{args.dataset}_{args.model}_{args.optimizer}_k{args.k}_{args.threshold}_{args.removal_criteria}_ma{args.moving_average_rate}_{iternum}'
        csvfile = file_name + ".csv"
        with open(csvfile, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Len of Training set', 'Avg Training Loss', 'Avg Test Loss', 'TestAcc'])

        # Initialize datasets and dataloaders
        train_dataset, test_dataset = get_dataset(args.root_dir, transform)
        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, args.batch_size, args.num_workers)
        full_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, drop_last=True)
        full_train_dataset = train_dataset

        # Loss history for each sample
        sample_loss_history = {idx: [] for idx in range(len(train_loader.dataset))}
        derivative_history = {idx: [] for idx in range(len(train_loader.dataset))}

        best_test_accuracy = 0
        remaining_loader = None

        total_saved_samples = 0
        total_samples = len(full_train_dataset)
        for epoch in range(args.epochs):

            start_epoch_time = time.time()
            model.train()
            running_loss = 0.0
            start_train_time = time.time()
            for inputs, labels, indices in train_loader:
                if inputs.size(0) == 1:
                    continue

                inputs, labels = inputs.to(device), labels.to(device)
                indices = indices.numpy()

                optimizer.zero_grad()

                outputs = model(inputs)
                criterion = nn.CrossEntropyLoss(reduction='none')
                losses = criterion(outputs, labels)

                for i, index in enumerate(indices):
                    sample_loss_history[index].append(losses[i].item())

                loss = losses.mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader.dataset)
            end_train_time = time.time()
            print(f'Training time for epoch {epoch + 1}: {end_train_time - start_train_time:.2f} seconds')

            scheduler.step()

            if remaining_loader is not None:
                start_eval_time = time.time()
                with torch.no_grad():
                    model.eval()
                    for inputs, labels, indices in remaining_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        indices = indices.numpy()

                        outputs = model(inputs)
                        criterion = nn.CrossEntropyLoss(reduction='none')
                        losses = criterion(outputs, labels)

                        for i, index in enumerate(indices):
                            sample_loss_history[index].append(losses[i].item())
                end_eval_time = time.time()
                print(f'Evaluation time for epoch {epoch + 1}: {end_eval_time - start_eval_time:.2f} seconds')

            if args.threshold > 0:
                start_calc_time = time.time()
                for idx, losses in sample_loss_history.items():
                    if len(losses) >= derivative_order + 1:
                        latest_derivative = calculate_derivative(losses, derivative_order)
                        derivative_history[idx].append(latest_derivative)
                        if len(derivative_history[idx]) > args.moving_average_rate + args.k:
                            derivative_history[idx] = derivative_history[idx][-(args.moving_average_rate + args.k):]
                end_calc_time = time.time()
                print(f'Derivative calculation time: {end_calc_time - start_calc_time:.2f} seconds')

                start_excluded_samples_time = time.time()
                excluded_samples = []
                for idx, derivatives in derivative_history.items():
                    if len(derivatives) >= args.moving_average_rate + args.k:
                        ma_derivatives = correct_moving_average_new_new(derivatives, args.moving_average_rate)
                        derivative_sum = np.abs(ma_derivatives[-args.k:]).sum()
                        if derivative_sum < threshold:
                            excluded_samples.append(idx)
                end_excluded_samples_time = time.time()
                print(
                    f'Excluded samples calculation time: {end_excluded_samples_time - start_excluded_samples_time:.2f} seconds')

                if excluded_samples:
                        excluded_set = set(excluded_samples)
                        all_indices = set(range(len(full_train_dataset)))
                        new_indices = list(all_indices - excluded_set)
                        train_dataset = CifarDataset(full_train_dataset.data, full_train_dataset.labels, new_indices,
                                                     transform)
                        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers)

                        remaining_indices = list(excluded_set)
                        remaining_dataset = cifar_dataset(full_train_dataset.data, full_train_dataset.labels,
                                                          remaining_indices,
                                                          transform)
                        remaining_loader = DataLoader(remaining_dataset, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=args.num_workers, pin_memory=True)
                        if excluded_samples:
                            total_saved_samples += len(excluded_samples)
            else:
                saved_ratio = 0
                print("Skipping second derivative calculations and sample exclusion (Baseline method)")

            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels, _ in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_test_loss = test_loss / len(test_loader.dataset)
            if correct > best_test_accuracy:
                best_test_accuracy = correct

            with open(csvfile, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [epoch + 1, len(train_loader.dataset), epoch_loss, avg_test_loss, correct])

            end_epoch_time = time.time()
            print(f'=============Finished epoch {epoch + 1}. Time taken: {end_epoch_time - start_epoch_time:.2f} seconds=============')

        print(f'Finished Iternum {iternum}. Best Test Accuracy: {best_test_accuracy:.4f}')
        all_best_accuracies.append(best_test_accuracy / (len(test_dataset) * 0.01))

        saved_ratio = total_saved_samples / (total_samples * args.epochs)
        all_saved_ratios.append(saved_ratio * 100)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate statistics
    mean_accuracy = np.mean(all_best_accuracies)
    std_accuracy = np.std(all_best_accuracies)
    mean_saved_ratio = np.mean(all_saved_ratios)

    with open(global_file_name + ".csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Threshold', 'Total Run Time', 'Best Accuracy', 'STD', 'Saved Samples Ratio'])
        writer.writerow([
            f"{args.threshold}",
            f"{total_time:.2f}",
            f"{mean_accuracy:.2f}%",
            f"{std_accuracy:.2f}%",
            f"{mean_saved_ratio:.2f}%"
        ])

    print(f"Total Running Time: {total_time:.2f} seconds")
    print(f"Best Accuracy: {mean_accuracy:.2f}%±{std_accuracy:.2f}%")
    print(f"Saved Samples Ratio: {mean_saved_ratio:.2f}%")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)