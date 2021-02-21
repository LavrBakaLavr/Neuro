# -*- coding: utf-8 -*-
"""
@author: Максим Лавринов
"""

# необходимые импорты
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import time

#Запускаем счетчик времени
start_time = time.time()

FOLDER_DATASET = "./archive/"

# создаем объект tranfrorm для трансформации изображений 
transform = transforms.Compose(
    [
     transforms.Resize((176, 88)),
     transforms.Grayscale(),
     transforms.ToTensor()
    ]
)

#Dataloader для тренировочного набора
class DriveData(Dataset):
    __xs = []
    __ys = []

    def __init__(self, folder_dataset, transform=transform):
        self.transform = transform
        # Open and load text file including the whole training data
        with open(folder_dataset + "train2.txt") as f:
            for line in f:
                # Image path
                self.__xs.append(folder_dataset + line.split()[1]) 
                # Steering wheel label
                self.__ys.append(np.float32(line.split()[0]))

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = Image.open(self.__xs[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        label = torch.from_numpy(np.asarray(self.__ys[index]))
        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)

#Dataloader для тестового набора
class DriveData1(Dataset):
    __xs = []
    __ys = []

    def __init__(self, folder_dataset, transform=transform):
        self.transform = transform
        # Open and load text file including the whole training data
        with open(folder_dataset + "test2.txt") as f:
            for line in f:
                # Image path
                self.__xs.append(folder_dataset + line.split()[1]) 
                # Steering wheel label
                self.__ys.append(np.float32(line.split()[0]))

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = Image.open(self.__xs[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        label = torch.from_numpy(np.asarray(self.__ys[index]))
        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)

dset_train = DriveData(FOLDER_DATASET)
train_loader = DataLoader(dset_train, batch_size=32, shuffle=True)

dset_test = DriveData1(FOLDER_DATASET)
test_loader = DataLoader(dset_test, batch_size=32, shuffle=True)

# построение сверточной нейронной сети на PyTorch

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    # стек сверточных слоев
    self.conv_layers = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, padding=1, stride=1), # (N, 1, 28, 28) 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), 
        
        nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        
        nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    # стек полносвязных слоев
    self.linear_layers = nn.Sequential(
        nn.Linear(in_features=192*22*11, out_features=1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        
        nn.Linear(in_features=1024, out_features=512),
        nn.ReLU(),
        nn.Dropout(0.1),
        
        nn.Linear(in_features=512, out_features=196) # количество выходных признаков равно количеству классов
    )
  
  # определение метода для прямого распространения сигналов по сети
  def forward(self, x):
    x = self.conv_layers(x)
    # перед отправкой в блок полносвязных слоев признаки необходимо сделать одномерными
    x = x.view(x.size(0), -1)
    x = self.linear_layers(x)
    return x


cnn = CNN()

# функция, отвечающая за обучение сети

def fit(model, 
          optimizer, 
          loss_function, 
          train_loader, 
          test_loader, 
          epochs, 
          device,
         ):
    # определяем количество батчей в тренировочной выборке
    total_step = len(train_loader)
    
    # пускаем цикл по эпохам
    for epoch in range(epochs):
        train_loss = 0
        # для каждого батча в тренировочном наборе
        for i, batch in enumerate(train_loader):  
            # извлекаем изображения и их метки
            images, labels = batch
            # отправляем их на устройство
            images = images.to(device)
            labels = labels.to(device).long()
            # вычисляем выходы сети
            outputs = model(images)
            # вычисляем потери на батче
            loss = loss_function(outputs, labels)
            # обнуляем значения градиентов
            optimizer.zero_grad()
            # вычисляем значения градиентов на батче
            loss.backward()
            # корректируем веса
            optimizer.step()
            
            # корректируем значение потерь на эпохе
            train_loss += loss.item()
            
            # логируем
            if (i+1) % 100 == 0:
                print ('Эпоха [{}/{}], Шаг [{}/{}], Тренировочные потери: {:.4f}' 
                       .format(epoch+1, epochs, i+1, total_step, loss.data.item()))
                
        
    # режим тестирования модели
    # для тестирования вычислять градиенты не обязательно, поэтому оборачиваем код
    # для теста в блок with torch.no_grad()
    with torch.no_grad():
        # заводим начальные значения корректно распознанных примеров и общего количества примеров
        correct = 0
        total = 0
        # для каждого батча в тестовой выборкй
        for batch in test_loader:
            # извлекаем изображения и метки
            print('.', end='')
            images, labels = batch
            # помещаем их на устройство
            images = images.to(device)
            labels = labels.to(device)
            # вычисление предсказаний сети
            outputs = model(images)
            # создание тензора предсказаний сети
            _, predicted = torch.max(outputs.data, 1)
            # корректировка общего значения примеров на величину батча
            total += labels.size(0)
            # корректировка значения верно классифицированных примеров
            correct += (predicted == labels).sum().item()
        
        # логирование
        print('\nТочность на тестовом наборе {} %'.format(100 * correct / total))
        
# определим функцию оптимизации
optimizer = optim.Adam(cnn.parameters(), lr=0.0005)

# определим функцию потерь
loss_function = nn.CrossEntropyLoss()

# определим устройство, на котором будет идти обучение
device = None
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

print(device)

# перемещение модели на устройство
cnn.to(device)

epochs=15

fit(cnn,       optimizer,      loss_function,      train_loader,      test_loader,      epochs,      device)

#Сохраняем модель
torch.save(cnn.state_dict(), FOLDER_DATASET + 'cnn+ann.ckpt')

print("--- %s seconds ---" % (time.time() - start_time))