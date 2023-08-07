import torch
import albumentations as A
from torch import nn
from torch.utils.data import DataLoader

from src.CustomizedResnet import CustomizedResNet50
from src.FarmerDataset import TrainDataset
from tools.dataSplit import data_split

BATCH_SIZE = 64
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
EPOCH = 10

model = CustomizedResNet50()
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), 0.0001)

trainDataset = TrainDataset('../dataset/split/train_val/train.csv', transform=A.Compose([
    A.RandomRotate90(),
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    # A.RandomContrast(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
]))
trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=False)


def train(m_DataLoader, m_Model, m_Criterion, m_Optimizer):
    m_Model.train()
    m_train_loss = 0.0
    for count, (img, target) in enumerate(m_DataLoader):
        img, target = img.to(DEVICE), target.to(DEVICE)
        output = m_Model(img)
        loss = m_Criterion(output, target)
        m_Optimizer.zero_grad()
        loss.backward()
        m_Optimizer.step()

        if count % 100 == 0:
            print('Train loss', loss.item())

        m_train_loss += loss.item()

    return m_train_loss / len(m_DataLoader.dataset)


valDataset = TrainDataset('../dataset/split/train_val/val.csv', transform=A.Compose([
    A.RandomRotate90(),
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    # A.RandomContrast(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
]))
valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=False)


def val(m_DataLoader, m_Model):
    m_Model.eval()
    accuracy = 0.0

    with torch.no_grad():
        for count, (img, target) in enumerate(m_DataLoader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            output = m_Model(img)
            accuracy += (output.max(dim=1).indices == target).sum().item()

    return accuracy / len(m_DataLoader.dataset)


if __name__ == '__main__':

    model.load_state_dict(torch.load("../result/pth/res50_accu_0.92904.pth"))

    for i in range(EPOCH):
        data_split()

        train_loss = train(m_Model=model, m_DataLoader=trainDataLoader, m_Optimizer=optimizer, m_Criterion=criterion)
        val_acc = val(m_Model=model, m_DataLoader=valDataLoader)
        print("Epoch:{}\t\t\ttrain_loss:{}\t\t\tval_acc:{}".format(i, train_loss, val_acc))

        torch.save(model.state_dict(), "../result/accu_{:.4f}.pth".format(val_acc))
