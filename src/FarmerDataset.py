from torch.utils.data import Dataset
import cv2


class TrainDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgList = []
        fh.readline()
        for line in fh:
            line = line.rstrip()
            words = line.split()
            label = words[0].split(',')
            imgList.append(('../dataset/train/' + label[0], int(label[1])))
        self.imgList = imgList
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgList[index]
        img = cv2.imread(fn)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        img = img.transpose([2, 0, 1])
        return img, label

    def __len__(self):
        return len(self.imgList)


class TestDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgList = []
        fh.readline()
        for line in fh:
            line = line.rstrip()
            words = line.split()
            label = words[0].split(',')
            imgList.append(('../dataset/test/' + label[0], int(label[1])))
        self.imgList = imgList
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgList[index]
        img = cv2.imread(fn)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        img = img.transpose([2, 0, 1])
        return img, label

    def __len__(self):
        return len(self.imgList)
