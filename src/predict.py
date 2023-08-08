import glob

import numpy as np
import pandas as pd
import torch
import albumentations as A

from src.CustomizedResnet import CustomizedResNet50
from FarmerDataset import TestDataset
from torch.utils.data import DataLoader
from torch import optim

test_path = glob.glob('../dataset/test/*')
test_path.sort()

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

model = CustomizedResNet50()
model.to(DEVICE)
optimizer = optim.AdamW(model.parameters(), 0.001)
model.load_state_dict(torch.load('../result/accu_0.9415.pth'))
model.eval()

test_loader = DataLoader(TestDataset('../dataset/test.csv',
                                     transform=A.Compose([
                                         A.Resize(256, 256),
                                         A.RandomCrop(224, 224),
                                         A.HorizontalFlip(p=0.5),
                                         A.RandomBrightnessContrast(p=0.5),
                                         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                     ])
                                     ), batch_size=2, shuffle=False, num_workers=1, pin_memory=False
                         )


def predict(MyDataLoader, MyModel):
    MyModel.eval()

    test_prediction = []
    with torch.no_grad():
        for i, (image, target) in enumerate(MyDataLoader):
            image = image.to(DEVICE)
            output = MyModel(image)
            test_prediction.append(output.data.cpu().numpy())

    return np.vstack(test_prediction)


if __name__ == '__main__':

    prediction = None

    for _ in range(1):
        if prediction is None:
            prediction = predict(test_loader, model)
        else:
            prediction += predict(test_loader, model)

    submit = pd.DataFrame(
        {
            'name': [x.split('/')[-1] for x in test_path],
            'label': prediction.argmax(1)
        })

    submit = submit.sort_values(by='name')
    submit.to_csv('../result/test_csv/submit.csv', index=False)
