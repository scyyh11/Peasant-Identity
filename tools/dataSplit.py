import pandas as pd


# 将25个类按比例分为训练集和验证集
def data_split():
    train = pd.DataFrame(columns=['name', 'label'])
    val = pd.DataFrame(columns=['name', 'label'])

    for i in range(25):
        data = pd.read_csv('../dataset/split/data_{}.csv'.format(i))
        temp_train = data.sample(frac=0.8)
        train = pd.concat([train, temp_train], ignore_index=True)
        temp_val = data[~data.index.isin(temp_train.index)]
        val = pd.concat([val, temp_val], ignore_index=True)

    train.sample(frac=1).to_csv('../dataset/split/train_val/train.csv', index=False)
    val.sample(frac=1).to_csv('../dataset/split/train_val/val.csv', index=False)


if __name__ == '__main__':
    data_split()
