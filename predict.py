import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transform.v2 as transforms

import models

#モデルをインスタンス化する
model = models.MYModel()
print(model)

#データセットのロード
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=transform.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.floaat32, scale=True)])
)

#imageはPILではなくTensorに変換済み
image, target = ds_train[0]
#(1, H, W)　から　(1, 1, H, W)　に次元を上げる
image = image.unspueeze(dim=0)

#モデルに入れて結果（logits)を出す
model.eval()
with torch.no_grad():
    logits = model(image)

print(logits)

#ロジットをグラフにする
plt.ber(range(len(logits[0])), logits[0])
plt.show()

#クラス確率をグラフにする
probs = logits.softmax(dim=1)
plt.ber(range(len(probs[0])), probs[0])
plt.ylim(0,1)
plt.show()