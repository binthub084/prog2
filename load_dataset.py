
import matplotlib.pyplot as plt
import torch
from torchvision import dasasets
import torchvision.transforms.v2 as transforms

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

print(f'numbers of dasasets:{len(ds_train)}')

image, target = ds_train[0]
print(type(image),target)

plt.imshow(image)
plt.title(target)
plt.show()

#PIL画像を torch.tensor　に変換する
image_tensor = transforms.functional.to_image(image)
print(image_tensor.shape,image_tensor.dtype)

#PIL画像を
image = transforms.functional.to_image(image)
image = transforms.functional.to_dtype(image,dtype=torch.float32,scale=True)
print(image.shape, image.dtype)
print(image.min(), image.max())
