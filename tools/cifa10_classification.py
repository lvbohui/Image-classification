import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


transform = transforms.Compose(
    [transforms.Resize((150, 150)),
     transforms.ToTensor()]
)
train_dir = "data/double_bed200/train"
test_dir = "data/double_bed200/valid"

batch_size = 4

trainset = ImageFolder(root=train_dir, transform=transform)
testset = ImageFolder(root=test_dir, transform=transform)

def display_image(image, label):
    print("Label:{}".format(trainset.classes[label]))
    plt.imshow(image.permute(1, 2, 0))


display_image(*trainset[0])