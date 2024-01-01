from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

# mean is tensor([0.6325, 0.6159, 0.5908]) and std is tensor([0.2962, 0.2915, 0.3144])

data_augmentation = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop((227,227)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.6325, 0.6159, 0.5908),(0.2962, 0.2915, 0.3144)),
])

def create_train_dataloader():
    """ creates the gemstone train data loader from train folder """
    dataset = ImageFolder('./gemstone dataset/train',transform=data_augmentation)
    dataloader = DataLoader(dataset,32,True)
    return dataloader

def create_test_dataloader():
    """ creates the gemstone train data loader from train folder """
    dataset = ImageFolder('./gemstone dataset/test',transform=data_augmentation)
    dataloader = DataLoader(dataset,32,True)
    return dataloader