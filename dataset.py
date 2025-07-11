import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class BModeDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, folder_list=None):
        
        self.root_dir = root_dir
        self.transform = transform

        # Gather list of subfolders that contain the expected images
        self.folder_list = [
            folder for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder))
        ]

        # Sort folders (assumes folder names are numeric; adjust key if necessary)
        self.folder_list = sorted(self.folder_list, key=lambda x: int(x) if x.isdigit() else x)

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        folder = self.folder_list[idx]
        folder_path = os.path.join(self.root_dir, folder)
        pre_path = os.path.join(folder_path, "pre.png")
        post_path = os.path.join(folder_path, "post.png")

        # Load as grayscale images
        pre_img = Image.open(pre_path).convert("L")
        post_img = Image.open(post_path).convert("L")

        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)
        else:
            pre_img = transforms.ToTensor()(pre_img)
            post_img = transforms.ToTensor()(post_img)

        return {'img1': pre_img, 'img2': post_img}