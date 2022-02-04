from dir_length import dir_length
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as T

class InitDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.dir_list = os.listdir(self.root_dir)

    def __len__(self):
        return dir_length(self.root_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.dir_list[idx])
        image = Image.open(img_name)

        if self.transform:
            tensor_image = self.transform(image)
        return tensor_image