import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from iic.transforms import ComposeTransformIIC, IntentityTransformIIC


class SateliteDataset(Dataset):
    def __init__(self, satelite_image, invariant_transforms=None):
        self.satelite_image = satelite_image
        if invariant_transforms is None:
            self.invariant_transforms = ComposeTransformIIC([IntentityTransformIIC()])
        else:
            self.invariant_transforms = ComposeTransformIIC(invariant_transforms)

    def __len__(self):
        return self.satelite_image.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        satelite_images1 = self.satelite_image[idx]

        satelite_images2 = self.invariant_transforms.forward_transform(satelite_images1)

        return {"img1": satelite_images1, "img2": satelite_images2,
                "inverse": self.invariant_transforms.copy()}
