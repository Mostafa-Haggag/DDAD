import os
from glob import glob
import torch.utils.data
from PIL import Image
from torchvision import transforms
import numpy as np

def scale_to_minus_one_one(tensor):
    return (tensor * 2) - 1


class Dataset_maker(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(), # Scales data into [0,1] 
                transforms.Lambda(scale_to_minus_one_one) # Scale between [-1, 1]
            ]
        )
        self.config = config
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(),
                # Scales data into [0,1]
            ]
        )
        # if is_train:
        root = root.replace("\\", "\\\\")

        self.image_files = glob(
                    os.path.join(root, "*.bmp")
        )

        # else:
        #     if category:
        #         self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
        #     else:
        #         self.image_files = glob(os.path.join(root, "test", "*", "*.png"))
        # self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if image.shape[0] == 1:
            # if black scale this is the solution
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
        # if self.is_train:
        label = 'good'
        return image, label
        # else:
        #     if self.config.data.mask:
        #         if os.path.dirname(image_file).endswith("good"):
        #             target = torch.zeros([1, image.shape[-2], image.shape[-1]])
        #             label = 'good'
        #         else :
        #             if self.config.data.name == 'MVTec':
        #                 target = Image.open(
        #                     image_file.replace("/test/", "/ground_truth/").replace(
        #                         ".png", "_mask.png"
        #                     )
        #                 )
        #             else:
        #                 target = Image.open(
        #                     image_file.replace("/test/", "/ground_truth/"))
        #             target = self.mask_transform(target)
        #             label = 'defective'
            # else:
            #     if os.path.dirname(image_file).endswith("good"):
            #         target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            #         label = 'good'
            #     else :
            #         target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            #         label = 'defective'
                
            # return image, target, label

    def __len__(self):
        return len(self.image_files)


class Dataset_maker_test(torch.utils.data.Dataset):
    def __init__(self, root, config):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(),  # Scales data into [0,1]
                transforms.Lambda(scale_to_minus_one_one)  # Scale between [-1, 1]
            ]
        )
        self.config = config
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(),
                # Scales data into [0,1]
            ]
        )
        root_mask = f"{root}_mask"

        # if is_train:
        root = root.replace("\\", "\\\\")

        self.image_files = glob(
            os.path.join(root, "*.bmp")
        )
        self.image_files_masks = glob(
            os.path.join(root_mask, "*.png")
        )

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_mask_file = self.image_files_masks[index]
        image = Image.open(image_file)
        mask = Image.open(image_mask_file).convert('L')
        mask_np = np.array(mask)
        binarized_mask_np = ((mask_np > 0).astype(np.uint8))
        mask_unormalized = (binarized_mask_np*255).astype(np.uint8)
        mask = Image.fromarray(mask_unormalized)
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        max_value = np.max(binarized_mask_np)
        if max_value == 0:
            label = 'good'
        elif max_value == 1:
            label = 'defective'
        else:
            raise ValueError(f"Unexpected maximum value in the binarized mask: {max_value} {binarized_mask_np.dtype}")
        return image, mask, label

    def __len__(self):
        return len(self.image_files)
