import os

import numpy as np
import torch
from PIL import Image


class SketchDataset(object):
    """
    @brief Dataset used for loading sketches for learning to detect
    sketches on images.
    """

    def __init__(self, root, transforms):
        """
        @brief Initializes the dataset.
        @param root Root directory containing images and image masks
        @param transforms Contains transformations
        """
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        """
        @brief Loads the image associated with the given index
        adds labels, masks and bounding boxes as tensors.
        @returns Images (target) and target information
        """
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        mask = np.array(mask)
        mask = mask[:, :, 1]

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        boxes = []
        num_objs = len(obj_ids)

        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            x_min = np.min(pos[1])
            x_max = np.max(pos[1])
            y_min = np.min(pos[0])
            y_max = np.max(pos[0])

            if x_min == x_max or y_min == y_max:
                obj_ids = np.delete(obj_ids, i)
                num_objs = num_objs - 1
            else:
                boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        is_crowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": is_crowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """
        @brief Length of the dataset.
        @returns length
        """
        return len(self.imgs)
