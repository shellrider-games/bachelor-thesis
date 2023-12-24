import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class DatasetPreparator(object):
    """
    @brief Class enabling the generation of training data based on the annotated
    SketchParse dataset.
    """

    def __init__(self, root, save_to):
        """
        @brief Constructor taking the root path that contains the subfolders 'images' and 'masks', see SketchParse
        dataset.
        """
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
        self.big_imgs = os.path.join(save_to, "images")
        self.big_masks = os.path.join(save_to, "masks")

    def prepare_data(self):
        """
        @brief Prepares the data and saves it to the given path to the subfolders "images" and "masks" respectively.
        @returns None
        """
        for idx in range(0, len(self.imgs) - 5, 5):
            num_objects = random.randint(2, 5)
            big_img = np.zeros([400, 800, 3])
            big_img[:, :, :] = 255
            big_mask = np.zeros([400, 1600]).astype(int)
            for obj_id in range(1, num_objects):
                x = random.randint(-obj_id, len(self.imgs))

                img_path = os.path.join(self.root, "images", self.imgs[np.minimum(idx + x, len(self.imgs) - 1)])
                mask_path = os.path.join(self.root, "masks", self.masks[np.minimum(idx + x, len(self.imgs) - 1)])
                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path)
                mask = np.array(mask)
                mask = (mask > 0).astype(int)
                mask[mask > 0] = obj_id

                rand_x = np.random.randint(np.size(big_img, 0) - np.size(img, 0))
                rand_y = np.random.randint(np.size(big_img, 1) - np.size(img, 1))

                big_img[rand_x:np.size(img, 0) + rand_x, rand_y:np.size(img, 1) + rand_y, :] *= img
                big_mask[rand_x:np.size(mask, 0) + rand_x, rand_y:np.size(mask, 1) + rand_y] = \
                    np.maximum(mask, big_mask[rand_x:np.size(mask, 0) + rand_x, rand_y:np.size(mask, 1) + rand_y])

            fig = plt.figure()
            plt.axis('off')
            plt.imshow(big_img)
            fig.savefig(self.big_imgs + "/image_" + str(idx) + ".jpg", bbox_inches='tight', transparent=True,
                        pad_inches=0)
            plt.close(fig)
            fig = plt.figure()
            plt.axis('off')
            plt.imshow(big_mask, cmap='gray', vmin=0, vmax=255)
            fig.savefig(self.big_masks + "/image_" + str(idx) + ".jpg", bbox_inches='tight', transparent=True,
                        pad_inches=0)
            plt.close(fig)


if __name__ == "__main__":
    dataset = DatasetPreparator("D:/git/bachelor-thesis/datasets/sketch-parse/", "D:/git/bachelor-thesis/datasets/sketch-parse/detection")
    dataset.prepare_data()
