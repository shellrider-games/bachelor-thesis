from skeletonization.skeletonizer import Skeletonizer
import numpy as np
import cv2

def create_skeleton(img, segmented_img):
    img = ~img
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    skeletonizer = Skeletonizer(img)
    skeleton = skeletonizer.skeletonize(30,3,10)
    segmented_img = segmented_img.astype(np.uint8)
    segmented_img = cv2.resize(segmented_img,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
    skeleton.identify(segmented_img)

    return skeleton