import cv2
import numpy as np

def image_to_mask(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (15,15), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        1,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        63,
        15
    )
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    h, w = closed.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(closed, mask, (0,0), 255)
    inverted = cv2.bitwise_not(closed)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    largest_contour_image = np.zeros_like(grayscale)
    cv2.drawContours(largest_contour_image, contours, max_index, 255, thickness=cv2.FILLED)
    return largest_contour_image