import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_most_accurate_obj(img_gray, template):
    height, width = template.shape

    # Minimum Square Difference (TM_SQDIFF) because we are looking for the min difference between the template image and the source image.
    res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)

    # Plot minimum difference map
    # plt.imshow(res, cmap='gray')
    # plt.show()

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = min_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)

    cv2.rectangle(img_rgb, top_left, bottom_right, (255, 0, 0), 2)
    cv2.waitKey()
    cv2.destroyAllWindows()


def detect_multiple_obj(img_gray, template, threshold=0.5):
    height, width = template.shape

    # TM_CCOEFF_NORMED because we need to get the maximum values, not the minimum values
    # larger values means good fit
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + width, pt[1] + height), (255, 0, 0), 1) 

    cv2.imshow("Matched image", img_rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()


# SWEET HOMEWORK


def multiscale_detect_multiple_obj(img_gray, template):
    # HINTS:
    # 1. LOOP over the input image at different scales
    # 2. APPLY matchtemplate    
    # Draw the corresponding bounding box on the inital image
    return


if __name__ == '__main__':
    img_rgb = cv2.imread('SourceIMG.jpeg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('TemplateIMG.jpeg', 0)

    # detect_most_accurate_obj(img_gray, template)
    detect_multiple_obj(img_gray, template, threshold=0.5)

