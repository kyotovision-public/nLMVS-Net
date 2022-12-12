import sys
import numpy as np
import cv2

argv = sys.argv

assert len(argv) == 4

in_file_1 = argv[1]
in_file_2 = argv[2]
out_file = argv[3]

img_1 = cv2.imread(in_file_1, -1)
img_2 = cv2.imread(in_file_2, -1)
log_img_1 = np.log1p(1000 * np.clip(img_1, 0, None))
log_img_2 = np.log1p(1000 * np.clip(img_2, 0, None))
error_map = np.mean(np.abs(log_img_1 - log_img_2), axis=-1)
error_map_gray = (255 * np.clip(error_map / np.log(1.5), 0, 1)).astype(np.uint8)
cmap = cv2.applyColorMap(error_map_gray, cv2.COLORMAP_JET)
cv2.imwrite(out_file, cmap)