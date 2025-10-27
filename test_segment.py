import cv2
import mediapipe as mp
import numpy as np

# 初始化分割模型
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# 读取图像
image = cv2.imread("val/161-11.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取分割掩码
results = segmentation.process(image_rgb)
mask = results.segmentation_mask  # float32, [0,1]

# 二值化
mask_binary = (mask > 0.5).astype(np.uint8)

# 提取最上方点
ys, xs = np.where(mask_binary==1)
forehead_point = (int(xs.mean()), int(ys.min()))  # x 平均值，y 最小

# 可视化
cv2.circle(image, forehead_point, 3, (0,0,255), -1)
cv2.imshow("Forehead", image)
cv2.waitKey(0)
cv2.destroyAllWindows()