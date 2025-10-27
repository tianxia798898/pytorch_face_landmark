import cv2
import mediapipe as mp

# 初始化 FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,     # 静态图片模式
    max_num_faces=1,            # 最大检测人脸数
    refine_landmarks=True,      # 精细标注（眼睛、唇）
    min_detection_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# 读取图片
image = cv2.imread("val/161-11.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 检测
results = face_mesh.process(image_rgb)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # 遍历 468 个关键点
        for idx, lm in enumerate(face_landmarks.landmark):
            x = int(lm.x * image.shape[1])
            y = int(lm.y * image.shape[0])
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

cv2.imshow("FaceMesh", image)
cv2.waitKey(0)
cv2.destroyAllWindows()