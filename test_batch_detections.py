import torch
import cv2
import numpy as np
import os
import glob
import argparse
from PIL import Image
import torchvision.transforms as transforms
from models.mobilefacenet import MobileFaceNet
from models.pfld_compressed import PFLDInference
from models.basenet import MobileNet_GDConv
from common.utils import BBox, drawLandmark_multiple
from face_parsing import inference as parsing_inference

# 设置参数
parser = argparse.ArgumentParser(description='PyTorch face landmark')
parser.add_argument('--backbone', default='MobileNet',
                    help='choose which backbone network to use: MobileNet, PFLD, MobileFaceNet')
parser.add_argument('--detector', default='MTCNN',
                    help='choose which face detector to use: MTCNN, FaceBoxes, Retinaface')
parser.add_argument('--output_dir', default='results',
                    help='output directory to save results')
args = parser.parse_args()
args.input_dir = 'val'  # 固定使用val2目录

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
map_location = device

def load_model():
    model = None
    checkpoint = None
    
    # 默认使用MobileFaceNet作为backbone
    def load_mobilefacenet():
        nonlocal model, checkpoint
        model = MobileFaceNet([112, 112], 136)   
        checkpoint = torch.load('checkpoint/mobilefacenet_model_best.pth.tar', map_location=map_location)      
        print('Use MobileFaceNet as backbone')
    
    try:
        if args.backbone == 'MobileNet':
            print('MobileNet backbone is not available, fallback to MobileFaceNet')
            load_mobilefacenet()
        elif args.backbone == 'PFLD':
            model = PFLDInference() 
            checkpoint = torch.load('checkpoint/pfld_model_best.pth.tar', map_location=map_location)
            print('Use PFLD as backbone') 
        elif args.backbone == 'MobileFaceNet':
            load_mobilefacenet()
        else:
            print(f'Warning: unsupported backbone "{args.backbone}", fallback to MobileFaceNet')
            load_mobilefacenet()
    except Exception as e:
        print(f'Error loading backbone "{args.backbone}": {e}. Fallback to MobileFaceNet.')
        load_mobilefacenet()

    if model is None or checkpoint is None:
        raise RuntimeError('Failed to initialize model and checkpoint!')

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def init_face_detector():
    if args.detector == 'MTCNN':
        from MTCNN.detector import detect_faces
        return detect_faces
    elif args.detector == 'FaceBoxes':
        from FaceBoxes.FaceBoxes import FaceBoxes
        return FaceBoxes()
    elif args.detector == 'Retinaface':
        from Retinaface.Retinaface import Retinaface
        return Retinaface()
    else:
        print(f'Warning: unsupported detector "{args.detector}", fallback to MTCNN')
        from MTCNN.detector import detect_faces
        return detect_faces

def process_image(image_path, model, face_detector, to_tensor, resize, parsing_model=None, parsing_device=None):
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # 如果分割模型可用，提前计算分割掩码并生成蒙版图像（恢复到原始分辨率）
    seg_mask = None
    seg_vis = None
    if parsing_model is not None:
        try:
            input_batch = parsing_inference.prepare_image(img_pil).to(parsing_device)
            with torch.no_grad():
                out = parsing_model(input_batch)[0]
            pred_mask = out.squeeze(0).cpu().numpy().argmax(0)
            mask_pil = Image.fromarray(pred_mask.astype(np.uint8))
            # resize back to original image size (width, height)
            restored_mask = mask_pil.resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
            seg_mask = np.array(restored_mask)
            
            # 生成分割可视化图像，使用颜色映射显示不同区域
            # 创建颜色遮罩
            segmentation_color = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
            
            # 颜色映射表（参考face_parsing/utils/common.py）
            COLOR_LIST = [
                [0, 0, 0],      # 背景
                [255, 85, 0],   # 皮肤
                [255, 170, 0],  # 左眉
                [255, 0, 85],   # 右眉
                [255, 0, 170],  # 左眼
                [0, 255, 0],    # 右眼
                [85, 255, 0],   # 眼镜
                [170, 255, 0],  # 左耳
                [0, 255, 85],   # 右耳
                [0, 255, 170],  # 耳环
                [0, 0, 255],    # 鼻子
                [85, 0, 255],   # 嘴
                [170, 0, 255],  # 上唇
                [0, 85, 255],   # 下唇
                [0, 170, 255],  # 颈部
                [255, 255, 0],  # 颈饰
                [255, 255, 85], # 衣服
                [255, 255, 170],# 头发
                [255, 0, 255],  # 帽子
            ]
            
            # 对每个类别应用颜色
            num_classes = len(COLOR_LIST) - 1
            for class_idx in range(num_classes + 1):
                indices = seg_mask == class_idx
                segmentation_color[indices] = COLOR_LIST[class_idx]
            
            # 将遮罩与原图混合
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            seg_vis = cv2.addWeighted(img_bgr, 0.6, segmentation_color, 0.4, 0)
        except Exception as e:
            print(f"警告：分割模型推理失败，继续不使用分割：{e}")
            seg_mask = None

    # 人脸检测
    if args.detector == 'MTCNN':
        boxes, landmarks = face_detector(img_pil)
        if len(boxes) == 0:
            print(f"未检测到人脸: {image_path}")
            return None
        boxes = boxes[:, 0:4].astype(np.int32)
    else:
        boxes = face_detector(img)
        if len(boxes) == 0:
            print(f"未检测到人脸: {image_path}")
            return None
        boxes = np.array([box[:4] for box in boxes]).astype(np.int32)

    result_img = img.copy()
    landmarks_all = []

    # 对每个检测到的人脸进行关键点检测
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(max([w, h])*1.1)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        # 处理边界情况
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)
        edx = max(0, x2 - img.shape[1])
        edy = max(0, y2 - img.shape[0])
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)
        cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]

        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)

        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped = Image.fromarray(cropped)
        test_face = resize(cropped)
        test_face = to_tensor(test_face)
        test_face.unsqueeze_(0)
        test_face = test_face.to(device)

        # 关键点检测
        with torch.no_grad():
            landmarks = model(test_face)
            if isinstance(landmarks, tuple):
                landmarks = landmarks[0]  # 如果模型返回tuple，取第一个元素
        landmarks = landmarks.cpu().numpy()
        landmarks = landmarks.reshape(-1, 2)
        landmarks = new_bbox.reprojectLandmark(landmarks)

        # 先尝试使用分割掩码：以第9个点为起点向上搜索掩码中的第一个像素作为额头点
        forehead_point = None
        try:
            if seg_mask is not None:
                x0 = int(round(landmarks[8][0]))
                y0 = int(round(landmarks[8][1]))
                x0 = np.clip(x0, 0, seg_mask.shape[1]-1)
                y0 = np.clip(y0, 0, seg_mask.shape[0]-1)
                mask_face = (seg_mask > 0)
                # 从 y0 向上搜索
                for yy in range(y0, -1, -1):
                    if mask_face[yy, x0]:
                        forehead_point = np.array([x0, yy], dtype=np.float32)
                        break
        except Exception:
            forehead_point = None

        # 备用：基于延长线交点的方法
        def get_line_intersection(p1, p2, p3, p4):
            """计算两条直线的交点"""
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:  # 平行线
                return None

            px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom

            return np.array([px, py])

        def extend_line(p1, p2, factor=2.0):
            """延长线段"""
            direction = p2 - p1
            return p2 + direction * factor

        if forehead_point is None:
            # 计算第一条线 l1（点2与点19、20的中点连线）
            p2 = landmarks[1]  # 点2 (索引1对应标号2)
            mid_point1 = (landmarks[18] + landmarks[19]) / 2  # 点19和20的中点
            l1_extended = extend_line(p2, mid_point1)

            # 计算第二条线 l2（点16与点24、25的中点连线）
            p16 = landmarks[15]  # 点16 (索引15对应标号16)
            mid_point2 = (landmarks[23] + landmarks[24]) / 2  # 点24和25的中点
            l2_extended = extend_line(p16, mid_point2)

            # 计算两条延长线的交点作为新的前额点
            forehead_point = get_line_intersection(p2, l1_extended, p16, l2_extended)

            # 如果没有找到有效的交点，使用两条延长线的中点作为后备
            if forehead_point is None:
                print("警告：无法计算延长线交点，使用备用方法计算前额点")
                forehead_point = (l1_extended + l2_extended) / 2

        # 将前额点添加到landmarks中
        landmarks = np.vstack([landmarks, forehead_point])
        landmarks_all.append(landmarks)

        # 在分割可视化图像上绘制关键点
        if seg_vis is not None:
            result_img = seg_vis.copy()  # 使用带紫色蒙版的图像
        
        # 绘制关键点
        result_img = drawLandmark_multiple(result_img, new_bbox, landmarks)

    return result_img

if __name__ == '__main__':
    try:
        # 初始化模型和预处理
        model = load_model()
        model = model.to(device)
        model.eval()
        print("模型加载成功！")
        
        # 初始化人脸检测器
        face_detector = init_face_detector()
        print(f"使用 {args.detector} 作为人脸检测器")

        # 尝试加载 face-parsing 分割模型（可选）
        parsing_model = None
        parsing_device = device
        try:
            parsing_model = parsing_inference.load_model('resnet18', 19, 'face_parsing/weights/resnet18.pt', parsing_device)
            parsing_model.eval()
            print('face-parsing 模型加载成功')
        except Exception as e:
            print(f'警告：无法加载 face-parsing 模型，继续不使用分割：{e}')
        
        # 设置图像预处理
        resize = transforms.Resize([112, 112])
        to_tensor = transforms.ToTensor()
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 获取输入目录中的所有图片
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(glob.glob(os.path.join(args.input_dir, f'*.{ext}')))
        
        if not image_files:
            print(f"在{args.input_dir}目录中未找到图片文件")
            exit(1)
            
        print(f"找到{len(image_files)}张图片")
        
        # 处理每张图片
        for image_path in image_files:
            print(f"处理图片: {image_path}")
            result_img = process_image(image_path, model, face_detector, to_tensor, resize, parsing_model, parsing_device)
            
            if result_img is not None:
                # 保存结果
                output_path = os.path.join(args.output_dir, os.path.basename(image_path))
                cv2.imwrite(output_path, result_img)
                print(f"结果已保存到: {output_path}")
        
        print("批量处理完成！")
        
    except Exception as e:
        print(f"错误: {str(e)}")
