import os
import cv2
import numpy as np
# 同时展示集成结果和若干单个模型的结果
# 模型太多看不清
# 按任意键切换下一张

images_folder = 'dataseta/jpgs'
labels_folder = 'dataseta/6/t6pe0/labels'
n=35 # 子模型数
image_files = sorted(os.listdir(images_folder))
label_files = sorted(os.listdir(labels_folder))
labels_folders=[f'dataseta/6/t6pe{i}/labels' for i in range(n)]
for image_file, label_file in zip(image_files, label_files):
    image_path = os.path.join(images_folder, image_file)
    images = []
    ww, hh = 1000, 640
    for i in range(n):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (ww,hh))
        label_path = os.path.join(labels_folders[i], label_file)
        try :
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                x = int((x_center - width / 2) * 1000)
                y = int((y_center - height / 2) * 640)
                w = int(width * 1000)
                h = int(height * 640)
                colorlst=[(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255),(255,255,0),(255,255,255),(0,0,0)]
                cv2.rectangle(image, (x, y), (x + w, y + h), colorlst[class_id], 2)
                cv2.putText(image, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except FileNotFoundError:
            pass
        images.append(image)
    canvas = np.zeros((6 * hh, 6 * ww, 3), dtype=np.uint8)
    for i in range(n):
        row = i // 6
        col = i % 6
        start_x = col * ww
        start_y = row * hh
        canvas[start_y:start_y + hh, start_x:start_x + ww, :] = images[i]
    cv2.namedWindow(f'{image_file}', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'{image_file}', 1530, 830)
    cv2.moveWindow(f'{image_file}', 0, 0)
    cv2.imshow(f'{image_file}', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()













