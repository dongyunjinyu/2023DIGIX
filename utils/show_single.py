import os
import cv2
# 展示推理结果
images_folder = '../dataseta/jpgs/'
labels_folder = '../dataseta/43/t4pe0/'
# images_folder = 'dataset1/images/'

# labels_folder = 'dataset1/labels/'
image_files = sorted(os.listdir(images_folder))
label_files = sorted(os.listdir(labels_folder))


for image_file, label_file in zip(image_files, label_files):
    image_path = os.path.join(images_folder, image_file)
    ww, hh = 1000, 640
    image = cv2.imread(image_path)
    image = cv2.resize(image, (ww,hh))
    try :
        with open(labels_folder+image_file.split('.')[0]+'.txt', 'r') as f:
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
            # cls=['横向裂缝','纵向裂缝','块状裂缝','龟裂','坑槽','修补网状裂缝','修补裂缝','修补坑槽']
            colorlst=[(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255),(255,255,0),(255,255,255),(0,0,0)]
            # colorlst=[(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255),(255,255,0),(0,255,255),(0,0,0)]
            cv2.rectangle(image, (x, y), (x + w, y + h), colorlst[class_id], 4)
            cv2.putText(image, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # 显示置信度
            #cv2.putText(image, "{:.2f}".format(float(parts[5])), (x+50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    except FileNotFoundError:
        print('FileNotFoundError:',labels_folder+image_file.split('.')[0]+'.txt')
        pass
    cv2.namedWindow(f'{image_file}', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'{image_file}', 1530, 830)
    cv2.moveWindow(f'{image_file}', 0, 0)
    cv2.imshow(f'{image_file}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
