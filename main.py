from ultralytics import YOLO
from ultralytics import RTDETR
from our_wbf import weighted_boxes_fusion1
import numpy as np
import os
import cv2

def ComputeHist(img):
    h, w = img.shape
    hist, bin_edge = np.histogram(img.reshape(1, w * h), bins=list(range(257)))
    return hist
def ComputeMinLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[i]
        if (sum >= (pnum * rate * 0.01)):
            return i
def ComputeMaxLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[255 - i]
        if (sum >= (pnum * rate * 0.01)):
            return 255 - i
def LinearMap(minlevel, maxlevel):
    if (minlevel >= maxlevel):
        return []
    else:
        newmap = np.zeros(256)
        for i in range(256):
            if (i < minlevel):
                newmap[i] = 0
            elif (i > maxlevel):
                newmap[i] = 255
            else:
                newmap[i] = (i - minlevel) / (maxlevel - minlevel) * 255
        return newmap
def CreateNewImg(img):
    h, w, d = img.shape
    newimg = np.zeros([h, w, d])
    for i in range(d):
        imghist = ComputeHist(img[:, :, i])
        minlevel = ComputeMinLevel(imghist, 8.3, h * w)
        maxlevel = ComputeMaxLevel(imghist, 2.2, h * w)
        newmap = LinearMap(minlevel, maxlevel)
        if (newmap.size == 0):
            continue
        for j in range(h):
            newimg[j, :, i] = newmap[img[j, :, i]]
    return newimg
def ProcessImages(input_folder, output_folder):
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print('defogging')
    for file_name in image_files:
        img = cv2.imread(os.path.join(input_folder, file_name))
        if img is not None:
            new_img = CreateNewImg(img)
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, new_img)
    print('defogged')

def run():
    # 1.参数设置------------------------------------------------------------------------
    source='' # 待推理图片文件夹
    source_cw='' # 存放数据增强的图片，无需手动创建，运行时自动创建，只需要指定位置
    os.makedirs(source_cw, exist_ok=True)
    src='runs/detect/'  # 单模型推理后存放的主目录
    save='output/'  # 保存最后txt的目录，自动创建
    ProcessImages(source,source_cw)
    iou_thr=0.3  # 集成预测的iou
    single_conf=0.1  # 单模型推理的conf
    pe_conf=0.06  # 集成预测后生成最后txt时的conf
    skip_box_thr=0.35  # 集成预测时的对于每一个单模型的conf
    # 2.RTDETR推理--------------------------------------------------------------------
    lstrt=['model/B1_/weights/best.pt',
           'model/B2_/weights/best.pt',
           'model/B3_/weights/best.pt',
           'model/B3x_/weights/best.pt',
           'model/B3x_2/weights/best.pt', #
           'model/B4_/weights/best.pt',
           'model/B4x_/weights/best.pt',
           'model/B5_/weights/best.pt',
           'model/B58_/weights/best.pt',
           'model/B68_/weights/best.pt', #
           'model/B11_7_/weights/best.pt',
           'model/hhhhh1_/weights/best.pt',
           'model/hhhhh3_/weights/best.pt',
           'model/hhhhh4_/weights/best.pt',
           'model/hhhhh5_/weights/best.pt', #
           'model/hf12_/weights/best.pt',
           'model/hf14_/weights/best.pt',
           'model/hf32_/weights/best.pt',
           'model/hf34_/weights/best.pt',
           'model/hf42_/weights/best.pt', #
           'model/nice1/weights/best.pt',
           'model/nice2/weights/best.pt',
           'model/nice3/weights/best.pt',
           'model/nice8/weights/best.pt',
           'model/train64/weights/best.pt',#
           'model/train69/weights/best.pt',
           'model/train88/weights/best.pt'] # 27
    lstrt_cw=['model/B2_/weights/best.pt',
              'model/B4_/weights/best.pt',
              'model/B4x_/weights/best.pt',
              'model/B58_/weights/best.pt',
              'model/B68_/weights/best.pt',
              'model/hf12_/weights/best.pt',
              'model/hf14_/weights/best.pt',
              'model/hf32_/weights/best.pt',
              'model/hf34_/weights/best.pt',
              'model/hf42_/weights/best.pt',]
    lstrt_zq=['model/train64/weights/best.pt',
             'model/train69/weights/best.pt',]
    print('inferring')
    for i in range(len(lstrt)):
        model=RTDETR(lstrt[i])
        if lstrt[i] in lstrt_cw:
            _=model(source=source_cw,save_conf=True,save_txt=True,
                     name=f'pe{i+1}',conf=single_conf)
        elif lstrt[i] in lstrt_zq:
            _=model(source=source_cw,save_conf=True,save_txt=True,
                     name=f'pe{i+1}',conf=single_conf,augment=True)
        else:
            _=model(source=source,save_conf=True,save_txt=True,
                     name=f'pe{i+1}',conf=single_conf)
    # 3.YOLO推理-----------------------------------------------------------------------
    lstyolo=['model/yolo1/weights/best.pt',
            'model/yolo3/weights/best.pt',
            'model/yolo4/weights/best.pt', # 30
            'model/yolo5/weights/best.pt',
            'model/A3_/weights/best.pt',
            'model/A3x_/weights/best.pt',
            'model/A4x_/weights/best.pt',
            'model/nice7/weights/best.pt', #
            'model/nice9/weights/best.pt',
            'model/train78/weights/best.pt',
            'model/train81/weights/best.pt',
            'model/train86/weights/best.pt',
            'model/train89/weights/best.pt',#
            'model/train96/weights/best.pt',
            'model/train699/weights/best.pt',
            'model/train888/weights/best.pt'] # 43
    lstyolo_cw=['model/A4x_/weights/best.pt']
    lstyolo_zq=['model/nice9/weights/best.pt']
    for i in range(len(lstrt),len(lstrt)+len(lstyolo)):
        model=YOLO(lstyolo[i-len(lstrt)])
        if lstyolo[i-len(lstrt)] in lstyolo_cw:
            _=model(source=source_cw,save_conf=True,save_txt=True,
                     name=f'pe{i+1}',conf=single_conf)
        elif lstyolo[i-len(lstrt)] in lstyolo_zq:
            _=model(source=source_cw,save_conf=True,save_txt=True,
                     name=f'pe{i+1}',conf=single_conf,augment=True)
        else:
            _=model(source=source,save_conf=True,save_txt=True,
                     name=f'pe{i+1}',conf=single_conf)
    print('inferred')
    # 4.权重--------------------------------------------------------------------------
    # 子模型权重矩阵，若不对类别进行单独赋权，使用向量权重，将下面的WBF函数后面的1去掉
    # 若对类别进行单独赋权，使用矩阵权重，将下面的WBF函数后面的1加上
    #weights=[1]*43
    weights=[[1,1,1,1,1,1,1,1],#B1_
             [1,1,1,1,1,1,1,1],#B2_
             [1,1,1,1,1,1,1,2],#B3_
             [1,1,1,1,1,1,1,2],#B3x_
             [1,1,1,1,1,1,1,2],#B3x_2
             [1,1,1,1,1,1,1,1],#B4_
             [1,1,1,1,1,1,1,2],#B4x_
             [1,1,1,1,1,1,1,2],#B5_
             [1,1,1,1,1,1,1,1],#B58_
             [1,1,1,1,1,1,1,2],#B68_
             [1,1,1,1,1,1,1,2],#B11_7_
             [1,1,1,1,1,1,1,2],#hhhhh1_
             [1,1,1,1,1,1,1,2],#hhhhh3_
             [1,1,1,1,1,1,1,1],#hhhhh4_
             [1,1,1,1,1,1,1,1],#hhhhh5_
             [1,1,1,1,1,1,1,1],#hf12_
             [1,1,1,1,1,1,1,1],#hf14_
             [1,1,1,1,1,1,1,1],#hf32_
             [1,1,1,1,1,1,1,2],#hf34_
             [1,1,1,1,1,1,1,2],#hf42_
             [1,1,1,1,1,1,1,1],#nice1
             [1,1,1,1,1,1,1,1],#nice2
             [1,1,1,1,1,1,1,1],#nice3
             [1,1,1,1,1,1,1,1],#nice8
             [1,1,1,1,1,1,1,1],#train64
             [1,1,1,1,1,1,1,1],#train69
             [1,1,1,1,1,1,1,1],#train88####
             [1,1,1,1,1,1,1,1],#yolo1
             [1,1,1,1,1,1,1,2],#yolo3
             [1,1,1,1,1,1,1,1],#yolo4
             [1,1,1,1,1,1,1,2],#yolo5
             [1,1,1,1,1,1,1,1],#A3_
             [1,1,1,1,1,1,1,1],#A3x_
             [1,1,1,1,1,1,1,2],#A4x_
             [1,1,1,1,1,1,1,1],#nice7
             [1,1,1,1,1,1,1,1],#nice9
             [1,1,1,1,1,1,1,1],#train78
             [1,1,1,1,1,1,1,1],#train81
             [1,1,1,1,1,1,1,1],#train86
             [1,1,1,1,1,1,1,1],#train89
             [1,1,1,1,1,1,1,2],#train96
             [1,1,1,1,1,1,1,1],#train699
             [1,1,1,1,1,1,1,1]]#train888
    n=len(weights)
    print('ensemble start')
    # 5.集成预测---------------------------------------------------------------------
    names=list(map(lambda x:x.split('.')[0],os.listdir(source)))
    pe_lst=[f'pe{i}' for i in range(1,n+1)]
    for name in names:
        labels_list,boxes_list,scores_list=[],[],[]
        for i in range(n):
            model_labels,model_boxes,model_scores=[],[],[]
            if os.path.exists(src+pe_lst[i]+'/labels/'+name+".txt"):
                with open(src+pe_lst[i]+'/labels/'+name+".txt", 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        line=line.split()
                        model_labels.append(int(line[0]))
                        x_center, y_center, width, height = map(float,line[1:5])
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        model_boxes.append([x1, y1, x2, y2])
                        model_scores.append(float(line[5]))
            labels_list.append(model_labels)
            boxes_list.append(model_boxes)
            scores_list.append(model_scores)
        boxes, scores, labels = weighted_boxes_fusion1(boxes_list,scores_list,labels_list,
            weights=weights,iou_thr=iou_thr,skip_box_thr=skip_box_thr)
        os.makedirs(save, exist_ok=True)
        with open(save + name + ".txt", 'w') as file:
            for j in range(len(scores)):
                if scores[j]>pe_conf:
                    box=boxes[j]
                    box_xywh=[]
                    box_xywh.append((box[0]+box[2])/2)
                    box_xywh.append((box[1]+box[3])/2)
                    box_xywh.append(box[2]-box[0])
                    box_xywh.append(box[3]-box[1])
                    file.write(" ".join([str(num) for num in [int(labels[j])]+box_xywh]) + "\n")
    print('ensemble done')

if __name__=='__main__':
    run()
