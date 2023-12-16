import os
import shutil
import random
import cv2
import numpy as np
# 数据集处理工具


def f10():
    """根据label筛选"""
    source_folder = 'dataseta/jpgs/'
    images_folder = 'dataseta/afeter_images/'
    labels_folder = 'dataseta/labels/'
    os.makedirs(images_folder, exist_ok=True)
    for name in os.listdir(labels_folder):
        shutil.copy(source_folder+name.split('.')[0]+'.jpg', images_folder)
    print('done')
f10()


def f1():
    """将原始数据集中的txt和jpg文件复制到dataset1"""
    source_folder = 'jsai_data'
    images_folder = 'dataset1/images'
    labels_folder = 'dataset1/labels'
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    for name in os.listdir(source_folder):
        if name.endswith('.txt'):
            source_file = os.path.join(source_folder, name)
            destination_file = os.path.join(labels_folder, name)
            shutil.copy(source_file, destination_file)
        if name.endswith('.jpg'):
            source_file = os.path.join(source_folder, name)
            destination_file = os.path.join(images_folder, name)
            shutil.copy(source_file, destination_file)
    print('done')

def f2():
    """划分训练集和验证集为yolo格式到dataset2"""
    source_folder = 'jsai_data'
    a = 'dataset2/train/images'
    b = 'dataset2/train/labels'
    c = 'dataset2/val/images'
    d = 'dataset2/val/labels'
    for i in [a,b,c,d]:
        os.makedirs(i, exist_ok=True)
    files = os.listdir(source_folder)
    names=list(set(map(lambda x:x.split('.')[0],files)))
    train_names = random.sample(names, k=int(len(names) * 0.8))
    val_names = [x for x in names if x not in train_names]
    for name in train_names:
        shutil.copy(source_folder+'/'+name+'.jpg', a+'/'+name+'.jpg')
        shutil.copy(source_folder+'/'+name+'.txt', b+'/'+name+'.txt')
    for name in val_names:
        shutil.copy(source_folder + '/' + name + '.jpg', c + '/' + name + '.jpg')
        shutil.copy(source_folder + '/' + name + '.txt', d + '/' + name + '.txt')
    print('done')

def f3():
    """检查"""
    a = 'dataset2/train/images'
    b = 'dataset2/train/labels'
    c = 'dataset2/val/images'
    d = 'dataset2/val/labels'
    e = 'dataset1/images'
    f = 'dataset1/labels'
    x = 'jsai_data'
    for i in [a,b,c,d,e,f,x]:
        files = os.listdir(i)
        print(len(files))



def f4():
    """归类"""
    source_folder = 'jsai_data'
    a = 'dataset3/train/images'
    b = 'dataset3/train/labels'
    c = 'dataset3/val/images'
    d = 'dataset3/val/labels'
    for i in [a,b,c,d]:
        os.makedirs(i, exist_ok=True)
    files = os.listdir(source_folder)
    names=list(set(map(lambda x:x.split('.')[0],files)))
    train_names = random.sample(names, k=int(len(names) * 0.8))
    val_names =  [x for x in names if x not in train_names]
    for name in names:
        with open(source_folder+'/'+name+".txt", 'r') as file:
            lines = file.readlines()
        modified_lines = []
        for line in lines:
            if line.strip():  # 忽略空行
                modified_line = '0' + line[1:]
                modified_lines.append(modified_line)
        if name in train_names:
            with open(b + '/' + name+".txt", 'w') as file:
                file.writelines(modified_lines)
        elif name in val_names:
            with open(d + '/' + name+".txt", 'w') as file:
                file.writelines(modified_lines)
    for name in train_names:
        shutil.copy(source_folder+'/'+name+'.jpg', a+'/'+name+'.jpg')
    for name in val_names:
        shutil.copy(source_folder + '/' + name + '.jpg', c + '/' + name + '.jpg')
    print('done')



def f5():
    """预处理后划分训练集和验证集为yolo格式到dataset4"""
    def process(image):
        """数据增强  -image:图片路径  -return:处理后的图片"""
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)  # 增加对比度
        image = cv2.equalizeHist(image)  # 直方图亮度均衡
        kernel = np.array([[-2,-1,-2],
                           [-1, 15,-2],
                           [-2,-1,-2]])  # 神奇的滤波核，可操作
        image = cv2.filter2D(image, -1, kernel)  # 锐化
        image = cv2.medianBlur(image, 3)  # 中值滤波
        # 进行高斯滤波,不如中值滤波
        # image = cv2.GaussianBlur(image, (3, 3), 1)
        return image
    source_folder = 'jsai_data'
    a = 'dataset4/train/images'
    b = 'dataset4/train/labels'
    c = 'dataset4/val/images'
    d = 'dataset4/val/labels'
    for i in [a,b,c,d]:
        os.makedirs(i, exist_ok=True)
    files = os.listdir(source_folder)
    names=list(set(map(lambda x:x.split('.')[0],files)))
    train_names = random.sample(names, k=int(len(names) * 0.8))
    val_names = [x for x in names if x not in train_names]
    for name in train_names:
        image=process(source_folder + '/' + name + '.jpg')
        cv2.imwrite(a + '/' + name + '.jpg',image)
        shutil.copy(source_folder+'/'+name+'.txt', b+'/'+name+'.txt')
        print(name)
    for name in val_names:
        image = process(source_folder + '/' + name + '.jpg')
        cv2.imwrite(c + '/' + name + '.jpg',image)
        shutil.copy(source_folder + '/' + name + '.txt', d + '/' + name + '.txt')
        print(name)
    print('done')


def f6():
    """拆分学习，数据集构建"""
    source_folder = 'jsai_data'
    a1 = 'dataset5/detect/train/images'
    b1 = 'dataset5/detect/train/labels'
    c1 = 'dataset5/detect/val/images'
    d1 = 'dataset5/detect/val/labels'
    a2 = 'dataset5/classify/train/images'
    b2 = 'dataset5/classify/train/labels'
    c2 = 'dataset5/classify/val/images'
    d2 = 'dataset5/classify/val/labels'
    for i in [a1, b1, c1, d1, a2, b2, c2, d2]:
        os.makedirs(i, exist_ok=True)
    files = os.listdir(source_folder)
    # detect todo
    names = list(set(map(lambda x: x.split('.')[0], files)))
    train_names = random.sample(names, k=int(len(names) * 0.8))
    val_names = [x for x in names if x not in train_names]
    for name in names:
        with open(source_folder + '/' + name + ".txt", 'r') as file:
            lines = file.readlines()
        modified_lines = []
        for line in lines:
            if line.strip():  # 忽略空行
                if line[0] in {'0','1','3','6'}:
                    modified_line = '0' + line[1:]
                    modified_lines.append(modified_line)
                else:
                    modified_line = line
                    modified_lines.append(modified_line)
        if name in train_names:
            with open(b1 + '/' + name + ".txt", 'w') as file:
                file.writelines(modified_lines)
        elif name in val_names:
            with open(d1 + '/' + name + ".txt", 'w') as file:
                file.writelines(modified_lines)
    for name in train_names:
        shutil.copy(source_folder + '/' + name + '.jpg', a1 + '/' + name + '.jpg')
    for name in val_names:
        shutil.copy(source_folder + '/' + name + '.jpg', c1 + '/' + name + '.jpg')
    print('done')


def f7(aa):
    source_folder = 'jsai_data'
    a = 'dataset6/train/images'
    b = 'dataset6/train/labels'
    c = 'dataset6/val/images'
    d = 'dataset6/val/labels'
    for i in [a, b, c, d]:
        os.makedirs(i, exist_ok=True)
    files = os.listdir(source_folder)
    names = list(set(map(lambda x: x.split('.')[0], files)))
    train_names = random.sample(names, k=int(len(names) * 0.8))
    val_names = [x for x in names if x not in train_names]
    for name in names:
        with open(source_folder + '/' + name + ".txt", 'r') as file:
            lines = file.readlines()
        modified_lines = []
        for line in lines:
            if line.strip():  # 忽略空行
                bb = int(line[0])
                if bb in aa:
                    modified_line = str(aa.index(bb)) + line[1:]
                    modified_lines.append(modified_line)
        if name in train_names:
            with open(b + '/' + name + ".txt", 'w') as file:
                file.writelines(modified_lines)
        elif name in val_names:
            with open(d + '/' + name + ".txt", 'w') as file:
                file.writelines(modified_lines)
    for name in train_names:
        shutil.copy(source_folder + '/' + name + '.jpg', a + '/' + name + '.jpg')
    for name in val_names:
        shutil.copy(source_folder + '/' + name + '.jpg', c + '/' + name + '.jpg')
    print('done')

