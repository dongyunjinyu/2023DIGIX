import os
from PIL import Image

source_folder_image = ''
source_folder_labels = ''
files = os.listdir(source_folder_labels)
names = list(set(map(lambda x: x.split('.')[0], files)))

a = ''
b = ''
os.makedirs(a, exist_ok=True)
os.makedirs(b, exist_ok=True)
count = 0

# 对部分类别进行数据增广
for name in names:
    with open(source_folder_labels + '/' + name + '.txt', 'r') as label_file:
        labels = label_file.readlines()
        for label in labels:
            category = int(label.split()[0])
            image_path = source_folder_image + '/' + name + '.jpg'
            image = Image.open(image_path)
            if category in [0, 2, 3, 4, 5, 7]:
                count += 1
                # 进行左右翻转
                flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_image.save(a + '/' + name + '_fz.jpg')
                # 更新标签文件的文件名和x_center
                flipped_labels = []
                for label in labels:
                    parts = label.split()
                    category = int(parts[0])
                    x_center = float(parts[1])
                    if category in [0, 1, 2, 3, 4, 5, 6, 7]:
                        x_center_flipped = 1 - x_center
                        parts[1] = str(x_center_flipped)
                        flipped_labels.append(' '.join(parts) + '\n')
                    else:
                        flipped_labels.append(label)
                flipped_labels_path = b + '/' + name + '_fz.txt'
                with open(flipped_labels_path, 'w') as flipped_labels_file:
                    flipped_labels_file.writelines(flipped_labels)
                break
print(count)