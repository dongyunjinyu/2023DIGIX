import os

# 根据逻辑来排除一些类别的框
def area(lst1,lst2):
    x1, y1, w1, h1 = lst1
    x2, y2, w2, h2 = lst2
    x1_min = x1 - w1 / 2
    y1_min = y1 - h1 / 2
    x1_max = x1 + w1 / 2
    y1_max = y1 + h1 / 2
    x2_min = x2 - w2 / 2
    y2_min = y2 - h2 / 2
    x2_max = x2 + w2 / 2
    y2_max = y2 + h2 / 2
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    overlap_w = max(0, x_max - x_min)
    overlap_h = max(0, y_max - y_min)
    overlap_area = overlap_w * overlap_h
    return overlap_area
src='dataseta/6/t6pe0/'
save='dataseta/6/result/'
os.makedirs(save, exist_ok=True)
label_files = os.listdir(src)
area_thres=0.8

for filename in label_files:
    with open(src+filename, 'r') as f:
        lines = f.readlines()
    lst=[] # 所有行
    for line in lines:
        alst=list(map(float,line.strip().split(' ')))
        alst[0]=int(alst[0])
        lst.append(alst)
    index=[] # 块状的索引
    for i in range(len(lst)):
        if lst[i][0]==2:
            index.append(i)
    yizhi_lst=[] # 会被抑制的索引
    try:
        if index: # 如果存在块状
            max_i,max_area=0,0
            for i in index: # 找到最大的块状
                if lst[i][3] * lst[i][4] > max_area:
                    max_i=i
                    max_area=lst[i][3] * lst[i][4]
            b=True
            for i in range(len(lst)):
                if lst[i][0] in [0, 1]:
                    if lst[i][5]>lst[max_i][5] and area(lst[i][1:5], lst[max_i][1:5]) / (lst[i][3] * lst[i][4]) > area_thres:
                        b=False
            if b:
                for i in range(len(lst)):
                    if lst[i][0] in [0, 1]:
                        if area(lst[i][1:5], lst[max_i][1:5]) / (lst[i][3] * lst[i][4]) > area_thres:
                            yizhi_lst.append(i)
    except:
        pass
    with open(save + filename, 'w') as file:
        for i in range(len(lst)):
            if i not in yizhi_lst:
                file.write(" ".join([str(num) for num in lst[i][:5]]) + "\n")



