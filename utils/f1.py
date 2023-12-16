# 计算f1
import numpy as np
import os
def calculate_iou(box1, box2):
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    x_intersection = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_intersection = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    intersection = x_intersection * y_intersection
    union = area_box1 + area_box2 - intersection
    iou = intersection / union
    return iou
def parse_boxes(txt_file):
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            lines = file.readlines()
    else:
        lines=[]
    boxes = []
    for line in lines:
        data = line.strip().split(' ')
        label = int(data[0])
        x_center, y_center, width, height = map(float, data[1:])
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes.append((label, x1, y1, x2, y2))
    return boxes
def calculate_cm(predict_folder, label_folder, nc, iou_thres=0.5):
    cm = np.zeros((nc + 1, nc + 1), dtype=int)
    for filename in os.listdir(label_folder):
        predict_file = os.path.join(predict_folder, filename)
        label_file = os.path.join(label_folder, filename)
        predictions = parse_boxes(predict_file)
        labels = parse_boxes(label_file)
        matched_preds = set()
        matched_labels = set()
        for i, pred in enumerate(predictions):
            for j, label in enumerate(labels):
                if calculate_iou(pred[1:], label[1:]) >= iou_thres:
                    if i not in matched_preds and j not in matched_labels:
                        cm[pred[0]][label[0]] += 1
                        matched_preds.add(i)
                        matched_labels.add(j)
                        break
        for i, pred in enumerate(predictions):
            if i not in matched_preds:
                cm[pred[0]][-1] += 1  
        for j, label in enumerate(labels):
            if j not in matched_labels:
                cm[-1][label[0]] += 1  
    return cm
def calculate_metrics(confusion_matrix):
    num_classes = confusion_matrix.shape[0] - 1
    metrics = []
    for class_id in range(num_classes):
        true_positives = confusion_matrix[class_id, class_id]
        false_positives = np.sum(confusion_matrix[:, class_id]) - true_positives
        false_negatives = np.sum(confusion_matrix[class_id, :]) - true_positives
        recall = true_positives / (true_positives + false_positives + 1e-9)
        precision = true_positives / (true_positives + false_negatives + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        metrics.append((precision, recall, f1))
    return metrics
#-------------------------------------------------------------------------------------------
cm=calculate_cm('/root/runs/detect/t1/','/root/hf_val/labels/',nc=8,
                iou_thres=0.5)
print(cm)
print(np.array(calculate_metrics(cm)))
print(np.array(calculate_metrics(cm))[:,2].sum()/8)