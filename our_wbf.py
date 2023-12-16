import numpy as np

def prefilter_boxes(boxes, scores, labels, weights, thr):
    new_boxes = dict()
    for t in range(len(boxes)):
        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])
            b = [int(label), float(score) * weights[t][label], weights[t][label], t, x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]
    return new_boxes


def get_weighted_box(boxes):
    box,conf,w,conf_list = np.zeros(8, dtype=np.float32),0,0,[]
    for b in boxes:
        box[4:] += (b[1] * b[4:])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0],box[1],box[2],box[3],box[4:] = boxes[0][0], conf / len(boxes),w,-1,box[4:]/conf
    return box


def find_matching_box_fast(boxes_list, new_box, match_iou):
    def bb_iou_array(boxes, new_box):
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])
        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)
        boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou
    if boxes_list.shape[0] == 0:
        return -1, match_iou
    boxes = boxes_list
    ious = bb_iou_array(boxes[:, 4:], new_box[4:])
    ious[boxes[:, 0] != new_box[0]] = -1
    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]
    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1
    return best_idx, best_iou


def weighted_boxes_fusion1(boxes_list,scores_list,labels_list,
        weights=None,iou_thr=0.55,skip_box_thr=0.0):
    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 8))

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box_fast(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index])
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = new_boxes[i]
            weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / sum(sublist[label] for sublist in weights)
        overall_boxes.append(weighted_boxes)
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels
