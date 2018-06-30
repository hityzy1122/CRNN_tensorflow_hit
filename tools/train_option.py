import numpy as np
from tools.inform_trans import sparse_tensor_to_str


def cal_accuracy(gt_labels, preds):

    preds = sparse_tensor_to_str(preds[0])
    gt_labels = sparse_tensor_to_str(gt_labels)

    accuracy_train = []
    for index, gt_label in enumerate(gt_labels):
        pred = preds[index]
        total_count = len(gt_label)
        correct_count = 0
        try:
            for i, tmp in enumerate(gt_label):
                if tmp == pred[i]:
                    correct_count += 1
        except IndexError:
            continue
        finally:
            try:
                accuracy_train.append(correct_count / total_count)
            except ZeroDivisionError:
                if len(pred) == 0:
                    accuracy_train.append(1)
                else:
                    accuracy_train.append(0)

    accuracy_train = np.mean(np.array(accuracy_train).astype(np.float32), axis=0)
    return accuracy_train, gt_labels, preds
