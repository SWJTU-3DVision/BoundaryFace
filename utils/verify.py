import os

if __name__ == '__main__':
    # 总的闭集
    closed_list = {}
    with open(r'C:\Users\swjtu\Desktop\noise_flip_30.txt') as f:
        lines = f.read().splitlines()
        for line in lines:
            key, label = line.split()
            closed_list[key] = label

    # 检测出来的闭集
    keys_label = {}
    with open(r'C:\Users\swjtu\Desktop\savedPath_6.txt') as f:
        lines = f.read().splitlines()
        for line in lines:
            key, label = line.split()
            keys_label[key] = label
    overlap = list(set(closed_list.items()) & set(keys_label.items()))
    precision = len(overlap) / len(keys_label)
    recall = len(overlap) / len(closed_list)
    print('The flip noise correct precision : {:.2f}'.format(precision * 100))
    print('The flip noise correct recall : {:.2f}'.format(recall * 100))