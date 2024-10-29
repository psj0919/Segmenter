import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    network = ['Seg-S']  # Seg-S, Seg-B
    result_path = '/storage/sjpark/vehicle_data/precision_recall_per_class_p_threshold/'

    image_sizes = ['256']

    class_names = glob(os.path.join(result_path, network[0], image_sizes[0], 'precision', '*'))
    class_names = [class_names[i].split('/')[-1] for i in range(len(class_names))]
    class_names.sort()

    probs = glob(os.path.join(result_path, network[0], image_sizes[0], 'precision', class_names[0], '*'))
    probs = [probs[i].split('/')[-1][:-4].split('_')[-1] for i in range(len(probs))]
    probs.sort()

    precision_dict = dict()
    recall_dict = dict()

    empty_classes = []
    for net_name in network:
        precision_dict[net_name] = dict()
        recall_dict[net_name] = dict()
        for img_s in image_sizes:
            precision_dict[net_name][img_s] = dict()
            recall_dict[net_name][img_s] = dict()
            for name in class_names:
                file_names = glob(os.path.join(result_path, net_name, img_s, 'precision', name, '*'))
                if len(file_names) == 0:
                    empty_classes.append(name)
                    pass
                precision_dict[net_name][img_s][name] = dict()
                recall_dict[net_name][img_s][name] = dict()
                for prob in probs:
                    precision_dict[net_name][img_s][name][prob] = dict(result_values=[], ap=[])
                    recall_dict[net_name][img_s][name][prob] = dict(result_values=[], ar=[])

    empty_classes = list(np.unique(empty_classes))
    for name_class in empty_classes:
        class_names.remove(name_class)

    for net_name in network:
        for img_s in image_sizes:
            for name in class_names:
                for prob in probs:
                    file_path = os.path.join(result_path, net_name, img_s, 'precision', name, name + '_' + prob + '.txt')
                    with open(file_path, 'r') as f:
                        result_values = f.readlines()
                    result_values = [float(result_values[i].split('\n')[0]) for i in range(len(result_values))]
                    precision_dict[net_name][img_s][name][prob]['result_values'] = result_values
                    precision_dict[net_name][img_s][name][prob]['ap'] = np.sum(result_values) / len(result_values)
                    #
                    file_path = os.path.join(result_path, net_name, img_s, 'recall', name, name + '_' + prob + '.txt')
                    with open(file_path, 'r') as f:
                        result_values = f.readlines()
                    result_values = [float(result_values[i].split('\n')[0]) for i in range(len(result_values))]
                    recall_dict[net_name][img_s][name][prob]['result_values'] = result_values
                    recall_dict[net_name][img_s][name][prob]['ar'] = np.sum(result_values) / len(result_values)


    # AP-AR curve
    # precision_curve = []
    # recall_curve = []
    #
    # for name in class_names:
    #     for prob in probs:
    #         precision_curve.append(precision_dict["Seg-S"]["512"][name][prob]['ap'])
    #         recall_curve.append(recall_dict["Seg-S"]["512"][name][prob]['ar'])
    #
    #     fig = plt.figure(figsize=(9, 6))
    #     plt.plot(recall_curve, precision_curve)
    #     plt.scatter(recall_curve, precision_curve)
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title('Precision_recall_{}_curve'.format(name))
    #     plt.savefig('/storage/sjpark/vehicle_data/curve/{}.png'.format(name))
    #     plt.close()
    #     precision_curve.clear()
    #     recall_curve.clear()


    # mAP
    mAP = dict()
    for net_name in network:
        mAP[net_name] = dict()
        for prob in probs:
            mAP[net_name][prob] = []
            x = 0
            for name in class_names:
                x += precision_dict[net_name]['512'][name][prob]['ap']
            mAP[net_name][prob].append(x / len(class_names))

    for name in class_names:
        print(precision_dict['Seg-S']['512'][name]['0.4']['ap'])


    # mAP per threshold_probability
    val = []
    for prob in probs:
        val.append(mAP['Seg-S'][prob][0])
    #
    plt.plot(probs, val)
    plt.scatter(probs, val)
    plt.xlabel('threshold_probability')
    plt.ylabel('mAP')
    plt.title('mAP per threshold_probability')
    plt.imshow()

    #F1-score
    precision_f1 = []
    recall_f1 = []
    f1_score = {}

    for name in class_names:
        for prob in probs:
            x = 2 * precision_dict["FCN"]["512"][name][prob]['ap'] * recall_dict["FCN"]["512"][name][prob]['ar'] / precision_dict["FCN"]["512"][name][prob]['ap'] * recall_dict["FCN"]["512"][name][prob]['ar']
            f1_score.setdefault(prob, x)



