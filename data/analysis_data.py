#!/bin/python

debug = False

def analyse_img(data_path):
    """
    analyse train and test raw images data. such as resolution (height, width)
    """
    # init
    resolution_dict = {}
    format_dict = {}
    topK = 5

    import os
    file_list = os.listdir(data_path)
    file_num = len(file_list)
    print("file_num:{}".format(file_num))

    from PIL import Image
    for idx in xrange(len(file_list)):
        file_name = file_list[idx]
        img_path = "/".join([data_path, file_name])
        img = Image.open(img_path)
        # check
        try:
            resolution_dict[img.size] += 1
        except:
            resolution_dict[img.size] = 1

        try:
            format_dict[img.format] += 1
        except:
            format_dict[img.format] = 1

        if debug:
            print("{}\timg_path:{}\tsize:{}\tformat:{}".format(idx+1, img_path, img.size, img.format))
        img.close()

    # sort result
    resolution_tuple_list = sorted(resolution_dict.items(), key=lambda item: -item[1])
    format_tuple_list = sorted(format_dict.items(), key=lambda item: -item[1])

    # statistic
    print("\n=== resolution TOP%d ===" % topK)
    for k, v in resolution_tuple_list[:5]:
        print("%d,%d\t%d\t%.4f" % (k[0], k[1], v, v/float(file_num)))
    print("-----\nsum:%d" % (file_num))
    print("\n=== format ===")
    for k, v in format_tuple_list:
        print("{}\t{}".format(k, v))
    print("-----\nsum:%d" % (file_num))


def analyse_label(label_file_path):
    print("\n=== label ===")
    import pandas as pd
    csv = pd.read_csv(label_file_path)
    print("%s" % csv.count())
    print("%s" % csv.head(1))

    label_dict = {}
    for sample_idx in xrange(csv['id'].count()):
        sample_id = csv["id"][sample_idx]
        sample_label = csv["breed"][sample_idx]
        try:
            label_dict[sample_label] += 1
        except:
            label_dict[sample_label] = 1
    print("\n=== more detailed ===")

    # sort
    label_tuple_list = sorted(label_dict.items(), key=lambda item: -item[1])

    # result
    for k, v in label_tuple_list:
        print("%-30s\t%d\t%.4f" % (k, v, float(v)/csv['id'].count()))

if __name__ == "__main__":
    analyse_img("./test")
    analyse_img("./train")
    analyse_label("./labels.csv")
