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
    idx = 0
    for k, v in label_tuple_list:
        idx += 1
        print("%d\t%-30s\t%d\t%.4f" % (idx, k, v, float(v)/csv['id'].count()))

def get_label_list(label_file_path):
    import pandas as pd
    csv = pd.read_csv(label_file_path)
    label_list = csv['breed'].unique()
    id_list = csv['id'].unique()
    print("-----\nlen(id_list):%d" % len(id_list))
    print("len(label_list):%d" % len(label_list))
    return label_list, id_list

def generate_synset_list(label_file_path, synset_file_path="./full-synset.txt"):
    label_list, id_list = get_label_list(label_file_path)
    synset_list = []
    for idx in xrange(len(label_list)):
        label = label_list[idx]
        line = " ".join([str(idx), label, "\n"])
        synset_list.append(line)

    with open(synset_file_path, "w") as f:
        f.writelines(synset_list)


def rearrange_train(label_file_path, train_dir="./train", new_train_dir="./rearranged-train"):
    import os
    if not os.path.exists(new_train_dir):
        os.mkdir(new_train_dir)
    label_list, id_list = get_label_list(label_file_path)

    import pandas as pd
    csv = pd.read_csv(label_file_path)

    import shutil
    label_dict = {}
    for idx in xrange(len(id_list)):
        img_id = id_list[idx]
        img_label = csv[csv['id'].isin([img_id])]['breed'].values[0]
        old_img_path = "/".join([train_dir, img_id + ".jpg"])
        new_img_path = "/".join([new_train_dir, img_label, img_id + ".jpg"])

        # cp
        label_path = "/".join([new_train_dir, img_label])

        # check
        #print(idx+1, old_img_path, new_img_path, label_path)

        if not os.path.exists(label_path):
            os.mkdir(label_path)
        shutil.copyfile(old_img_path, new_img_path)
        print("cp %d / %d img" % (idx+1, len(id_list)))


def raw2MX(rearranged_train_dir, resolution=224, train_ratio=0.95, prefix_name="mx", quality=95):
    import os
    thread_num_output = os.popen('cat /proc/cpuinfo| grep "processor"| wc -l')
    thread_num = int(thread_num_output.read())

    prefix_name_list = map(str, [prefix_name, resolution, "qua", quality, "ratio", train_ratio])
    prefix_name = "-".join(prefix_name_list)

    lst_cmd = "python tools/im2rec.py --list True --recursive True --train-ratio {train_ratio} {prefix_name} {img_data_dir}".format(train_ratio=train_ratio, prefix_name=prefix_name, img_data_dir=rearranged_train_dir)
    rec_cmd = "python tools/im2rec.py --resize {size} --quality {img_quality} --num-thread {thread_num} {prefix_name} {img_data_dir}".format(size=resolution, thread_num=thread_num, prefix_name=prefix_name, img_data_dir=rearranged_train_dir, img_quality=quality)
    # check
    print(lst_cmd)
    print(rec_cmd)

    # execute
    os.system(lst_cmd)
    os.system(rec_cmd)



if __name__ == "__main__":
    # init
    train_dir = "./train"
    test_dir = "./test"
    label_file_path = "./labels.csv"
    synset_file_path = "./full-synset.txt" # generated file

    rearranged_train_dir = './rearranged-train'
    resolution = 512
    prefix_name = "mx"
    train_ratio = 0.95
    quality = 99

    # analyse
    '''
    analyse_img(test_dir)
    analyse_img(train_dir)
    analyse_label(label_file_path)
    '''

    # prepare data for mxnet format
    generate_synset_list(label_file_path, synset_file_path)
    #rearrange_train(label_file_path, train_dir, rearranged_train_dir)
    raw2MX(rearranged_train_dir,
           resolution,
           train_ratio,
           prefix_name,
           quality)
