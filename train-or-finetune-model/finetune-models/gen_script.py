#!/bin/python

def init():
    global starter_cmd
    global model_prefix_list
    global model_epoch_list

    global max_run_epoch
    global gpu_num
    global batch_size

    global img_size_list
    global lr_list
    global momentum_list
    
    ############################################ init ######################################
    starter_cmd = "python run_finetune.py"
    model_prefix_list = ["../../models/pretrained-models/caffenet/caffenet",\
                         "../../models/pretrained-models/resnet-152/resnet-152",\
                         "../../models/pretrained-models/vgg-19/vgg19",\
                         "../../models/pretrained-models/vgg-16/vgg16"]
    model_epoch_list = [0, 0, 0, 0]
    max_run_epoch = 50
    gpu_num = 3
    batch_size = 20

    img_size_list = [512]
    lr_list = [0.01, 0.05]
    momentum_list = [0.9, 0.7]
    ############################################ init ######################################

def create_cmd():
    global cmd_list
    cmd_list = []

    for midx in xrange(len(model_prefix_list)):
        model_prefix = model_prefix_list[midx]
        model_epoch = model_epoch_list[midx]
        
        for img_size in img_size_list:
            for lr in lr_list:
                for momentum in momentum_list:
                    cmd = [starter_cmd,   # python run_finetine.py
                           model_prefix,  # load model prefix
                           model_epoch,   # load model epoch
                           model_prefix+"-"+str(img_size)+"-lr-"+str(lr)+"-mom-"+str(momentum),  # save model prefix
                           max_run_epoch,
                           gpu_num,
                           batch_size,]

                    cmd = map(str, cmd)
                    cmd = " ".join(cmd)
                    cmd_list.append(cmd)

    # check
    for cmd in cmd_list: print cmd

def run_cmd():
    import os
    for cmd in cmd_list:
        print(cmd)
        os.system(cmd)
        print("\n\n\n\n")

def main():
    init()
    create_cmd()
    #run_cmd()

if __name__ == "__main__":
    main()
