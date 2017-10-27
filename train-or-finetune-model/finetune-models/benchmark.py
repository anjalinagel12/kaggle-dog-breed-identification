#!/bin/python

def init():
    global starter_cmd
    global model_prefix_list
    global model_epoch_list
    global run_epoch_num_list
    global gpus_num_list
    global batch_size_list
    
    ############################################ init ######################################
    starter_cmd = "python run_finetune.py"
    model_prefix_list = ["../../models/pretrained-models/caffenet/caffenet",\
                         "../../models/pretrained-models/resnet-152/resnet-152",\
                         "../../models/pretrained-models/vgg-19/vgg19",\
                         "../../models/pretrained-models/vgg-16/vgg16"]
    model_epoch_list = [0, 0, 0, 0]
    run_epoch_num_list = [50, 50, 50, 50]
    gpus_num_list = [3]
    batch_size_list = [20]#map(lambda base: 2**base, xrange(0, 11)) #1~1024
    ############################################ init ######################################

def create_cmd():
    global cmd_list
    cmd_list = []

    for midx in xrange(len(model_prefix_list)):
        model_prefix = model_prefix_list[midx]
        model_epoch = model_epoch_list[midx]
        run_epoch_num = run_epoch_num_list[midx]
        
        for gpu_num in gpus_num_list:
            for batch_size in batch_size_list:
                cmd = [starter_cmd,   # python run_finetine.py
                       model_prefix,  # load model prefix
                       model_epoch,   # load model epoch
                       model_prefix+"-gpunum-"+str(gpu_num)+"-bsize-"+str(batch_size),  # save model prefix
                       run_epoch_num, # run train epoch number
                       gpu_num,       # gpu numbers
                       batch_size]    # batch size
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
