#!/bin/bash

python run_finetune.py \
'../../models/pretrained-models-based-on-stanford-dogs/inception-resnet-v2-50-standford-512-lr-0.01/inception-resnet-v2-50-standford-512-lr-0.01' \
'27' \
'./models/kaggle-inception-resnet-v2-50-standford-512-lr-0.01' \
'60' \
'2' \
'3'

#python run_finetune.py \
#'../../models/pretrained-models/caffenet/caffenet' \
#'0' \
#'../../models/pretrained-models/caffenet/caffenet-train-224-adam-lr-0.01' \
#'30' \
#'3' \
#'10'

#python run_finetune.py \
#'../../models/pretrained-models/caffenet/caffenet' \
#'0' \
#'../../models/pretrained-models/caffenet/caffenet-train-224-adam-lr-0.01' \
#'30' \
#'3' \
#'10'

#python run_finetune.py \
#'../../models/pretrained-models/resnet-152/resnet-152' \
#'0' \
#'../../models/pretrained-models/resnet-152/resnet-152-train-224-adam-lr-0.01' \
#'30' \
#'3' \
#'10'

#python run_finetune.py \
#'../../models/pretrained-models/vgg-19/vgg19' \
#'0' \
#'../../models/pretrained-models/vgg-19/vgg19-train-224-lr-0.01' \
#'30' \
#'3' \
#'10'

#python run_finetune.py \
#'../../models/pretrained-models/caffenet/caffenet' \
#'0' \
#'../../models/pretrained-models/caffenet/caffenet-train-224-lr-0.01' \
#'30' \
#'3' \
#'10'
