#!/bin/bash

#printf "[inception-resnet-v2]\n"
#python train_ccs-train.py \
#--lr-factor 0.01 \
#--lr-step-epochs 10 \
#--optimizer 'adam' \
#--mom 0.9 \
#--wd 0.9 \
#--batch-size 48 \
#--disp-batches 100 \
#--network inception-resnet-v2 \
#--num-layers 50 \
#--model-prefix ./models/inception-resnet-v2-50-standford-512-adam-lr-0.01/inception-resnet-v2-50-standford-512-adam-lr-0.01

#printf "[inception-resnet-v2]\n"
#python train_ccs-train.py \
#--network inception-resnet-v2 \
#--num-layers 50 \
#--model-prefix ./models/inception-resnet-v2-50-standford-512-lr-0.01/inception-resnet-v2-50-standford-512-lr-0.01

#printf "[resnext]\n"
#python train_ccs-train.py \
#--network resnext \
#--num-layers 50 \
#--model-prefix ./models/resnext-50-standford-512-lr-0.01

printf "[resnet-50]\n"
python train_ccs-train.py \
--num-epochs 1000 \
--lr 0.1 \
--lr-factor 0.1 \
--lr-step-epochs 60 \
--optimizer 'adam' \
--mom 0.9 \
--wd 0.0001 \
--batch-size 64 \
--gpus '0,1,2,3' \
--disp-batches 100 \
--network 'resnet' \
--num-layers 50 \
--model-prefix ./models/resnet-50-standford-512-adam-lr-0.01/resnet-50-standford-512-adam-lr-0.01

#printf "[resnet-34]\n"
#python train_ccs-train.py \
#--network resnet \
#--num-layers 34 \
#--model-prefix ./models/resnet-34-standford-512-lr-0.01

#printf "[resnet-152]\n"
#python train_ccs-train.py \
#--network resnet \
#--num-layers 152 \
#--model-prefix ./models/resnet-152-standford-512-lr-0.01/resnet-152-standford-512-lr-0.01

#printf "[resnet-200]\n"
#python train_ccs-train.py \
#--network resnet \
#--num-layers 200 \
#--model-prefix ./models/resnet-200-standford-512-lr-0.01/resnet-200-standford-512-lr-0.01

#printf "[resnet-50]\n"
#python train_ccs-train.py \
#--network resnet \
#--num-layers 50 \
#--model-prefix ./models/resnet-50-standford-512-lr-0.01/resnet-50-standford-512-lr-0.01

#printf "[resnet-34]\n"
#python train_ccs-train.py \
#--network resnet \
#--num-layers 34 \
#--model-prefix ./models/resnet-34-standford-512-lr-0.01/resnet-34-standford-512-lr-0.01

#printf "[resnet-18]\n"
#python train_ccs-train.py \
#--network resnet \
#--num-layers 18 \
#--model-prefix ./models/resnet-18-standford-512-lr-0.01/resnet-18-standford-512-lr-0.01

#printf "[resnxet-18]\n"
#python train_ccs-train.py \
#--network resnext \
#--num-layers 18 \
#--model-prefix ./models/resnext-18-standford-512-lr-0.01/resnext-18-standford-512-lr-0.01

#printf "[resnxet-34]\n"
#python train_ccs-train.py \
#--network resnext \
#--num-layers 34 \
#--model-prefix ./models/resnext-34-standford-512-lr-0.01/resnext-34-standford-512-lr-0.01

#printf "[resnxet-50]\n"
#python train_ccs-train.py \
#--network resnext \
#--num-layers 50 \
#--model-prefix ./models/resnext-50-standford-512-lr-0.01/resnext-50-standford-512-lr-0.01

#printf "[resnxet-101]\n"
#python train_ccs-train.py \
#--network resnext \
#--num-layers 101 \
#--model-prefix ./models/resnext-101-standford-512-lr-0.01/resnext-101-standford-512-lr-0.01

#printf "[resnxet-152]\n"
#python train_ccs-train.py \
#--network resnext \
#--num-layers 152 \
#--model-prefix ./models/resnext-152-standford-512-lr-0.01/resnext-152-standford-512-lr-0.01

#printf "[resnxet-200]\n"
#python train_ccs-train.py \
#--network resnext \
#--num-layers 200 \
#--model-prefix ./models/resnext-200-standford-512-lr-0.01/resnext-200-standford-512-lr-0.01

