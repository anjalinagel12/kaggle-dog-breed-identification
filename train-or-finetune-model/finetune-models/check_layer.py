import sys
sys.path.append("/home/yuanshuai/software/opencv-3.2.0/lib/python2.7/dist-packages")

import mxnet as mx

load_model_prefix = sys.argv[1]
load_model_epoch = int(sys.argv[2])
sym, arg_params, aux_params = mx.model.load_checkpoint(load_model_prefix, load_model_epoch)
outputs_layers = sym.get_internals().list_outputs()
for idx in xrange(len(outputs_layers)):
	print idx+1, outputs_layers[idx]
    
