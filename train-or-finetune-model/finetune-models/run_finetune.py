import sys
# sys.path.append('/mnt/sdcard/lizhihao/install/opencv-3.1.0/lib/python2.7/dist-packages')
sys.path.append("/home/yuanshuai/software/opencv-3.2.0/lib/python2.7/dist-packages")
# download data
'''
import os, urllib
def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)
download('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec')
download('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec')
'''


# define data iterators
import mxnet as mx

def get_iterators(batch_size, data_shape=(3, 128, 128)):
    train = mx.io.ImageRecordIter(
        path_imgrec         = '../../data/mx-128-qua-95-ratio-0.95_train.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        preprocess_threads  = 8,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True)
    val = mx.io.ImageRecordIter(
        path_imgrec         = '../../data/mx-128-qua-95-ratio-0.95_val.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        preprocess_threads  = 8,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False)
    return (train, val)

# download pretrained models
'''
def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))
get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)
'''

#sym, arg_params, aux_params = mx.model.load_checkpoint('../../model/resnet-50/resnet-50', 0)
load_model_prefix = sys.argv[1]
load_model_epoch  = int(sys.argv[2])
sym, arg_params, aux_params = mx.model.load_checkpoint(load_model_prefix, load_model_epoch)



# Train

# replace the last fully-connected layer for a given network 
def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    try:
        net = all_layers[layer_name+'_output']
    except:
        print "note: caffe model don't have 'flatten0', but 'flatten_0'"
        try:
            layer_name = 'flatten_0'
            net = all_layers[layer_name+'_output']
        except:
            print "note: don't have flatten_0, but flatten"
            layer_name = 'flatten'
            net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)

# Now we create a module. We pass the argument parameters of the pre-trained model to replace all parameters except for the last fully-connected layer. For the last fully-connected layer, we use an initializer to initialize.


import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus, save_model_prefix):
    devs = [mx.gpu(i) for i in xrange(num_gpus)]
    #devs = [mx.gpu(1), mx.gpu(2)]
    #devs = mx.cpu()
    mod = mx.mod.Module(symbol=new_sym, context=devs)
    mod.fit(train, val, 
        num_epoch=int(sys.argv[4]),
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 100),
        epoch_end_callback=mx.callback.do_checkpoint(save_model_prefix),
        kvstore='device',
        optimizer='adam',
        optimizer_params={'learning_rate':0.01,},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)

# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
num_classes = 120
batch_per_gpu = int(sys.argv[6])
num_gpus = int(sys.argv[5])
save_model_prefix = sys.argv[3]

(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)
batch_size = batch_per_gpu * num_gpus
data_shape = (3, 128, 128)
(train, val) = get_iterators(batch_size, data_shape)
mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus, save_model_prefix)
assert mod_score > 0.77, "Low training accuracy."
