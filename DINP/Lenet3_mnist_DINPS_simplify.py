# Licensed to the Apache Software Foundation (ASF under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import division
import argparse, time
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet import kvstore as kvs

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='dataset to use. options are mnist, cifar10, and dummy.')
parser.add_argument('--batch-size', type=int, default=100,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('--gid', type=int, default=0,
                    help='the gpu id')
parser.add_argument('--epochs1', type=int, default=5,
                    help='number of training epochs.')
parser.add_argument('--epochs2', type=int, default=10,
                    help='number of training epochs.')
parser.add_argument('--loops1', type=int, default=20,
                    help='number of training epochs.')
parser.add_argument('--loops', type=int, default=30,
                    help='number of training loops outsides.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.01.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123.')
parser.add_argument('--benchmark', action='store_true',
                    help='whether to run benchmark.')
parser.add_argument('--use_thumbnail', action='store_true',
                    help='use thumbnail or not in resnet. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer/module.')
parser.add_argument('--log-interval', type=int, default=100, help='Number of batches to wait before logging.')
parser.add_argument('--nworkers', type=int, default=1,
                    help='number of worker machines')
parser.add_argument('--wid', type=int, default=-1, help='the worker ID for loading data')
parser.add_argument('--gamma', type=float, default=0, help='the gamma parameter')
parser.add_argument('--const', type=float, default=1, help='the constant parameter')

opt = parser.parse_args()

print(opt)
opt.lrdense = opt.lr
opt.lrsparse = opt.lr
mx.random.seed(opt.seed)

dataset_classes = {'mnist': 10, 'cifar10': 10, 'imagenet': 1000, 'dummy': 1000}

batch_size, dataset, classes = opt.batch_size, opt.dataset, dataset_classes[opt.dataset]

gpus = opt.gpus

gid = opt.gid

wid = opt.wid

gamma = opt.gamma #
const = opt.const # 0 only avg weight


batch_size *= 1


""" data iterator for dist mnist """

def mnist_iterator_dist(wid, batch_size, input_shape):
    """return train and val iterators for mnist"""
    flat = False if len(input_shape) == 3 else True

    train_dataiter = mx.io.MNISTIter(
        image="../data_storage/data" + str(opt.nworkers) + "/train-images-idx3-ubyte_" + str(wid),
        label="../data_storage/data" + str(opt.nworkers) + "/train-labels-idx1-ubyte_" + str(wid),
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        flat=flat)
    wholetrain_dataiter = mx.io.MNISTIter(
        image="../data_storage/data/train-images-idx3-ubyte",
        label="../data_storage/data/train-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        flat=flat)
    val_dataiter = mx.io.MNISTIter(
        image="../data_storage/data/t10k-images-idx3-ubyte",
        label="../data_storage/data/t10k-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        flat=flat)

    return train_dataiter,wholetrain_dataiter,val_dataiter


def getmodel():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(300, activation='relu', use_bias=False))
        net.add(nn.Dense(100, activation='relu', use_bias=False))
        net.add(nn.Dense(10, use_bias=False))
    return net


net = getmodel()

grad_key = []
weight_key = []
for key in net.collect_params().keys():
    weight_key.append(key)
    key_tmp = key + '_grad'
    grad_key.append(key_tmp)
    print(key)


# get dataset iterators
def get_data():
    logging.info('Worker ID %d' % wid)
    train_data, wholetrain_data, val_data = mnist_iterator_dist(wid, batch_size, (1, 32, 32))
    return train_data, wholetrain_data, val_data

train_data, wholetrain_data, val_data = get_data()

def test(ctx):
    metric = mx.metric.Accuracy()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()

def gettrainingloss(ctx):
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    n_data = 0
    L = 0.0
    wholetrain_data.reset()
    for batch in wholetrain_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        n_data += label[0].shape[0]
        for x, y in zip(data, label):
            temploss = loss(net(x), y)
            temploss = temploss.asnumpy().tolist()
            L += sum(temploss)
    return L/float(n_data)


def calgrad(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    grad_w = {}
    temp_w = {}
    params = net.collect_params()
    for key in weight_key:
        temp_w[key] = mx.nd.zeros(params[key].data(ctx[0]).shape, ctx[0])
        grad_w[key] = mx.nd.zeros(params[key].data(ctx[0]).shape, ctx[0])
    train_data.reset()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    n_data = 0
    for i, batch in enumerate(train_data):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        Ls = []
        n_data += label[0].shape[0]
        with ag.record():
            for x, y in zip(data, label):
                z = net(x)
                L = loss(z, y)
                # store the loss and do backward after we have done forward
                # on all GPUs for better speed on multiple GPUs.
                Ls.append(L)
                outputs.append(z)
            for L in Ls:
                L.backward()
        for key in weight_key:
            temp_w[key] += params[key].grad(ctx[0])
    for key in weight_key:
        grad_w[key] = temp_w[key]/n_data
    return grad_w



def train_sparse_PGP(grad_avg, grad_w, const, gamma, weight_s, inds_all, wid, ctx, epochs=1, loopiter=0, subloop=-1):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    logging.info('Worker[%d] starts fully correcting' % wid)
    params = net.collect_params()
    trainer = None
    if loopiter == -1:
        trainer = gluon.Trainer(params, 'sgd', {'learning_rate': 0, 'wd': opt.wd,
                                                'momentum': opt.momentum}, kvstore='device')
        epochs = 1
    else:
        trainer = gluon.Trainer(params, 'sgd', {'learning_rate': opt.lrsparse, 'wd': opt.wd,
                                                'momentum': opt.momentum}, kvstore='device')
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    compressratio = [0.5, 0.2]

    for epoch in range(epochs):
        tic = time.time()
        train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            if const != 0:  # and gamma != 0:
                for l, key in enumerate(weight_key):
                    grad = const * (grad_avg[l] - grad_w[key]) + gamma * (params[key].data(ctx[0]) - weight_s[l]) + \
                           params[key].grad(ctx[0])  # ????? eq SM2 gamma(w - wt-1)
                    params[key].grad(ctx[0])[:] = grad
            trainer.step(batch.data[0].shape[0])

            # prunning
            for key in weight_key:
                if 'conv' in key in key:
                    ratio = compressratio[0]
                elif 'dense' in key:
                    ratio = compressratio[1]
                else:
                    ratio = 1
                inds = inds_all[key]
                weight_tmp = params[key].data(ctx[0]).asnumpy()
                weight_tmp = weight_tmp.flatten()
                nonzero_inds = int(params[key].data().size * ratio)
                weight_np = np.zeros(params[key].data().shape)
                weight_np.flat[inds[0:nonzero_inds]] = weight_tmp[inds[0:nonzero_inds]]
                mx.nd.array(weight_np).copyto(params[key].data(ctx[0]))

            metric.update(label, outputs)
            if opt.log_interval and not (i + 1) % opt.log_interval:
                name, acc = metric.get()
                logging.info('[Worker %d Epoch %d Batch %d] speed: %f samples/s, training: %s=%f' % (wid, epoch, i, batch_size /(time.time() - btic),
                                                                                                     name, acc))
            btic = time.time()

        logging.info('[Worder %d Loop %d, subLoop %d Epoch %d Sparse Model] compress ratio=%s' % (wid, loopiter, subloop, epoch, str(compressratio)))
        logging.info('Worker %d Loop %d, subLoop %d Epoch %d Sparse Model] learning rate = %f' % (wid, loopiter, subloop, epoch, trainer.learning_rate))
        name, acc = metric.get()
        logging.info('[Worker %d Loop %d, subLoop %d Epoch %d Sparse Model] training: %s=%f' % (wid, loopiter, subloop, epoch, name, acc))
        logging.info('[Worker %d Loop %d, subLoop %d Epoch %d Sparse Model] time cost: %f' % (wid, loopiter, subloop, epoch, time.time() - tic))
        name, val_acc = test(ctx)
        logging.info('[Worker %d, Loop %d, subLoop %d Epoch %d Sparse Model] validation: %s=%f' % (wid, loopiter, subloop, epoch, name, val_acc))

    # return weight_w
    weight_w = {}
    for key in weight_key:
        weight_w[key] = params[key].data(ctx[0])
    return weight_w


def train_dense_PGP(grad_avg, grad_w, const, gamma, weight_s, wid, ctx, epochs=1, loopiter=0, subloop=0):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    logging.info('Worker [%d] start to train dense net' % wid)
    trainer = None
    params = net.collect_params()

    if loopiter == -1:
        trainer = gluon.Trainer(params, 'sgd', {'learning_rate': 0, 'wd': opt.wd,
                                                'momentum': opt.momentum, }, kvstore='device')
        epochs = 0
    else:
        trainer = gluon.Trainer(params, 'sgd', {'learning_rate': opt.lrdense, 'wd': opt.wd,
                                                'momentum': opt.momentum, }, kvstore='device')
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    for epoch in range(epochs):
        tic = time.time()
        train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            # update weight using info sent back by server
            if const != 0:  # and  gamma != 0:
                for l, key in enumerate(weight_key):
                    grad = const * (grad_avg[l] - grad_w[key]) + gamma * (params[key].data(ctx[0]) - weight_s[l]) + \
                           params[key].grad(ctx[0])  # ????? eq SM2 gamma(w - wt-1)
                    params[key].grad(ctx[0])[:] = grad
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if opt.log_interval and not (i + 1) % opt.log_interval:
                name, acc = metric.get()
                logging.info('[Worker %d Loop %d SubLoop %d Epoch %d Batch %d] speed: %f samples/s, training: %s=%f' % (
                    wid, loopiter, subloop, epoch, i, batch_size / (time.time() - btic), name, acc))
            btic = time.time()
        logging.info('[Worder %d  Loop %d SubLoop %d Epoch %d Dense Model] compress ratio=%s' % (wid, loopiter, subloop, epoch, str(compressratio)))
        logging.info('[Worker %d  Loop %d SubLoop %d Epoch %d Dense Model] learning rate=%f' % (wid, loopiter, subloop, epoch, trainer.learning_rate))
        name, acc = metric.get()
        logging.info('[Worker %d  Loop %d SubLoop %d Epoch %d Dense Model] training: %s=%f' % (wid, loopiter, subloop, epoch, name, acc))
        logging.info('[Epoch %d] time cost: %f' % (epoch, time.time() - tic))
        name, val_acc = test(ctx)
        logging.info('[Worker %d  Loop %d SubLoop %d Epoch %d Dense Model] validation: %s=%f' % (wid, loopiter, subloop, epoch, name, val_acc))
    weight_w = {}
    for key in weight_key:
        shape = params[key].data().shape
        new_weight1 = np.zeros(shape)
        weight1 = params[key].data()
        weight1_np = weight1.asnumpy()
        new_weight1.flat[:] = weight1_np.flat[:]
        weight_w[key] = mx.nd.array(new_weight1)
    return weight_w

if __name__ == '__main__':
    logging.info('[Worker %d] gets started\n' % wid)

    kv = kvs.create(opt.kvstore)
    kv.set_optimizer(mx.optimizer.Test())
    ctx = mx.gpu(gid)
    for i in range(opt.nworkers):
        model_name = 'LeNet3-mnist_fed_best_BW_w' + str(opt.nworkers) + '.params'
        net.load_params(model_name, ctx)

    name, val_acc = test([ctx])
    logging.info('Initial Model acc before training: %s=%f' % (name, val_acc))

    params = net.collect_params()
    bestacc = 0
    """Initialization Training"""
    grad_avg = []
    grad_w = {}
    grad_key = []
    weight_s = []
    weight_s_prev = []
    weight_s_current = []
    grad_avg_current = []
    grad_avg_prev = []
    weight_w = train_dense_PGP(grad_avg, grad_w, 0, 0, weight_s, wid, ctx, epochs=0, loopiter=-1)  # training in dense and applied IHT in the last step
    val_loss = gettrainingloss([ctx])
    weight_w_list = [weight_w[key] for key in weight_key]
    inds_all = {}
    weight_w = {}
    for key in weight_key:
        if 'conv' in key:
            ratio = 0.5
        elif 'dense' in key:
            ratio = 0.2

        shape = params[key].data().shape
        size = params[key].data().size
        nonzero_inds = int(size * ratio)
        new_weight1 = np.zeros(shape)
        weight1 = params[key].data()
        weight1_np = weight1.asnumpy()
        weight1_np_abs = -(np.abs(weight1.asnumpy()))
        weight1_np_abs = weight1_np_abs.flatten()
        inds = np.argsort(weight1_np_abs.ravel())
        new_weight1.flat[inds[0:nonzero_inds]] = weight1_np.flat[inds[0:nonzero_inds]]
        mx.nd.array(new_weight1).copyto(params[key].data(ctx))
        inds_all[key] = inds
        weight_w[key] = mx.nd.array(new_weight1)

    params = net.collect_params()
    weight_w = train_sparse_PGP(grad_avg, grad_w, 0, 0, weight_s, inds_all, wid, ctx, epochs=opt.epochs2, loopiter=0)  # fully corrective: SM2 with inds return by previous
    name, val_acc = test([ctx])
    logging.info('[Worker %d] Initial acc: %s=%f' % (wid, name, val_acc))
    train_loss = gettrainingloss([ctx])
    logging.info('[Worker %d] Initial loss: %f' % (wid, train_loss))
    """initialization KVStore"""
    for l, key in enumerate(weight_key):
        key_tmp = key + '_grad'
        grad_key.append(key_tmp)
        logging.info(key_tmp)
        grad_avg.append(mx.nd.zeros(params[key].data(ctx).shape, ctx))
        kv.init(grad_key[l], grad_avg[l])
        kv._barrier()
        kv.pull(grad_key[l], grad_avg[l])
        grad_avg_current.append(mx.nd.zeros(params[key].data(ctx).shape, ctx))
        grad_avg_prev.append(mx.nd.zeros(params[key].data(ctx).shape, ctx))
        grad_avg_prev[l][:] = grad_avg[l]
        grad_avg_current[l][:] = grad_avg[l]

        logging.info(key)
        weight_s.append(mx.nd.zeros(params[key].data(ctx).shape, ctx))
        weight_s[l][:] = params[key].data(ctx)
        weight_s_prev.append(mx.nd.zeros(params[key].data(ctx).shape, ctx))
        weight_s_current.append(mx.nd.zeros(params[key].data(ctx).shape, ctx))
        kv.init(weight_key[l], weight_s[l])
        kv._barrier()
        kv.pull(weight_key[l], weight_s[l])
        weight_s_prev[l][:] = weight_s[l]
        weight_s_current[l][:] = weight_s[l]
    logging.info('init is done')
    compressratio = [0.5, 0.2]
    commiter = 0
    for iter in range(opt.loops):
        if iter%50 == 0 and iter != 0:
            opt.lrsparse *= 0.1
            opt.lrdense *= 0.1
        for subiter in range(opt.loops1):
            grad_w = calgrad(ctx)
            grad_w_list = [grad_w[key] for key in weight_key]  #
            kv.push(grad_key, grad_w_list)
            kv._barrier()
            kv.pull(grad_key, grad_avg_current)
            for l in range(len(grad_key)):
                grad_avg[l] = (grad_avg_current[l] - grad_avg_prev[l])
                grad_avg_prev[l][:] = grad_avg_current[l]

            for l in range(len(grad_key)):
                grad_avg[l] /= opt.nworkers

            weight_w = train_dense_PGP(grad_avg, grad_w, const, gamma, weight_s, wid, ctx, epochs=opt.epochs1, loopiter=iter-1, subloop=subiter)  # training in dense and applied IHT in the last step
            weight_w_list = [weight_w[key] for key in weight_key]

            params = net.collect_params()
            inds_all = {}
            weight_w = {}
            for key in weight_key:
                if 'conv' in key:
                    ratio = compressratio[0]
                elif 'dense' in key:
                    ratio = compressratio[1]

                shape = params[key].data().shape
                size = params[key].data().size
                nonzero_inds = int(size * ratio)
                new_weight1 = np.zeros(shape)
                weight1 = params[key].data()
                weight1_np = weight1.asnumpy()
                weight1_np_abs = -(np.abs(weight1.asnumpy()))
                weight1_np_abs = weight1_np_abs.flatten()
                inds = np.argsort(weight1_np_abs.ravel())
                new_weight1.flat[inds[0:nonzero_inds]] = weight1_np.flat[inds[0:nonzero_inds]]
                mx.nd.array(new_weight1).copyto(params[key].data(ctx))
                inds_all[key] = inds
                weight_w[key] = mx.nd.array(new_weight1)

            print('solving problem on worker')
            weight_w = train_sparse_PGP(grad_avg, grad_w, const, gamma, weight_s, inds_all, wid, ctx, epochs=opt.epochs2, loopiter=iter, subloop=subiter)  # fully corrective: SM2 with inds return by previous
            train_loss = gettrainingloss([ctx])
            name, val_acc = test([ctx])
            logging.info('Worker:[%d] Before Round %d val_acc monitor %f, training_loss: %f' % (wid, commiter, val_acc, train_loss))
            weight_w_list = [weight_w[key] for key in weight_key]
            kv.push(weight_key, weight_w_list)
            kv._barrier()
            kv.pull(weight_key, weight_s_current)
            for l in range(len(weight_key)):
                weight_s[l] = weight_s_current[l] - weight_s_prev[l]
                weight_s_prev[l][:] = weight_s_current[l]

            # iht operation
            for l, key in enumerate(weight_key):
                if 'conv' in key:
                    ratio = compressratio[0]
                elif 'dense' in key:
                    ratio = compressratio[1]

                weight_s[l] /= opt.nworkers  # average
                shape = weight_s[l].shape
                size = weight_s[l].size
                nonzero_inds = int(size * ratio)
                new_weight1 = np.zeros(shape)
                weight1_np = weight_s[l].asnumpy()
                weight1_np_abs = -(np.abs(weight1_np))
                weight1_np_abs = weight1_np_abs.flatten()
                inds = np.argsort(weight1_np_abs.ravel())
                new_weight1.flat[inds[0:nonzero_inds]] = weight1_np.flat[inds[0:nonzero_inds]]
                mx.nd.array(new_weight1).copyto(params[key].data(ctx))
            name_tmp, val_acc = test([ctx])
            logging.info('Worker:[%d] Communication Round %d val_acc monitor:%f' % (wid, commiter, val_acc))

            train_loss = gettrainingloss([ctx])
            if bestacc < val_acc:
                bestacc = val_acc
                if wid == 0:
                    model_name = 'mnist_sdnn_best_BW_w' + str(opt.nworkers) + 'compressed.params'
                    net.save_params(model_name)
            logging.info('Worker %d Communication Round %d training_loss monitor:%f  best_acc = %f' % (wid, commiter, train_loss, bestacc))
            commiter += 1
    if wid == 0:
        # save model
        model_name = 'LeNet3-mnist_sdnn_BW_w' + str(opt.nworkers) + 'compressed.params'
        net.save_params(model_name)
