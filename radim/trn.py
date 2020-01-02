'''Pytorch training procedure for the FashionMNIST

Usage:
  %(prog)s  [--no-cuda]
            [--json-path=<path>]
            [--skip-tst-dataset]
            [--seed=<int>]

  %(prog)s (--help | -h)

Options:
    --seed=<int>                            random seed [default: 1]
    --json-path=<path>                      path to the json config file
    --skip-tst-dataset                      skips testing on the testing dataset
    --no-cuda                               disables CUDA training
    -h, --help                              should be help but none is given

See '%(prog)s --help' for more information.
'''
import torch.nn.functional as F


class Criterion:
    MSE = 'mean_squared_error'
    BCE = 'binary_cross_entropy'
    CE = 'cross_entropy'


class BestModel(object):
    def __init__(self, measure_names):
        self.measures = {}
        for measure_name in measure_names:
            self.measures[measure_name] = {'path_to_model': '', 'value': None, 'epoch': None}

    def store_path(self, measure_name, path_to_model, epoch=None, value=None):
        self.measures[measure_name]['path_to_model'] = path_to_model
        self.measures[measure_name]['value'] = value
        self.measures[measure_name]['epoch'] = epoch

    def get_best_model_path(self, measure_name):
        return self.measures[measure_name]['path_to_model']

    def get_best_model_value(self, measure_name):
        return self.measures[measure_name]['value']

    def get_best_model_epoch(self, measure_name):
        return self.measures[measure_name]['epoch']


def trn(model_to_train, model_optimizer, training_epoch, trn_ds, config):
    start = time.time()

    model_to_train.train()

    trn_losses = []
    accs = []
    for batch_idx, (data, target) in enumerate(trn_loader):
        target = target.type(torch.LongTensor)
        trn_ds.cnn_augmentation.to_deterministic()
        if cuda:
            data, target = data.cuda(), target.cuda()

        # import cv2
        # for i in range(data.size()[0]):
        #     cv2.imwrite(os.path.join('c:', os.sep, 'Users', 'jarmi', 'fashion-mnist', 'radim', 'experiments', 'sanity',
        #                              '{:.2f}.jpg'.format(target[i].cpu().numpy())),
        #                 np.transpose(data[i].cpu().numpy(), (1, 2, 0)).astype(np.uint8))
        # raise RuntimeException("hooo")

        model_optimizer.zero_grad()

        batch_output = model_to_train(data)

        if config['trn']['criterion'] == Criterion.CE:
            trn_loss = F.cross_entropy(batch_output.squeeze(), target.squeeze())
        else:
            raise RuntimeError('Unknown criterion {:s}'.format(config['trn']['criterion']))

        val, ind = batch_output.max(dim=1)
        accs.append(float((ind == target).sum()) / float(len(ind)))

        batch_output = batch_output.detach().squeeze().float().cpu().numpy()

        trn_loss.backward()
        model_optimizer.step()

        if len(trn_losses) == 0:
            trn_losses = trn_loss.clone().view(-1)
        else:
            trn_losses = torch.cat((trn_losses, trn_loss.clone().view(-1)), dim=0)

        del data, batch_output, trn_loss

    end = time.time()

    return trn_losses.data.cpu().numpy(), np.mean(accs), end - start


def validate(model_to_validate, loader, training_epoch, val_ds, log_prefix='val'):
    start = time.time()

    model_to_validate.eval()
    with torch.no_grad():
        validation_losses = []
        accs = []
        for batch_idx, (data, target) in enumerate(loader):
            target = target.type(torch.LongTensor)
            if cuda:
                data, target = data.cuda(), target.cuda()

            batch_output = model_to_validate(data)

            if config['trn']['criterion'] == Criterion.CE:
                validation_loss = F.cross_entropy(batch_output.squeeze(), target.squeeze())
            else:
                raise RuntimeError('Unknown criterion {:s}'.format(config['trn']['criterion']))

            val, ind = batch_output.max(dim=1)
            accs.append(float((ind == target).sum()) / float(len(ind)))

            batch_output = batch_output.squeeze().float().cpu().numpy()

            if len(validation_losses) == 0:
                validation_losses = validation_loss.clone().view(-1)
            else:
                validation_losses = torch.cat((validation_losses, validation_loss.clone().view(-1)), dim=0)

            del data, validation_loss, batch_output

    end = time.time()

    return validation_losses.data.cpu().numpy(), np.mean(accs),  end - start


def evaluate_model(model, model_name, losses, epoch_shift, experiments_directory, trn_ds, val_ds):
    lr = float(config['trn']['learning_rate'])

    if float(config['trn']['momentum']) < 10**-7:
        model_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, weight_decay=0.0)
    elif float(config['trn']['momentum']) > 10**3:
        model_optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, weight_decay=0.0)
    else:
        model_optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=lr, momentum=float(config['trn']['momentum']))

    best_models = BestModel(['loss'])

    logger.info(' Learning with the model %s...' % model_name)
    validate(model, val_loader, epoch_shift, val_ds)
    for epoch in range(epoch_shift, int(config['trn']['epochs']) + 1):
        trn_losses, trn_acc, training_time = trn(model, model_optimizer, epoch, trn_ds, config)
        val_losses, val_acc, validation_time = validate(model, val_loader, epoch, val_ds)
        torch.cuda.empty_cache()

        logger.info('[{:04d}][TRN] {:.6f} ACC: {:.2f} ({:.0f}s)'.format(epoch, trn_losses.mean(), trn_acc, training_time))
        logger.info('[{:04d}][VAL] {:.6f} ACC: {:.2f} ({:.0f}s)'.format(epoch, val_losses.mean(), val_acc, validation_time))

        losses[epoch, :] = [trn_losses.mean(), np.median(trn_losses),
                            val_losses.mean(), np.median(val_losses),
                            trn_acc, val_acc]

        summary.add_scalar('trn/loss_avg', losses[epoch, 0], epoch)
        summary.add_scalar('trn/loss_med', losses[epoch, 1], epoch)
        summary.add_scalar('val/loss_avg', losses[epoch, 2], epoch)
        summary.add_scalar('val/loss_med', losses[epoch, 3], epoch)
        summary.add_scalar('trn/acc', losses[epoch, 4], epoch)
        summary.add_scalar('val/acc', losses[epoch, 5], epoch)

        if epoch > 0:
            if losses[epoch, 2] < losses[:epoch, 2].min():
                if not os.path.isdir(os.path.join(experiments_directory, model_name)):
                    os.mkdir(os.path.join(experiments_directory, model_name))
                path_to_save_model = os.path.join(experiments_directory, model_name, 'epoch=%d_val_avg-loss-best' % epoch)
                torch.save(model.state_dict(), path_to_save_model)
            if losses[epoch, 5] > losses[:epoch, 5].max():
                if not os.path.isdir(os.path.join(experiments_directory, model_name)):
                    os.mkdir(os.path.join(experiments_directory, model_name))
                path_to_save_model = os.path.join(experiments_directory, model_name, 'epoch=%d_val_acc-best' % epoch)
                torch.save(model.state_dict(), path_to_save_model)
                best_models.store_path('loss', path_to_save_model, epoch=epoch, value=losses[epoch, 5])

    if best_models.get_best_model_epoch('loss') is not None:
        logger.info(' Restoring %02.1f model: %s' % (
            best_models.get_best_model_value('loss'), best_models.get_best_model_path('loss')))
        model.load_state_dict(torch.load(best_models.get_best_model_path('loss')))

    tst_losses, *_ = validate(model, tst_loader, epoch_shift, val_ds, log_prefix='tst')
    logger.info('[TST] %.6f, ACC: %.2f' % (tst_losses.mean(), _[0]))

    return losses


def prepare_summary(experiments_directory, basename):
    summary_filepath = os.path.join(experiments_directory, 'tensorboard', basename)
    # if os.path.exists(summary_filepath):
    #     raise FileExistsError('Are you sure you want to overwrite %s ?' % summary_filepath)
    summary = SummaryWriter(summary_filepath)

    return summary


def prepare_loaders(config, logger, skip_tst_dataset, multiply_by_number_of_gpu=False):
    from radim.FashionMnistDataset import FashionMnistDataset
    batch_size = int(config['trn']['batch_size'])
    if multiply_by_number_of_gpu:
        batch_size = torch.cuda.device_count() * int(config['trn']['batch_size'])

    logger.info(" Batch size: %d" % batch_size)

    from utils.mnist_reader import load_mnist
    X, y = load_mnist(config['trn']['dataset_dir'], kind='train')
    X_tst, y_tst = load_mnist(config['trn']['dataset_dir'], kind='t10k')

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    trn_idx, val_idx = list(sss.split(X, y))[0]

    logger.info(' There is {:d} samples in the TRN dataset.'.format(len(trn_idx)))
    logger.info(' There is {:d} samples in the VAL dataset.'.format(len(val_idx)))

    trn_ds = FashionMnistDataset(config, X[trn_idx], y[trn_idx], is_trn=True)
    val_ds = FashionMnistDataset(config, X[val_idx], y[val_idx], is_trn=False)

    if not skip_tst_dataset:
        tst_ds = FashionMnistDataset(config, X_tst, y_tst, is_trn=False)
    else:
        logger.warning(' Using val instead of tst!')
        tst_ds = val_ds

    epochs = int(config['trn']['epochs'])
    if torch.cuda.device_count() > 1:
        logger.warning(' Multiplying the size of the batch by the number of GPUs (%d)!' % torch.cuda.device_count())
    trn_loader = torch.utils.data.DataLoader(trn_ds, batch_size=batch_size, num_workers=12 if epochs > -1 else 1)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=12 if epochs > -1 else 1)
    tst_loader = torch.utils.data.DataLoader(tst_ds, batch_size=batch_size, shuffle=False, num_workers=12)

    return trn_loader, val_loader, tst_loader, trn_ds, val_ds, tst_ds


def load_models_prepare_losses_file(experiments_directory):
    epoch_shift = 0
    model_name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')) + '_' + basename

    model_path = os.path.join(experiments_directory, model_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    fileHandler = logging.FileHandler(os.path.join(model_path, 'log'))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    model = load_model(config, cuda)

    losses_filepath = os.path.join(model_path, model_name + '_losses.npy')
    losses = np.zeros((int(config['trn']['epochs']) + 1, 6))
    if os.path.isfile(losses_filepath):
        losses = np.load(losses_filepath)
        if losses.shape[0] != int(config['trn']['epochs']) + 1:
            oldl = losses
            losses = np.zeros((int(config['trn']['epochs']) + 1, losses.shape[1]))
            losses[:oldl.shape[0], :] = oldl

    logger.info(" Models loaded.")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(' The model has %d parameters.' % pytorch_total_params)

    return model_name, model, losses, epoch_shift


def load_model(config, cuda, *model_args):
    import re
    base_model_classname = re.sub(r'[0-9]+', '', config['model']['architecture'])
    base_model_classname = re.sub(r'Metropolis', '', base_model_classname)
    module = __import__('radim.model.%s' % (config['model']['architecture']),
                        fromlist=[base_model_classname])
    class_ = getattr(module, base_model_classname)
    model = class_(cuda, config['model']['input_channels'], *model_args)

    continue_model_path = config['trn']['continue-model']
    continue_model_path_partly = config['trn']['continue-model-partly']
    if len(continue_model_path) > 0:
        logger.info(' Loading model %s' % continue_model_path)
        model.load_state_dict(torch.load(continue_model_path))
    elif len(continue_model_path_partly) > 0:
        logger.info(' Partly loading model %s' % continue_model_path_partly)
        model.load_state_dict_partly(torch.load(continue_model_path_partly))
    if cuda:
        model.cuda()

    # Let's use more GPU!
    if cuda and torch.cuda.device_count() > 1:
        logger.info(" Let's use %d GPUs!" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    return model


def setup_logger():
    __logging_format__ = '[%(levelname)s]%(message)s'
    logFormatter = logging.Formatter(__logging_format__)
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    return logger, logFormatter


if __name__ == '__main__':
    import sys
    import logging
    from docopt import docopt
    import datetime
    import torch
    import torch.optim as optim
    import matplotlib
    import random
    import time
    import json
    import os.path
    import numpy as np
    from tensorboardX import SummaryWriter

    matplotlib.use('agg')
    prog = os.path.basename(sys.argv[0])
    completions = dict(
        prog=prog,
    )
    args = docopt(
        __doc__ % completions,
        argv=sys.argv[1:],
        version='RADIM',
    )

    logger, logFormatter = setup_logger()

    cuda = not bool(args['--no-cuda']) and torch.cuda.is_available()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    with open(args['--json-path']) as config_buffer:
        config = json.loads(config_buffer.read())

    additional_info = ''
    basename = 'arch=%s_lr=%.0E_mom=%.0E_aug=%d_bs=%d_c=%s%s' % (
        config['model']['architecture'],
        float(config['trn']['learning_rate']), float(config['trn']['momentum']),
        int(config['trn']['augment']), int(config['trn']['batch_size']),
        config['trn']['criterion'], additional_info)

    logger.info(config)

    experiments_directory = config['trn']['experiments_dir']

    logger.info(' %s' % basename)
    if config['trn']['augment']:
        logger.info(' Using augmentation...')

    summary = prepare_summary(experiments_directory, basename)

    model_name, model, losses, epoch_shift = load_models_prepare_losses_file(experiments_directory)
    logger.info(config)
    trn_loader, val_loader, tst_loader, trn_ds, val_ds, tst_ds = prepare_loaders(config, logger, args['--skip-tst-dataset'])

    start_time = time.time()
    losses = evaluate_model(model, model_name, losses, epoch_shift, experiments_directory, trn_ds, val_ds)
    logger.info(' The whole learning finished in %.0f s.' % (time.time() - start_time))

    logger.info(' Closing summary...')
    summary.close()

    logger.info(' Succesfully finished...')
