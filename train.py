import os
import sys
import time
import torch
import numpy as np
import copy
import math
import argparse
from typing import List, Any
from torch.utils.data import DataLoader
from utils import *
from alg import *
from dataloader import get_img_dataloader
from Networks import DSANN,GeneDistrNet,ReverseLayerF

def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam hyper-param')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dim of sonnet')
    parser.add_argument('--input_size', type=int, default=2048, help='the input size of distribution generator')
    parser.add_argument('--latent_size', type=int, default=4096, help='the latent size of distribution generator')
    parser.add_argument('--checkpoint_freq', type=int, default=4, help='Checkpoint every N epoch')
    parser.add_argument('--data_file', type=str, default='/kaggle/input/pacsppppp/Homework3-PACS-master/',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='PACS')
    parser.add_argument('--data_dir', type=str, default='PACS', help='data dir')
    parser.add_argument('--dis_hidden', type=int, default=256, help='discriminator hidden dimension')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--layer', type=str, default="bn", choices=["ori", "bn"], help='bottleneck normalization style')
    parser.add_argument('--lr0', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr1', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_gamma', type=float, default=0.0003, help='for optimizer')
    parser.add_argument('--lr_decay1', type=float, default=1.0, help='for pretrained featurizer')
    parser.add_argument('--epochs', type=int, default=50, help="max iterations of pretrain")
    parser.add_argument('--momentum', type=float, default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet18',
                        help="featurizer: resnet18, resnet50, resnet101, AlexNet")
    parser.add_argument('--N_WORKERS', type=int, default=2)
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='start',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--test_envs', type=int, nargs='+', default=[1], help='target domains')
    parser.add_argument('--output', type=str, default="train_output", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--alpha', type=float, default=1, help='transfer_loss_weight')
    args = parser.parse_args()
    args.step_per_epoch = 100
    args.data_dir = args.data_file + args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)
    eval_loaders: List[DataLoader[Any]]
    train_loaders, eval_loaders = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)
    s = print_args(args, [])
    print('=======hyper-parameter used========')
    print(s)
    model = DSANN(args.net, len(args.domains) - len(args.test_envs), args.num_classes).cuda()
    generator = GeneDistrNet(args.input_size, model.input_size, args.num_classes, args.latent_size).cuda()
    train_minibatches_iterator = zip(*train_loaders)

    # adapt_train(update D->update F,H,C->update G)
    params = model.get_parameters1(initial_lr=args.lr0)
    params.append({'params': generator.parameters(), 'lr': args.lr0})
    opt = torch.optim.SGD(params, lr=args.lr0, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    sch = get_scheduler(opt, args)
    criterion_cls = torch.nn.CrossEntropyLoss()
    entropy_criterion = HLoss()
    criterion_dis = torch.nn.BCELoss()

    best_valid_acc = 0
    acc_type_list = ['train', 'valid', 'target']
    print('===========start training===========')
    sss = time.time()
    for epoch in range(args.epochs):
        p = epoch / (args.epochs - 1)
        alpha = (2 / (1 + math.exp(-10 * p)) - 1) * 1
        for iter_num in range(args.step_per_epoch):
            minibatches_device = [(data) for data in next(train_minibatches_iterator)]
            # for i,data in enumerate(minibatches_device):
            fea_source1 = model.base_network(minibatches_device[0][0].cuda().float())
            fea_source2 = model.base_network(minibatches_device[1][0].cuda().float())
            fea_source3 = model.base_network(minibatches_device[2][0].cuda().float())
            y_source1 = torch.zeros(minibatches_device[0][1].size(0), args.num_classes).cuda()
            y_source1.scatter_(1, minibatches_device[0][1].cuda().view(-1, 1), 0.7).cuda()
            y_source2 = torch.zeros(minibatches_device[1][1].size(0), args.num_classes).cuda()
            y_source2.scatter_(1, minibatches_device[1][1].cuda().view(-1, 1), 0.7).cuda()
            y_source3 = torch.zeros(minibatches_device[2][1].size(0), args.num_classes).cuda()
            y_source3.scatter_(1, minibatches_device[2][1].cuda().view(-1, 1), 0.7).cuda()

            pred_source1 = model.classifiers[0](fea_source1)
            pred_source2 = model.classifiers[1](fea_source2)
            pred_source3 = model.classifiers[2](fea_source3)

            classifier_loss = criterion_cls(pred_source1, minibatches_device[0][1].cuda().long())
            classifier_loss += criterion_cls(pred_source2, minibatches_device[1][1].cuda().long())
            classifier_loss += criterion_cls(pred_source3, minibatches_device[2][1].cuda().long())

            input_source1 = ReverseLayerF.apply(fea_source1, alpha)
            input_source2 = ReverseLayerF.apply(fea_source2, alpha)
            input_source3 = ReverseLayerF.apply(fea_source3, alpha)
            randomn = torch.rand(minibatches_device[0][1].size(0), args.input_size).cuda()
            fea_DG1 = generator(y=y_source1, x=randomn)
            input_DG1 = ReverseLayerF.apply(fea_DG1, 1)
            fea_DG2 = generator(y=y_source2, x=randomn)
            input_DG2 = ReverseLayerF.apply(fea_DG2, 1)
            fea_DG3 = generator(y=y_source3, x=randomn)
            input_DG3 = ReverseLayerF.apply(fea_DG3, 1)
            realz1 = model.Discriminators[0](y_source1, input_DG1)
            fakez1 = model.Discriminators[0](y_source1, input_source1)
            loss_dis = criterion_dis(fakez1, torch.ones(y_source1.size(0), 1).float().cuda()) \
                       + criterion_dis(realz1, torch.zeros(y_source1.size(0),1).float().cuda())
            realz2 = model.Discriminators[1](y_source2, input_DG2)
            fakez2 = model.Discriminators[1](y_source2, input_source2)
            loss_dis += criterion_dis(fakez2, torch.ones(y_source2.size(0), 1).float().cuda()) \
                        + criterion_dis(realz2, torch.zeros( y_source2.size(0),1).float().cuda())
            realz3 = model.Discriminators[2](y_source3, input_DG3)
            fakez3 = model.Discriminators[2](y_source3, input_source3)
            loss_dis += criterion_dis(fakez3, torch.ones(y_source3.size(0), 1).float().cuda()) \
                        + criterion_dis(realz3, torch.zeros( y_source3.size(0),1).float().cuda())
            gen_loss = criterion_dis(realz1, torch.zeros(y_source1.size(0), 1).float().cuda()) \
                       + criterion_dis(realz2, torch.zeros( y_source1.size(0), 1).float().cuda()) \
                       + criterion_dis(realz3, torch.zeros(y_source1.size(0), 1).float().cuda())


            all_x = torch.cat([data[0].cuda().float() for data in minibatches_device])
            all_feature = model.base_network(all_x)
            all_pred1 = model.classifiers[0](all_feature)
            all_pred2 = model.classifiers[1](all_feature)
            all_pred3 = model.classifiers[2](all_feature)
            l1_loss = torch.mean(torch.abs(F.softmax(all_pred1, dim=1) - F.softmax(all_pred2, dim=1)))
            l1_loss += torch.mean(torch.abs(F.softmax(all_pred1, dim=1) - F.softmax(all_pred3, dim=1)))
            l1_loss += torch.mean(torch.abs(F.softmax(all_pred2, dim=1) - F.softmax(all_pred3, dim=1)))
            loss = classifier_loss + loss_dis * 0.5 + alpha * l1_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            if sch:
                sch.step()

        if (epoch in [int(args.epochs * 0.7), int(args.epochs * 0.9)]) and (not args.schuse):
            print('manually descrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr'] * 0.1

        if (epoch == (args.epochs - 1)) or (epoch % args.checkpoint_freq == 0):
            print('===========epoch %d===========' % (epoch))
            s = ''
            # s += ('cls_loss:%.4f,' % loss)
            s += ('cls_loss:%.4f,' % classifier_loss) + ('dis_loss:%.4f,' % loss_dis) + (
                        'gen_loss:%.4f,' % gen_loss) + ('l1_loss:%.4f,' % l1_loss)
            print(s[:-1])
            s = ''
            acc_record = {}
            model.eval()
            with torch.no_grad():
                for item in acc_type_list:
                    acc_record[item] = np.mean(
                        np.array([test(model.cuda(), eval_loaders[i], mode='avg') for i in eval_name_dict[item]]))
                    s += (item + '_acc:%.4f,' % acc_record[item])
                print(s[:-1])
            if acc_record['valid'] > best_valid_acc:
                best_valid_acc = acc_record['valid']
                best_model_msa = copy.deepcopy(model.state_dict())
                best_model_generator = copy.deepcopy(generator.state_dict())
            print('total cost time: %.4f' % (time.time() - sss))
            model.train()

    model.load_state_dict(best_model_msa)
    generator.load_state_dict(best_model_generator)
    save_checkpoint('msa1.pkl', model, args)
    save_checkpoint('generator1.pkl', generator, args)
    print('result: %.4f' % best_valid_acc)
    with open(os.path.join(args.output, f'done.txt'), 'w') as f:
        f.write('done\n')
        f.write('total cost time:%s\n' % (str(time.time() - sss)))
        f.write('best_valid_acc:%.4f' % (best_valid_acc))

    # test
    with torch.no_grad():
        target_acc = np.mean(np.array([test(model.cuda().eval(), eval_loaders[i]) for i in eval_name_dict['target']]))
        print('target_acc:%.4f' % target_acc)
        target_acc = np.mean(
            np.array([test(model.cuda().eval(), eval_loaders[i], mode='avg') for i in eval_name_dict['target']]))
        print('target_acc:%.4f' % target_acc)