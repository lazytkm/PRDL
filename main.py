import argparse
import copy
import logging
import sys
import warnings
import os
import random

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, f1_score

from model.model_clam import CLAM_SB, CLAM_MB
from model.transmil import TransMIL
from loader import WSI_Dataset


def aug(feats, mu, log_var, aug_masks, rate):
    if np.random.rand() <= rate:
        # sample from N(0, 1)
        shift = torch.randn(feats.size(2)).cuda()
        
        cutoff = torch.rand(1)

        weight = torch.rand(feats.size(1))
        i = (weight<=cutoff).nonzero(as_tuple=True)[0]
        
        aug_masks = torch.sigmoid(aug_masks)
        mask_tokens = torch.zeros_like(feats[0, i])
        aug_num = 0
        for j in range(aug_masks.shape[0]):
            if np.random.rand() <= rate:
                mask_tokens = mask_tokens + aug_masks[j]
                aug_num += 1
        if aug_num > 1:
            mask_tokens = mask_tokens / aug_num 

        feats[0, i] = mu[0, i] + shift * torch.exp(log_var[0, i] / 2) * mask_tokens
        
    return feats


def train_dist(loader, milnet, aug_masks, criterion, optimizer, args):
    milnet.train()
    total_loss = 0
    for i, (feats, label, mu, logvar) in enumerate(loader):
        optimizer.zero_grad()
        feats = feats.cuda()
        label = label.cuda()
        if args.aug:
            mu = mu.cuda()
            logvar = logvar.cuda()
            feats = aug(feats, mu, logvar, aug_masks, args.rate)
        if args.model == 'clam_sb' or args.model == 'clam_mb':
            logits, Y_prob, Y_hat, _, instance_dict = milnet(feats, label=label, instance_eval=True)
            loss = criterion(logits, label)
            instance_loss = instance_dict['instance_loss']    
            total_loss = 0.7 * loss + 0.3 * instance_loss 
            loss = total_loss
        elif args.model == 'transmil':
            logits, Y_prob, Y_hat, _, _ = milnet(feats)
            loss = criterion(logits, label)
        else:
            raise NotImplementedError
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] loss: %.4f' % (i, len(loader), loss.item()))
    sys.stdout.write('\n')
    return total_loss / len(loader)

def train_feat(loader, milnet, mu_head, log_var_head, aug_masks, criterion, optimizer, args):
    milnet.train()
    total_loss = 0
    for i, (feats, label, _, _) in enumerate(loader):
        optimizer.zero_grad()
        feats = feats.cuda()
        label = label.cuda()
        if args.aug:
            mu = mu_head(feats)
            logvar = log_var_head(feats)
            feats = aug(feats, mu, logvar, aug_masks, args.rate)
        if args.model == 'clam_sb' or args.model == 'clam_mb':
            logits, Y_prob, Y_hat, _, instance_dict = milnet(feats, label=label, instance_eval=True)
            loss = criterion(logits, label)
            instance_loss = instance_dict['instance_loss']    
            total_loss = 0.7 * loss + 0.3 * instance_loss 
            loss = total_loss
        elif args.model == 'transmil':
            logits, Y_prob, Y_hat, _, _ = milnet(feats)
            loss = criterion(logits, label)
        else:
            raise NotImplementedError
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] loss: %.4f' % (i, len(loader), loss.item()))
    sys.stdout.write('\n')
    return total_loss / len(loader)


def test(loader, milnet, criterion, args):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_probs = []
    test_predicts = []
    with torch.no_grad():
        for i, (feats, label, _, _) in enumerate(loader):
            bag_label = label.cuda()
            bag_feats = feats.cuda()
            if args.model == 'clam_sb' or args.model == 'clam_mb':
                logits, Y_prob, Y_hat, _, instance_dict = milnet(bag_feats, label=bag_label, instance_eval=True)
            elif args.model == 'transmil':
                logits, Y_prob, Y_hat, _, _ = milnet(bag_feats)
            else:
                raise NotImplementedError
            test_predicts.append(int(Y_hat))
            test_labels.append(bag_label.item())
            test_probs.append(Y_prob.squeeze().tolist())
    if args.num_classes == 2:
        acc = np.sum(np.array(test_predicts)==np.array(test_labels)) / len(test_labels)
        auc = roc_auc_score(test_labels, np.array(test_probs)[:, 1])
        test_f1_score = f1_score(test_labels, test_predicts, average='macro')
        return auc, test_f1_score, acc
    else:
        acc = np.sum(np.array(test_predicts)==np.array(test_labels)) / len(test_labels)
        micro_auc = roc_auc_score(test_labels, test_probs, multi_class='ovr')
        macro_auc = roc_auc_score(test_labels, test_probs, multi_class='ovo')
        test_f1_score = f1_score(test_labels, test_predicts, average='macro')
        return micro_auc, macro_auc, test_f1_score, acc


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='Train MIL Models with ReMix')
    parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes')
    parser.add_argument('--feats_size', default=384, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs')
    parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 24)')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0, ), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--model', default='clam_mb', type=str,
                        choices=['clam_sb', 'clam_mb', 'transmil'], help='MIL model')
    
    parser.add_argument('--aug', default=True, action='store_true', help='Augmentation')
    parser.add_argument('--status', default='dist', type=str,
                        choices=['feat', 'dist'], help='input')
    parser.add_argument('--weight', default='',
                    help='path to load weights of distribution estimator and masks')
    parser.add_argument('--num_aug', type=int, default=6, help='number of augmentations')
    parser.add_argument('--rate', default=0.7, type=float, help='Augmentation rate')
    
    # Utils
    parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
    parser.add_argument('--list-dir', type=str, default='',
                    help='The directory where slide lists are stored')
    parser.add_argument('--slide-list', type=str, default='',
                        help='slide list')
    parser.add_argument('--train-list', type=str, default='',
                        help='train list')
    parser.add_argument('--data-dir', type=str, default='',
                        help='data directory')
    parser.add_argument('--checkpoint', default='./checkpoint',
                    help='path to save checkpoints')
    args = parser.parse_args()
    
    seed_torch(args.seed)

    print('\nLoading Dataset')

    train_dataset = WSI_Dataset(list_path=os.path.join(args.list_dir, args.train_list),
                                    data_path=args.data_dir, status=args.status)
    val_dataset = WSI_Dataset(list_path=os.path.join(args.list_dir, args.slide_list + '_val_split.csv'),
                                    data_path=args.data_dir)
    test_dataset = WSI_Dataset(list_path=os.path.join(args.list_dir, args.slide_list + '_test.csv'),
                                    data_path=args.data_dir)
    
    train_weights = train_dataset.get_weights()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, sampler=torch.utils.data.WeightedRandomSampler(train_weights, len(train_weights)))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=args.workers)

    # prepare model
    if args.model == 'clam_sb':
        instance_loss_fn = nn.CrossEntropyLoss()
        milnet = CLAM_SB(n_classes=args.num_classes, instance_loss_fn=instance_loss_fn, feat_dim=args.feats_size, k_sample=8, subtyping=True).cuda()
    elif args.model == 'clam_mb':
        instance_loss_fn = nn.CrossEntropyLoss()
        milnet = CLAM_MB(n_classes=args.num_classes, instance_loss_fn=instance_loss_fn, feat_dim=args.feats_size, k_sample=8, subtyping=True).cuda()
    elif args.model == 'transmil':
        milnet = TransMIL(n_classes=args.num_classes, feat_dim=args.feats_size).cuda()

    aug_masks = nn.Parameter(torch.zeros(args.num_aug, args.feats_size), requires_grad = False)

    checkpoint = torch.load(args.weight, map_location=torch.device('cpu'))
    state_dict = checkpoint['teacher']
    for k in list(state_dict.keys()):
        if k.startswith('pra.aug_masks'):
            aug_masks = state_dict[k]
    aug_masks = aug_masks.cuda()
    if args.aug and args.status == 'feat':
        log_var_head = nn.Linear(args.feats_size, args.feats_size)
        mu_head = nn.Linear(args.feats_size, args.feats_size)
        for name, param in log_var_head.named_parameters():
            param.requires_grad = False
        for name, param in mu_head.named_parameters():
            param.requires_grad = False
        state_dict_log_var = {}
        state_dict_mu = {}
        for k in list(state_dict.keys()):
            if k.startswith('pra.log_var_head'):
                state_dict_log_var[k[len("pra.log_var_head."):]] = state_dict[k]
            elif k.startswith('pra.mu_head'):
                state_dict_mu[k[len("pra.mu_head."):]] = state_dict[k]
        msg1 = log_var_head.load_state_dict(state_dict_log_var, strict=False)
        print(msg1)
        msg2 = mu_head.load_state_dict(state_dict_mu, strict=False)
        print(msg2)
        log_var_head = log_var_head.cuda()
        log_var_head.eval()
        mu_head = mu_head.cuda()
        mu_head.eval()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    
    epoch_best = 0
    acc_best = 0
    auc_best = 0
    auc_test = 0
    macro_auc_test = 0
    f1_score_test = 0
    acc_test = 0
    for epoch in range(1, args.num_epochs + 1):
        if args.status == 'dist':
            train_loss_bag = train_dist(train_loader, milnet, aug_masks, criterion, optimizer, args)
        elif args.status == 'feat':
            train_loss_bag = train_feat(train_loader, milnet, mu_head, log_var_head, aug_masks, criterion, optimizer, args)
        logging.info('Epoch [%d/%d] train loss: %.4f' % (epoch, args.num_epochs, train_loss_bag))
        scheduler.step()

        if args.num_classes == 2:
            auc, test_f1_score, acc = test(val_loader, milnet, criterion, args)
            print('Val ACC:', acc, 'AUC:', auc, 'F1 Score:', test_f1_score)
        else:
            auc, macro_auc, test_f1_score, acc = test(val_loader, milnet, criterion, args)
            print('Val ACC:', acc, 'Micro-AUC:', auc, 'Macro-AUC:', macro_auc, 'F1 Score:', test_f1_score)

        if auc > auc_best:
            epoch_best = epoch
            acc_best = acc
            auc_best = auc
            if args.num_classes == 2:
                auc, test_f1_score, acc = test(test_loader, milnet, criterion, args)
                auc_test = auc
                f1_score_test = test_f1_score
                acc_test = acc
                print('Test ACC:', acc, 'AUC:', auc, 'F1 Score:', test_f1_score)
            else:
                auc, macro_auc, test_f1_score, acc = test(test_loader, milnet, criterion, args)
                auc_test = auc
                macro_auc_test = macro_auc
                f1_score_test = test_f1_score
                acc_test = acc
                print('Test ACC:', acc, 'Micro-AUC:', auc, 'Macro-AUC:', macro_auc, 'F1 Score:', test_f1_score)

            os.makedirs(os.path.join(args.checkpoint, args.exp_code), exist_ok=True)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': milnet.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/{}/checkpoint_{:04d}.pth.tar'.format(args.checkpoint, args.exp_code, epoch))

    
    print('best epoch:', epoch_best)
    if args.num_classes == 2:
        print('Test ACC:', acc_test, 'AUC:', auc_test, 'F1 Score:', f1_score_test)
    else:
        print('Test ACC:', acc_test, 'Micro-AUC:', auc_test, 'Macro-AUC:', macro_auc_test, 'F1 Score:', f1_score_test)

    


if __name__ == '__main__':
    main()
