import sys

from torch.utils.data import DistributedSampler
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data.distributed
import torch.utils.data
import torch.multiprocessing as mp
import torch.optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import torch
import os
import argparse
import time
import pickle
import timm
import cv2
import numpy as np
import csv
import math
from PIL import Image
from sklearn.cluster import KMeans

import vision_transformer as vits


# 'Large':40X, 'Medium':20X, 'Small':10X, 'Overview':5X
scales = ['Large', 'Medium', 'Small', 'Overview']
# label map
LABEL_DICT = {'0': 0, '0-0': 0, '0-1': 0, '0-2': 0, '0-3': 0, '0-4': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}

PATCH_NUMBER_PER_ANCHOR = [36, 64, 100, 144, 256, 400]

parser = argparse.ArgumentParser('Extract cnn freatures of whole slide images')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--dist-url', default='tcp://localhost:10002', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--level', type=int, default=1,
                    help='Magnification level. 0-40X 1-20X 2-10X 3-5X')
parser.add_argument('--mask-level', type=int, default=3,
                    help='Magnification level of mask')
parser.add_argument('--tile-size', type=int, default=512,
                    help='The size of tiles')
parser.add_argument('--imsize', type=int, default=224,
                    help='The size of the sampled patches')
parser.add_argument('--step', type=int, default=224,
                    help='The feature extracting step')
parser.add_argument('--feat-dim', type=int, default=384,
                    help='The dimension of features')
parser.add_argument('--max-nodes', type=int, default=4096,
                    help='The maximum number of the extracted features per slide')
parser.add_argument('--intensity-thred', type=int, default=25,
                    help='Intensity threshold for sampling')
parser.add_argument('--num-classes', default=6, type=int, help='num classes')
parser.add_argument('--arch', default='vit_small', type=str, help='model architecture')
parser.add_argument('--cl', default='prdl', type=str, help='use contrastive learning model')
parser.add_argument('--ckp-path', default='', type=str, help='checkpoint path')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='use pretrained model')
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--list-dir', type=str, default='',
                    help='The directory where slide lists are stored')
parser.add_argument('--slide-dir', type=str, default='',
                    help='The directory where the slides are stored')
parser.add_argument('--slide-list', type=str, default='',
                    help='The slide list for sampling')
parser.add_argument('--feat-dir', type=str, default='',
                    help='The directory where the datasets of the extracted features are stored')
parser.add_argument('--invert-rgb', action='store_true', default=False,
                    help='Adjust the format between RGB and BGR\
                        The default color format of the patch is BGR')
parser.add_argument('--invert-xy', action='store_true', default=False)
parser.add_argument('--suffix', type=str, default='')


class SlideLocalTileDataset(data.Dataset):
    def __init__(self, image_dir, position_list, step, transform, class_mat,
            tile_size=512, imsize=224, od_mode=False, invert_rgb=False, invert_xy=False):
        self.transform = transform

        self.im_dir = image_dir
        self.pos = position_list
        self.step = step
        self.od = od_mode
        self.ts = tile_size
        self.imsize = imsize
        self.inv_rgb = invert_rgb
        self.class_mat = class_mat
        self.invert_xy = invert_xy

    def __getitem__(self, index):
        img = extract_tile(self.im_dir, self.ts, self.pos[index][1] * self.step, self.pos[index][0] * self.step, self.imsize, self.imsize, self.invert_xy)
        label = self.class_mat[self.pos[index][0]][self.pos[index][1]]
        if len(img) == 0:
            img = np.ones((self.imsize, self.imsize, 3), np.uint8) * 240
        # The default format of opencv is BGR but PIL.Image is RGB. 
        # So, a cvtColor is required here, to make sure the color 
        # channels are consistent with the trained model.
        if not self.inv_rgb: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        img = self.transform(img)

        if self.od:
            img = -torch.log(img + 1.0/255.0)

        return img, label

    def __len__(self):
        return self.pos.shape[0]


def main(args):
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.distributed:
        if args.dist_url == "env://" and args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])

    # slurmd settings
    # args.rank = int(os.environ["SLURM_PROCID"])
    # args.world_size = int(os.environ["SLURM_NPROCS"])

    ngpus_per_node = torch.cuda.device_count()

    args.wsi_feat_dir = get_feature_path(args)
    if not os.path.exists(args.wsi_feat_dir):
        os.makedirs(args.wsi_feat_dir)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    start_time = time.time()
    
    if args.gpu is not None:
        print("Use GPU: {} for encoding".format(args.gpu))

    if args.distributed:
        if args.rank == -1:
            if args.dist_url == "env://":
                args.rank = int(os.environ["RANK"])
            elif 'SLURM_PROCID' in os.environ:
                args.rank = int(os.environ['SLURM_PROCID'])
                
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    
    # create model
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=16, num_classes=args.num_classes)
        
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=torch.device('cpu'))
            model.load_state_dict({k.replace('module.', ''):v for k, v in checkpoint.items()})
    elif args.cl == 'prdl':
        print("=> loading cl checkpoint '{}'".format(args.cl))
        checkpoint = torch.load(args.ckp_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['teacher']
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

        log_var_head = nn.Linear(args.feat_dim, args.feat_dim)
        mu_head = nn.Linear(args.feat_dim, args.feat_dim)
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
        

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int(args.num_workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
            if args.cl == 'prdl':
                mu_head.cuda(args.gpu)
                mu_head = torch.nn.parallel.DistributedDataParallel(
                    mu_head, device_ids=[args.gpu])
                mu_head.eval()
                log_var_head.cuda(args.gpu)
                log_var_head = torch.nn.parallel.DistributedDataParallel(
                    log_var_head, device_ids=[args.gpu])
                log_var_head.eval()
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    print('Load model time', time.time() - start_time)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    model.eval()
    if 'vit' in args.arch:
        inference_model = model.module if args.distributed else model
    else:
        raise NotImplementedError('The network {} is not supported. \
            You may need to write the feature extraction code \
            for the network you choose.'.format(args.arch))
            
    with open(os.path.join(args.list_dir, args.slide_list + '_train.csv')) as f:
        slide_list_train = list(csv.reader(f))
    with open(os.path.join(args.list_dir, args.slide_list + '_test.csv')) as f:
        slide_list_test = list(csv.reader(f))
    slide_list = slide_list_train + slide_list_test
    
    current_slide_list = []
    for s_id, s_info in enumerate(slide_list):
        feat_save_path = os.path.join(args.wsi_feat_dir, '{}.pkl'.format(s_info[0]))
        slide_path = os.path.join(args.slide_dir, s_info[0])
        if not os.path.exists(feat_save_path) and os.path.exists(slide_path):
            current_slide_list.append(s_info)
    
    # print(current_slide_list)

    for s_id, s_info in enumerate(current_slide_list):
        porc_start = time.time()
        s_guid, s_label = s_info
        s_label = LABEL_DICT[s_label]
        if args.distributed:
            # skip the slides the other gpus are working on
            if not s_id % args.world_size == args.rank:
                continue
        
        feat_save_path = os.path.join(args.wsi_feat_dir, '{}_feat.pkl'.format(s_guid))
        dist_save_path = os.path.join(args.wsi_feat_dir, '{}_dist.pkl'.format(s_guid))
        if os.path.exists(feat_save_path) and os.path.exists(dist_save_path):
            continue

        slide_path = os.path.join(args.slide_dir, s_guid)
        image_dir = os.path.join(slide_path, scales[args.level])

        if os.path.exists(os.path.join(slide_path, 'TissueMask.png')):
            tissue_mask = cv2.imread(os.path.join(slide_path, 'TissueMask.png'), 0)
        else:
            tissue_mask = get_tissue_mask(cv2.imread(
                    os.path.join(slide_path, 'Overview.jpg')))

        content_mat = cv2.blur(
            tissue_mask, ksize=args.filter_size, anchor=(0, 0))
        content_mat = content_mat[::args.frstep, ::args.frstep] > args.intensity_thred
        
        patches_in_graph = np.sum(content_mat)
        if patches_in_graph < 1:
            continue
        
        class_mat = np.full(tissue_mask.shape, s_label)

        mask_path = os.path.join(slide_path, 'AnnotationMask.png')
        
        class_mat = class_mat[::args.frstep, ::args.frstep]
        

        # grid sampling
        sampling_mat = np.copy(content_mat)
        down_factor = 1
        if patches_in_graph > args.max_nodes:
            down_factor = int(np.sqrt(patches_in_graph/args.max_nodes)) + 1
            tmp = np.zeros(sampling_mat.shape, np.uint8) > 0
            tmp[::down_factor,::down_factor] = sampling_mat[::down_factor,::down_factor]
            sampling_mat = tmp
            patches_in_graph = np.sum(sampling_mat)

        # patch position
        patch_pos = np.transpose(np.asarray(np.where(sampling_mat)))
        
        # patch feature
        slide_dataset = SlideLocalTileDataset(image_dir, patch_pos, args.step, transform, class_mat,
                                                args.tile_size, args.imsize, args.invert_rgb)
        slide_loader = torch.utils.data.DataLoader(
            slide_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False)
        features = []
        mus = []
        logvars = []
        infer_start = time.time()
        for images, labels in slide_loader:
            images = images.cuda(args.gpu, non_blocking=True)
            with torch.no_grad():
                if 'vit' in args.arch:
                    intermediate_output = inference_model.get_intermediate_layers(images, 1)
                    x = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                else:
                    raise NotImplementedError('The network {} is not supported. \
                        You may need to write the feature extraction code \
                        for the network you choose.'.format(args.arch))
                features.append(x.cpu().numpy())

                if args.cl == 'prdl':
                    mu = mu_head(x)
                    logvar = log_var_head(x)
                    mus.append(mu.cpu().numpy())
                    logvars.append(logvar.cpu().numpy())
        features = np.concatenate(features, axis=0)
        if args.cl == 'prdl':
            mus = np.concatenate(mus, axis=0)
            logvars = np.concatenate(logvars, axis=0)
        print(features.shape, mus.shape, logvars.shape)

        # save features
        with open(feat_save_path, 'wb') as f:
            graph = {
                'cm':content_mat,
                'feats':features,
                'pos':patch_pos,
                'down_factor':down_factor,
                'label':s_label,
            }
            pickle.dump(graph, f)
        # save distributions
        with open(dist_save_path, 'wb') as f:
            graph = {
                'cm':content_mat,
                'mu':mus,
                'logvar':logvars,
                'pos':patch_pos,
                'down_factor':down_factor,
                'label':s_label,
            }
            pickle.dump(graph, f)

        print('Processer #{}: {}/{} {}'.format(args.rank, s_id, len(current_slide_list), s_guid),
            '#patch:', patch_pos.shape[0], 
            'df:', down_factor, 
            'labels:', s_label,
            'time:', time.time() - porc_start,
            'inference time:', time.time() - infer_start
        )


def get_feature_path(args):
    if args.pretrained:
        prefix = '[{}_pre][fs{}fts{}]'.format(args.arch, args.step, args.feat_dim)
    else:
        if args.cl:
            prefix = '[{}_{}][fs{}fts{}]'.format(args.arch, args.cl, args.step, args.feat_dim)
        else:
            prefix = '[{}][fs{}fts{}]'.format(args.arch, args.step, args.feat_dim)

    return os.path.join(args.feat_dir, prefix + args.suffix)


def get_tissue_mask(temp_image, scale=30):
    image_hsv = cv2.cvtColor(temp_image, cv2.COLOR_BGR2HSV)
    _, tissueMask = cv2.threshold(image_hsv[:, :, 1],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * scale + 1, 2 * scale + 1))
    tissueMask = cv2.morphologyEx(tissueMask, cv2.MORPH_CLOSE, element)

    return tissueMask


def extract_tile(image_dir, tile_size, x, y, width, height, invert_xy):
    x_start_tile = x // tile_size
    y_start_tile = y // tile_size
    x_end_tile = (x+width) // tile_size
    y_end_tile = (y+height) // tile_size

    tmp_image = np.ones(
        ((y_end_tile-y_start_tile+1)*tile_size, (x_end_tile-x_start_tile+1)*tile_size, 3),
        np.uint8)*240

    for y_id, col in enumerate(range(x_start_tile, x_end_tile + 1)):
        for x_id, row in enumerate(range(y_start_tile, y_end_tile + 1)):
            img_path = os.path.join(image_dir, '{:04d}_{:04d}.jpg'.format(row,col))
            if invert_xy:
                img_path = os.path.join(image_dir, '{:04d}_{:04d}.jpg'.format(col,row))
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            tmp_image[(x_id*tile_size):(x_id*tile_size + h), (y_id*tile_size):(y_id*tile_size + w),:] = img

    x_off = x % tile_size
    y_off = y % tile_size
    output = tmp_image[y_off:y_off+height, x_off:x_off+width]
    
    return output


if __name__ == '__main__':
    args = parser.parse_args()

    args.rl = args.mask_level - args.level
    args.frstep = args.step>>args.rl
    args.filter_size = (args.imsize >> args.rl, args.imsize >> args.rl)

    main(args)
