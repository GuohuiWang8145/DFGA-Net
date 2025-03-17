import sys

import torch
import torch.nn.functional as F
import argparse
import os

from torch.utils.data import DataLoader
from dataloader.datasets import build_dataset
from model.DFGANet import DFGANet
# from torch.utils.tensorboard import SummaryWriter
from utils import misc
from loss.stereo_metric import d1_metric
from utils.visualization import disp_error_img, save_images
from evaluate import (validate_things, validate_kitti15, validate_eth3d,
                      validate_middlebury, create_sceneflow_submission, create_kitti_submission,
                      create_eth3d_submission,
                      create_middlebury_submission,
                      inference_stereo,
                      )
from utils.utils import EarlyStopping

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='kitti15mix', type=str,
                        help='training stage on different datasets')
    parser.add_argument('--val_dataset', default=['kitti15'], type=str, nargs='+')
    parser.add_argument('--max_disp', default=400, type=int,
                        help='exclude very large disparity in the loss function')
    parser.add_argument('--img_height', default=288, type=int)
    parser.add_argument('--img_width', default=512, type=int)
    parser.add_argument('--padding_factor', default=16, type=int)

    # training
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--accum_iter', default=2, type=int)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--minlr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--task', default='stereo', choices=['flow', 'stereo', 'depth'], type=str)
    parser.add_argument('--num_scales', default=1, type=int,
                        help='feature scales: 1/8 or 1/8 + 1/4')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--reg_refine', default=False, type=bool,
                        help='optional task-specific local regression refinement')
    parser.add_argument('--num_reg_refine', default=3, type=int,
                        help='number of additional local regression refinement')
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--patience', default=7, type=int)

    # model: parameter-free
    parser.add_argument('--attn_type', default='self_swin2d_cross_1d', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[1], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')

    # eval
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+')
    parser.add_argument('--count_time', action='store_true')
    parser.add_argument('--save_vis_disp', action='store_true')
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--middlebury_resolution', default='F', choices=['Q', 'H', 'F'])

    # submission
    parser.add_argument('--submission', action='store_true')
    parser.add_argument('--eth_submission_mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--middlebury_submission_mode', default='training', type=str, choices=['training', 'test'])
    parser.add_argument('--output_path', default='output', type=str)

    # log
    parser.add_argument('--summary_freq', default=10, type=int, help='Summary frequency to tensorboard (iterations)')
    parser.add_argument('--save_ckpt_freq', default=1, type=int, help='Save checkpoint frequency (steps)')
    parser.add_argument('--val_freq', default=200, type=int, help='validation frequency in terms of training steps')
    parser.add_argument('--save_latest_ckpt_freq', default=1, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)

    # resume
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--no_resume_optimizer', action='store_true')
    parser.add_argument('--resume_exclude_upsampler', action='store_true')

    # distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--launcher', default='none', type=str)
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # inference
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_dir_left', default=None, type=str)
    parser.add_argument('--inference_dir_right', default=None, type=str)
    parser.add_argument('--pred_bidir_disp', action='store_true',
                        help='predict both left and right disparities')
    parser.add_argument('--pred_right_disp', action='store_true',
                        help='predict right disparity')
    parser.add_argument('--save_pfm_disp', action='store_true',
                        help='save predicted disparity as .pfm format')

    parser.add_argument('--debug', action='store_true')

    return parser


def main(args):
    log_file = os.path.join(args.checkpoint_dir, 'log_train.txt')
    if 'things' in args.val_dataset:
        early_stopping_things = EarlyStopping(patience=args.patience, verbose=True)
    if 'kitti15' in args.val_dataset:
        early_stopping_kitti15 = EarlyStopping(patience=args.patience, verbose=True)
    if 'kitti12' in args.val_dataset:
        early_stopping_kitti12 = EarlyStopping(patience=args.patience, verbose=True)
    if 'eth3d' in args.val_dataset:
        early_stopping_eth3d = EarlyStopping(patience=args.patience, verbose=True)
    if 'middlebury' in args.val_dataset:
        early_stopping_middlebury = EarlyStopping(patience=args.patience, verbose=True)

    print_info = not args.eval and not args.submission and args.inference_dir is None and \
                 args.inference_dir_left is None and args.inference_dir_right is None

    if print_info and args.local_rank == 0:
        print(args)

        misc.save_args(args)
        misc.check_path(args.checkpoint_dir)
        misc.save_command(args.checkpoint_dir)

    misc.check_path(args.output_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = False

    flag = torch.cuda.device_count()
    print(flag)

    if args.reg_refine:
        print("refine:%d" % args.num_reg_refine)
    else:
        print("no refine")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model
    model = DFGANet(feature_channels=args.feature_channels,
                    num_scales=args.num_scales,
                    upsample_factor=args.upsample_factor,
                    num_head=args.num_head,
                    num_transformer_layers=args.num_transformer_layers,
                    reg_refine=args.reg_refine
                    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)
    with open(log_file, 'a') as f:
        f.write('Number of params: %d\n' % num_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # summary_writer = SummaryWriter(args.checkpoint_dir)

    model.to(device)
    model_without_ddp = model
    # model_without_ddp = torch.compile(model, mode="max-autotune")

    train_data = build_dataset(args)
    start_step = 0
    start_epoch = 0

    if args.resume:
        print("=> Load checkpoint: %s" % args.resume)

        loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)
        # for key, value in checkpoint["model"].items():
        #     print(key, value.size(), sep="  ")

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not \
                args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']

        if print_info:
            print('start_epoch: %d' % start_epoch)

    if args.submission:
        if 'things' in args.val_dataset:
            create_sceneflow_submission(model_without_ddp,
                                        dataset=args.val_dataset,
                                        output_path=args.output_path,
                                        padding_factor=args.padding_factor,
                                        attn_type=args.attn_type,
                                        attn_splits_list=args.attn_splits_list,
                                        corr_radius_list=args.corr_radius_list,
                                        prop_radius_list=args.prop_radius_list,
                                        num_reg_refine=args.num_reg_refine,
                                        inference_size=args.inference_size,
                                        )
        if 'kitti15' in args.val_dataset or 'kitti12' in args.val_dataset:
            create_kitti_submission(model_without_ddp,
                                    dataset=args.val_dataset,
                                    output_path=args.output_path,
                                    padding_factor=args.padding_factor,
                                    attn_type=args.attn_type,
                                    attn_splits_list=args.attn_splits_list,
                                    corr_radius_list=args.corr_radius_list,
                                    prop_radius_list=args.prop_radius_list,
                                    num_reg_refine=args.num_reg_refine,
                                    inference_size=args.inference_size,
                                    )

        if 'eth3d' in args.val_dataset:
            create_eth3d_submission(model_without_ddp,
                                    output_path=args.output_path,
                                    padding_factor=args.padding_factor,
                                    attn_type=args.attn_type,
                                    attn_splits_list=args.attn_splits_list,
                                    corr_radius_list=args.corr_radius_list,
                                    prop_radius_list=args.prop_radius_list,
                                    num_reg_refine=args.num_reg_refine,
                                    inference_size=args.inference_size,
                                    submission_mode=args.eth_submission_mode,
                                    save_vis_disp=args.save_vis_disp,
                                    )

        if 'middlebury' in args.val_dataset:
            create_middlebury_submission(model_without_ddp,
                                         output_path=args.output_path,
                                         padding_factor=args.padding_factor,
                                         attn_type=args.attn_type,
                                         attn_splits_list=args.attn_splits_list,
                                         corr_radius_list=args.corr_radius_list,
                                         prop_radius_list=args.prop_radius_list,
                                         num_reg_refine=args.num_reg_refine,
                                         inference_size=args.inference_size,
                                         submission_mode=args.middlebury_submission_mode,
                                         save_vis_disp=args.save_vis_disp,
                                         )

        return

    if args.eval:
        val_results = {}

        if 'things' in args.val_dataset:
            results_dict = validate_things(model_without_ddp,
                                           max_disp=args.max_disp,
                                           padding_factor=args.padding_factor,
                                           inference_size=args.inference_size,
                                           attn_type=args.attn_type,
                                           attn_splits_list=args.attn_splits_list,
                                           corr_radius_list=args.corr_radius_list,
                                           prop_radius_list=args.prop_radius_list,
                                           num_reg_refine=args.num_reg_refine,
                                           )

            if args.local_rank == 0:
                val_results.update(results_dict)

        if 'kitti15' in args.val_dataset or 'kitti12' in args.val_dataset:
            results_dict = validate_kitti15(model_without_ddp,
                                            dataset=args.val_dataset,
                                            padding_factor=args.padding_factor,
                                            inference_size=args.inference_size,
                                            attn_type=args.attn_type,
                                            attn_splits_list=args.attn_splits_list,
                                            corr_radius_list=args.corr_radius_list,
                                            prop_radius_list=args.prop_radius_list,
                                            num_reg_refine=args.num_reg_refine,
                                            count_time=args.count_time,
                                            debug=args.debug,
                                            )

            if args.local_rank == 0:
                val_results.update(results_dict)

        if 'eth3d' in args.val_dataset:
            results_dict = validate_eth3d(model_without_ddp,
                                          padding_factor=args.padding_factor,
                                          inference_size=args.inference_size,
                                          attn_type=args.attn_type,
                                          attn_splits_list=args.attn_splits_list,
                                          corr_radius_list=args.corr_radius_list,
                                          prop_radius_list=args.prop_radius_list,
                                          num_reg_refine=args.num_reg_refine,
                                          )

            if args.local_rank == 0:
                val_results.update(results_dict)

        if 'middlebury' in args.val_dataset:
            results_dict = validate_middlebury(model_without_ddp,
                                               padding_factor=args.padding_factor,
                                               inference_size=args.inference_size,
                                               attn_type=args.attn_type,
                                               attn_splits_list=args.attn_splits_list,
                                               corr_radius_list=args.corr_radius_list,
                                               prop_radius_list=args.prop_radius_list,
                                               num_reg_refine=args.num_reg_refine,
                                               resolution=args.middlebury_resolution,
                                               )

            if args.local_rank == 0:
                val_results.update(results_dict)

        return

    if args.inference_dir or (args.inference_dir_left and args.inference_dir_right):
        inference_stereo(model_without_ddp,
                         inference_dir=args.inference_dir,
                         inference_dir_left=args.inference_dir_left,
                         inference_dir_right=args.inference_dir_right,
                         output_path=args.output_path,
                         padding_factor=args.padding_factor,
                         inference_size=args.inference_size,
                         attn_type=args.attn_type,
                         attn_splits_list=args.attn_splits_list,
                         corr_radius_list=args.corr_radius_list,
                         prop_radius_list=args.prop_radius_list,
                         num_reg_refine=args.num_reg_refine,
                         pred_bidir_disp=args.pred_bidir_disp,
                         pred_right_disp=args.pred_right_disp,
                         save_pfm_disp=args.save_pfm_disp,
                         )

        return

    print('=> {} training samples found in the training set'.format(len(train_data)))

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True,
                              sampler=None,
                              )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.minlr
    )
    # total_steps = start_step
    epoch = start_epoch

    accum_iter = args.accum_iter
    print('=> Start training...')

    while epoch <= args.num_epochs:
        model.train()

        # if args.local_rank == 0:
        # summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], epoch + 1)

        early_stopping = False
        # validation
        if epoch % args.val_freq == 0 and epoch > 0:
            val_results = {}

            if 'things' in args.val_dataset:
                results_dict = validate_things(model_without_ddp,
                                               max_disp=args.max_disp,
                                               padding_factor=args.padding_factor,
                                               inference_size=args.inference_size,
                                               attn_type=args.attn_type,
                                               attn_splits_list=args.attn_splits_list,
                                               corr_radius_list=args.corr_radius_list,
                                               prop_radius_list=args.prop_radius_list,
                                               num_reg_refine=args.num_reg_refine,
                                               count_time=args.count_time,
                                               )
                val_loss = results_dict['things_epe']
                early_stopping = early_stopping_things(val_loss, model_without_ddp)
                if early_stopping_things.early_stop:
                    print("Early stopping(things)")
                    with open(log_file, 'a') as f:
                        f.write("Early stopping:%d epoch" % epoch)
                    break
                if args.local_rank == 0:
                    val_results.update(results_dict)

            if 'kitti15' in args.val_dataset or 'kitti12' in args.val_dataset:
                results_dict = validate_kitti15(model_without_ddp,
                                                dataset=args.val_dataset,
                                                padding_factor=args.padding_factor,
                                                inference_size=args.inference_size,
                                                attn_type=args.attn_type,
                                                attn_splits_list=args.attn_splits_list,
                                                corr_radius_list=args.corr_radius_list,
                                                prop_radius_list=args.prop_radius_list,
                                                num_reg_refine=args.num_reg_refine,
                                                count_time=args.count_time,
                                                )
                val_loss = results_dict['kitti15_epe']
                early_stopping = early_stopping_kitti15(val_loss, model_without_ddp)
                if early_stopping_kitti15.early_stop:
                    print("Early stopping(kitti15)")
                    with open(log_file, 'a') as f:
                        f.write("Early stopping:%d epoch" % epoch)
                    break

                if args.local_rank == 0:
                    val_results.update(results_dict)

            # if 'kitti12' in args.val_dataset:
            #     results_dict = validate_kitti12(model_without_ddp,
            #                                     padding_factor=args.padding_factor,
            #                                     inference_size=args.inference_size,
            #                                     attn_type=args.attn_type,
            #                                     attn_splits_list=args.attn_splits_list,
            #                                     corr_radius_list=args.corr_radius_list,
            #                                     prop_radius_list=args.prop_radius_list,
            #                                     num_reg_refine=args.num_reg_refine,
            #                                     count_time=args.count_time,
            #                                     )
            #     val_loss = results_dict['kitti12_epe']
            #     early_stopping = early_stopping_kitti12(val_loss, model_without_ddp)
            #     if early_stopping_kitti12.early_stop:
            #         print("Early stopping(kitti12)")
            #         with open(log_file, 'a') as f:
            #             f.write("Early stopping:%d epoch" % epoch)
            #         break
            #
            #     if args.local_rank == 0:
            #         val_results.update(results_dict)

            if 'eth3d' in args.val_dataset:
                results_dict = validate_eth3d(model_without_ddp,
                                              padding_factor=args.padding_factor,
                                              inference_size=args.inference_size,
                                              attn_type=args.attn_type,
                                              attn_splits_list=args.attn_splits_list,
                                              corr_radius_list=args.corr_radius_list,
                                              prop_radius_list=args.prop_radius_list,
                                              num_reg_refine=args.num_reg_refine,
                                              count_time=args.count_time,
                                              )
                val_loss = results_dict['eth3d_epe']
                early_stopping = early_stopping_eth3d(val_loss, model_without_ddp)
                if early_stopping_eth3d.early_stop:
                    print("Early stopping(eth3d)")
                    with open(log_file, 'a') as f:
                        f.write("Early stopping:%d epoch" % epoch)
                    break
                if args.local_rank == 0:
                    val_results.update(results_dict)

            if 'middlebury' in args.val_dataset:
                results_dict = validate_middlebury(model_without_ddp,
                                                   padding_factor=args.padding_factor,
                                                   inference_size=args.inference_size,
                                                   attn_type=args.attn_type,
                                                   attn_splits_list=args.attn_splits_list,
                                                   corr_radius_list=args.corr_radius_list,
                                                   prop_radius_list=args.prop_radius_list,
                                                   num_reg_refine=args.num_reg_refine,
                                                   )
                val_loss = results_dict['middlebury_epe']
                early_stopping = early_stopping_middlebury(val_loss, model_without_ddp)
                if early_stopping_middlebury.early_stop:
                    print("Early stopping(middlebury)")
                    with open(log_file, 'a') as f:
                        f.write("Early stopping:%d epoch" % epoch)
                    break
                if args.local_rank == 0:
                    val_results.update(results_dict)

            if args.local_rank == 0:
                # save to tensorboard
                for key in val_results:
                    tag = key.split('_')[0]
                    tag = tag + '/' + key
                    # summary_writer.add_scalar(tag, val_results[key], epoch)

                # save validation results to file
                val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
                with open(val_file, 'a') as f:
                    f.write('epoch: %06d\n' % epoch)
                    # order of metrics
                    metrics = ['things_epe', 'things_d1',
                               'kitti15_epe', 'kitti15_d1', 'kitti15_3px',
                               'kitti12_epe', 'kitti12_3px', 'kitti12_4px',
                               'eth3d_epe', 'eth3d_1px',
                               'middlebury_epe', 'middlebury_2px',
                               ]

                    eval_metrics = []
                    for metric in metrics:
                        if metric in val_results.keys():
                            eval_metrics.append(metric)

                    metrics_values = [val_results[metric] for metric in eval_metrics]

                    num_metrics = len(eval_metrics)

                    f.write(("| {:>20} " * num_metrics + '\n').format(*eval_metrics))
                    f.write(("| {:20.4f} " * num_metrics).format(*metrics_values))

                    f.write('\n\n')
            model.train()
        # training
        if epoch < args.num_epochs and not early_stopping:
            for total_steps, sample in enumerate(train_loader, 0):
                left = sample['left'].to(device)  # [B, 3, H, W]
                right = sample['right'].to(device)
                gt_disp = sample['disp'].to(device)  # [B, H, W]

                mask = (gt_disp > 0) & (gt_disp < args.max_disp)

                if not mask.any():
                    continue

                pred_disps = model(left, right,
                                   attn_type=args.attn_type,
                                   attn_splits_list=args.attn_splits_list,
                                   corr_radius_list=args.corr_radius_list,
                                   prop_radius_list=args.prop_radius_list,
                                   num_reg_refine=args.num_reg_refine,
                                   task='stereo',
                                   )['disp_preds']
                disp_loss = 0

                # loss weights
                loss_weights = [0.9 ** (len(pred_disps) - 1 - power) for power in
                                range(len(pred_disps))]

                for k in range(len(pred_disps)):
                    pred_disp = pred_disps[k]
                    weight = loss_weights[k]
                    disp_loss += weight * F.smooth_l1_loss(pred_disp[mask], gt_disp[mask],
                                                           reduction='mean')

                total_loss = disp_loss

                total_loss.backward()

                if ((total_steps + 1) % accum_iter == 0) or (total_steps + 1 == len(train_loader)):

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()

                    # more efficient zero_grad
                    for param in model.parameters():
                        param.grad = None

                    pred_disp = pred_disps[-1]

                    epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')

                    d1 = d1_metric(pred_disp, gt_disp, mask)
                    if ((total_steps + 1) / accum_iter) % args.summary_freq == 0 or total_steps == 0:
                        print('[epoch:%2d]step: %03d \t epe: %.3f \t d1: %0.3f \t total_loss:%.3f' % (epoch + 1,
                                                                                                      (
                                                                                                                  total_steps + 1) / accum_iter,
                                                                                                      epe.item(),
                                                                                                      d1.item(),
                                                                                                      total_loss))
                        with open(log_file, 'a') as f:
                            f.write('[epoch:%2d]step: %03d/%03d \t epe: %.3f \t total_loss:%.3f\n' % (epoch + 1,
                                                                                                      (
                                                                                                                  total_steps + 1) / accum_iter,
                                                                                                      len(train_loader) / accum_iter,
                                                                                                      epe.item(),
                                                                                                      total_loss))

        lr_scheduler.step()
        # always save the latest model for resuming training
        if epoch % args.save_latest_ckpt_freq == 0:
            # Save lastest checkpoint after each epoch
            checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

            save_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

            torch.save(save_dict, checkpoint_path)

        # save checkpoint of specific epoch
        if epoch % args.save_ckpt_freq == 0:
            print('Save checkpoint at epoch: %d' % epoch)
            # sys.stdout.flush()
            checkpoint_path = os.path.join(args.checkpoint_dir, 'epoch_%03d.pth' % epoch)

            save_dict = {
                'model': model_without_ddp.state_dict(),
            }

            torch.save(save_dict, checkpoint_path)

        if epoch >= args.num_epochs or early_stopping:
            print('Training done')
            with open(log_file, 'a') as f:
                f.write('Training done')
            return

        epoch += 1

    return


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
