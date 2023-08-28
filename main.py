import os
import time
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, \
    get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, StyleTranslator, get_twin_image
import wandb
from contrastive_loss import *


def main(args):

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() \
        if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    if 0.1 * args.Iteration < 500:
        eval_it_pool.append(round(0.9 * args.Iteration))
    if args.Iteration not in eval_it_pool:
        eval_it_pool.append(args.Iteration)
    print('eval_it_pool: ', eval_it_pool)

    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = get_dataset(args.dataset, args.data_path) # _: class_names, dst_test
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    if not SWEEP:
        wandb.init(sync_tensorboard=False,
                   project="DC-Fac",
                   job_type="CleanRepo",
                   config=args
                   )

    for key in args.__dict__:
        wandb.config.update({key: args.__dict__[key]})

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])
    
    data_save = []
    best_acc = []
    best_std = []

    loss_recorder = torch.zeros(size=(7, args.Iteration+1), device='cpu')
    # record the loss of each iter. 7 columns for 7 losses

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_base = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], 
            dtype=torch.long, 
            requires_grad=False, 
            device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        ### ### ### ### ###
        styles = nn.ModuleList([StyleTranslator(in_channel=1 if args.single_channel else 3, mid_channel=channel, out_channel=channel, kernel_size=3)
                                for _ in range(args.n_style)])
        sim_content_net = Extractor(num_classes)

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_base.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')

        # Initialize the best recorder
        best_acc.append({m: 0 for m in model_eval_pool})
        best_std.append({m: 0 for m in model_eval_pool})

        ''' training '''
        ### ### ### ### ###
        image_base = image_base.detach().to(args.device).requires_grad_(True)
        styles = styles.to(args.device) # Exactly, the so called Hallucinator.
        sim_content_net = sim_content_net.to(args.device) # Extractor, return both feature vectors and logits

        optimizer_img = torch.optim.SGD([image_base, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        # optimizer_img = torch.optim.SGD([base_image], lr=args.lr_img, momentum=0.95) # from ddfac
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)

        ### ### ### ### ###
        optimizer_style = torch.optim.SGD(styles.parameters(), lr=args.lr_style, momentum=0.95)
        optimizer_sim_content = torch.optim.SGD(sim_content_net.parameters(), lr=args.lr_extractor, momentum=0.9)
        contrast = SupConLoss().to(args.device)

        start_time = time.time()
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):
            save_this_it = False

            wandb.log({"Progress": it}, step=it)

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) 
                        # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_base_eval, label_syn_eval = copy.deepcopy(image_base.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, _, acc_test = evaluate_synset(it_eval, net_eval, image_base_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)

                    # pack these into a function in utils.py
                    accs_test = np.array(accs)
                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    if acc_test_mean > best_acc[exp][model_eval]:
                        best_acc[exp][model_eval] = acc_test_mean
                        best_std[exp][model_eval] = acc_test_std
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    
                    wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                    wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[exp][model_eval]}, step=it)
                    wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                    wandb.log({'Max_Std/{}'.format(model_eval): best_std[exp][model_eval]}, step=it)

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

            if it in eval_it_pool and (save_this_it or it % 1000 == 0):
                with torch.no_grad():
                    image_save = image_base.detach()

                    save_to_local = False
                    if save_to_local:
                        save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                        torch.save(styles.state_dict(), os.path.join(save_dir, "styles_{}.pt".format(it)))

                        if save_this_it:
                            torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt"))
                            torch.save(styles.state_dict(), os.path.join(save_dir, "weights_best.pt"))

                    wandb.log({"Pixels": wandb.Histogram(image_base.detach().cpu())}, step=it)

                    if args.ipc < 50 or args.force_save:
                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Synthetic_Images": wandb.Image(grid.detach().cpu())}, step=it)
                        wandb.log({'Synthetic_Pixels': wandb.Histogram(image_save.detach().cpu())}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(
                                grid.detach().cpu())}, step=it)


            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            ### ### ### ### ###
            sim_content_net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  
            # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in order 
            # to be consistent with DC paper.


            for ol in range(args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer


                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                ### ### ### ### ###
                matching_loss = torch.tensor(0.0).to(args.device)

                ### ### ### ### ###
                style_idx = random.randint(0, args.n_style - 1)
                style = styles[style_idx]
                image_syn = style(image_base)

                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    ### ### ### ### ###
                    matching_loss += match_loss(gw_syn, gw_real, args) / num_classes

                _, image_syn_twin = get_twin_image(image_base, label_syn, styles, args, num_classes, shuffle=True) 
                # At the next step, logits and feature vectors are computed. Two composed image in one tensor computed. 
                if args.dsa and (not args.no_aug):
                    image_syn_twin = DiffAugment(image_syn_twin, args.dsa_strategy, param=args.dsa_param)
                _, embed_c = sim_content_net(image_syn_twin)
                club_content_loss = ((torch.nn.functional.cosine_similarity(embed_c[:image_base.shape[0]], embed_c[image_base.shape[0]:]) + 1.) / 2.).mean()
                loss = matching_loss + club_content_loss * args.lambda_club_content

                optimizer_img.zero_grad()

                ### ### ### ### ###
                optimizer_style.zero_grad()

                loss.backward()
                optimizer_img.step()

                ### ### ### ### ###
                optimizer_style.step()               
                loss_avg += loss.item()

                ### ### ### ### ###
                label, image_syn_twin = get_twin_image(image_base, label_syn, styles, args, num_classes, shuffle=True)
                if args.dsa and (not args.no_aug):
                    image_syn_twin = DiffAugment(image_syn_twin, args.dsa_strategy, param=args.dsa_param)
                logits_c, embed_c = sim_content_net(image_syn_twin)
                cls_content_loss = criterion(logits_c, torch.cat([label, label], dim=0))
                likeli_content_loss = ((1 - torch.nn.functional.cosine_similarity(embed_c[:image_base.shape[0]], embed_c[image_base.shape[0]:])) / 2.).mean()
                embed_c_0 = torch.nn.functional.normalize(embed_c[:image_base.shape[0]])
                embed_c_1 = torch.nn.functional.normalize(embed_c[image_base.shape[0]:])
                contrast_content_loss = contrast(torch.stack([embed_c_0, embed_c_1], dim=1), label)
                sim_content_loss = cls_content_loss * args.lambda_cls_content + likeli_content_loss * args.lambda_likeli_content + contrast_content_loss * args.lambda_contrast_content
                
                optimizer_sim_content.zero_grad()
                sim_content_loss.backward()
                optimizer_sim_content.step()

                wandb.log({"Grand_Loss": loss.detach().cpu(),
                    "Matching_loss": matching_loss.detach().cpu(),
                    "Club_Content_Loss": club_content_loss.detach().cpu(),
                    "Sim_Content_Loss": sim_content_loss.detach().cpu(),
                    "Cls_Content_Loss": cls_content_loss.detach().cpu(),
                    "Likeli_Content_Loss": likeli_content_loss.detach().cpu(),
                    "Contrast_Content_Loss": contrast_content_loss.detach().cpu()}, step = it)
                
                loss_list = [loss.detach().cpu(), 
                             matching_loss.detach().cpu(), 
                             club_content_loss.detach().cpu(), 
                             sim_content_loss.detach().cpu(), 
                             cls_content_loss.detach().cpu(), 
                             likeli_content_loss.detach().cpu(), 
                             contrast_content_loss.detach().cpu()]
                
                for i in range(loss_recorder.shape[0]):
                    loss_recorder[i, it] = loss_list[i]
                
                if ol == args.outer_loop - 1:
                    break


                ''' update network '''
                image_base_train, label_syn_train = copy.deepcopy(image_base.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                image_syn_train = image_syn = styles[0](image_base_train)
                label_syn_train = label_syn_train.repeat(5)
                for i in range(len(styles) - 1):
                    image_syn_train = torch.cat([image_syn_train, styles[i + 1](image_base_train)], dim=0)

                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)

            loss_avg /= (args.outer_loop)

            wandb.log({"Time_Cost": time.time() - start_time}, step = it)

            if it == 0:
                wandb.log({"Time_Cost_per_Iter": time.time() - start_time}, step = it)
            else:
                wandb.log({"Time_Cost_per_Iter": time.time() - time_stamp}, step = it)
            time_stamp = time.time()

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

        wandb.log({"Avg_Time_Cost_per_Iter": (time.time() - start_time) / args.Iteration})
        std_list = torch.var(loss_recorder, dim=1).tolist() 
        wandb.log({"Grand_Loss_std": std_list[0],
                "Matching_loss_std": std_list[1],
                "Club_Content_Loss_std": std_list[2],
                "Sim_Content_Loss_std": std_list[3],
                "Cls_Content_Loss_std": std_list[4],
                "Likeli_Content_Loss_std": std_list[5],
                "Contrast_Content_Loss_std": std_list[6]})

    if not SWEEP:
        wandb.finish()  

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))


# if __name__ == '__main__':

SWEEP = False
if not SWEEP:

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    ####################################################################################
    parser.add_argument('--n_style', type=int, default=5, help='the number of styles')
    parser.add_argument('--single_channel', action='store_true', help="using single-channel but more basis")

    parser.add_argument('--lr_img', type=float, default=0.5, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.05, help='learning rate for updating network parameters')
    parser.add_argument('--lr_style', type=float, default=0.005, help='learning rate for updating style translator')
    parser.add_argument('--lr_extractor', type=float, default=0.005, help='learning rate for updating extractor')
    parser.add_argument('--lambda_club_content', type=float, default=10)
    parser.add_argument('--lambda_likeli_content', type=float, default=1)
    parser.add_argument('--lambda_cls_content', type=float, default=1)
    parser.add_argument('--lambda_contrast_content', type=float, default=1)

    parser.add_argument('--Iteration', type=int, default=1500, help='training iterations')    
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    ####################################################################################
    args = parser.parse_args()

    main(args)

else:
    def parser():

        wandb.init(sync_tensorboard=False, project="DC-Fac")

        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
        parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
        parser.add_argument('--model', type=str, default='ConvNet', help='model')
        parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
        parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
        parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
        parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
        parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
        parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
        parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
        parser.add_argument('--data_path', type=str, default='data', help='dataset path')
        parser.add_argument('--save_path', type=str, default='result', help='path to save results')
        parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
        ####################################################################################
        parser.add_argument('--n_style', type=int, default=5, help='the number of styles')
        parser.add_argument('--single_channel', action='store_true', help="using single-channel but more basis")
        # parser.add_argument('--lr_img', type=float, default=0.5, help='learning rate for updating synthetic images')
        # parser.add_argument('--lr_net', type=float, default=0.05, help='learning rate for updating network parameters')
        # parser.add_argument('--lr_style', type=float, default=0.003, help='learning rate for updating style translator')
        # parser.add_argument('--lr_extractor', type=float, default=0.007, help='learning rate for updating extractor')
        # parser.add_argument('--lambda_club_content', type=float, default=0.14)
        # parser.add_argument('--lambda_likeli_content', type=float, default=1.2)
        # parser.add_argument('--lambda_cls_content', type=float, default=10.)
        # parser.add_argument('--lambda_contrast_content', type=float, default=1.)

        parser.add_argument('--lr_img', type=float, default=wandb.config.lr_img, help='learning rate for updating synthetic images')
        parser.add_argument('--lr_net', type=float, default=wandb.config.lr_net, help='learning rate for updating network parameters')
        parser.add_argument('--lr_style', type=float, default=wandb.config.lr_style, help='learning rate for updating style translator')
        parser.add_argument('--lr_extractor', type=float, default=wandb.config.lr_extractor, help='learning rate for updating extractor')
        parser.add_argument('--lambda_club_content', type=float, default=wandb.config.lambda_club_content)
        parser.add_argument('--lambda_likeli_content', type=float, default=wandb.config.lambda_likeli_content)
        parser.add_argument('--lambda_cls_content', type=float, default=wandb.config.lambda_cls_content)
        parser.add_argument('--lambda_contrast_content', type=float, default=wandb.config.lambda_likeli_content)

        parser.add_argument('--Iteration', type=int, default=wandb.config.Iteration, help='training iterations')    
        parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
        ####################################################################################
        args = parser.parse_args()
        main(args)


    sweep_configuration = {
        # 'name': 'sweep-initial-screening-6',
        'name': 'sweep-dummy',
        'method': 'random',
        'metric': 
        {
            'goal': 'maximize', 
            'name': 'Max_Accuracy'
            },
        'parameters': 
        {   
            'Iteration': {'value': 1500},

            'lr_img': {'values': [0.5]},
            'lr_net': {'values': [0.05]},
            'lr_style': {'values': [0.005]},
            'lr_extractor': {'values': [0.05]},
            'lambda_club_content': {'value': 10},  
            'lambda_likeli_content': {'value': 10},
            'lambda_cls_content': {'value': 10},

            # 'lr_img': {'distribution': 'log_uniform', 'max': 0, 'min': -4.6},
            # 'lr_net': {'distribution': 'log_uniform', 'max': -2.3, 'min': -6.9},
            # 'lr_style': {'distribution': 'log_uniform', 'max': -3, 'min': -9.2},
            # 'lr_extractor': {'distribution': 'log_uniform', 'max': -1.6, 'min': -3.9},
            # 'lambda_club_content': {'distribution': 'log_uniform', 'max': 4.6, 'min': 0},  
            # 'lambda_likeli_content': {'distribution': 'log_uniform', 'max': 6.9, 'min': -2.3},
            # 'lambda_cls_content': {'distribution': 'log_uniform', 'max': 2.99, 'min': 1.6},
            # 'lambda_contrast_content': {'distribution': 'log_uniform', 'max': 4.6, 'min': -4.6}
        }
    }

    # 3: Start the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='DC-Fac'
        )

    wandb.agent(sweep_id, function=parser, count=1)