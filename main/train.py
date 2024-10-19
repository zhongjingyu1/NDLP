import tqdm
import argparse
import warnings
from datasets import build_dataset1
from mllt.datasets import build_dataset
from pml_data import generate_uniform_cv_candidate_labels
from PML_Confidence import PML_Confidence
from build_optimizer import build_optimizer
import os
from test_ephco import test_epoch
from utils_algo import *
from mmcv import Config
from mllt.models import build_classifier
from mmcv.parallel import MMDataParallel
from mllt.datasets import build_dataloader
from collections import OrderedDict
from mllt.core.evaluation.mean_ap import eval_map
import torch.optim.lr_scheduler as lr_scheduler
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Noise Correction and Distribution Fine-Tuning', add_help=False)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default='0', type=int, help='seed')
    parser.add_argument('--config', default='../configs/voc/LT_resnet50_pfc_DB.py', choices=['../configs/voc/LT_resnet50_pfc_DB.py', '../configs/coco/LT_resnet50_pfc_DB.py'], help='train config file path')
    parser.add_argument('--dataset', default='voc-lt', type=str, choices=['voc-lt', 'coco-lt'], help='dataset name')
    parser.add_argument('--epochs', default=10, type=int, help='train epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate for optim')
    parser.add_argument('--test_epochs', default=8, type=int, help='train epochs')
    parser.add_argument('--gamma', default=3, type=float, help='gamma of loss function')
    parser.add_argument('--data_dir', default='../codes/', type=str, help='experiment directory for loading pre-generated data')
    parser.add_argument('--partial_rate', default=0.3, type=float, choices=[0.05, 0.1, 0.3, 0.5], help='{COCO: 0.05, 0.1} // {VOC: 0.3, 0.5}')
    parser.add_argument('--eta', default=0.9, type=float, help='final weight of reliable sample loss')
    parser.add_argument('--alpha_range', default='0.4,0.8', type=str, help='ratio of clean labels (alpha)')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use ''(only applicable to non-distributed training)')
    return parser

def main(args):
    print(args)
    cfg = Config.fromfile(args.config)
    torch.manual_seed(args.seed)

    def parse_losses(losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    '{} is not a tensor or list of tensors'.format(loss_name))
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = loss
        for name in log_vars:
            log_vars[name] = log_vars[name].item()
        return loss, log_vars

    def batch_processor(model, data):
        losses, output11 = model(**data)
        loss, log_vars = parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data),losses_all=losses)
        return outputs,output11

    """ 
        model
    """
    if args.dataset=='coco-lt':
        args.num_class = 80
    elif args.dataset=='voc-lt':
        args.num_class = 20

    model = build_classifier(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.CLASSES = args.num_class
    model = MMDataParallel(model, device_ids=range(args.gpus)).cuda()
    model.train()

    """
    optimizer
    """
    optimizer = build_optimizer(model, cfg.optimizer)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='max', verbose=True, min_lr=1e-7)
    """
    dataset and dataloader
    """
    train_dataset = build_dataset1(dataset=args.dataset, split='train')
    data_dir_prod = os.path.join(args.data_dir, 'pre-processed-data')
    if not os.path.exists(data_dir_prod):
        os.makedirs(data_dir_prod)

    print('==> Loading local data copy in the partial multi-label setup')
    data_file = "{ds}_{pr}_sd{sd}.npy".format(
        ds=args.dataset,
        pr=args.partial_rate,
        sd=args.seed)

    save_path = os.path.join(data_dir_prod, data_file)
    if not os.path.exists(save_path):
        Partial_Labels = generate_uniform_cv_candidate_labels(train_dataset.targets, args.partial_rate)
        data_dict = {
            'partial_labels': Partial_Labels
        }
        save_path = os.path.join(data_dir_prod, data_file)
        with open(save_path, 'wb') as f:
            np.save(f, data_dict)
        print('local data saved at ', save_path)
    else:
        data_dict = np.load(save_path, allow_pickle=True).item()
        Partial_Labels = data_dict['partial_labels']

    partialY = torch.from_numpy(Partial_Labels)
    if torch.sum(partialY * torch.from_numpy(train_dataset.targets)) == torch.sum(torch.from_numpy(train_dataset.targets)):
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')
    # check partial labels
    print('Average candidate num: ', partialY.squeeze(1).sum(1).mean())

    Partial_Labels_index = train_dataset.ind1
    train_dataset = build_dataset(cfg.data.train)
    sampler_cfg = cfg.data.get('sampler_cfg', None)
    sampler = cfg.data.get('sampler', 'Group')

    train_dataset.targets = Partial_Labels
    train_dataset.targets_index = Partial_Labels_index
    data_loaders = build_dataloader(
        train_dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        args.gpus,
        dist=False,
        sampler=sampler,
        sampler_cfg=sampler_cfg)

    """
    loss function
    """
    loss_fn = PML_Confidence(partialY.squeeze(1).cuda())

    def get_high_confidence(loss_vec,  pseudo_label_idx, nums_vec):
        idx_chosen = []
        loss_vec_mean = loss_vec.sum(1)/loss_vec.size(1)
        chosen_flags = torch.zeros(len(loss_vec_mean)).cuda()
        for j, nums in enumerate(nums_vec):
            indices = np.where(pseudo_label_idx[:,j].cpu().numpy() == 1)[0]
            if len(indices) == 0:
                continue
            loss_vec_j = loss_vec_mean[indices]
            sorted_idx_j = loss_vec_j.sort()[1].cpu().numpy()
            partition_j = max(min(int(math.ceil(nums)), len(indices)), 1)
            idx_chosen.append(indices[sorted_idx_j[:partition_j]])
        idx_chosen = np.concatenate(idx_chosen)
        chosen_flags[idx_chosen] = 1
        idx_chosen = torch.where(chosen_flags == 1)[0]
        return idx_chosen

    def get_loss(inputs, data_batch, ce_label, partial_label, model, loss_fn, emp_dist, alpha, eta, epoch):
        bs = inputs.shape[0]

        Threshold_phi = loss_fn.confidence.sum(0) / ((loss_fn.confidence != 0) * 1).sum(0)
        ce_label = torch.pow(ce_label,args.gamma) / torch.pow(ce_label,args.gamma).sum(1).unsqueeze(1)
        pseudo_label_idx = (ce_label > Threshold_phi) * 1
        data_batch.update({'gt_labels':pseudo_label_idx})

        loss1, output_two = batch_processor(model, data_batch)
        outputs = output_two[0]
        loss_pseu = loss1.get('loss')
        ce_loss_vec = loss1.get('losses_all').get('loss_cls')

        pseudo_label_idx = (ce_label > Threshold_phi)*1
        r_vec = emp_dist * bs * alpha
        idx_chosen = get_high_confidence(ce_loss_vec, pseudo_label_idx, r_vec.tolist())

        prediction_soft = F.softmax(outputs.detach(), dim=1)
        prediction_adj = prediction_soft * partial_label
        prediction_adj_soft = prediction_adj / prediction_adj.sum(dim=1, keepdim=True)

        if epoch < 1 or idx_chosen.shape[0] == 0:
            loss = loss_pseu
        else:
            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            X_w_c = inputs[idx_chosen]
            ce_label_c = ce_label[idx_chosen]
            idx = torch.randperm(X_w_c.size(0))
            X_w_c_rand = X_w_c[idx]
            ce_label_c_rand = ce_label_c[idx]
            X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand
            ce_label_c_mix = l * ce_label_c + (1 - l) * ce_label_c_rand
            dic = dict(img=X_w_c_mix, img_meta=None, gt_labels=ce_label_c_mix)
            loss_mix_means, logits_mix = model(**dic)
            loss_mix,_ = parse_losses(loss_mix_means)
            loss = loss_pseu + loss_mix * eta
        return loss, prediction_adj_soft, outputs

    """
    training
    """
    sf = nn.Softmax(dim=1)

    [args.alpha_start, args.alpha_end] = [float(item) for item in args.alpha_range.split(',')]
    for epoch in range(args.epochs):
        model.train()
        gt_labels = []
        predict_p = []
        eta = args.eta * linear_rampup(epoch, args.epochs)
        alpha = args.alpha_start + (args.alpha_end - args.alpha_start) * linear_rampup(epoch, args.epochs)
        emp_dist = torch.Tensor([1 / args.num_class for _ in range(args.num_class)]).cuda()

        for i, data_batch in enumerate(tqdm.tqdm(data_loaders)):

            labels = data_batch.pop('truelabel')
            index = data_batch.pop('index')
            partial_label=data_batch.get('gt_labels')
            inputs_datacenter = data_batch.get('img')
            inputs=inputs_datacenter.data[0]
            inputs, labels, partial_label, index = inputs.cuda(), labels.cuda(), partial_label.cuda(), index.cuda()
            optimizer.zero_grad()

            q_soft_label = loss_fn.confidence[index]
            loss, prediction_adj,outputs = get_loss(inputs, data_batch, q_soft_label, partial_label, model, loss_fn, emp_dist, alpha, eta, epoch)

            epsion = loss_fn.confidence[index].sum(0) / loss_fn.confidence[index].sum(0).sum()
            epsion[(epsion == 0)] = emp_dist[(epsion == 0)].to(torch.float64)
            emp_dist1 = emp_dist / (epsion + 1e-8)
            emp_dist = emp_dist1 / emp_dist1.sum()
            loss_fn.confidence_move_update(prediction_adj, index)

            gt_labels.extend(labels.cpu().numpy().tolist())
            predict_p.extend(sf(outputs).cpu().detach().numpy())

            loss.backward()
            optimizer.step()
        try:
            mAP, APs = eval_map(predict_p, gt_labels, None, print_summary=False)
            print("train epoch[{}/{}] loss:{:.3f} train mAP:{}".format(epoch + 1, args.epochs, loss, mAP))
        except:
            print('ValueError: Input contains NaN.')
            print("train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, loss))

        if epoch > args.test_epochs:
            metrics_all = test_epoch(model, cfg)
            for split, mAP, micro_f1, macro_f1, acc in metrics_all:
                print('Split:{:>6s} mAP:{:.4f}  acc:{:.4f}  micro:{:.4f}  macro:{:.4f}'.format(
                    split, mAP, acc, micro_f1, macro_f1))
            current_mAP = metrics_all[3][1]
            exp_lr_scheduler.step(current_mAP)
            checkpoint_path = f'../checkpoint/{args.dataset}_best_result.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'checkpoint saved at: {checkpoint_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Noise Correction and Distribution Fine-Tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
