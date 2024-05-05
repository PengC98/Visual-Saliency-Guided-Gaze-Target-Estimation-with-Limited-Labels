from model.model import *
from dataset.dataset import *
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from statistics import mean
from torch.nn import functional as F
from dataset.dataset_utils import get_multi_hot_map, get_auc,get_heatmap_peak_coords,get_l2_dist,get_angular_error
from skimage.transform import resize
from model.SalGan import SaliencyNet
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class Train:
    def __init__(self, args):

        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log
        self.accumulation_steps = args.accumulation_steps

        self.dir_data = args.dir_data
        #self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.wgt_c_a = args.wgt_c_a
        self.wgt_c_b = args.wgt_c_b

        self.wgt_pore = args.wgt_pore
        self.wgt_po = args.wgt_po

        self.beta1 = args.beta1

        self.gpu_ids = args.gpu_ids

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.name_data = args.name_data
        torch.manual_seed(1)
        np.random.seed(1)

        self.train_dataset = self.select_dataset(self.name_data, True)
        train_size_l = int(0.9 * len(self.train_dataset))
        train_size_n = len(self.train_dataset) - train_size_l
        self.train_dataset, self.train_dataset_sp = torch.utils.data.random_split(self.train_dataset, [train_size_l, train_size_n])


        self.test_dataset = self.select_dataset(self.name_data, False)

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")
        
        #self.pose_model = WHENet('WHENet\\WHENet.h5')

    def save(self, dir_chck, scene_net, optimG, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'scene_net': scene_net.state_dict(),
                    'optimG': optimG.state_dict(),},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))
    def select_dataset(self,name, train):
        if name == 'GazeFollow':
            if train:
                labels = os.path.join(self.dir_data, "train_annotations_release.txt")
                dataset = GazeFollow(self.dir_data, labels, input_size=224, output_size=56,is_test_set=False)
            else:
                labels = os.path.join(self.dir_data, "test_annotations_release.txt")
                dataset = GazeFollow(self.dir_data, labels, input_size=224, output_size=56,is_test_set=True)

        return dataset

    def load(self, dir_chck,scene_net, optimG=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])
            print(epoch)

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch),map_location='cuda:1')

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            scene_net.load_state_dict(dict_net['scene_net'])
            optimG.load_state_dict(dict_net['optimG'])

            return scene_net, optimG, epoch

        elif mode == 'test':
            scene_net.load_state_dict(dict_net['scene_net'])

            return scene_net, epoch


    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G


        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_log_train = os.path.join(self.dir_log, self.scope, name_data, 'train')

        loader_train_sp = torch.utils.data.DataLoader(self.train_dataset_sp, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


        num_train = len(self.train_dataset_sp)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        scene_net = Sence_B().to(device)

        mse_loss=nn.MSELoss().to(device)

        info_loss = HLoss().to(device)

        paramsG_scene = scene_net.parameters()


        optimG = torch.optim.Adam(paramsG_scene, lr=lr_G)


        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            scene_net,optimG, st_epoch = \
                self.load(dir_chck, scene_net, optimG, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)

        saliency_detector = SaliencyNet(True).to(device)
        saliency_detector.eval()

        for epoch in range(st_epoch + 1, num_epoch + 1):


            scene_net.train()

            loss_ss_train = []
            loss_info_train = []


            for i, data in enumerate(loader_train_sp, 1):

                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)


                head = data['head']
                mask = data['mask_img']
                node_num_list = data['node_num']
                node_human_list = data['human_num']
                rgb_img_o = data['img_o']
                rgb_img = data['img']
                depth = data['depth']
                t_heatmap = data['true_label_heatmap']
                grid224 = data['g224']
                face = data['face']
                edge_feature = data['edge_feature']
                bound_remove = data['bound_remove']

                depth = depth.to(device)
                head = head.to(device)
                bound_remove = bound_remove.to(device).float()

                grid224 = grid224.to(device)
                rgb_img_o = rgb_img_o.to(device)
                t_heatmap = t_heatmap.to(device)
                t_heatmap = t_heatmap.float()
                rgb_img = rgb_img.to(device)
                face = face.to(device)
                head = head.float()

                with torch.no_grad():
                    #X = X.to(device)

                    saliency,_ = saliency_detector(rgb_img_o)
                    del rgb_img_o
                    pooling = torch.nn.MaxPool2d(56)

                    saliency_map224 = F.grid_sample(saliency, grid224).to(device)

                    saliency_map224 = saliency_map224*bound_remove
                    saliency_map224_s = saliency_map224


                target_mask_Graph = plable_gen(mask, node_num_list, node_human_list, edge_feature)
                target_mask_Graph = F.grid_sample(target_mask_Graph.to(device), grid224)


                target_saliency,patch_pre = scene_net(rgb_img,depth,head,face)


                saliency_map224_l = torch.mul(saliency_map224_s,target_mask_Graph.to(device))
                m = pooling.forward(saliency_map224_l)+1e-8

                saliency_map224_lm = saliency_map224_l/m


                loss_info = info_loss(target_saliency)

                ##pre-train
                #loss_ss = mse_loss(target_saliency, saliency_map224_lm) * 100

                loss_ss = mse_loss(target_saliency, t_heatmap) * 100
                loss_patch = mse_loss(patch_pre, saliency_map224_s) * 100




                optimG.zero_grad()
		
                loss_G = loss_ss + 0.1*loss_info + 0.001*loss_patch


                ##pre-train
                #loss_G = loss_ss + 0.1 * loss_info + 0.001 * loss_patch

                loss_G.backward()
                optimG.step()

                loss_ss_train += [loss_ss.item()]
                loss_info_train += [loss_info.item()]


                print('TRAIN:EPOCH %d: BATCH %04d/%04d: '
                        'lossSS: %.4f lossinfo: %.4f '
                        % (epoch, i, num_batch_train, mean(loss_ss_train), mean(loss_info_train)))


            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, scene_net, optimG, epoch)

        writer_train.close()

    def test(self):
        mode = self.mode

        batch_size = self.batch_size
        device = self.device
        gpu_ids = self.gpu_ids

        name_data = self.name_data

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)


        loader_test = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_te)

        num_test = len(self.test_dataset)
        scene_net = Sence_B().to(device)
        saliency_detector = SaliencyNet(True).to(device)



        #init_net(netG_a2b, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        #init_net(netG_b2a, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## load from checkpoints
        st_epoch = 0

        scene_net, st_epoch = self.load(dir_chck, scene_net, mode=mode)
        AUC = []
        min_Dis = []
        avg_Dis = []
        min_Ang = []
        avg_Ang = []
        ## test phase
        with torch.no_grad():
            scene_net.eval()
            saliency_detector.eval()

            for i, data in enumerate(loader_test, 1):

                face = data['face']
                head = data['head']

                rgb_img = data['img']
                depth = data['depth']
                gaze_coords = data['gaze_coords']
                eye_coords = data['eye_coords']
                img_size = data['img_size']

                t_heatmap = data['gaze_heatmap']

                #mask = mask.to(device)
                face = face.to(device)
                head = head.to(device)

                head = head.float()

                rgb_img= rgb_img.to(device)

                depth = depth.to(device)



                target_saliency,_ = scene_net(rgb_img, depth, head, face)

                heat_pre = target_saliency.squeeze(1).cpu()




                for gt_point,eye_point, i_size, pred,h_m in zip(gaze_coords,eye_coords, img_size, heat_pre,t_heatmap):
                    valid_gaze = gt_point[gt_point != -1].view(-1, 2)
                    valid_eyes = eye_point[eye_point != -1].view(-1, 2)
                    if len(valid_gaze) == 0:
                        continue
                    multi_hot = get_multi_hot_map(valid_gaze, i_size)

                    scaled_heatmap = resize(pred, (i_size[1], i_size[0]))
                    auc_score = get_auc(scaled_heatmap, multi_hot)
                    pred_x, pred_y = get_heatmap_peak_coords(pred)
                    norm_p = torch.tensor([pred_x / float(56), pred_y / float(56)])
                    all_distances = []
                    all_angular_errors = []

                    for index, gt_gaze in enumerate(valid_gaze):
                        all_distances.append(get_l2_dist(gt_gaze, norm_p))
                        all_angular_errors.append(
                            get_angular_error(gt_gaze - valid_eyes[index], norm_p - valid_eyes[index]))

                    # Average distance: distance between the predicted point and human average point
                    mean_gt_gaze = torch.mean(valid_gaze, 0)
                    avg_distance = get_l2_dist(mean_gt_gaze, norm_p)

                    #print(auc_score)
                    AUC.append(auc_score)
                    min_Dis.append(min(all_distances))
                    avg_Dis.append(avg_distance)
                    min_Ang.append(min(all_angular_errors))
                    avg_Ang.append(np.mean(all_angular_errors))
                    #avg_Ang.append(all_angular_errors)

            print('test_AUC_e: %s , test_min_Dis: %s , test_avg_Dis: %s, test_min_Ang: %s, test_avg_Ang: %s ' %(np.mean(AUC),np.mean(min_Dis),np.mean(avg_Dis),np.mean(min_Ang),np.mean(avg_Ang)))







                #recon_b = netG_a2b(output_a)
                #recon_a = netG_b2a(output_b)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)




class KLLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(KLLoss, self).__init__()

  def forward(self, pred, target):
      loss = F.kl_div(pred, target, reduction='none')
      loss = loss.sum(-1).sum(-1).sum(-1)
      return loss
class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = torch.softmax(x, dim=-1)
        ind = torch.ones_like(x, requires_grad=True)
        ind = torch.div(ind , ind.sum(-1, keepdim=True))

        pos_mask = torch.nn.ReLU()((x - x.mean(-1, keepdim=True).expand_as(x)))
        pos_mask = torch.nn.Softsign()(torch.div(pos_mask, (x.var(-1, keepdim=True)+1e-8)) * 1e3)

        # saliency value of region
        pos_x = torch.mul(pos_mask, x).sum(-1, keepdim=True)
        neg_x = torch.mul((1.-pos_mask), x).sum(-1, keepdim=True)
        p = torch.cat([pos_x, neg_x], dim=-1)

        # pixel percentage of region
        pos_ind = torch.mul(pos_mask, ind).sum(-1, keepdim=True)
        neg_ind = torch.mul((1. - pos_mask), ind).sum(-1, keepdim=True)
        ratio_w = torch.cat([pos_ind, neg_ind], dim=-1)

        b = F.softmax(ratio_w, dim=-1) * F.log_softmax(p, dim=-1)
        b = -1.0 * b.sum(dim=-1)
        return b.mean()


def plable_gen(mask, node_num_list, node_human_list,edge_feature):
    cos_sim = edge_feature[:, -1].unsqueeze(1)
    inter_prob = torch.ones_like(cos_sim)

    index_k = ((np.array(node_num_list) - 1) * np.array(node_human_list)).tolist()
    cos_tuple = cos_sim.split(index_k, dim=0)
    inter_tuple = inter_prob.split(index_k, dim=0)
    inter_up = torch.randn(1, 1)
    for i, c, node_num in zip(inter_tuple, cos_tuple, node_num_list):
        cos = c[:node_num - 1]
        # print(cos - cos.mean().expand_as(cos))
        if node_num == 2:
            mean = torch.zeros_like(cos)
        else:
            mean = cos.mean()

        i[:node_num - 1] = torch.nn.ReLU()(cos - mean.expand_as(cos))+0.1

        # c[:node_num - 1] = ((1 + F.cosine_similarity(p, grad, dim=1)) / 2).unsqueeze(1)

        inter_up = torch.cat((inter_up, i), dim=0)
    inter_up = inter_up[1:]

    split_index = np.array(node_num_list).tolist()

    mask_tuple = mask.split(split_index, dim=0)

    new_mask = torch.cat(
        [torch.cat([maskss[torch.arange(maskss.size(0)) != i] for i in range(node_human)], dim=0)
         for node_num, node_human, maskss in zip(node_num_list, node_human_list, mask_tuple)], dim=0)

    # new_mask = torch.cat(
    #			[torch.cat([maskss[torch.arange(maskss.size(0))] for i in range(node_human)], dim=0)
    #				 for node_num, node_human, maskss in zip(node_num_list, node_human_list,mask_tuple)], dim=0)

    new_mask = new_mask.reshape(-1, 224 * 224)

    interat = torch.mul(new_mask, inter_up).reshape(-1, 224, 224)

    index_b = ((np.array(node_num_list) - 1) * np.array(node_human_list)).tolist()

    interat_batch_tuple = interat.split(index_b, dim=0)

    target_mask = torch.stack(
        [torch.sum(i[:node_num - 1], dim=0) for i, node_num in zip(interat_batch_tuple, node_num_list)], dim=0)

    target_mask_Graph = target_mask.unsqueeze(1)


    return target_mask_Graph
