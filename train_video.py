import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from statistics import mean
from torch.nn import functional as F
from dataset.dataset_utils import get_auc,get_heatmap_peak_coords,get_l2_dist,get_angular_error
from skimage.transform import resize
from SalGan import SaliencyNet
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

        labels = os.path.join(self.dir_data, "annotations/train")

        all_data = glob.glob(os.path.join(labels, "*"))

        sample_number = int(len(all_data) * 0.85)
        sample_list = [i for i in range(len(all_data))]
        np.random.shuffle(sample_list)

        sample_list_u = sample_list[:sample_number]
        sample_list_sp = sample_list[sample_number:]

        self.train_dataset = self.select_dataset(self.name_data, True, sample_list_u)
        self.train_dataset_sp = self.select_dataset(self.name_data, True, sample_list_sp)

        #self.train_dataset = self.select_dataset(self.name_data, True)
        #train_size_l = int(0.9 * len(self.train_dataset))
        #train_size_n = len(self.train_dataset) - train_size_l
        #self.train_dataset, self.train_dataset_sp = torch.utils.data.random_split(self.train_dataset, [train_size_l, train_size_n])
        self.test_dataset = self.select_dataset(self.name_data, False,sample_list_u)

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

    def select_dataset(self, name, train, sel):
        if name == 'Videoattention':
            if train:
                labels = os.path.join(self.dir_data, "annotations/train")
                dataset = VideoAttentionTargetImages(self.dir_data, labels, sel, input_size=224, output_size=56,
                                                     is_test_set=False)
            else:
                labels = os.path.join(self.dir_data, "annotations/test")
                dataset = VideoAttentionTargetImages(self.dir_data, labels, sel, input_size=224, output_size=56,
                                                     is_test_set=True)

        return dataset

    def load(self, dir_chck,scene_net, optimG=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])
            print(epoch)

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch),map_location='cuda:0')

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            scene_net.load_state_dict(dict_net['scene_net'])
            optimG.load_state_dict(dict_net['optimG'])

            return scene_net, optimG, epoch

        elif mode == 'test':
            #plg.load_state_dict(dict_net['plg'])
            scene_net.load_state_dict(dict_net['scene_net'])

            return scene_net, epoch


    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch
        accumulation_steps = self.accumulation_steps

        lr_G = self.lr_G
        lr_D = self.lr_D

        wgt_c_a = self.wgt_c_a
        wgt_c_b = self.wgt_c_b

        wgt_pore = self.wgt_pore
        wgt_po = self.wgt_po

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

        transform_inv = transforms.Compose(
            [
                transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                ToNumpy()
            ]
        )

        #loader_train = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        loader_train_sp = torch.utils.data.DataLoader(self.train_dataset_sp, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        #data_iter_sup = iter(loader_train)

        num_train = len(self.train_dataset_sp)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        scene_net = Sence_B().to(device)
        #model_dict = scene_net.state_dict()
        #snapshot = torch.load('initial_weights_for_spatial_training.pt')
        #snapshot = snapshot['model']
        #model_dict.update(snapshot)
        #scene_net.load_state_dict(model_dict, strict=False)
        #for para in scene_net.face_backbone.parameters():
        #    para.requires_grad = False
        #plg = GeneratorAtoB(device).to(device)
        #scene_net = GeneratorAtoB().to(device)
       # init_net(scene_net, init_type='normal', init_gain=0.05, gpu_ids=gpu_ids)
       # init_net(plg, init_type='normal', init_gain=0.05, gpu_ids=gpu_ids)
        mse_loss=nn.MSELoss().to(device)
        #ag_loss = AG_loss().to(device)
        #mask_loss = RRLoss().to(device)
        #prob_loss = CLLoss().to(device)
        info_loss = HLoss().to(device)

        paramsG_scene = scene_net.parameters()
        #paramsG_plg = filter(lambda p: p.requires_grad, plg.parameters())


        #paramsFace = face_net.parameters()

        optimG = torch.optim.Adam(paramsG_scene, lr=lr_G)


        # schedG = get_scheduler(optimG, self.opts)
        # schedD = get_scheduler(optimD, self.opts)

        # schedG = torch.optim.lr_scheduler.ExponentialLR(optimG, gamma=0.9)
        # schedD = torch.optim.lr_scheduler.ExponentialLR(optimD, gamma=0.9)
	#schedG = torch.optim.lr_scheduler.StepLR(optimG,step_size=5,gamma=0.1,last_epoch=-1)

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            scene_net,optimG, st_epoch = \
                self.load(dir_chck, scene_net, optimG, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)

        saliency_detector = SaliencyNet(True).to(device)
        #face_net = Facenet().to(device)
        saliency_detector.eval()

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            #face_net.eval()

            scene_net.train()

            loss_ss_train = []
            loss_prob_train = []
            loss_mask_train = []
            loss_masks_train = []
            loss_info_train = []
            loss_patch_train = []
            loss_update_train = []

            for i, data in enumerate(loader_train_sp, 1):

                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                grad = data['grad']
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
                #grad = grad.to(device)
                head = head.to(device)
                bound_remove = bound_remove.to(device).float()

                #edge_feature = edge_feature.to(device)
                #mask = mask.to(device)
                grid224 = grid224.to(device)
                rgb_img_o = rgb_img_o.to(device)
                t_heatmap = t_heatmap.to(device)
                t_heatmap = t_heatmap.float()
                rgb_img = rgb_img.to(device)
                face = face.to(device)
                head = head.float()

                split_index = np.array(node_num_list).tolist()
                mask_tuple = mask.split(split_index, dim=0)

                batch_m = torch.stack([torch.sum(i, dim=0) for i in mask_tuple], dim=0).unsqueeze(1)
                #print(torch.max(mask.reshape(-1,224*224),dim=1)[0])
                batch_m = torch.clamp(batch_m,0,1)
                t_heatmap = F.grid_sample(t_heatmap, grid224).float()

                with torch.no_grad():
                    #X = X.to(device)

                    saliency,_ = saliency_detector(rgb_img_o)
                    del rgb_img_o
                    pooling = torch.nn.MaxPool2d(56)
                    #avgpool = torch.nn.AvgPool2d(32)
                    #ups = torch.nn.Upsample(scale_factor=32, mode='nearest')
                    saliency_map224 = F.grid_sample(saliency, grid224).to(device)
                    #t_heatmaps_b = torch.where(t_heatmap > 0, 1., 0.)
                   # t_heatmap = torch.mul(t_heatmaps_b, saliency_map224)
                    #t = pooling.forward(t_heatmap) + 1e-8
                   # t_heatmap = t_heatmap / t
                    #saliency_map224 = saliency_map224*bound_remove
                    saliency_map224_s = saliency_map224
                    #saliency_map224_s = pooling.forward(saliency_map224_s)

                    #saliency_map224_s = saliency_map224/saliency_map224_s
                    #head_inv = torch.where(head > 0, 0., 1.)

                    #pose = face_net(face)
                   # po_region = (1 + F.cosine_similarity(pose, grad, dim=2).view(-1, 224, 224).transpose(2, 1).unsqueeze(1)) / 2

                    #po_region = po_region*head_inv

                    #po_region_b = torch.where(po_region>0.5, 1., 0.)
                    #p_t = torch.where(po_region > 0.5, po_region, 0.)
                    #po_region = ((p_t-0.5)/0.5)*po_region_b
                    #po_region = po_region.float()

                    #po_region_a = avgpool.forward(saliency_map224)
                    #po_region_patch = saliency_map224_s#ups.forward(po_region_a)
                    #mask_ = batch_m*po_region_b
                   # mask_sp = batch_m*po_region
                   # mask_sp = mask_sp.float()
                    #po_region_patch.float()

                #heat_pre, mask_pre,patch_pre,gus_map = scene_net(rgb_img, depth, head, po_region)

                target_mask_Graph = plable_gen(mask, node_num_list, node_human_list, edge_feature)
                target_mask_Graph = F.grid_sample(target_mask_Graph.to(device), grid224)


                target_saliency,patch_pre = scene_net(rgb_img,depth,head,face)


                saliency_map224_l = torch.mul(saliency_map224_s,target_mask_Graph.to(device))
                m = pooling.forward(saliency_map224_l)+1e-8

                saliency_map224_lm = saliency_map224_l/m


                loss_info = info_loss(target_saliency)

                '''
                try:
                    data_sp = next(data_iter_sup)
                except StopIteration:
                    data_iter_sup = iter(loader_train_sp)
                    data_sp = next(data_iter_sup)

                grads = data_sp['grad']
                heads = data_sp['head']
                masks = data_sp['mask_img']
                node_num_lists = data_sp['node_num']
                node_human_lists = data_sp['human_num']
                rgb_img_os = data_sp['img_o']
                rgb_imgs = data_sp['img']
                depths = data_sp['depth']
                t_heatmaps = data_sp['true_label_heatmap']
                grid224s = data_sp['g224']
                faces = data_sp['face']
                node_features = data_sp['node_feature']
                edge_features = data_sp['edge_feature']


                depths = depths.to(device)
                heads = heads.to(device)
                node_features = node_features.to(device)
                edge_features = edge_features.to(device)
                masks = masks.to(device)
                t_heatmaps = t_heatmaps.to(device)
                grid224s = grid224s.to(device)
                rgb_img_os = rgb_img_os.to(device)

 

                t_heatmaps = t_heatmaps.float()
                rgb_imgs = rgb_imgs.to(device)

                faces = faces.to(device)
                heads = heads.float()

                with torch.no_grad():
                    saliency_detector.eval()
                    #X = X.to(device)

                    saliencys = saliency_detector(rgb_img_os)
                    pooling = torch.nn.MaxPool2d(64)

                    saliency_map224s = F.grid_sample(saliencys, grid224s).to(device)
                    saliency_map224_ss = saliency_map224s
                    saliency_map224_ss = pooling.forward(saliency_map224_ss)

                    saliency_map224_ss = saliency_map224s/saliency_map224_ss


                target_mask_Graphs, inter_probs, cos_sims = plg(masks, node_num_lists, node_human_lists, node_features,
                                                             edge_features)

                target_saliencys, patch_pres = scene_net(rgb_imgs, depths, heads, faces,target_mask_Graphs)

                index_bs = ((np.array(node_num_lists) - 1)).tolist()
                interat_batch_tuples = inter_probs.split(index_bs, dim=0)
                cos_tuples = cos_sims.split(index_bs, dim=0)
                '''

                #loss_prob = torch.mean(torch.stack([torch.mean(prob_loss(c, i), dim=0) for c, i, n in
                                                    #zip(cos_tuple, interat_batch_tuple, node_num_list)], dim=0), dim=0)
                            #+ torch.mean(torch.stack([torch.mean(prob_loss(c, i), dim=0) for c, i, n in
                            #                        zip(cos_tuples, interat_batch_tuples, node_num_lists)], dim=0), dim=0)


                loss_ss = mse_loss(target_saliency,t_heatmap) * 100
                loss_patch = mse_loss(patch_pre, saliency_map224_s) * 100
                
                #loss_sp = mse_loss(target_saliencys, t_heatmaps)*100
                #loss_sss = mse_loss(patch_pres, saliency_map224_ss)*100


                optimG.zero_grad()
		
                loss_G = loss_ss + 0.1*loss_info + 0.01*loss_patch #+ loss_sp + loss_sss

                loss_G.backward()
                optimG.step()

                loss_ss_train += [loss_ss.item()]
                #loss_prob_train+=[loss_prob.item()]
                #loss_mask_train += [loss_sp.item()]
                #loss_masks_train += [loss_sss.item()]
                loss_info_train += [loss_info.item()]
                #loss_patch_train +=[loss_patch.item()]
                #loss_update_train += [loss_update.item()]

                print('TRAIN:EPOCH %d: BATCH %04d/%04d: '
                        'lossSS: %.4f lossinfo: %.4f '
                        % (epoch, i, num_batch_train, mean(loss_ss_train), mean(loss_info_train)))

                if should(num_freq_disp):
                    ## show output
                    true = t_heatmap.cpu().numpy().transpose(0, 2, 3, 1)

                    #depth = depth.cpu().numpy().transpose(0, 2, 3, 1)
                    s224 = saliency_map224.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    #s224k = saliency_map224_l.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    s224km = saliency_map224_lm.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    #t_map = t_heatmaps.detach().cpu().numpy().transpose(0, 2, 3, 1)#[:,:,:,0:3]
                    #po_region = po_region.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    #po_region_patch = po_region_patch.detach().cpu().numpy().transpose(0, 2, 3, 1)

                    rgb_img = transform_inv(rgb_img)

                   # patch_pre = patch_pre.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
                    #target_saliencys = target_saliencys.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
                    target_mask_Graph = target_mask_Graph.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
                    heat_pre = target_saliency.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
                    batch_m = batch_m.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

                    face = transform_inv(face)

                    writer_train.add_images('true', true, num_batch_train * (epoch - 1) + i,
                                            dataformats='NHWC')
                    #writer_train.add_images('t_map', t_map, num_batch_train * (epoch - 1) + i,
                                            #dataformats='NHWC')
                   # writer_train.add_images('po_region', po_region, num_batch_train * (epoch - 1) + i,
                    #                        dataformats='NHWC')
                    #writer_train.add_images('po_region_patch', po_region_patch, num_batch_train * (epoch - 1) + i,
                    #                        dataformats='NHWC')

                   # writer_train.add_images('depth', depth, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

                    #writer_train.add_images('patch_pred', patch_pre, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('s224', s224, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    #writer_train.add_images('s224_super', s224k, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('s224_superm', s224km, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('batch_m', batch_m, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

                    writer_train.add_images('rgb_img', rgb_img, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    #writer_train.add_images('target_saliencys', target_saliencys, num_batch_train * (epoch - 1) + i,
                     #                       dataformats='NHWC')
                    writer_train.add_images('target_mask_Graph', target_mask_Graph, num_batch_train * (epoch - 1) + i,
                                            dataformats='NHWC')
                    writer_train.add_images('heat_pre', heat_pre, num_batch_train * (epoch - 1) + i,
                                            dataformats='NHWC')
                    writer_train.add_images('face', face, num_batch_train * (epoch - 1) + i,
                                            dataformats='NHWC')
                    del rgb_img
                    del depth
                    del head
                    del face


            #writer_train.add_scalar('loss_SS', mean(loss_ss_train), epoch)
            #writer_train.add_scalar('loss_prob', mean(loss_prob_train), epoch)
            #writer_train.add_scalar('loss_sp', mean(loss_mask_train), epoch)
            #writer_train.add_scalar('loss_patchs', mean(loss_masks_train), epoch)
            #writer_train.add_scalar('loss_info', mean(loss_info_train), epoch)
            #writer_train.add_scalar('loss_patch', mean(loss_patch_train), epoch)
            #writer_train.add_scalar('loss_update', mean(loss_update_train), epoch)




            # # update schduler
            # # schedG.step()
            # # schedD.step()

            ## save
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


        transform_inv = transforms.Compose(
            [
                transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                ToNumpy()
            ]
        )

        loader_test = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_te)

        num_test = len(self.test_dataset)

        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        scene_net = Sence_B().to(device)


        saliency_detector = SaliencyNet(True).to(device)

        mse_loss = nn.MSELoss().to(device)


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

            # netG_a2b.train()
            # netG_b2a.train()

            gen_loss_l1_test = 0
            for i, data in enumerate(loader_test, 1):
                #k=100
                #if i <=k:
                # continue
                face = data['face']
                head = data['head']
                #mask = data['mask_img']
                #node_num_list = data['node_num']
                grad = data['grad']
                rgb_img = data['img']
                depth = data['depth']
                gaze_coords = data['gaze_coords']
                eye_coords = data['eye_coords']
                img_size = data['img_size']

                #edge_feature = data['edge_feature']
                #node_human_list = data['human_num']
                #rgb_img_o = data['img_o']
                grid224 = data['g224']
                bound_remove = data['bound_remove']
                t_heatmap = data['gaze_heatmap']

                #mask = mask.to(device)
                face = face.to(device)
                head = head.to(device)
                #t_heatmap = t_heatmap.to(device)
                head = head.float()

                #edge_feature = edge_feature.to(device)
                bound_remove = bound_remove.to(device)
                bound_remove = bound_remove.float()



                rgb_img= rgb_img.to(device)
                #rgb_img_o = rgb_img_o.to(device)
                grid224 = grid224.to(device)
                grad = grad.to(device)
                depth = depth.to(device)
                #split_index = np.array(node_num_list).tolist()
                #mask_tuple = mask.split(split_index, dim=0)


                    #X = X.to(device)
                #saliency,_ = saliency_detector(rgb_img_o)
                pooling = torch.nn.MaxPool2d(224)
                #batch_m = torch.stack([torch.sum(i, dim=0) for i in mask_tuple], dim=0).unsqueeze(1)
                # print(torch.max(mask.reshape(-1,224*224),dim=1)[0])
                #batch_m = torch.clamp(batch_m, 0, 1)
                #pooling1 = torch.nn.MaxPool2d(32)

                #saliency_map224 = F.grid_sample(saliency, grid224).to(device)
                #t_heatmap = F.grid_sample(t_heatmap.unsqueeze(1), grid224).to(device)
                #saliency_map224_s = saliency_map224
                #saliency_map224_s = pooling.forward(saliency_map224_s)

                #saliency_map224_s = saliency_map224 / saliency_map224_s
                #saliency_map224_c = pooling1.forward(saliency_map224)
                #ups = torch.nn.Upsample(scale_factor=32, mode='nearest')
                #saliency_map224_c = ups(saliency_map224_c)
                #saliency_map224_s = saliency_map224
                #saliency_map224_s = pooling.forward(saliency_map224_s)
                #saliency_map224_s = saliency_map224/saliency_map224_s
                #target_mask_Graph = plable_gen(mask, node_num_list, node_human_list,
                 #                                            edge_feature)

                target_saliency,_ = scene_net(rgb_img, depth, head, face)
                #target_saliency = target_saliency
                #saliency_map224_l = torch.mul(saliency_map224, target_mask_Graph.to(device))
                #m = pooling.forward(saliency_map224_l) + 1e-8

                #saliency_map224_lm = saliency_map224_l / m
                heat_pre = target_saliency.squeeze(1).cpu()#*bound_remove.cpu()


                # forward netG

                #heat_pre, mask_pre, patch_pre, gus_map = scene_net(rgb_img, depth, head, po_region)
                #saliency_map224_l = torch.mul(saliency_map224_s, target_mask_Graph)

                #saliency_map224_l = torch.clamp(saliency_map224_l, 0, 1)

                '''
                print(node_num_list)

                heat_pre = heat_pre.double()

                #heat_pre = torch.where(heat_pre > 0.1, heat_pre, 0.)


                img_r = transform_inv(rgb_img)
                print(gaze_coords)
                img_r = np.squeeze(img_r,0)
                #img_r = cv2.cvtColor(np.array(img_r), cv2.COLOR_RGB2BGR)
                img_r *= 250.0
                img_r = img_r.astype(np.uint8)
                img_r = transforms.ToPILImage()(img_r)#Image.fromarray(img_r)#transforms.ToPILImage()(img_r)
                #img_r.show()
                #k = Image.fromarray(heat_pre)  # transforms.ToPILImage()(img_r)
                #k.show()
                gt = transforms.ToPILImage()(heat_pre)
                #gt.show()
                plt.figure()
                plt.imshow(img_r, cmap='bone')
                plt.imshow(gt,cmap='rainbow', alpha=0.4)
                plt.show()





                #img_r = numpy.asarray(img_r)
                #rgb_mask = np.uint8(rgb_mask)

                #img_r = np.uint8(img_r)

                #img_r = cv2.addWeighted(img_r, 1, rgb_mask, 1.2, 0)
                #cv2.imshow("capture", img_r)
                #cv2.waitKey(0)




                if i == 110:
                 break;

                '''

                for gt_point, eye_point, i_size, pred, h_m in zip(gaze_coords, eye_coords, img_size, heat_pre,
                                                                  t_heatmap):
                    # valid_gaze = gt_point[gt_point != -1].view(-1, 2)
                    # valid_eyes = eye_point[eye_point != -1].view(-1, 2)
                    # if len(valid_gaze) == 0:
                    #    continue
                    # multi_hot = get_multi_hot_map(valid_gaze, i_size)
                    # print(h_m.shape)
                    scaled_heatmap = resize(pred, (64, 64))  # resize(pred, (i_size[1], i_size[0]))

                    multi_hot = h_m
                    # print(h_m.shape)
                    multi_hot = torch.where(multi_hot > 0, 1., 0.).float().numpy() * 1
                    # print(multi_hot.shape)
                    auc_score = get_auc(scaled_heatmap, multi_hot)
                    pred_x, pred_y = get_heatmap_peak_coords(pred)
                    norm_p = torch.tensor([pred_x / float(56), pred_y / float(56)])
                    # all_distances = []
                    # all_angular_errors = []
                    # print(gt_point)
                    # print(norm_p)
                    avg_distance = get_l2_dist(gt_point, norm_p)
                    all_angular_errors = get_angular_error(gt_point - eye_point, norm_p - eye_point)
                    # for index, gt_gaze in enumerate(valid_gaze):
                    #    all_distances.append(get_l2_dist(gt_gaze, norm_p))
                    #    all_angular_errors.append(
                    #        get_angular_error(gt_gaze - valid_eyes[index], norm_p - valid_eyes[index]))

                    # Average distance: distance between the predicted point and human average point
                    # mean_gt_gaze = torch.mean(valid_gaze, 0)
                    # avg_distance = get_l2_dist(mean_gt_gaze, norm_p)

                    # print(auc_score)
                    AUC.append(auc_score)
                    # min_Dis.append(min(all_distances))
                    avg_Dis.append(avg_distance)
                    # min_Ang.append(min(all_angular_errors))
                    # avg_Ang.append(np.mean(all_angular_errors))
                    avg_Ang.append(all_angular_errors)

                # print('test_AUC_e: %s , test_min_Dis: %s , test_avg_Dis: %s, test_min_Ang: %s, test_avg_Ang: %s ' %(np.mean(AUC),np.mean(min_Dis),np.mean(avg_Dis),np.mean(min_Ang),np.mean(avg_Ang)))
                print('test_AUC_e: %s  , test_avg_Dis: %s, , test_avg_Ang: %s ' % (
                    np.mean(AUC), np.mean(avg_Dis), np.mean(avg_Ang)))
            test_AUC = np.mean(AUC)
            # print('test_AUC_e: %s , test_min_Dis: %s , test_avg_Dis: %s, test_min_Ang: %s, test_avg_Ang: %s ' %(np.mean(AUC),np.mean(min_Dis),np.mean(avg_Dis),np.mean(min_Ang),np.mean(avg_Ang)))
            print('test_AUC_e: %s  , test_avg_Dis: %s, , test_avg_Ang: %s ' % (
                np.mean(AUC), np.mean(avg_Dis), np.mean(avg_Ang)))
            print(test_AUC)
            




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


class AG_loss(torch.nn.Module):
    def __init__(self):
        super(AG_loss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity()
    def forward(self,d1, d2):
        cos = self.cosine_similarity(d1, d2)
        loss = 1-cos
        return torch.mean(loss)

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

class RRLoss(torch.nn.Module):
    def __init__(self):
        super(RRLoss, self).__init__()

    def forward(self, output,region_mask):
        #region_mask = region_mask*10
        #region_mask = torch.where(region_mask>0.5,1.0,0.0)
        #region_mask = torch.clamp(region_mask, 0, 1)
        dif = output-region_mask
        dif = torch.nn.ReLU()(dif)
        b = torch.div(1,1+1e-8-dif)-1
        return b.mean()

class CLLoss(torch.nn.Module):
    def __init__(self):
        super(CLLoss, self).__init__()

    def forward(self, cos, inter_prob):

        #x = x.squeeze(2)
        batch_size = inter_prob.size(0)
        #if (inter_prob.size(1) == 1):
        #    return torch.tensor([0.0]).cuda()

        inter_prob = inter_prob.view(-1, batch_size)
        cos = cos.view(-1, batch_size)


        #pos_mask = torch.nn.ReLU()((x - x.mean(-1, keepdim=True).expand_as(x)))
        pos_mask = torch.nn.ReLU()(cos - cos.mean(-1, keepdim=True).expand_as(cos))
        neg_mask = torch.nn.ReLU()(cos.mean(-1, keepdim=True).expand_as(cos) - cos)
        #pos_mask = torch.nn.Softsign()(torch.div(pos_mask, (x.var(-1, keepdim=True) + 1e-8)) * 1e3)

        # saliency value of region


        pos_mask = torch.where(pos_mask>0,1.,0.)
        neg_mask = torch.where(neg_mask > 0, 1., 0.)
        pos_x = torch.mul(pos_mask, inter_prob)
        neg_x = torch.mul(neg_mask, inter_prob)

        pn_x = (pos_x+neg_x+1e-8)/0.3
        epn_x = torch.exp(pn_x).sum(-1,keepdim=True)

        loss = torch.log(torch.exp(pn_x)/epn_x)*pos_mask
        loss = -1.0*loss.sum(dim=-1)

        return loss.mean()


def get_coloured_mask(mask,n):
  """
  random_colour_masks
	parameters:
	  - image - predicted masks
	method:
	  - the masks of each predicted object is given random colour for visualization
  """
  colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [125, 125, 125]]

  r = np.ones_like(mask).astype(np.uint8)*colours[n][0]
  g = np.ones_like(mask).astype(np.uint8)*colours[n][1]
  b = np.ones_like(mask).astype(np.uint8)*colours[n][2]
  #r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
  coloured_mask = np.stack([r, g, b], axis=2)
  coloured_mask = coloured_mask.squeeze(3)
  coloured_mask = coloured_mask*mask

  return coloured_mask

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
