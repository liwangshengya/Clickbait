from itertools import count
import os
import argparse
import time
import copy
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from DataLoad import MyDataset
from Model import MMGCN

torch.set_num_threads(2)
#选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 2020
torch.manual_seed(seed) # cpu
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed) #gpu
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True # cudnn
np.random.seed(seed) #numpy
random.seed(seed) #random and transforms

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)

class Net:
    def __init__(self, args):
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.save_path = args.save_path
        self.log_name = args.log_name
        self.learning_rate = args.l_r
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.concat = args.concat
        self.num_workers = args.num_workers
        self.num_epoch = args.num_epoch
        self.early_stop = args.early_stop
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.dim_latent = args.dim_latent
        self.aggr_mode = args.aggr_mode
        self.num_layer = args.num_layer
        self.has_id = args.has_id
        self.dim_v = 768
        self.dim_a = 768
        self.dim_t = 768
        self.alpha = args.alpha
#################################################################################################################################
        print('Data loading ...')
        self.train_dataset = MyDataset(self.data_path, self.num_user, self.num_item)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
        self.edge_index = np.load(self.data_path+'train.npy')
        self.user_item_dict = np.load(self.data_path+'user_item_dict.npy', allow_pickle=True).item()
        self.val_dataset = np.load(self.data_path+'val_full.npy', allow_pickle=True)
        self.test_dataset = np.load(self.data_path+'test_full.npy', allow_pickle=True)
        self.d_feat_tensor = torch.load(self.data_path+'feat_des.pt')
        print("d_feat_tensor", self.d_feat_tensor.size())
        self.b_feat_tensor = torch.load(self.data_path+'feat_body.pt')
        print("b_feat_tensor", self.b_feat_tensor.size())
        self.t_feat_tensor = torch.load(self.data_path+'feat_title.pt')
        print("t_feat_tensor", self.t_feat_tensor.size())
        print('Data has been loaded.')
#################################################################################################################################

        if self.model_name == 'MMGCN':
            self.model = MMGCN(self.d_feat_tensor, self.b_feat_tensor, self.t_feat_tensor, self.edge_index, self.batch_size, self.num_user, self.num_item, self.aggr_mode, self.concat, self.num_layer, self.has_id, self.user_item_dict, self.dim_latent, self.alpha).to(device)
        
#         elif self.model_name == 'VBPR':
#             self.model = VBPR_model(self.d_feat_tensor, self.b_feat_tensor, self.t_feat_tensor, self.num_user, self.num_item, self.user_item_dict, self.dim_latent).cuda()

#         elif self.model_name == 'NGCF':
#             self.model = NGCF(self.d_feat_tensor, self.b_feat_tensor, self.t_feat_tensor, self.edge_index, self.batch_size, self.num_user, self.num_item, self.user_item_dict, self.dim_latent).cuda()

        if args.PATH_weight_load and os.path.exists(args.PATH_weight_load):
            self.model.load_state_dict(torch.load(args.PATH_weight_load))
            print('module weights loaded....')
#################################################################################################################################

        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.learning_rate}], weight_decay=self.weight_decay)

#################################################################################################################################


    def run(self,num_run):
        max_recall = [0]*4
        max_rec = [0]*4
        max_pre = [0]*4
        max_ndcg = [0]*4
        num_des = 0

        count = 0
        for epoch in range(self.num_epoch):
            self.model.train()
            print('Now, training start ...')
            pbar = tqdm(total=len(self.train_dataset))
            sum_loss = 0.0
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                self.loss = self.model.loss(data,count,args.loss_type)
                self.loss.backward()
                self.optimizer.step()
                pbar.update(self.batch_size)
                sum_loss += self.loss
            count += 1
            print(sum_loss/self.batch_size)
            pbar.close()

            print('Validation start...')
            self.model.eval()
            with torch.no_grad():
                precision, recall, ndcg_score = self.model.full_ranking(self.val_dataset, topk=[10, 20, 50, 100])
                print('---------------Valid:第{}次训练 epoch: {0} ---------------'.format(num_run,epoch))
                print('Precision: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(precision[0], precision[1], precision[2], precision[3]))
                print('Recall: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(recall[0], recall[1], recall[2], recall[3]))
                print('NDCG: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(ndcg_score[0], ndcg_score[1], ndcg_score[2], ndcg_score[3]))

            with torch.no_grad():
                test_precision, test_recall, test_ndcg_score = self.model.full_ranking(self.test_dataset, topk=[10, 20, 50, 100])
                print('---------------Test:第{}次训练 epoch: {0} ---------------'.format(num_run,epoch))
                print('Precision: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(test_precision[0], test_precision[1], test_precision[2], test_precision[3]))
                print('Recall: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(test_recall[0], test_recall[1], test_recall[2], test_recall[3]))
                print('NDCG: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(test_ndcg_score[0], test_ndcg_score[1], test_ndcg_score[2], test_ndcg_score[3]))
            # if not os.path.exists(self.save_path):
            #     os.makedirs(self.save_path)
            # torch.save(self.model, '{}{}_{}_{}.pth'.format(self.save_path, self.model_name, self.log_name, epoch))
            if recall[0] > max_recall[0]:
                max_recall = copy.deepcopy(recall)
                max_rec = copy.deepcopy(test_recall)
                max_pre = copy.deepcopy(test_precision)
                max_ndcg = copy.deepcopy(test_ndcg_score)
                num_des = 0
                print("########### Best Validation ###########")
                if not os.path.exists(self.save_path):
                    os.mkdir(self.save_path)
                torch.save(self.model, '{}_{}{}_{}.pth'.format(num_run,self.save_path, self.model_name, self.log_name))
            else:
                num_des += 1    
                if num_des >= self.early_stop:
                    print("Early stopping!")
                    return max_pre, max_rec, max_ndcg
        return max_pre, max_rec, max_ndcg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--model_name', default='MMGCN', help='Model name.')
    parser.add_argument('--data_path', default='..//data/mind/', help='Dataset path')
    parser.add_argument('--save_path', default='./models/', help='saved model path')
    parser.add_argument('--log_name', default='mind', help='training log name')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('--alpha', type=float, default=1, help='alpha.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--dim_latent', type=int, default=64, help='Latent dimension.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number.')
    parser.add_argument('--early_stop', type=int, default=10, help='early_stop Epoch number.')
    parser.add_argument('--num_workers', type=int, default=2, help='Workers number.')
    parser.add_argument('--num_user', type=int, default=5964, help='User number.')
    parser.add_argument('--num_item', type=int, default=4781, help='Item number.')
    parser.add_argument('--aggr_mode', default='mean', help='Aggregation mode.')
    parser.add_argument('--num_layer', type=int, default=2, help='Layer number.')
    parser.add_argument('--has_id', type=bool, default=True, help='Has id_embedding')
    parser.add_argument('--concat', type=bool, default=False, help='Concatenation')
    parser.add_argument('--loss_type', type=float, default=1, help='loss_type:0:BPR,1:BPR+T_CE,2:CE,3:T_CE')
    args = parser.parse_args()
    print("arguments: %s " %(args))
    egcn = Net(args)
    #run n次，结果取平均
    n = 10
    test_precision_list = []
    test_recall_list = []
    test_ndcg_score_list = []
    for i in range(n):
        print("第{}次".format(i))
        test_precision, test_recall, test_ndcg_score = egcn.run(i)
        test_precision_list.append(test_precision) 
        test_recall_list.append(test_recall)
        test_ndcg_score_list.append(test_ndcg_score)
    test_precision_list = np.array(test_precision_list)
    test_recall_list = np.array(test_recall_list)
    test_ndcg_score_list = np.array(test_ndcg_score_list)
    
    print('---------------Training Done !- ---------------'.format(args.num_epoch))
    print("----------------平均结果----------------")
    print('Precision: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(np.mean(test_precision_list[:,0]),
            np.mean(test_precision_list[:,1]),
            np.mean(test_precision_list[:,2]),
            np.mean(test_precision_list[:,3])))
    print('Recall: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(np.mean(test_recall_list[:,0]),
            np.mean(test_recall_list[:,1]),
            np.mean(test_recall_list[:,2]),
            np.mean(test_recall_list[:,3])))
    print('NDCG: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(np.mean(test_ndcg_score_list[:,0]),
            np.mean(test_ndcg_score_list[:,1]),
            np.mean(test_ndcg_score_list[:,2]),
            np.mean(test_ndcg_score_list[:,3])))
    print('----------------------------------------------')

    #选出值最大一次的结果
    max=np.argmax(test_recall_list[:,0])
    print('---------------运行10次，最好的一次为第{}次---------------'.format(max+1))
    #取出最好的结果
    max_test_precision = test_precision_list[max]
    max_test_recall = test_recall_list[max]
    max_test_ndcg_score = test_ndcg_score_list[max]
    print("-------------------最好的结果为-------------------")
    print('Precision: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(max_test_precision[0], max_test_precision[1], max_test_precision[2], max_test_precision[3]))
    print('Recall: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(max_test_recall[0], max_test_recall[1], max_test_recall[2], max_test_recall[3]))
    print('NDCG: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(max_test_ndcg_score[0], max_test_ndcg_score[1], max_test_ndcg_score[2], max_test_ndcg_score[3]))
    print('---------------Training Done !---------------')

    # print('Precision: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(test_precision[0], test_precision[1], test_precision[2], test_precision[3]))
    # print('Recall: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(test_recall[0], test_recall[1], test_recall[2], test_recall[3]))
    # print('NDCG: {:.4f}-{:.4f}-{:.4f}-{:.4f}'.format(test_ndcg_score[0], test_ndcg_score[1], test_ndcg_score[2], test_ndcg_score[3]))
