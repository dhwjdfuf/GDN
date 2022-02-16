import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from GDN import GDN
from train_test import *
from evaluate import *

class TimeDataset(Dataset): 

    def __init__(self, data_df, mode, config):
        self.data_df=data_df
        self.config=config
        self.mode=mode

        self.feature, self.label, self.attack = self.process()
        

    
    def process(self):
        win=self.config['slide_win']
        stride=self.config['slide_stride']

      
        if self.mode=='test':
            attack_col=torch.tensor(self.data_df['attack'])
            self.data_df=self.data_df.drop(columns=['attack'])


        num_nodes=len(self.data_df.columns)
        timestamp_len=len(self.data_df.iloc[:,1])


        ran=range(win,timestamp_len,stride) if self.mode =='train' else range(win,timestamp_len)
        data_num=len(ran)
        feature=torch.zeros((data_num,num_nodes,win)) 
        label=torch.zeros((data_num,num_nodes))
        attack=torch.zeros((data_num))

        for cnt,i in enumerate(ran): 
            mat_i=torch.zeros((num_nodes,win))
            label_i=torch.zeros((num_nodes))
            for j in range(num_nodes):
                column=torch.tensor(self.data_df.iloc[:,j])
                mat_i[j]=column[i-win:i] 
                label_i[j]=column[i]

                if j==0 and self.mode=='test':
                    attack[cnt]=attack_col[i]
                    

            feature[cnt]=mat_i
            label[cnt]=label_i


        return feature, label, attack

    def __len__(self):
        return len(self.feature)

    def __getitem__(self,idx): 

        return self.feature[idx], self.label[idx], self.attack[idx]

def train_val_loader(train_dataset, batch, val_ratio=0.1): 
        dataset_len = int(len(train_dataset)) 

        train_use_len = int(dataset_len * (1 - val_ratio)) 
        val_use_len = int(dataset_len * val_ratio) 

        val_start_index = random.randrange(train_use_len) 
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]]) 
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len] 
        val_subset = Subset(train_dataset, val_sub_indices) 

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader

        


train_df=pd.read_csv('./data/train.csv',index_col='timestamp')
test_df=pd.read_csv('./data/test.csv',index_col='timestamp')
list_txt=open('./data/list.txt','r')

config={
    'slide_win': 15,
    'slide_stride': 5,
    'batch': 128,
    'dim': 64,
    'val_ratio': 0.1,
    'topk': 20, #including itself. 
    'out_layer_num': 1,
    'out_layer_inter_dim': 256,
    'decay': 0,
    'epoch': 100,
    'report': 'best' # or 'val'

}

nodes_list=[] 
for node in list_txt:
    nodes_list.append(node.strip())

full_edges=[]

for i in range(len(nodes_list)):
    for j in range(len(nodes_list)):
        if i==j:
            continue
        full_edges.append([i,j])

full_edges=torch.tensor(full_edges).T 

train_dataset=TimeDataset(train_df,'train',config)
test_dataset=TimeDataset(test_df,'test',config)




train_dataloader, val_dataloader = train_val_loader(train_dataset, config['batch'], config['val_ratio'])
test_dataloader = DataLoader(test_dataset, batch_size=config['batch'], shuffle=False, num_workers=0)

model = GDN(full_edges, len(nodes_list), 
                embed_dim=config['dim'], 
                input_dim=config['slide_win'],
                out_layer_num=config['out_layer_num'],
                out_layer_inter_dim=config['out_layer_inter_dim'],
                topk=config['topk'],
            )


train_log=train(model, config,  train_dataloader, val_dataloader, nodes_list, test_dataloader, test_dataset, train_dataset, full_edges)


best_model=model

_, test_result= test(best_model,test_dataloader,full_edges)
_, val_result = test(best_model,val_dataloader, full_edges)

get_score(test_result,val_result,config['report'])









