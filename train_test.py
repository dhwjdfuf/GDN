import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
from evaluate import *


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss

def train(model, config,  train_dataloader, val_dataloader, nodes_list, 
test_dataloader, test_dataset, train_dataset, full_edges):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 15

    model.train() 

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        for x, labels, _ in dataloader: 
            _start = time.time()

            x, labels = [item.float() for item in [x, labels]]

            optimizer.zero_grad()
            out = model(x, full_edges).float()
            loss = loss_func(out, labels)
            
            loss.backward()
            optimizer.step()

            
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1


        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        acu_loss/len(dataloader), acu_loss), flush=True
            )

        # use val dataset to judge
        if val_dataloader is not None: 

            val_loss, val_result = test(model, val_dataloader,full_edges)

            if val_loss < min_loss:
                torch.save(model.state_dict(),'model.pt')

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1


            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(),'model.pt')
                min_loss = acu_loss



    return train_loss_list

def test(model, dataloader,full_edges):
    # test
    loss_func = nn.MSELoss(reduction='mean')

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels in dataloader:
        x, y, labels= [item.float() for item in [x, y, labels]]
        
        with torch.no_grad():
            predicted = model(x, full_edges).float() 
            
            
            loss = loss_func(predicted, y)
            

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        i += 1



    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      
    
    avg_loss = sum(test_loss_list)/len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]




def get_score(test_result, val_result,report):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) 
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)


        print('=========================**MY Result **============================\n')

        info = None
        if report== 'best':
            info = top1_best_info
        elif report == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n') 

    
    





