from seqnet.seqmodel import SEQMODEL
from seqnet.seqnetmodel import SEQNETMODEL
from seqnet.eval_metrics import *

import argparse
from data.ML100K.process import dataset as ML100K_dataset
from data.Beauty.process import dataset as cikm_dataset
from data.ML100K.bert_process import dataset as ML100K_bert_dataset
from data.Beauty.bert_process import dataset as cikm_bert_dataset
import numpy as np
import torch
import random
import torch.nn as nn
import logging
import torch.nn.functional as F

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def run_seqmodel(config):
    if config.data.split("/")[2] != "ML100K":
        if config.model == "gru" or config.model == "hgn":
            data_dic, num_items, num_user, feature_num = cikm_dataset(config)
        elif config.model == "bert":
            data_dic, num_items, feature_num = cikm_bert_dataset(config)

    elif config.data.split("/")[2] == "ML100K":
        if config.model == "gru" or config.model == "hgn":
            data_dic, num_items, num_user, feature_num = ML100K_dataset(config)
        elif config.model == "bert":
            data_dic, num_items, feature_num = ML100K_bert_dataset(config)

    if config.model == "hgn":
        seq_model = SEQMODEL(config, num_items, num_user)
    else:
        seq_model = SEQMODEL(config, num_items, 0)

    if torch.cuda.is_available():
        device_ids_l = [0, 1, 2, 3]
        seq_model = seq_model.to(device)
        seq_model = torch.nn.DataParallel(seq_model, device_ids=device_ids_l)
    optimizer = torch.optim.Adam(seq_model.parameters(), lr=1e-3, weight_decay=1e-3)
    if config.model == "bert":
        optimizer = torch.optim.Adam(seq_model.parameters(), lr=1e-3, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=1.0)
    train_id = data_dic["train_id"]
    train_data = data_dic["train_data"]
    train_label = data_dic["train_label"]
    neg = data_dic["train_neg"]

    train_id_np = np.array(train_id)
    train_data_np = np.array(train_data)
    train_label_np = np.array(train_label)
    neg_np = np.array(neg)


    n_train = train_data_np.shape[0]

    record_indexes = np.arange(n_train)
    batch_size = config.batch_size
    num_batches = int(n_train / batch_size) + 1

    max_ans = 0
    end_model = None
    #recall, MRR, ndcg = seqmodel_evaluation(seq_model, data_dic, ttype="val")
    #print("=================")
    #recall, MRR, ndcg = seqmodel_evaluation(seq_model, data_dic, ttype="test")
    #print("=================")

    loss_train_list = []
    loss_val_list = []

    loss_max_list = []
    loss_mid_list = []
    loss_min_list = []

    for epoch_num in range(config.n_iter):
        seq_model.train()
        if config.model == "bert":
            data_dic2, num_items, feature_num = cikm_bert_dataset(config)
            train_id = data_dic2["train_id"]
            train_data = data_dic2["train_data"]
            train_label = data_dic2["train_label"]
            train_id_np = np.array(train_id)
            train_data_np = np.array(train_data)
            train_label_np = np.array(train_label)
            lr_scheduler.step()
        np.random.shuffle(record_indexes)

        epoch_loss = 0.0
        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < n_train:
                    end = n_train
                else:
                    break

            batch_record_index = record_indexes[start:end]

            batch_id = train_id_np[batch_record_index]
            batch_data = train_data_np[batch_record_index]
            batch_label = train_label_np[batch_record_index]
            batch_neg = neg_np[batch_record_index]

            batch_id = torch.from_numpy(batch_id).type(torch.LongTensor).to(device)
            batch_data = torch.from_numpy(batch_data).type(torch.LongTensor).to(device)
            batch_label = torch.from_numpy(batch_label).type(torch.LongTensor).to(device)
            batch_negatives = torch.from_numpy(batch_neg).type(torch.LongTensor).to(device)



            if config.model == "hgn":
                items_to_predict = torch.cat((batch_label.unsqueeze(1), batch_negatives), 1)
                prediction_score = seq_model(batch_data, batch_id, items_to_predict)
                (targets_prediction, negatives_prediction) = torch.split(
                    prediction_score, [batch_label.unsqueeze(1).size(1), batch_negatives.size(1)], dim=1)
                loss = -torch.log(torch.sigmoid(targets_prediction - negatives_prediction) + 1e-8)
                loss = torch.mean(torch.sum(loss))
            else:
                prediction_score = seq_model(batch_data)
                if config.model == "gru":
                    loss_fun = nn.CrossEntropyLoss(ignore_index=0)
                    loss_ = loss_fun(prediction_score, batch_label)
                    loss = torch.mean(torch.sum(loss_))

                elif config.model == "bert":
                    loss_fun = nn.CrossEntropyLoss(ignore_index=0)
                    logits = prediction_score.view(-1, prediction_score.size(-1))  # (B*T) x V
                    labels = batch_label.view(-1)  # B*T
                    loss = loss_fun(logits, labels)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss /= num_batches

        output_str = "Epoch %d  loss=%.4f" % (epoch_num + 1, epoch_loss)
        logger.info(output_str)

        if (epoch_num + 1) % 1 == 0:
            seq_model.eval()
            with torch.no_grad():
                recall, MRR, ndcg, loss_val, loss_max, loss_mid, loss_min = seqmodel_evaluation(seq_model, data_dic, ttype="val")
                loss_train_list.append(float(format(epoch_loss, '.2f')))
                loss_val_list.append(float(format(loss_val, '.2f')))
  
                loss_max_list.append(float(format(loss_max, '.2f')))
                loss_mid_list.append(float(format(loss_mid, '.2f')))
                loss_min_list.append(float(format(loss_min, '.2f')))

            if ndcg[1] > max_ans:
                max_ans = ndcg[1]
                output_str = "Epoch %d  loss=%.4f" % (epoch_num + 1, epoch_loss)
                logger.info(output_str)
                #logger.info('p: '+', '.join(str(e) for e in precision))
                logger.info('r: ' + ', '.join(str(e) for e in recall))
                logger.info('MRR: '+', '.join(str(e) for e in MRR))
                logger.info('ndcg: ' + ', '.join(str(e) for e in ndcg))

                checkPoints = {
                    'model': seq_model,
                    'epoch': epoch_num,
                    #'precision': precision,
                    'recall': recall,
                    'MRR': MRR,
                    'ndcg': ndcg,
                }
                modelName = "./result/{}_{}".format(config.data.split("/")[2], str(epoch_num + 1))
                torch.save(checkPoints, modelName)
                end_model = seq_model
                #if len(loss_val_list)>=20:
                #    if loss_val >= loss_val_list[len(loss_val_list)-3] and loss_val >= loss_val_list[len(loss_val_list)-2]:
                #        break

    logger.info("===test===test===test===test===test===test===test===test===test===")
    print(loss_train_list)
    print(loss_val_list)
    print(loss_max_list)
    print(loss_mid_list)
    print(loss_min_list)
    recall, MRR, ndcg,_,_,_,_ = seqmodel_evaluation(end_model, data_dic, ttype="test")
    #logger.info('p: ' + ', '.join(str(e) for e in precision))
    logger.info('r: ' + ', '.join(str(e) for e in recall))
    logger.info('MRR: ' + ', '.join(str(e) for e in MRR))
    logger.info('ndcg: ' + ', '.join(str(e) for e in ndcg))

def seqmodel_evaluation(seq_model, data_dic, ttype="val"):

    iid = data_dic[ttype+"_id"]
    data = data_dic[ttype+"_data"]
    label = data_dic[ttype+"_label"]


    id_np = np.array(iid)
    data_np = np.array(data)
    label_np = np.array(label)
    neg_np = data_dic["neg_data"]

    num_users = len(data)
    batch_size = 256
    num_batches = int(num_users / batch_size) + 1
    pred_list = None

    candidate_np = np.concatenate((label_np, neg_np), axis=1)

    loss_list = 0.0
    lis = []
    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        data_sequences = data_np[start:end]
        id_sequences = id_np[start:end]
        candidate_sequences = candidate_np[start:end]
        label_sequences = label_np[start:end]

        data_sequences = torch.from_numpy(data_sequences).type(torch.LongTensor).to(device)
        candidate_sequences = torch.from_numpy(candidate_sequences).type(torch.LongTensor).to(device)
        id_sequences = torch.from_numpy(id_sequences).type(torch.LongTensor).to(device)
        label_sequences = torch.from_numpy(label_sequences).type(torch.LongTensor).to(device)

        if config.model == "hgn":
            rating_pred = seq_model(data_sequences, id_sequences, candidate_sequences)
            scores = rating_pred
        else:
            rating_pred = seq_model(data_sequences)

            if config.model == "gru":
                scores = rating_pred.gather(1, candidate_sequences)
            elif config.model == "bert":
                rating_pred = rating_pred[:, -1, :]
                scores = rating_pred.gather(1, candidate_sequences)

        loss_fun = nn.CrossEntropyLoss(ignore_index=0)
        loss_ = loss_fun(rating_pred, label_sequences.squeeze(1))
        #print(rating_pred)
        #print(label_sequences.squeeze(1))
        #input()
        loss_list += loss_.item()
        lis.append(loss_.item())

        scores = scores.cpu().data.numpy().copy()
        ind = np.argsort(-scores)

        if batchID == 0:
            pred_list = ind
        else:
            pred_list = np.append(pred_list, ind, axis=0)

    label = np.array([[0]] * (len(pred_list)))
    recall, MRR, ndcg = [], [], []
    for k in [5, 10, 15, 20]:
        #precision.append(precision_at_k(label, pred_list, k))
        recall.append(recall_at_k(label, pred_list, k))
        MRR.append(mrr(pred_list,label))
        ndcg.append(ndcg_k(label, pred_list, k))
    #print(lis)
    lis.sort(reverse=True)
    #print(lis[:10])
    return recall, MRR, ndcg, loss_list/num_batches, lis[0], lis[int(len(lis)/2)], lis[-1]

def run_seqnetmodel(config):
    if config.data.split("/")[2] != "ML100K":
        if config.model == "gru" or config.model == "hgn":
            data_dic, num_items, num_user, feature_num = cikm_dataset(config)
        elif config.model == "bert":
            data_dic, num_items, feature_num = cikm_bert_dataset(config)

    elif config.data.split("/")[2] == "ML100K":
        if config.model == "gru" or config.model == "hgn":
            data_dic, num_items, num_user, feature_num = ML100K_dataset(config)
        elif config.model == "bert":
            data_dic, num_items, feature_num = ML100K_bert_dataset(config)

    if config.model == "hgn":
        seqnet_model = SEQNETMODEL(config, num_items, num_user)
    else:
        seqnet_model = SEQNETMODEL(config, num_items, feature_num)

    if torch.cuda.is_available():
        device_ids_l = [1, 2, 3]
        seqnet_model = seqnet_model.to(device)
        seqnet_model = torch.nn.DataParallel(seqnet_model, device_ids=device_ids_l)
    optimizer = torch.optim.Adam(seqnet_model.parameters(), lr=1e-3, weight_decay=1e-3)
    if config.model == "bert":
        optimizer = torch.optim.Adam(seqnet_model.parameters(), lr=1e-3, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=1.0)

    train_id = data_dic["train_id"]
    train_data = data_dic["train_data"]
    train_label = data_dic["train_label"]

    train_id_np = np.array(train_id)
    train_data_np = np.array(train_data)
    train_label_np = np.array(train_label)

    feature_data = data_dic["train_feature"]
    feature_label = data_dic["train_feature_label"]

    train_feature0_np = np.array(feature_data[0])
    train_feature0_label_np = np.array(feature_label[0])

    n_train = train_data_np.shape[0]

    record_indexes = np.arange(n_train)
    batch_size = config.batch_size
    num_batches = int(n_train / batch_size) + 1

    max_ans = 0
    end_model = None
    #recall, MRR, ndcg = seqnetmodel_evaluation(seqnet_model, data_dic, ttype="val")
    #print("=========================")
    #recall, MRR, ndcg = seqnetmodel_evaluation(seqnet_model, data_dic, ttype="test")
    #print("==========================")
    #input()
    loss_train_list = []
    loss_val_list = []

    loss_max_list = []
    loss_mid_list = []
    loss_min_list = []

    for epoch_num in range(config.n_iter):
        if config.model == "bert":
            data_dic, num_items, feature_num = cikm_bert_dataset(config)
            train_id = data_dic["train_id"]
            train_data = data_dic["train_data"]
            train_label = data_dic["train_label"]
            train_id_np = np.array(train_id)
            train_data_np = np.array(train_data)
            train_label_np = np.array(train_label)
            feature_data = data_dic["train_feature"]
            feature_label = data_dic["train_feature_label"]
            train_feature0_np = np.array(feature_data[0])
            train_feature0_label_np = np.array(feature_label[0])
            lr_scheduler.step()
        seqnet_model.train()
        np.random.shuffle(record_indexes)

        epoch_loss = 0.0
        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < n_train:
                    end = n_train
                else:
                    break

            batch_record_index = record_indexes[start:end]

            batch_id = train_id_np[batch_record_index]
            batch_data = train_data_np[batch_record_index]
            batch_label = train_label_np[batch_record_index]

            batch_id = torch.from_numpy(batch_id).type(torch.LongTensor).to(device)
            batch_data = torch.from_numpy(batch_data).type(torch.LongTensor).to(device)
            batch_label = torch.from_numpy(batch_label).type(torch.LongTensor).to(device)

            batch_feature0 = train_feature0_np[batch_record_index]
            batch_feature0_label = train_feature0_label_np[batch_record_index]

            batch_feature0 = torch.from_numpy(batch_feature0).type(torch.LongTensor).to(device)
            batch_feature0_label = torch.from_numpy(batch_feature0_label).type(torch.LongTensor).to(device)
            batch_feature0_label = F.one_hot(batch_feature0_label, num_classes=feature_num[0]).squeeze(1)
            #print(batch_feature0.size())
            prediction_score, prediction_score0 = seqnet_model(batch_data, batch_feature0)

            if config.model == "gru":
                loss_fun_mul = nn.BCEWithLogitsLoss()
                if len(batch_feature0_label.size()) == 3:
                    loss_0 = loss_fun_mul(prediction_score0, ((torch.sum(batch_feature0_label,dim=1)+2001)/2002).float())
                    loss_0 = torch.mean(torch.sum(loss_0))
                else:
                    loss_0 = loss_fun_mul(prediction_score0, batch_feature0_label.float())
                    loss_0 = torch.mean(torch.sum(loss_0))
                loss_fun = nn.CrossEntropyLoss(ignore_index=0)
                loss_ = loss_fun(prediction_score, batch_label)
                loss_ = torch.mean(torch.sum(loss_))
                loss = loss_ + loss_0

            elif config.model == "bert":
                loss_fun_mul = nn.BCEWithLogitsLoss()
                if len(batch_feature0_label.size()) == 4:
                    loss_0 = loss_fun_mul(prediction_score0,
                                          ((torch.sum(batch_feature0_label, dim=2) + 2001) / 2002).float())
                    loss_0 = torch.mean(torch.sum(loss_0))
                else:
                    loss_0 = loss_fun_mul(prediction_score0, batch_feature0_label.float())
                    loss_0 = torch.mean(torch.sum(loss_0))

                loss_fun = nn.CrossEntropyLoss(ignore_index=0)
                logits = prediction_score.view(-1, prediction_score.size(-1))  # (B*T) x V
                labels = batch_label.view(-1)  # B*T
                loss_ = loss_fun(logits, labels)

                loss = loss_ + loss_0*100

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss /= num_batches

        output_str = "Epoch %d  loss=%.4f" % (epoch_num + 1, epoch_loss)
        logger.info(output_str)

        if (epoch_num + 1) % 1 == 0:
            seqnet_model.eval()
            recall, MRR, ndcg, loss_val, loss_max, loss_mid, loss_min = seqnetmodel_evaluation(seqnet_model, data_dic, ttype="val")
            loss_train_list.append(float(format(epoch_loss, '.2f')))
            loss_val_list.append(float(format(loss_val, '.2f')))

            loss_max_list.append(float(format(loss_max, '.2f')))
            loss_mid_list.append(float(format(loss_mid, '.2f')))
            loss_min_list.append(float(format(loss_min, '.2f')))
            if ndcg[1] > max_ans:
                max_ans = ndcg[1]
                output_str = "Epoch %d  loss_=%.4f, loss_0=%.4f" % \
                             (epoch_num + 1, loss_, loss_0)
                logger.info(output_str)
                #logger.info('p: '+', '.join(str(e) for e in precision))
                logger.info('r: ' + ', '.join(str(e) for e in recall))
                logger.info('MRR: '+', '.join(str(e) for e in MRR))
                logger.info('ndcg: ' + ', '.join(str(e) for e in ndcg))

                checkPoints = {
                    'model': seqnet_model,
                    'epoch': epoch_num,
                    #'precision': precision,
                    'recall': recall,
                    'MRR': MRR,
                    'ndcg': ndcg,
                }
                modelName = "./result/{}_{}_{}".format(config.data.split("/")[2], str(config.use_feature),str(epoch_num + 1))
                torch.save(checkPoints, modelName)
                end_model = seqnet_model
                #if len(loss_val_list)>=20:
                #    if loss_val >= loss_val_list[len(loss_val_list)-3] and loss_val >= loss_val_list[len(loss_val_list)-2]:
                #        break

    logger.info("===test===test===test===test===test===test===test===test===test===")
    print(loss_train_list)
    print(loss_val_list)
    print(loss_max_list)
    print(loss_mid_list)
    print(loss_min_list)
    recall, MRR, ndcg,_,_,_,_ = seqnetmodel_evaluation(end_model, data_dic, ttype="test")
    #logger.info('p: ' + ', '.join(str(e) for e in precision))
    logger.info('r: ' + ', '.join(str(e) for e in recall))
    logger.info('MRR: ' + ', '.join(str(e) for e in MRR))
    logger.info('ndcg: ' + ', '.join(str(e) for e in ndcg))

def seqnetmodel_evaluation(seq_model, data_dic, ttype="val"):

    iid = data_dic[ttype+"_id"]
    data = data_dic[ttype+"_data"]
    label = data_dic[ttype+"_label"]

    feature_data = data_dic[ttype+"_feature"]

    id_np = np.array(iid)
    data_np = np.array(data)
    label_np = np.array(label)
    neg_np = data_dic["neg_data"]

    train_feature0_np = np.array(feature_data[0])

    num_users = len(data)
    batch_size = 256
    num_batches = int(num_users / batch_size) + 1
    pred_list = None

    candidate_np = np.concatenate((label_np, neg_np), axis=1)
    loss_list = 0.0
    lis = []
    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        data_sequences = data_np[start:end]
        candidate_sequences = candidate_np[start:end]
        label_sequences = label_np[start:end]

        data_sequences = torch.from_numpy(data_sequences).type(torch.LongTensor).to(device)
        candidate_sequences = torch.from_numpy(candidate_sequences).type(torch.LongTensor).to(device)
        label_sequences = torch.from_numpy(label_sequences).type(torch.LongTensor).to(device)

        batch_feature0 = train_feature0_np[start:end]

        batch_feature0 = torch.from_numpy(batch_feature0).type(torch.LongTensor).to(device)
        #print(batch_feature0.size())
        rating_pred, _ = seq_model(data_sequences, batch_feature0)

        if config.model == "bert":
            rating_pred = rating_pred[:, -1, :]

        loss_fun = nn.CrossEntropyLoss(ignore_index=0)
        loss_ = loss_fun(rating_pred, label_sequences.squeeze(1))
        loss_list += loss_.item()
        lis.append(loss_.item())


        scores = rating_pred.gather(1, candidate_sequences)
        scores = scores.cpu().data.numpy().copy()
        ind = np.argsort(-scores)

        if batchID == 0:
            pred_list = ind
        else:
            pred_list = np.append(pred_list, ind, axis=0)

    label = np.array([[0]] * (len(pred_list)))
    recall, MRR, ndcg = [], [], []
    for k in [5, 10, 15, 20]:
        #precision.append(precision_at_k(label, pred_list, k))
        recall.append(recall_at_k(label, pred_list, k))
        MRR.append(mrr(pred_list, label))
        ndcg.append(ndcg_k(label, pred_list, k))
    lis.sort(reverse=True)
    return recall, MRR, ndcg, loss_list/num_batches, lis[0], lis[int(len(lis)/2)], lis[-1]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--L', type=int, default=10)
    parser.add_argument('--T', type=int, default=1)

    # train arguments
    parser.add_argument('--n_iter', type=int, default=120)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--neg_samples', type=int, default=10)
    parser.add_argument('--sets_of_neg_samples', type=int, default=20)

    # Dataloader
    parser.add_argument('--dataloader_code', type=str, default='bert')
    parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)

    # seq model arguments
    parser.add_argument('--rnn_type', type=str, default="GRU")
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.8)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--return_sequence', type=bool, default=False)

    # Trainer
    parser.add_argument('--trainer_code', type=str, default='bert')
    # optimizer #
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
    parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
    # lr scheduler #
    parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')

    # Model
    parser.add_argument('--model_code', type=str, default='bert')
    parser.add_argument('--model_init_seed', type=int, default=0)
    # BERT #
    parser.add_argument('--bert_max_len', type=int, default=None, help='Length of sequence for bert')
    parser.add_argument('--bert_num_items', type=int, default=None, help='Number of total items')
    parser.add_argument('--bert_hidden_units', type=int, default=256, help='Size of hidden vectors (d_model)')
    parser.add_argument('--bert_num_blocks', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--bert_num_heads', type=int, default=4, help='Number of heads for multi-attention')
    parser.add_argument('--bert_dropout', type=float, default=0.1,
                        help='Dropout probability to use throughout the model')
    parser.add_argument('--bert_mask_prob', type=float, default=0.15,
                        help='Probability for masking items in the training sequence')


    # train data arguments
    parser.add_argument('--run_data', type=str, default="Beauty")
    parser.add_argument('--data', type=str, default="./data/Beauty/seq_data.pkl")
    parser.add_argument('--use_feature', type=int, default=1)
    parser.add_argument('--feature', type=str, default="./data/Beauty/feature.pkl")
    parser.add_argument('--model', type=str, default="gru")

    # model dependent arguments
    parser.add_argument('--dim', type=int, default=256)

    config = parser.parse_args()

    # seed all
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.run_data == 'Beauty':
        config.data = "./data/Beauty/seq_data.pkl"
        config.feature = "./data/Beauty/feature.pkl"
        config.bert_max_len = 7
    elif config.run_data == 'Sports':
        config.data = "./data/Sports/seq_data.pkl"
        config.feature = "./data/Sports/feature.pkl"
        config.bert_max_len = 7*5
    elif config.run_data == 'Toys':
        config.data = "./data/Toys/seq_data.pkl"
        config.feature = "./data/Toys/feature.pkl"
        config.bert_max_len = 7*5
    elif config.run_data == 'Yelp':
        config.data = "./data/Yelp/seq_data.pkl"
        config.feature = "./data/Yelp/feature.pkl"
        config.bert_max_len = 9*5
    elif config.run_data == 'LastFM':
        config.data = "./data/LastFM/seq_data.pkl"
        config.feature = "./data/LastFM/feature.pkl"
        config.bert_max_len = 46
    elif config.run_data == 'ML100K':
        config.data = "./data/ML100K/seq_data.pkl"
        config.feature = "./data/ML100K/feature.pkl"
        config.bert_max_len = 104
    if config.model == "bert":
        config.bert_max_len += 1
    print(config.use_feature)
    if config.use_feature:
        run_seqnetmodel(config)
    else:
        run_seqmodel(config)
