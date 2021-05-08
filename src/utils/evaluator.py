import torch
import logging
import numpy as np
from collections import defaultdict
from src.preprocess.processor import ENTITY_TYPES

from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report

logger = logging.getLogger(__name__)

def get_base_out(model, loader, device):
    """
    每一个任务的 forward 都一样，封装起来
    """
    model.eval()

    with torch.no_grad():
        for idx, _batch in enumerate(loader):

            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)

            tmp_out = model.inference(**_batch)

            yield tmp_out


def crf_decode(decode_tokens, raw_text, id2ent):
    """
    CRF 解码，用于解码 time loc 的提取
    """
    predict_entities = {}

    decode_tokens = decode_tokens[1:-1]  # 除去 CLS SEP token

    index_ = 0

    while index_ < len(decode_tokens):

        token_label = id2ent[decode_tokens[index_]].split('-')

        if token_label[0].startswith('S'):
            token_type = token_label[1]
            tmp_ent = raw_text[index_]

            if token_type not in predict_entities:
                predict_entities[token_type] = [(tmp_ent, index_)]
            else:
                predict_entities[token_type].append((tmp_ent, int(index_)))

            index_ += 1

        elif token_label[0].startswith('B'):
            token_type = token_label[1]
            start_index = index_

            index_ += 1
            while index_ < len(decode_tokens):
                temp_token_label = id2ent[decode_tokens[index_]].split('-')

                if temp_token_label[0].startswith('I') and token_type == temp_token_label[1]:
                    index_ += 1
                elif temp_token_label[0].startswith('E') and token_type == temp_token_label[1]:
                    end_index = index_
                    index_ += 1

                    tmp_ent = raw_text[start_index: end_index + 1]

                    if token_type not in predict_entities:
                        predict_entities[token_type] = [(tmp_ent, start_index)]
                    else:
                        predict_entities[token_type].append((tmp_ent, int(start_index)))

                    break
                else:
                    break
        else:
            index_ += 1

    return predict_entities


# 严格解码 baseline
def span_decode(start_logits, end_logits, raw_text, id2ent):
    predict_entities = defaultdict(list)

    start_pred = np.argmax(start_logits, -1)
    end_pred = np.argmax(end_logits, -1)

    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                tmp_ent = raw_text[i:i + j + 1]
                predict_entities[id2ent[s_type]].append((tmp_ent, i))
                break

    return predict_entities

# 严格解码 baseline
def mrc_decode(start_logits, end_logits, raw_text):
    predict_entities = []
    start_pred = np.argmax(start_logits, -1)
    end_pred = np.argmax(end_logits, -1)

    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                tmp_ent = raw_text[i:i+j+1]
                predict_entities.append((tmp_ent, i))
                break

    return predict_entities


def calculate_metric(gt, predict):
    """
    计算 tp fp fn
    """
    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        flag = 0
        for entity_gt in gt:
            if entity_predict[0] == entity_gt[0] and entity_predict[1] == entity_gt[1]:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1

    fn = len(gt) - tp

    return np.array([tp, fp, fn])


def get_p_r_f(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])




def crf_evaluation(model, dev_info, device, ent2id):
    dev_loader, dev_callback_info = dev_info

    pred_tokens = []

    true_tokens = []
    
    id2ent = {ent2id[key]: key for key in ent2id.keys()}



    # print('dev_callback_info_eva',len(dev_callback_info))
    # print('dev_callback_info_eva',dev_callback_info)
    # print('dev_callback_info_eva',dev_callback_info[0])

    for tmp_pred in get_base_out(model, dev_loader, device):
        
        pred_tokens.extend(tmp_pred[0])
    # print('pred_tokens',len(pred_tokens))
    assert len(pred_tokens) == len(dev_callback_info[0])

    # print('pred_tokens',pred_tokens)

    
    # print('new_tokens',new_tokens)
        

    # print('pred_tokens',pred_tokens)
    # for dev in get_true_label(dev_loader):
    #     true_tokens.extend(dev[0].labels)
    # print('dev_loader',len(dev_loader))
    info = []
    for feature in (dev_loader)  :
            
            # print('token_ids',token_ids)
            # print('attention_masks',attention_masks)
            # print('token_type_ids',token_type_ids)
            # print('label_ids',label_ids)
            labels = feature['labels']
            print('labels',labels)
            # token_ids = feature['token_ids']
            # print('token_ids',token_ids)
            # num = len(token_ids)
            # info.append(num)
            labels=labels.to('cpu').numpy()
            for label in labels:
                true_tokens.append(label)

    # print('true_tokens',true_tokens)
    # print('true_tokens',len(true_tokens))        
    new_tokens=[]
    new_tokens2=[]
    # print('dev_callback_info',len(dev_callback_info))
    # print('dev_callback_info',dev_callback_info)
    for i,m,j in zip(pred_tokens,true_tokens,dev_callback_info[0]):
        # for k in range(len(j)):
        #     new_tokens.append(id2ent[i[k]])
        tmp1,_ = j  
        # print('tmp1',tmp1)   
        num = len(tmp1)
        i = i[:num]
        m=m[:num]
        # print('i',i)
        # print('leni',len(i))
        
        new_tokens.extend(i) 
        
        new_tokens2.extend(m) 


    for i in range(len(new_tokens)):
        new_tokens[i]= id2ent[new_tokens[i]]  

    for i in range(len(new_tokens2)):
        new_tokens2[i]= id2ent[new_tokens2[i]]    
                        

    

    # assert len(pred_tokens) == len(dev_callback_info)
    # print('pred_tokens',pred_tokens)
    
    # for tmp_tokens, tmp_callback in zip(pred_tokens, true_tokens):

    #     # for i in range(len(tmp_tokens)):
    #     #     tmp_tokens[i]=id2ent[tmp_tokens[i]]
        
    # for i in range(len(true_tokens)):
    #     true_tokens[i]=id2ent[true_tokens[i]]
    

        
        
    # print('iftew equal',new_tokens)
    # print('111',new_tokens2)
    # print('true_tokens',len(new_tokens))
    # print('new_tokens',len(new_tokens2))
    p = precision_score(new_tokens2,new_tokens)
    f1= f1_score(new_tokens2,new_tokens)
    report =classification_report(new_tokens2,new_tokens)



    # role_metric = np.zeros([13, 3])

    # mirco_metrics = np.zeros(3)

    # for tmp_tokens, tmp_callback in zip(pred_tokens, dev_callback_info):

    #     text, gt_entities = tmp_callback

    #     tmp_metric = np.zeros([13, 3])

    #     pred_entities = crf_decode(tmp_tokens, text, id2ent)

    #     for idx, _type in enumerate(ENTITY_TYPES):
    #         if _type not in pred_entities:
    #             pred_entities[_type] = []

    #         tmp_metric[idx] += calculate_metric(gt_entities[_type], pred_entities[_type])

    #     role_metric += tmp_metric

    # for idx, _type in enumerate(ENTITY_TYPES):
    #     temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1], role_metric[idx][2])

    #     mirco_metrics += temp_metric * type_weight[_type]

    # metric_str = f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
    #              f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    # return metric_str, mirco_metrics[2]
    return report,f1

