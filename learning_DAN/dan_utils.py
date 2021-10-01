import numpy as np
import torch
import random
import tensorflow as tf
import pandas as pd

def mape_loss_func(preds, labels, m):
    mask = labels > m
    return np.mean(np.fabs(labels[mask]-preds[mask])/labels[mask])

def smape_loss_func(preds, labels, m):
    mask= labels > m
    return np.mean(2*np.fabs(labels[mask]-preds[mask])/(np.fabs(labels[mask])+np.fabs(preds[mask])))

def mae_loss_func(preds, labels, m):
    mask= labels > m
    return np.mean(np.fabs((labels[mask]-preds[mask])))

def nrmse_loss_func(preds, labels, m):
    mask= labels > m
    return np.sqrt(np.sum((preds[mask] - labels[mask])**2)/preds[mask].flatten().shape[0])/(preds[mask].max() - preds[mask].min())

def eliminate_nan(b):
    a = np.array(b)
    c = a[~np.isnan(a)]
    return c

def get_node(det, seg):
    # det is one single detector id
    # node is one single node id
    
    # seg = pd.read_csv('./data/segement.csv', header=None)
    try:
        node_info = seg[seg[6]==det]
        node = node_info.iloc[0, 0]
    except:
        node_info = seg[seg[7]==det]
        node = node_info.iloc[0, 0]
        
    return node

def get_class_with_node(seg, v_class):
    det_list_class = np.array([])
    try:
        v_class.insert(1, 'id2', '')  # id2 mean node id
    except:
        v_class['id2'] = ''
        
    for i in range(len(v_class)):
        det_list_class = np.append(det_list_class, v_class.iloc[i, 0])
        v_class.iloc[i, 1] = get_node(v_class.iloc[i, 0], seg)
    
    return det_list_class, v_class

def rds_mat(old_dist_mat, det_ids, seg):
    # get a matrix that contains n raods that have specified node id s
    node_ids = np.array([])
    for i in det_ids:
        node_ids = np.append(node_ids, get_node(i, seg))
        
    new_dist_mat = old_dist_mat.loc[node_ids, node_ids]
    old_dist_mat = np.array(old_dist_mat)
    new_near_id_mat = np.argsort(new_dist_mat)
    return new_near_id_mat

def sliding_window(flow, near_road, from_day, to_day, prop, num_links, k=5, t_input=12, t_pre=6):
    t_p = to_day - from_day
    flow = np.array(flow)
    # 选数据
    flow = flow[:, 144*(from_day-1):144*(to_day)]
#     print(flow.shape)

    # 利用滑动窗口的方式，重构数据为(n，最近路段数，输入时间窗，总路段数)的形式
    '''
    k # 参数k为需考虑的最近路段数
    t_p # 参数t_p为总时间序列长度（天）
    t_input #参数t_input为输入时间窗(10min颗粒度)
    t_pre #参数t_pre为预测时间窗(10min颗粒度)
    num_links #参数num_links为总路段数
    '''

    image = []
    for i in range(np.shape(near_road)[0]):
        road_id = []
        for j in range(k):
            road_id.append(near_road[i][j])
        image.append(flow[road_id, :])
#     ipdb.set_trace()
    image1 = np.reshape(image, [-1, k, flow.shape[1]])
    image2 = np.transpose(image1,(1,2,0))
#     print(image1.shape)
    image3 = []
    label = []
    day = []

    for i in range(1, t_p):
        for j in range(144-t_input-t_pre):
            image3.append(image2[:, i*144+j:i*144+j+t_input, :][:])
            label.append(flow[:, i*144+j+t_input:i*144+j+t_input+t_pre][:])
            day.append(flow[:, (i-1)*144+j+t_input:(i-1)*144+j+t_input+t_pre][:])

#     ipdb.set_trace()
    image3 = np.asarray(image3)
    label = np.asarray(label)
    day =  np.asarray(day)

    print(np.shape(image3))

    #划分前90%数据为训练集，最后10%数据为测试集
    image_train = image3[:int(np.shape(image3)[0]*prop)]
    image_test = image3[int(np.shape(image3)[0]*prop):]
    label_train = label[:int(np.shape(label)[0]*prop)]
    label_test = label[int(np.shape(label)[0]*prop):]

    day_train = day[:int(np.shape(day)[0]*prop)]
    day_test = day[int(np.shape(day)[0]*prop):]

    print(image_train.shape)
    print(image_test.shape)
    print(label_train.shape)
    print(label_test.shape)
    
    return image_train, image_test, day_train, day_test, label_train, label_test


def setup_seed(seed):
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)    
        torch.backends.cudnn.deterministic = True
    except:
        tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_np(array, name):
    np.savetxt(name, array, delimiter=',')

def norm_data(vec):
#     ipdb.set_trace()
    # vec = vec.flatten()
    vec_res = (vec.T - np.min(vec.T, axis=0))/(np.max(vec.T, axis=0) - np.min(vec.T, axis=0))
    return vec_res.T, np.min(vec.T, axis=0).T, np.max(vec.T, axis=0).T

def denorm_data(vec, min_val, max_val):
    return vec*(max_val - min_val) + min_val

def load_data(class_set, res=11, randseed=25):
    # Return:
    # det_list_class: all the detectors of class i before manual selection
    # 把矩阵分类
    setup_seed(randseed)

    v = pd.read_csv('../data/q_20_aggragated.csv')
    v = v.rename(columns={'Unnamed: 0': 'id'})
    id_402 = pd.read_csv('../res/%i_res%i_id_402_withclass.csv'%(randseed, res), index_col=0)
    part1 = pd.read_csv('../res/%i_res%i_det_partition_results1.csv'%(randseed, res))
    part2 = pd.read_csv('../res/%i_res%i_det_partition_results2.csv'%(randseed, res))

    v['class_i'] = ''
    for i in range(len(v)):
        v.loc[i, 'class_i'] = id_402[id_402['id']==v.loc[i, 'id']].iloc[0, 5]  # 5 stands for 'class_i'

    num_class = id_402['class_i'].drop_duplicates().size

    v_class = []
    for i in range(num_class):
        v_class.append(v[v['class_i']==i])

    print('There are %i class(es)'%num_class)

    # 制作 nearest_road_id.csv 和speed.csv
    dist_mat = pd.read_csv('../data/dist_mat.csv', index_col=0)
    id_info = pd.read_csv('../data/id2000.csv', index_col=0)
    dist_mat.index = id_info['id2']
    dist_mat.columns = id_info['id2']
    for i in range(len(dist_mat)):
        for j in range(len(dist_mat)):
            if i==j:
                dist_mat.iloc[i, j] = 0

    near_id = pd.DataFrame(np.argsort(np.array(dist_mat)), index = id_info['id2'], columns = id_info['id2'])

    # ## 以上做好了near_road矩阵，接下来做flow/speed矩阵
    seg = pd.read_csv('../data/segement.csv', header=None)

    det_list_class = []
    for i in range(num_class):
        det_list_class_temp, v_class_temp = get_class_with_node(seg, v_class[i])
        det_list_class.append(det_list_class_temp[:])
        v_class_temp = v_class_temp.loc[v_class_temp['id'].isin(det_list_class_temp[:])]
        v_class[i] = v_class_temp

    # Select detectors manually, then assemble v_class matrix
    selected_dets1 = pd.read_csv('../network_classification/selected_dets1.csv', index_col=0)
    selected_dets2 = pd.read_csv('../network_classification/selected_dets2.csv', index_col=0)
    selected_nodes = pd.read_csv('../network_classification/selected_nodes.csv', index_col=0)

    # selected_nodes = pd.DataFrame([], index=range(len(selected_dets)), columns=['class', 'node'])
    # for i in range(len(selected_dets)):
    #     det_ = selected_dets.loc[i, 'det']
    #     class_ = selected_dets.loc[i, 'class']
    #     node_ = seg.loc[seg[6]==det_, 0].values[0]
    #     selected_nodes.loc[i, 'class'] = class_
    #     selected_nodes.loc[i, 'node'] = node_
    # selected_nodes.to_csv('../network_classification/selected_nodes.csv')

    # selected_dets2 = pd.DataFrame([], index=range(len(selected_dets)), columns=['class', 'det'])
    # for i in range(len(selected_nodes)):
    #     node_ = selected_nodes.loc[i, 'node']
    #     class_ = selected_nodes.loc[i, 'class']
    #     det = seg.loc[seg[0]==node_, 7]
    #     selected_dets2.loc[i, 'class'] = class_
    #     selected_dets2.loc[i, 'det'] = det_
    # selected_dets2.to_csv('../network_classification/selected_dets2.csv')

    # filt, so that only selected dets remain
    for i in range(len(class_set)):
        cls_ = class_set[i]
        det_set = selected_nodes.loc[selected_nodes['class']==cls_, 'node'].values
        v_class[cls_] = v_class[cls_].loc[v_class[cls_]['id2'].isin(det_set)]  # id means detID, id2 memans nodeID
    det_num_min = min([v_class[i].shape[0] for i in range(5)])  # for all classes
    for i in range(5):
        v_class[i] = v_class[i].iloc[:det_num_min, :]

    # make near_road matrix
    near_road_set = []
    for i in range(num_class):
        det_set = v_class[i]['id'].values
    #     ipdb.set_trace()
        near_road_set.append(rds_mat(dist_mat, det_set, seg))

    return v, v_class, id_402, part1, part2, seg, det_list_class, near_road_set