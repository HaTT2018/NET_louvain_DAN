import numpy as np
import torch
import random
import tensorflow as tf

def mape_loss_func(preds, labels):
    mask = labels > .05
    return np.mean(np.fabs(labels[mask]-preds[mask])/labels[mask])

def smape_loss_func(preds, labels):
    mask= labels > .05
    return np.mean(2*np.fabs(labels[mask]-preds[mask])/(np.fabs(labels[mask])+np.fabs(preds[mask])))

def mae_loss_func(preds, labels):
    mask= labels > .05
    return np.fabs((labels[mask]-preds[mask])).mean()

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

def sliding_window(flow, near_road, from_day, to_day, prop, k, t_p, t_input, t_pre, num_links):
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

