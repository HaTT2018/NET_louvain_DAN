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
    print(flow.shape)

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
    image1 = np.reshape(image, [-1, k, len(flow[0,:])])
    image2 = np.transpose(image1,(1,2,0))
    image3 = []
    label = []
    day = []

    for i in range(1, t_p):
        for j in range(144-t_input-t_pre):
            image3.append(image2[:, i*144+j:i*144+j+t_input, :][:])
            label.append(flow[:, i*144+j+t_input:i*144+j+t_input+t_pre][:])
            day.append(flow[:, (i-1)*144+j+t_input:(i-1)*144+j+t_input+t_pre][:])

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

def cal_L2_dist(total):
    try:
        total0 = np.broadcast_to(np.expand_dims(total, axis=0), (int(total.shape[0]), int(total.shape[0]), int(total.shape[1])))
        total1 = np.broadcast_to(np.expand_dims(total, axis=1), (int(total.shape[0]), int(total.shape[0]), int(total.shape[1])))
        # total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
    except:
        total_cpu = total
        len_ = total_cpu.shape[0]
        L2_distance = np.zeros([len_, len_])
        for i in range(total_cpu.shape[1]):
            total0 = np.broadcast_to(np.expand_dims(total_cpu[:, i], axis=0), (int(total_cpu.shape[0]), int(total_cpu.shape[0])))
            total1 = np.broadcast_to(np.expand_dims(total_cpu[:, i], axis=1), (int(total_cpu.shape[0]), int(total_cpu.shape[0])))
            # total0 = total_cpu[:, i].unsqueeze(0).expand(int(total_cpu.size(0)), int(total_cpu.size(0)))
            # total1 = total_cpu[:, i].unsqueeze(1).expand(int(total_cpu.size(0)), int(total_cpu.size(0)))
            L2_dist = (total0 - total1)**2
            L2_distance += L2_dist
    return L2_distance

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    #source = source.cpu()
    #target = target.cpu()
    n_samples = int(source.size)+int(target.size)  # number of samples
    total = np.concatenate([source, target], axis=0)
    L2_distance = cal_L2_dist(total)
                       
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = np.sum(L2_distance.data) / (n_samples**2-n_samples)  # 可能出问题
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [np.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  #/len(kernel_val)

def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size)
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size)
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = np.mean(XX + YY - XY -YX)
    return loss