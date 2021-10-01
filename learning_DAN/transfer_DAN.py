#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning on a network, where roads are clustered into classes

# In[1]:


import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipdb


# In[2]:


import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer
import dan_models
import dan_utils

def main(class_src, class_tar):
    # In[3]:


    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


    # # Load data

    # In[4]:


    class_set = [2, 3, 4]
    randseed = 25
    res = 11
    v, v_class, id_402, part1, part2, seg, det_list_class, near_road_set          = dan_utils.load_data(class_set, res, randseed)

    # ind, class
    # 0  , blue
    # 1  , green
    # 2  , yellow  <--
    # 3  , black   <--
    # 4  , red     <--
    class_color_set = ['b', 'g', 'y', 'black', 'r']

    # ## Evaluation of 2 datasets

    # In[8]:


    def get_NSk(set1, set2):
        # designated for v_class1 and 2
        set1_v_mean = set1.iloc[:, 2:-1].T.mean().T
        set2_v_mean = set2.iloc[:, 2:-1].T.mean().T
        
        var1 = set1_v_mean.std()**2
        var2 = set2_v_mean.std()**2
        
        u1 = set1_v_mean.mean()
        u2 = set2_v_mean.mean()
        
        return 2*var1 / (var1 + var2 + (u1 - u2)**2)


    # In[9]:


    NSk_set = np.array([])

    for i in class_set:
        for j in class_set:
            if i!=j:
                NSk = get_NSk(v_class[i], v_class[j])
                NSk_set = np.append(NSk_set, NSk)

    print(NSk_set.mean())


    # # 源代码如下 （训练）
    # # Input classes here

    # In[10]:


    # ind, class
    # 0  , blue
    # 1  , green
    # 2  , yellow  <--
    # 3  , black   <--
    # 4  , red     <--
    # class_src = 2
    v_class1 = v_class[class_src]  # source
    near_road1 = np.array(near_road_set[class_src])

    # class_tar = 4
    v_class2 = v_class[class_tar]  # target
    near_road2 = np.array(near_road_set[class_tar])

    k, t_input, t_pre, num_links = 5, 12, 3, v_class1.shape[0]


    # In[11]:


    near_road_src = near_road1
    flow_src = v_class1.iloc[:, 2:-1]
    prop = 1  # proportion of training data
    from_day = 1
    to_day = 24
    t_p = to_day - from_day + 1

    image_train_source, image_test_source, day_train_source, day_test_source, label_train_source, label_test_source= dan_utils.sliding_window(
        flow_src, near_road_src, from_day, to_day, prop, 
        k, t_p, t_input, t_pre, num_links
    )


    # In[12]:


    near_road_tar = near_road2
    flow_tar = v_class2.iloc[:, 2:-1]
    prop = 1/3
    from_day = 22
    to_day = 30
    t_p = to_day - from_day + 1

    image_train_target, image_test_target, day_train_target, day_test_target, label_train_target, label_test_target= dan_utils.sliding_window(
        flow_tar, near_road_tar, from_day, to_day, prop, 
        k, t_p, t_input, t_pre, num_links
    )

    dup_mul = image_train_source.shape[0]//image_train_target.shape[0]
    dup_r   = image_train_source.shape[0]%image_train_target.shape[0]

    image_train_target, day_train_target, label_train_target = np.concatenate((np.tile(image_train_target, [dup_mul, 1, 1, 1]), image_train_target[:dup_r, :, :, :]), axis=0),np.concatenate((np.tile(day_train_target, [dup_mul, 1, 1]), day_train_target[:dup_r, :, :]), axis=0),np.concatenate((np.tile(label_train_target, [dup_mul, 1, 1]), label_train_target[:dup_r, :, :]), axis=0),


    # In[13]:


    print(image_train_target.shape)
    print(image_test_target.shape)
    print(day_train_target.shape)
    print(day_test_target.shape)
    print(label_train_target.shape)
    print(label_test_target.shape)


    # In[14]:


    #模型构建
    input_data = keras.Input(shape=(k,t_input,num_links), name='input_data')
    input_HA = keras.Input(shape=(num_links, t_pre), name='input_HA')

    finish_model = dan_models.build_model(input_data, input_HA)


    # In[15]:


    class_src


    # In[16]:


    #参数加载
    finish_model.load_weights('../model/source_%s.h5'%class_color_set[class_src])
    #模型预测
    model_pre = finish_model.predict([image_test_target, day_test_target])


    # In[17]:


    model_pre.shape


    # In[18]:


    #预测结果存储
    dan_utils.save_np(model_pre.reshape(model_pre.shape[0], -1), '../model/middle_res/%i_res%i_modelpre_%s_%s.csv'%(randseed, res, class_color_set[class_src], class_color_set[class_tar]))


    # In[19]:


    #transfer without FT 预测精度计算
    mape_mean = dan_utils.mape_loss_func(model_pre, label_test_target)
    smape_mean = dan_utils.smape_loss_func(model_pre, label_test_target)
    mae_mean = dan_utils.mae_loss_func(model_pre, label_test_target)

    print('mape = ' + str(mape_mean) + '\n' + 'smape = ' + str(smape_mean) + '\n' + 'mae = ' + str(mae_mean))


    # In[20]:


    # from sklearn import metrics
    def mmd(x, y):
        return np.abs(x.mean() - y.mean())


    # In[21]:


    import scipy.stats
    def norm_data(data):
        min_ = min(data)
        max_ = max(data)
        normalized_data = data - min_ / (max_ - min_)
        return normalized_data
        
    def js_divergence(set1, set2):
        p = np.array(set1.iloc[:, 2:-1].T.mean().T)
        q = np.array(set2.iloc[:, 2:-1].T.mean().T)
        M=(p+q)/2
        return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)
        # return scipy.stats.entropy(p, q)  # kl divergence


    # In[22]:


    def cal_L2_dist(total):
    #     ipdb.set_trace()
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
    #     ipdb.set_trace()
        return L2_distance

    def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        #source = source.cpu()
        #target = target.cpu()
    #     ipdb.set_trace()
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
    #     ipdb.set_trace()
        print(source.shape)
        print(target.shape)
        batch_size = int(source.size)
        kernels = guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = 0
        for i in range(batch_size):
            s1, s2 = i, (i+1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
    #     ipdb.set_trace()
        return loss / float(batch_size)

    def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    #     ipdb.set_trace()
        batch_size = int(source.shape[0])  # ?
        kernels = guassian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
    #     ipdb.set_trace()
        loss = np.mean(XX + YY - XY -YX)
        return loss


    # In[23]:


    middle1 = Model(inputs=[input_data, input_HA], outputs=finish_model.get_layer('dense_1').output)
    middle2 = Model(inputs=[input_data, input_HA], outputs=finish_model.get_layer('dense_2').output)

    middle_result_source1 = middle1.predict([image_train_source, day_train_source])
    middle_result_target1 = middle1.predict([image_train_target, day_train_target])

    middle_result_source2 = middle2.predict([image_train_source, day_train_source])
    middle_result_target2 = middle2.predict([image_train_target, day_train_target])

    # save intermidiate results
    dan_utils.save_np(middle_result_source1, '../model/middle_res/%i_res%i_middle_result_source1_%s_%s.csv'                 %(randseed, res, class_color_set[class_src], class_color_set[class_tar]))
    dan_utils.save_np(middle_result_target1, '../model/middle_res/%i_res%i_middle_result_target1_%s_%s.csv'                 %(randseed, res, class_color_set[class_src], class_color_set[class_tar]))
    dan_utils.save_np(middle_result_source2, '../model/middle_res/%i_res%i_middle_result_source2_%s_%s.csv'                 %(randseed, res, class_color_set[class_src], class_color_set[class_tar]))
    dan_utils.save_np(middle_result_target2, '../model/middle_res/%i_res%i_middle_result_target2_%s_%s.csv'                 %(randseed, res, class_color_set[class_src], class_color_set[class_tar]))


    lamb = js_divergence(v_class1.iloc[:, 2:-1], v_class2.iloc[:, 2:-1])
    # lamb = 0

    def new_loss(output_final, label_train_target):
        loss0 = K.mean(K.square(output_final - label_train_target), axis=-1) 
        loss1 = mmd_rbf_noaccelerate(middle_result_source1, middle_result_target1)
        loss2 = mmd_rbf_noaccelerate(middle_result_source2, middle_result_target2)
    #     loss2 = lamb * ( mmd(middle_result_source1, middle_result_target1) + mmd(middle_result_source2, middle_result_target2) )
    #     loss2 = 0.001 * mmd(middle_result_source2, middle_result_target2)
    #     print('Lambda is %.4f'%lamb)
        print(middle_result_source1.shape)
        print(middle_result_target1.shape)
        overall_loss = loss0 + lamb* (loss1 + loss2)
        
        return overall_loss


    # In[24]:


    finish_model.compile(optimizer='adam', loss=new_loss)


    # In[25]:


    finish_model.fit([image_train_target, day_train_target], label_train_target, epochs=300, batch_size=4620,
    validation_data=([image_test_target,day_test_target], label_test_target))


    # In[26]:


    model_pre = finish_model.predict([image_test_target, day_test_target])


    # In[27]:


    #模型保存
    finish_model.save_weights('../model/transfer_DAN_%s_%s_mape=%.5f.h5'%(class_color_set[class_src], class_color_set[class_tar], dan_utils.mape_loss_func(model_pre, label_test_target)))


    # In[28]:


    #transfer with DAN 预测精度计算

    mape_mean = dan_utils.mape_loss_func(model_pre, label_test_target)
    smape_mean = dan_utils.smape_loss_func(model_pre, label_test_target)
    mae_mean = dan_utils.mae_loss_func(model_pre, label_test_target)

    print('mape = ' + str(mape_mean) + '\n' + 'smape = ' + str(smape_mean) + '\n' + 'mae = ' + str(mae_mean))

for i in range(5):
    for j in range(5):
        if i != j:
            main(i, j)