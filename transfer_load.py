import time
import math
import tensorflow as tf
import keras 
import numpy as np
import pandas as pd
import ipdb
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import load_model,Model
from keras.engine.topology import Layer
from keras import backend as K


# 定义融合层，将深度学习算法与历史均值算法融合
class Merge_Layer(Layer):
    def __init__(self, **kwargs):
        super(Merge_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.para1 = self.add_weight(shape=(input_shape[0][1], input_shape[0][2]),
                                     initializer='uniform', trainable=True,
                                     name='para1')
        self.para2 = self.add_weight(shape=(input_shape[1][1], input_shape[1][2]),
                                     initializer='uniform', trainable=True,
                                     name='para2')
        super(Merge_Layer, self).build(input_shape)

    def call(self, inputs):
        mat1 = inputs[0]
        mat2 = inputs[1]
        output = mat1 * self.para1 + mat2 * self.para2
        # output = mat1 * 0.1 + mat2 * 0.9
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

#定义精度评价指标。为防止0值附近相对误差过大而导致的异常，定义mask层。
def mape_loss_func(preds, labels):
    mask = labels > 5
    return np.mean(np.fabs(labels[mask]-preds[mask])/labels[mask])

def smape_loss_func(preds, labels):
    mask= labels > 5
    return np.mean(2*np.fabs(labels[mask]-preds[mask])/(np.fabs(labels[mask])+np.fabs(preds[mask])))

def mae_loss_func(preds, labels):
    mask= labels > 5
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

def get_NSk(set1, set2):
    # designated for v_class1 and 2
    set1_v_mean = set1.iloc[:, 2:-1].T.mean().T
    set2_v_mean = set2.iloc[:, 2:-1].T.mean().T
    
    var1 = set1_v_mean.std()**2
    var2 = set2_v_mean.std()**2
    
    u1 = set1_v_mean.mean()
    u2 = set2_v_mean.mean()
    
    return 2*var1 / (var1 + var2 + (u1 - u2)**2)

def mmd (x, y):
    return metrics.mean_squared_error(x,y)

def kl_divergence(set1, set2):
    set1_v_mean = np.array(set1.iloc[:, 2:-1].T.mean().T)
    set2_v_mean = np.array(set2.iloc[:, 2:-1].T.mean().T)
    return np.sum(np.where(set1_v_mean != 0, set1_v_mean * np.log(set1_v_mean / set2_v_mean), 0))

def main(randseed, class1, class2, class3, class4):
    def new_loss(output_final, label_train_target):
        middle = Model(inputs=[input_data, input_HA],outputs=finish_model.get_layer('dense_%i'%(randseed+1)).output)
        middle_result_source = middle.predict([image_train_source, day_train_source])
        middle_result_target = middle.predict([image_train_target, day_train_target])

        loss1 = K.mean(K.square(output_final - label_train_target), axis=-1) 
        loss2 = 0.001 * mmd (middle_result_source, middle_result_target)
        overall_loss = loss1 + loss2
        return overall_loss

    # randseed = 3
    # class1 = 0
    # class2 = 1
    v = pd.read_csv('./data/v_20_aggragated.csv')
    v = v.rename(columns={'Unnamed: 0': 'id'})
    det_with_class = pd.read_csv('./res/%i_id_402_withclass.csv'%randseed, index_col=0)

    v['class_i'] = ''
    for i in range(len(v)):
        v.loc[i, 'class_i'] = det_with_class[det_with_class['id']==v.loc[i, 'id']].iloc[0, 5]  # 5 stands for 'class_i'

    v_class1 = v[v['class_i']==class1]
    v_class2 = v[v['class_i']==class2]
    v_class3 = v[v['class_i']==class3]
    v_class4 = v[v['class_i']==class4]

    dist_mat = pd.read_csv('./data/dist_mat.csv', index_col=0)
    id_info = pd.read_csv('./data/id2000.csv', index_col=0)
    dist_mat.index = id_info['id2']
    dist_mat.columns = id_info['id2']
    for i in range(len(dist_mat)):
        for j in range(len(dist_mat)):
            if i==j:
                dist_mat.iloc[i, j] = 0

    near_id = pd.DataFrame(np.argsort(np.array(dist_mat)), index = id_info['id2'], columns = id_info['id2'])

    seg = pd.read_csv('./data/segement.csv', header=None)

    det_list_class1, v_class1 = get_class_with_node(seg, v_class1)
    det_list_class2, v_class2 = get_class_with_node(seg, v_class2)
    det_list_class3, v_class3 = get_class_with_node(seg, v_class3)
    det_list_class4, v_class4 = get_class_with_node(seg, v_class4)

    num_dets = 30

    near_road1 = rds_mat(dist_mat, det_list_class1[:num_dets], seg)
    near_road2 = rds_mat(dist_mat, det_list_class2[:num_dets], seg)
    near_road3 = rds_mat(dist_mat, det_list_class3[:num_dets], seg)
    near_road4 = rds_mat(dist_mat, det_list_class4[:num_dets], seg)

    v_class1 = v_class1[v_class1['id'].isin(det_list_class1[:num_dets])]
    v_class2 = v_class2[v_class2['id'].isin(det_list_class2[:num_dets])]
    v_class3 = v_class3[v_class3['id'].isin(det_list_class3[:num_dets])]
    v_class4 = v_class4[v_class4['id'].isin(det_list_class4[:num_dets])]

    v_class_set = [v_class1, v_class2, v_class3, v_class4]
    NSk_set = np.array([])
    for i in range(4):
        for j in range(4):
            if i!=j:
                NSk = get_NSk(v_class_set[i], v_class_set[j])
                NSk_set = np.append(NSk_set, NSk)

    print('NSk is %.3f'%NSk_set.mean())
    '''
    ########################
    # near_road = np.array(pd.read_csv('./data/network/2small_network_nearest_road_id.csv',header = 0))
    # flow = np.array(pd.read_csv('./data/network/2small_network_speed.csv', header= 0)) #注意header=0 or None
    near_road = np.array(near_road1)
    flow = np.array(v_class1.iloc[:, 2:-1])

    # 利用滑动窗口的方式，重构数据为(n，最近路段数，输入时间窗，总路段数)的形式
    k = 5 # 参数k为需考虑的最近路段数
    t_p = 24 # 参数t_p为总时间序列长度（天）
    t_input = 12 #参数t_input为输入时间窗(5min颗粒度)
    t_pre = 3 #参数t_pre为预测时间窗(5min颗粒度)
    num_links = 30 #参数num_links为总路段数

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

    for i in range(1,t_p):
        for j in range(180-t_input-t_pre):
            image3.append(image2[:, i*180+j:i*180+j+t_input, :][:])
            label.append(flow[:, i*180+j+t_input:i*180+j+t_input+t_pre][:])
            day.append(flow[:, (i-1)*180+j+t_input:(i-1)*180+j+t_input+t_pre][:])
            

    image3 = np.asarray(image3)
    label = np.asarray(label)
    day =  np.asarray(day)

    #划分前90%数据为训练集，最后10%数据为测试集
    image_train_source = image3[:np.shape(image3)[0]*1//10]
    image_test_source = image3[np.shape(image3)[0]*1//10:]
    label_train_source = label[:np.shape(label)[0]*1//10]
    label_test_source = label[np.shape(label)[0]*1//10:]

    day_train_source = day[:np.shape(day)[0]*1//10]
    day_test_source = day[np.shape(day)[0]*1//10:]
    ########################


    ########################
    # near_road = np.array(pd.read_csv('./data/transfer_learning_traffic_data/small_network_nearest_road_id.csv',header = 0))
    # flow = np.array(pd.read_csv('./data/transfer_learning_traffic_data/small_network_speed.csv', header= 0)) #注意header=0 or None
    near_road = np.array(near_road2)
    flow = np.array(v_class2.iloc[:, 2:-1])

    # 利用滑动窗口的方式，重构数据为(n，最近路段数，输入时间窗，总路段数)的形式
    k = 5 # 参数k为需考虑的最近路段数
    t_p = 24 # 参数t_p为总时间序列长度（天）
    t_input = 12 #参数t_input为输入时间窗(5min颗粒度)
    t_pre = 3 #参数t_pre为预测时间窗(5min颗粒度)
    num_links = 30 #参数num_links为总路段数

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

    for i in range(1,t_p):
        for j in range(180-t_input-t_pre):
            image3.append(image2[:, i*180+j:i*180+j+t_input, :][:])
            label.append(flow[:, i*180+j+t_input:i*180+j+t_input+t_pre][:])
            day.append(flow[:, (i-1)*180+j+t_input:(i-1)*180+j+t_input+t_pre][:])

    image3 = np.asarray(image3)
    label = np.asarray(label)
    day =  np.asarray(day)

    #划分前80%数据为训练集，最后20%数据为测试集
    image_train_target = image3[:np.shape(image3)[0]*1//10]
    image_test_target = image3[np.shape(image3)[0]*1//10:]
    label_train_target = label[:np.shape(label)[0]*1//10]
    label_test_target = label[np.shape(label)[0]*1//10:]

    day_train_target = day[:np.shape(day)[0]*1//10]
    day_test_target = day[np.shape(day)[0]*1//10:]
    ########################

    #模型构建
    input_data = keras.Input(shape=(k,t_input,num_links), name='input_data')
    input_HA = keras.Input(shape=(num_links, t_pre), name='input_HA')

    x = keras.layers.BatchNormalization(input_shape =(k,t_input,num_links))(input_data)

    x = keras.layers.Conv2D(
                            filters = num_links,
                            kernel_size = 3,
                            strides = 1,
                            padding="SAME",
                            activation='relu')(x)

    x = keras.layers.AveragePooling2D(pool_size = (2,2),
                                    strides = 1,
                                    padding = "SAME",
                                    )(x)

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(
                        filters = num_links,
                        kernel_size = 3,
                        strides = 1,
                        padding="SAME",
                        activation='relu')(x)

    x = keras.layers.AveragePooling2D(pool_size = (2,2),
                                    strides = 1,
                                    padding = "SAME",
                                    )(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_links*t_pre, activation='relu')(x)

    output = keras.layers.Reshape((num_links,t_pre))(x)

    output_final = Merge_Layer()([output, input_HA])

    # construct model
    finish_model = keras.models.Model([input_data,input_HA], [output_final])

    finish_model.summary()

    #参数加载
    finish_model.load_weights('./model/source.h5')

    #模型预测
    model_pre = finish_model.predict([image_test_target,day_test_target])

    #transfer without FT 预测精度计算
    mape_mean1 = mape_loss_func(model_pre, label_test_source)
    smape_mean1 = smape_loss_func(model_pre, label_test_source)
    mae_mean1 = mae_loss_func(model_pre, label_test_source)

    print('mape = ' + str(mape_mean1) + '\n' + 'smape = ' + str(smape_mean1) + '\n' + 'mae = ' + str(mae_mean1))

    middle = Model(inputs=[input_data, input_HA]
    ,outputs=finish_model.get_layer('dense_%i'%(randseed+1)).output)

    middle_result_source = middle.predict([image_train_source, day_train_source])
    middle_result_target = middle.predict([image_train_target, day_train_target])

    lamb = kl_divergence(v_class1, v_class2)

    loss1 = K.mean(K.square(output_final - label_train_target), axis=-1) 
    loss2 = lamb * mmd (middle_result_source, middle_result_target)
    overall_loss = loss1 + loss2

    finish_model.compile(optimizer='adam',loss=new_loss)

    finish_model.fit([image_train_target, day_train_target], label_train_target, epochs=100, batch_size=462,
    validation_data=([image_test_target,day_test_target], label_test_target))

    model_pre = finish_model.predict([image_test_target,day_test_target])

    #transfer with DAN 预测精度计算

    mape_mean2 = mape_loss_func(model_pre, label_test_target)
    smape_mean2 = smape_loss_func(model_pre, label_test_target)
    mae_mean2 = mae_loss_func(model_pre, label_test_target)

    print('mape = ' + str(mape_mean2) + '\n' + 'smape = ' + str(smape_mean2) + '\n' + 'mae = ' + str(mae_mean2))

    mape_list = []
    for i in range(num_links):
        a1 = mape_loss_func(model_pre[:,i,:], label_test_target[:,i,:])
        mape_list.append(a1)

    mape_pd = pd.Series(mape_list)
    # mape_pd.sort_values()
    return NSk_value, mape_mean1, mape_mean2
    '''
    return NSk_set.mean()

if __name__ == '__main__':
    class1 = 0
    class2 = 1
    class3 = 2
    class4 = 3
    NSk_value_set, mape_mean1_set, mape_mean2_set = [], [], []
    plot_len = 10
    for randseed in range(plot_len):
        '''
        NSk_value, mape_mean1, mape_mean2 = main(randseed, class1, class2, class3, class4)
        NSk_value_set.append(NSk_value)
        mape_mean1_set.append(mape_mean1)
        mape_mean2_set.append(mape_mean2)
        '''
        NSk_value = main(randseed, class1, class2, class3, class4)
    
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.plot(range(plot_len), NSk_value_set)
    ax2 = fig.add_subplot(132)
    ax2.plot(range(plot_len), mape_mean1_set)
    ax3 = fig.add_subplot(133)
    ax3.plot(range(plot_len), mape_mean2_set)
    plt.show()
    '''