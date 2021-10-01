#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import pandas as pd


# In[2]:


import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Layer
import tensorflow as tf
import dan_models
import dan_utils


# In[3]:

def main(class_i):
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


    # In[4]:


    class_set = [2, 3, 4]
    v, v_class, id_402, part1, part2, seg, det_list_class, near_road_set          = dan_utils.load_data(class_set, res=11, randseed=25)


    # In[5]:


    # ind, class
    # 0  , blue
    # 1  , green
    # 2  , yellow  <--
    # 3  , black   <--
    # 4  , red     <--
    class_color_set = ['b', 'g', 'y', 'black', 'r']

    near_road = np.array(near_road_set[class_i])
    flow = v_class[class_i].iloc[:, 2:-1]

    prop = 0.8  # proportion of training data
    from_day = 1
    to_day = 24
    num_links = v_class[class_i].shape[0]

    image_train, image_test, day_train, day_test, label_train, label_test= dan_utils.sliding_window(
        flow, near_road, from_day, to_day, prop, num_links
    )

    t_input = image_train.shape[2]
    t_pre = label_train.shape[2]
    k = image_train.shape[1]


    # In[8]:


    input_data = keras.Input(shape=(k,t_input,num_links), name='input_data')
    input_HA = keras.Input(shape=(num_links, t_pre), name='input_HA')

    finish_model = dan_models.build_model(input_data, input_HA)


    # In[9]:


    finish_model.compile(optimizer='adam', loss='mean_squared_error')


    # In[10]:


    X_train = image_train
    X_HA_train = day_train
    label_train = label_train


    # In[12]:


    #模型拟合与评估
    finish_model.fit([X_train,X_HA_train], label_train, epochs=700, batch_size=2048,
    validation_data=([image_test,day_test], label_test))
    # finish_model.evaluate(image_test, label_test)


    # In[13]:


    #模型预测
    model_pre = finish_model.predict([image_test,day_test])


    # In[15]:


    #计算各项误差指标

    m = 5
    nrmse_mean = dan_utils.nrmse_loss_func(model_pre, label_test, m)
    mape_mean = dan_utils.mape_loss_func(model_pre, label_test, m)
    smape_mean = dan_utils.smape_loss_func(model_pre, label_test, m)
    mae_mean = dan_utils.mae_loss_func(model_pre, label_test, m)

    print('nrmse = ' + str(nrmse_mean) + '\n' + 'mape = ' + str(mape_mean) + '\n' + 'smape = ' + str(smape_mean) + '\n' + 'mae = ' + str(mae_mean))


    # In[16]:


    #模型保存
    # finish_model.save_weights('../model/base_ST-DTNN_%s_mape=%.5f_nrmse=%.5f.h5'%(class_color_set[class_i], mape_mean, nrmse_mean))
    finish_model.save_weights('../model/source_%s.h5'%(class_color_set[class_i]))




for i in range(5):
    main(i)

