import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import matplotlib.cm as cm
import networkx.algorithms.community as nxcom
from community import community_louvain
import os

def main(randseed, resolution):

    def get_corr(data):
        data = np.array(q)
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data = (data-data_mean) / data_std
        data_c = np.corrcoef(data)
        return data_c


    def assemble_det_relation_df(relation_df, seg, raw0, side):
        # side should be 0 or 1
        if side==1:
            side_ind = 6
        elif side==2:
            side_ind = 7
        det_relation_df = pd.DataFrame(columns=['from_det', 'to_det', 'flow', 'linkID'], index=range(len(relation_df)))
        for i in range(len(relation_df)):
            from_node = relation_df.iloc[i, 0]
            to_node = relation_df.iloc[i, 1]
            from_det = seg.loc[seg[0]==from_node].iloc[0, side_ind]  # 0 indicates node
            to_det = seg.loc[seg[0]==to_node].iloc[0, side_ind]
            det_relation_df.loc[i, 'from_det'] = from_det
            det_relation_df.loc[i, 'to_det'] = to_det
            try:
                # flow是两个detector的flow的平均
                det_relation_df.iloc[i, 2] = (q_det[from_det] + q_det[to_det]) / 2
            except:
                # 假如det没有，就用node信息找到flow
                try:
                    from_node = seg[seg[6]==from_det].iloc[0, 0]
                    to_node = seg[seg[6]==to_det].iloc[0, 0]
                except:
                    from_node = seg[seg[7]==from_det].iloc[0, 0]
                    to_node = seg[seg[7]==to_det].iloc[0, 0]
                    
                from_det = raw0[raw0['id2']==from_node]['id'].iloc[0]
                to_det = raw0[raw0['id2']==to_node]['id'].iloc[0]
                
                det_relation_df.loc[i, 'flow'] = (q_det[from_det] + q_det[to_det]) / 2
            
            det_relation_df.loc[i, 'linkID'] = i
        return det_relation_df


    def get_det_partition_results(seg, partition_results, side):
        if side==1:
            side_ind = 6
        elif side==2:
            side_ind = 7
        
        det_partition_results = pd.DataFrame([], columns=[0, 'det'], index=range(len(partition_results)))
        for i in range(len(partition_results)):
            det_partition_results.loc[i, 'det'] = seg[seg[0]==partition_results.loc[i, 'node']].iloc[0, side_ind]
        
        det_partition_results.loc[:, 0] = partition_results.loc[:, 0]
        return det_partition_results


    def get_bound_x_df(relation_df, partition_results):
        relation_df['if boundary'] = ''

        for i in range(len(relation_df)):
            class1 = partition_results[partition_results.iloc[:, 1]==relation_df.iloc[i, 0]].iloc[0, 0]
            class2 = partition_results[partition_results.iloc[:, 1]==relation_df.iloc[i, 1]].iloc[0, 0]
            if class1 != class2:
                relation_df.loc[i, 'if boundary'] = 1
            else:
                relation_df.loc[i, 'if boundary'] = 0

        bound_x_df = relation_df[relation_df['if boundary']==1].iloc[:, 0:2]
        # bound_nodes = np.array(bound_x_df).flatten()
        return bound_x_df


    def compare(det_id, b1, b2, det_partition_results):
        if_adjust = 0
        
        ini_var1 = np.var(np.array(b1[0]))  # initial variance
        ini_var2 = np.var(np.array(b2[0]))
        ini_mean1 = np.mean(np.array(b1[0]))  # initial mean
        ini_mean2 = np.mean(np.array(b2[0]))
        
        
        # 把det1从class1里面挑出来
        b1_ = b1[b1['det']!=det_id][0]
        # 计算不含det1的b11_的variance
        var1 = np.var(np.array(b1_))
        # 把det1加到class2
        b2_ = np.append(np.array(b2[0]), np.array(b1[b1['det']==det_id][0]))
        # 计算class2的variance
        var2 = np.var(b2_)
        # 比较两个variance，假如variance减小，则保留更改，反之恢复原位
        if var1<ini_var1 and var2<ini_var2:
            #ipdb.set_trace()
            class2 = b2.iloc[0, 3]  # 3 stands for class
            b_det.loc[b_det['det']==det_id, 'class'] = class2
            det_partition_results.loc[det_partition_results['det']==det_id, 0] = class2
            if_adjust = 1
        
        return b_det, det_partition_results, if_adjust


    seg = pd.read_csv('./data/segement.csv', header=None)
    raw0 = pd.read_csv(open('./data/id2000.csv'), header=0, index_col=0)
    q = pd.read_csv('./data/q_20_aggragated.csv', index_col = 0)
    b = pd.read_csv('./data/b_20_aggragated.csv', index_col = 0)  # time occupancy, (density)

    q_det = q.T.mean()
    b_det = b.T.mean()
    nodes = np.array(raw0['id2'])
    relation_df = pd.read_csv('./data/edges_all.csv', header = None)

    # 1 and 2 are different directions
    det_relation_df1 = assemble_det_relation_df(relation_df, seg, raw0, side=1)
    det_relation_df2 = assemble_det_relation_df(relation_df, seg, raw0, side=2)

    relation_df['flow'] = ''
    relation_df['linkID'] = ''

    for i in range(len(relation_df)):
        #ipdb.set_trace()
        det1 = raw0[raw0['id2']==relation_df.iloc[i, 0]]['id'].iloc[0]
        det2 = raw0[raw0['id2']==relation_df.iloc[i, 1]]['id'].iloc[0]
        relation_df.loc[i, 'flow'] = (q_det[det1] + q_det[det2]) / 2
        relation_df.loc[i, 'linkID'] = i

    relation = np.array(relation_df.iloc[:, :3])  # relation and flow (weight)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(relation)  # add weight from flow

    pos0 = raw0.iloc[:, 1:3]
    pos0 = np.array(pos0)

    vnode = pos0
    npos = dict(zip(nodes, vnode))  # 获取节点与坐标之间的映射关系，用字典表示

    partition = community_louvain.best_partition(G, resolution=resolution, weight='weight', random_state=randseed)

    # draw the graph
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)

    fig_net = plt.figure(figsize = (10,10))
    nx.draw_networkx(G, pos = npos, node_size=20, node_color=list(partition.values()), with_labels=False)


    partition_results = pd.DataFrame(data = list(partition.values()))
    partition_results['node'] = nodes

    det_partition_results1 = get_det_partition_results(seg, partition_results, side=1)
    det_partition_results2 = get_det_partition_results(seg, partition_results, side=2)


    partition_results.to_csv('./res/%i_res%i_partition_results.csv'%(randseed, resolution), index=False)
    det_partition_results1.to_csv('./res/%i_res%i_det_partition_results1.csv'%(randseed, resolution), index=False)
    det_partition_results2.to_csv('./res/%i_res%i_det_partition_results2.csv'%(randseed, resolution), index=False)

    b_det = pd.DataFrame(b_det)
    b_det['det'] = b_det.index
    b_det['node']=''
    b_det.index=range(402)

    for i in range(len(b_det)):
        try:
            b_det.iloc[i, 2] = seg[seg[6]==b_det.iloc[i, 1]].iloc[0, 0]
        except:
            b_det.iloc[i, 2] = seg[seg[7]==b_det.iloc[i, 1]].iloc[0, 0]

    b_det['class'] = ''
    for i in range(len(b_det)):
        b_det.iloc[i, 3] = partition_results[partition_results['node']==b_det.iloc[i, 2]].iloc[0, 0]

    org_bound_dets_df1 = get_bound_x_df(det_relation_df1, det_partition_results1)
    org_bound_dets_df2 = get_bound_x_df(det_relation_df2, det_partition_results2)
    org_det_partition_results1 = det_partition_results1.copy()
    org_det_partition_results2 = det_partition_results2.copy()

    # Boundary adjustment
    for i in range(len(org_bound_dets_df1)):
        adj_time = 0
        while 1:
            bound_nodes_df = get_bound_x_df(relation_df, partition_results)
            bound_dets_df1 = get_bound_x_df(det_relation_df1, det_partition_results1)
            bound_dets_df2 = get_bound_x_df(det_relation_df2, det_partition_results2)

            n1d1 = bound_dets_df1.iloc[i, 0]
            n1d2 = bound_dets_df2.iloc[i, 0]
            n2d1 = bound_dets_df1.iloc[i, 1]
            n2d2 = bound_dets_df2.iloc[i, 1]
            
            try:
                node1 = b_det[b_det['det']==n1d1].iloc[0, 2]  # 2 means node
            except:
                node1 = b_det[b_det['det']==n1d2].iloc[0, 2]  # 2 means node
            try:
                node2 = b_det[b_det['det']==n2d1].iloc[0, 2]
            except:
                node2 = b_det[b_det['det']==n2d2].iloc[0, 2]

            class1 = b_det[b_det['node']==node1].iloc[0, 3]  # 3 means class
            class2 = b_det[b_det['node']==node2].iloc[0, 3]

            b1 = b_det[b_det['class']==class1]
            b2 = b_det[b_det['class']==class2]
            
            #ipdb.set_trace()
            b_det, det_partition_results1, if_adjust11 = compare(n1d1, b1, b2, det_partition_results1)
            b_det, det_partition_results2, if_adjust12 = compare(n1d2, b1, b2, det_partition_results2)
            b_det, det_partition_results1, if_adjust21 = compare(n2d1, b2, b1, det_partition_results1)
            b_det, det_partition_results2, if_adjust22 = compare(n2d2, b2, b1, det_partition_results2)

            if_adjust = if_adjust11 + if_adjust12 + if_adjust21 + if_adjust22
            adj_time += if_adjust
            #ipdb.set_trace()
            if if_adjust==0:
                break
                # print('%i times done for boundary adjustment'%adj_time)
            # else:
                # print('%i times done for boundary adjustment'%adj_time)
                # print(b_det[b_det['class']==class1].shape[0])
                # print(det_partition_results1[det_partition_results1[0]==class1].shape[0])


    b_det.to_csv('./res/%i_res%i_b_det.csv'%(randseed, resolution))

    id_2000 = pd.read_csv('./data/id2000.csv', index_col=0)
    id_402 = pd.read_csv('./data/selected_id.csv')
    seg = pd.read_csv('./data/segement.csv', header=None)
    partition_results = pd.read_csv('./res/%i_res%i_partition_results.csv'%(randseed, resolution))

    id_402['id_node'] = ''
    for i in range(len(id_402)):
        id_402.iloc[i, 4] = b_det[b_det['det']==id_402.iloc[i, 0]].iloc[0, 2]
            
    id_402['class_i'] = ''
    for i in range(len(id_402)):
        id_402.iloc[i, 5] = b_det[b_det['det']==id_402.iloc[i, 0]].iloc[0, 3]
    
    id_402.to_csv('./res/%i_res%i_id_402_withclass.csv'%(randseed, resolution))

    q = pd.read_csv('./data/q_20_aggragated.csv', index_col = 0)
    b = pd.read_csv('./data/b_20_aggragated.csv', index_col = 0)  # time occupancy, (density)
    v = pd.read_csv('./data/v_20_aggragated.csv', index_col = 0)
    id_class = pd.read_csv('./res/%i_res%i_id_402_withclass.csv'%(randseed, resolution))

    id_class0 = id_class[id_class.class_i == 0]
    q_c0 = q.loc[id_class0.id]
    b_c0 = b.loc[id_class0.id]

    id_class1 = id_class[id_class.class_i == 1]
    q_c1 = q.loc[id_class1.id]
    b_c1 = b.loc[id_class1.id]

    id_class2 = id_class[id_class.class_i == 2]
    q_c2 = q.loc[id_class2.id]
    b_c2 = b.loc[id_class2.id]

    id_class3 = id_class[id_class.class_i == 3]
    q_c3 = q.loc[id_class3.id]
    b_c3 = b.loc[id_class3.id]

    id_class4 = id_class[id_class.class_i == 4]
    q_c4 = q.loc[id_class4.id]
    b_c4 = b.loc[id_class4.id]

    id_class5 = id_class[id_class.class_i == 5]
    q_c5 = q.loc[id_class5.id]
    b_c5 = b.loc[id_class5.id]
    
    id_class5 = id_class[id_class.class_i == 5]
    q_c5 = q.loc[id_class5.id]
    b_c5 = b.loc[id_class5.id]
    
    id_class5 = id_class[id_class.class_i == 5]
    q_c5 = q.loc[id_class5.id]
    b_c5 = b.loc[id_class5.id]
    
    id_class6 = id_class[id_class.class_i == 6]
    q_c6 = q.loc[id_class6.id]
    b_c6 = b.loc[id_class6.id]
    
    id_class7 = id_class[id_class.class_i == 7]
    q_c7 = q.loc[id_class7.id]
    b_c7 = b.loc[id_class7.id]
    
    id_class8 = id_class[id_class.class_i == 8]
    q_c8 = q.loc[id_class8.id]
    b_c8 = b.loc[id_class8.id]
    
    id_class9 = id_class[id_class.class_i == 9]
    q_c9 = q.loc[id_class9.id]
    b_c9 = b.loc[id_class9.id]
    
    id_class10 = id_class[id_class.class_i == 10]
    q_c10 = q.loc[id_class10.id]
    b_c10 = b.loc[id_class10.id]

    fig_MFD = plt.figure(figsize=[5, 5])
    ax2 = fig_MFD.add_subplot(111)
    ax2.set_xlim([0, 30])
    ax2.set_ylim([0, 700])

    ax2.scatter(x = b_c0.iloc[:,:432].mean(), y = q_c0.iloc[:,:432].mean(), s=1, c = 'b')
    ax2.scatter(x = b_c1.iloc[:,:432].mean(), y = q_c1.iloc[:,:432].mean(), s=1, c = 'g')
    ax2.scatter(x = b_c2.iloc[:,:432].mean(), y = q_c2.iloc[:,:432].mean(), s=1, c = 'y')
    ax2.scatter(x = b_c3.iloc[:,:432].mean(), y = q_c3.iloc[:,:432].mean(), s=1, c = 'black')
    ax2.scatter(x = b_c4.iloc[:,:432].mean(), y = q_c4.iloc[:,:432].mean(), s=1, c = 'r')
    ax2.scatter(x = b_c5.iloc[:,:432].mean(), y = q_c5.iloc[:,:432].mean(), s=1, c = 'grey')
    ax2.scatter(x = b_c6.iloc[:,:432].mean(), y = q_c6.iloc[:,:432].mean(), s=1, c = 'pink')
    ax2.scatter(x = b_c7.iloc[:,:432].mean(), y = q_c7.iloc[:,:432].mean(), s=1, c = 'purple')
    ax2.scatter(x = b_c8.iloc[:,:432].mean(), y = q_c8.iloc[:,:432].mean(), s=1, c = 'lightblue')
    ax2.scatter(x = b_c9.iloc[:,:432].mean(), y = q_c9.iloc[:,:432].mean(), s=1, c = 'coral')
    ax2.scatter(x = b_c10.iloc[:,:432].mean(), y = q_c10.iloc[:,:432].mean(), s=1, c = 'lightgreen')


    # CH index
    cluster_set = b_det['class'].drop_duplicates()

    Nu1 = 0  # numerator 1
    c = b_det[0].mean()
    for c in cluster_set:
        cluster = b_det[b_det['class']==c]
        nk = len(cluster)
        ck = cluster[0].mean()
        Nu1 += nk*abs(ck-c)**2

    Nu2 = 0  # numerator 2
    for c in cluster_set:
        cluster = b_det[b_det['class']==c]
        nk = len(cluster)
        ck = cluster[0].mean()
        for i in range(nk):
            Nu2 += (cluster.iloc[i, 0] - ck)**2
            

    CH = Nu1*Nu2/((len(cluster_set)-1) * (len(b_det)-len(cluster_set)))

    # Total variance TV
    cluster_set = b_det['class'].drop_duplicates()

    TV = 0
    for c in cluster_set:
        cluster = b_det[b_det['class']==c]
        NA = len(cluster)
        TV += NA*cluster[0].std()**2

    #####################################
    # new file
    #####################################
    # NSk value
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


    def rds_mat(old_dist_mat, det_ids):
        # get a matrix that contains n raods that have specified node id s
        node_ids = np.array([])
        for i in det_ids:
            node_ids = np.append(node_ids, get_node(i, seg))
            
        new_dist_mat = old_dist_mat.loc[node_ids, node_ids]
        old_dist_mat = np.array(old_dist_mat)
        new_near_id_mat = np.argsort(new_dist_mat)
        return new_near_id_mat



    v = pd.read_csv('./data/v_20_aggragated.csv')
    v = v.rename(columns={'Unnamed: 0': 'id'})
    det_with_class = pd.read_csv('./res/%i_res%i_id_402_withclass.csv'%(randseed, resolution), index_col=0)

    v['class_i'] = ''
    for i in range(len(v)):
        v.loc[i, 'class_i'] = det_with_class[det_with_class['id']==v.loc[i, 'id']].iloc[0, 5]  # 5 stands for 'class_i'

    num_class = det_with_class['class_i'].drop_duplicates().size

    v_class = []
    for i in range(num_class):
        v_class.append(v[v['class_i']==i])

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
    num_dets = 30

    det_list_class = []
    for i in range(num_class):
        det_list_class_temp, v_class_temp = get_class_with_node(seg, v_class[i])
        det_list_class.append(det_list_class_temp)
        v_class_temp = v_class_temp[v_class_temp['id'].isin(det_list_class_temp[:num_dets])]
        v_class[i] = v_class_temp

    near_road_set = []
    for i in range(num_class):
        near_road_set.append(rds_mat(dist_mat, det_list_class[i][:num_dets]))

    def get_NSk(set1, set2):
        # designated for v_class1 and 2
        set1_v_mean = set1.iloc[:, 2:-1].T.mean().T
        set2_v_mean = set2.iloc[:, 2:-1].T.mean().T
        
        var1 = set1_v_mean.std()**2
        var2 = set2_v_mean.std()**2
        
        u1 = set1_v_mean.mean()
        u2 = set2_v_mean.mean()
        
        return 2*var1 / (var1 + var2 + (u1 - u2)**2)

    NSk_set = np.array([])
    for i in range(num_class):
        for j in range(num_class):
            if i!=j:
                NSk = get_NSk(v_class[i], v_class[j])
                NSk_set = np.append(NSk_set, NSk)

    # print(NSk_set.mean())
    # save result if it is good
    if NSk_set.mean() < 92:
        fig_net.savefig('./res/img/%i_res%i_net'%(randseed, resolution))
        fig_MFD.savefig('./res/img/%i_res%i_MFD'%(randseed, resolution))

        ind_df = pd.DataFrame([], index=range(1), columns=['NSk_mean', 'NSk_min', 'NSk_max', 'NSk_std', 'TV', 'CH'])
        ind_df.loc[0, 'NSk_mean'] = NSk_set.mean()
        ind_df.loc[0, 'NSk_min'] = NSk_set.min()
        ind_df.loc[0, 'NSk_max'] = NSk_set.max()
        ind_df.loc[0, 'NSk_std'] = NSk_set.std()

        ind_df.loc[0, 'TV'] = TV
        ind_df.loc[0, 'CH'] = CH

        ind_df.to_csv('./res/%i_res%i_index.csv'%(randseed, resolution))
    else:
        os.remove('./res/%i_res%i_b_det.csv'%(randseed, resolution))
        os.remove('./res/%i_res%i_det_partition_results1.csv'%(randseed, resolution))
        os.remove('./res/%i_res%i_det_partition_results2.csv'%(randseed, resolution))
        os.remove('./res/%i_res%i_partition_results.csv'%(randseed, resolution))
        os.remove('./res/%i_res%i_id_402_withclass.csv'%(randseed, resolution))

if __name__ == '__main__':
    for resolution in range(11, 20):
        for i in range(100):
            print('Doing %i %i'%(resolution, i))
            main(i, resolution)
            plt.close('all')