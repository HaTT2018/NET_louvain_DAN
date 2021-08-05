import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import matplotlib.cm as cm
import networkx.algorithms.community as nxcom
from community import community_louvain


def get_corr(data):
    data = np.array(q)
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data = (data-data_mean) / data_std
    data_c = np.corrcoef(data)
    return data_c

def main(randseed):
    seg = pd.read_csv('./data/segement.csv', header=None)
    raw0 = pd.read_csv(open('./data/id2000.csv'), header=0, index_col=0)
    q = pd.read_csv('./data/q_20_aggragated.csv', index_col = 0)
    b = pd.read_csv('./data/b_20_aggragated.csv', index_col = 0)  # time occupancy, (density)

    q_det = q.T.mean()
    b_det = b.T.mean()

    nodes = np.array(raw0['id2'])

    relation_df = pd.read_csv('./data/edges_all.csv', header = None)

    relation_df['flow'] = ''

    for i in range(len(relation_df)):
        #ipdb.set_trace()
        det1 = raw0[raw0['id2']==relation_df.iloc[i, 0]]['id'].iloc[0]
        det2 = raw0[raw0['id2']==relation_df.iloc[i, 1]]['id'].iloc[0]
        relation_df.iloc[i, 2] = (q_det[det1] + q_det[det2]) / 2

    relation = np.array(relation_df)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(relation)

    pos0 = raw0.iloc[:, 1:3]
    pos0 = np.array(pos0)

    vnode = pos0
    npos = dict(zip(nodes, vnode))  # 获取节点与坐标之间的映射关系，用字典表示

    partition = community_louvain.best_partition(G, resolution=10, weight='weight', random_state=randseed)

    # draw the graph
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)

    partition_results = pd.DataFrame(data = list(partition.values()))
    partition_results['node'] = nodes

    partition_results.to_csv('./res/partition_results1.csv', index=False)

    relation_df['if boundary nodes'] = ''

    for i in range(len(relation_df)):
        class1 = partition_results[partition_results['node']==relation_df.iloc[i, 0]].iloc[0, 0]
        class2 = partition_results[partition_results['node']==relation_df.iloc[i, 1]].iloc[0, 0]
        if class1 != class2:
            relation_df.iloc[i, 3] = 1
        else:
            relation_df.iloc[i, 3] = 0

    bound_nodes_df = relation_df[relation_df['if boundary nodes']==1].iloc[:, 0:2]
    bound_nodes = np.array(bound_nodes_df).flatten()

    bound_dets_df = pd.DataFrame([], columns=['n1d1','n1d2','n2d1','n2d2'], index=range(len(bound_nodes_df)))
    for i in range(len(bound_nodes_df)):
        node1 = bound_nodes_df.iloc[i, 0]
        node2 = bound_nodes_df.iloc[i, 1]
        bound_dets_df.iloc[i, 0] = seg[seg[0]==node1].iloc[0, 6]
        bound_dets_df.iloc[i, 1] = seg[seg[0]==node1].iloc[0, 7]
        bound_dets_df.iloc[i, 2] = seg[seg[0]==node2].iloc[0, 6]
        bound_dets_df.iloc[i, 3] = seg[seg[0]==node2].iloc[0, 7]

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

    for i in range(len(bound_nodes_df)):
        node1 = bound_nodes_df.iloc[i, 0]
        node2 = bound_nodes_df.iloc[i, 1]
        class1 = partition_results[partition_results['node']==node1].iloc[0, 0]
        class2 = partition_results[partition_results['node']==node2].iloc[0, 0]
        
        b1 = b_det[b_det['class']==class1]
        b2 = b_det[b_det['class']==class2]
        
        ini_var1 = np.var(np.array(b1[0]))  # initial variance 1
        ini_var2 = np.var(np.array(b2[0]))
        ini_mean1 = np.mean(np.array(b1[0]))
        ini_mean2 = np.mean(np.array(b2[0]))
        
        # 把node1从class1里面挑出来
        b1_ = b1[b1['node']!=node1][0]
        # 计算class1的variance
        var1 = np.var(np.array(b1_))
        # 把node1加到class2
        b2_ = np.append(np.array(b2[0]), np.array(b1[b1['node']==node1][0]))
        # 计算class2的variance
        var2 = np.var(b2_)
        # 比较两个variance，假如varianc减小，则保留更改，反之恢复原位
        if var1<ini_var1 and var2<ini_var2:
            b_det[b_det['node']==node1]['class'] = class2
            
        # 对node2重复以上五步
        b2_ = b2[b2['node']!=node2][0]
        var2 = np.var(np.array(b2_))
        b1_ = np.append(np.array(b1[0]), np.array(b2[b2['node']==node2][0]))
        var1 = np.var(b1_)
        if var1<ini_var1 and var2<ini_var2:
            b_det[b_det['node']==node2]['class'] = class1

    b_det.to_csv('./res/b_det.csv')

    id_2000 = pd.read_csv('./data/id2000.csv', index_col=0)
    id_402 = pd.read_csv('./data/selected_id.csv')
    seg = pd.read_csv('./data/segement.csv', header=None)
    partition_results = pd.read_csv('./res/partition_results1.csv')

    id_402['id_node'] = ''
    for i in range(len(id_402)):
        id_402.iloc[i, 4] = b_det[b_det['det']==id_402.iloc[i, 0]].iloc[0, 2]
            
    id_402['class_i'] = ''
    for i in range(len(id_402)):
        id_402.iloc[i, 5] = b_det[b_det['det']==id_402.iloc[i, 0]].iloc[0, 3]

    id_402.to_csv('./res/id_402_withclass.csv')

    q = pd.read_csv('./data/q_20_aggragated.csv', index_col = 0)
    b = pd.read_csv('./data/b_20_aggragated.csv', index_col = 0)  # time occupancy, (density)
    v = pd.read_csv('./data/v_20_aggragated.csv', index_col = 0)
    id_class = pd.read_csv('./res/id_402_withclass.csv')

    id_class0 = id_class[id_class.class_i == 0]
    q_c0 = q.loc[id_class0.id]
    b_c0 = b.loc[id_class0.id]

    id_class0 = id_class[id_class.class_i == 1]
    q_c1 = q.loc[id_class0.id]
    b_c1 = b.loc[id_class0.id]

    id_class0 = id_class[id_class.class_i == 2]
    q_c2 = q.loc[id_class0.id]
    b_c2 = b.loc[id_class0.id]

    id_class0 = id_class[id_class.class_i == 3]
    q_c3 = q.loc[id_class0.id]
    b_c3 = b.loc[id_class0.id]

    fig = plt.figure(figsize=[16, 16])
    #ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(111)
    #ax3 = fig.add_subplot(223)
    #ax4 = fig.add_subplot(224)
    ax2.set_xlim([0, 30])
    ax2.set_ylim([0, 700])

    ax2.scatter(x = b_c0.iloc[:,:432].mean(), y = q_c0.iloc[:,:432].mean(), s=1, c = 'b')
    ax2.scatter(x = b_c1.iloc[:,:432].mean(), y = q_c1.iloc[:,:432].mean(), s=1, c = 'g')
    ax2.scatter(x = b_c2.iloc[:,:432].mean(), y = q_c2.iloc[:,:432].mean(), s=1, c = 'y')
    ax2.scatter(x = b_c3.iloc[:,:432].mean(), y = q_c3.iloc[:,:432].mean(), s=1, c = 'black')

    plt.savefig('./res/img/%i.png'%randseed)

for i in range(20):
    main(i)