from torch_geometric.nn import MessagePassing
from torch_geometric.data import InMemoryDataset, Data
import torch
import torch.nn as nn
import torch.nn.functional as F

def input_to_graph_input(edge_num, node_num, state, edge_index, edge_attr):
    edge_state = torch.stack([state, edge_attr, torch.ones_like(state)], dim=1).squeeze(-1)
    node_state = torch.zeros([node_num, edge_state.shape[-1]], device=edge_state.device)
    graph_x = torch.cat([edge_state, node_state])
    node_marker = torch.zeros([edge_num+node_num], dtype=torch.bool, device=edge_state.device)
    node_marker[edge_num:]=1
    graph_data = Data(x=graph_x, edge_index=edge_index, edge_attr=edge_attr,
                      edge_num=edge_num,node_num=node_num, node_marker=node_marker)
    return graph_data

def parse_edge_outputs(graphs, outputs, actions, mode):
    results = []
    graph_start_pointer=0
    for i in range(graphs.num_graphs):
        edge_num = graphs.edge_num[i]
        node_num = graphs.node_num[i]

        output = outputs[graph_start_pointer:graph_start_pointer+edge_num]
        if mode=='action':
            action = actions[i]
        else:
            action=None
        result = parse_edge_output(edge_num, node_num, output, action, mode)
        results.append(result)
        graph_start_pointer = graph_start_pointer + edge_num + node_num
    return torch.stack(results)

def parse_edge_output(edge_num, node_num, output, action, mode='action'):
    edge_output = output[:edge_num]
    #base_q_value = edge_output[:, 0].sum()
    #q_value = edge_output[:, 1] + base_q_value
    q_value = edge_output[:,0]
    #print('q=',q_value)
    if mode=='action':
        #print('\nparse with action, q=%s, action=%d'%(q_value,action))
        return q_value[action]
    elif mode=='max':
        return q_value.max(dim=0).values
    elif mode=='argmax':
        #print('\nq=',q_value)
        return q_value.argmax()
    elif mode=='softmax':
        return F.softmax(q_value)
    elif mode=='default':
        return q_value
    else:
        raise RuntimeError('invalid mode in dqn.data_util.parse_output()')

def parse_node_outputs(graphs, outputs, actions, mode):
    results = []
    graph_start_pointer=0
    for i in range(graphs.num_graphs):
        edge_num = graphs.edge_num[i]
        node_num = graphs.node_num[i]

        output = outputs[graph_start_pointer+edge_num:graph_start_pointer+edge_num+node_num]
        if mode=='action':
            action = actions[i]
        else:
            action=None
        result = parse_node_output(edge_num, node_num, output, action, mode)
        results.append(result)
        graph_start_pointer = graph_start_pointer + edge_num + node_num
    return torch.stack(results)

def parse_node_output(edge_num, node_num, output, action, mode='action'):
    edge_output = output[:edge_num]
    #base_q_value = edge_output[:, 0].sum()
    #q_value = edge_output[:, 1] + base_q_value
    q_value = edge_output[:,0]
    #print('q=',q_value)
    if mode=='action':
        #print('\nparse with action, q=%s, action=%d'%(q_value,action))
        return q_value[action]
    elif mode=='max':
        return q_value.max(dim=0).values
    elif mode=='argmax':
        #print('\nq=',q_value)
        return q_value.argmax()
    elif mode=='softmax':
        return F.softmax(q_value)
    elif mode=='sum':
        return q_value.sum()
    else:
        raise RuntimeError('invalid mode in dqn.data_util.parse_output()')

class ReplayGraphDataset(InMemoryDataset):
    def __init__(self, root, states, edge_nums, node_nums, edge_indexes, edge_attrs):
        self.states = states
        self.edge_nums = edge_nums
        self.node_nums = node_nums
        self.edge_indexes = edge_indexes
        self.edge_attrs = edge_attrs
        super(ReplayGraphDataset, self).__init__(root)
        self.data = self.my_data
        self.slices = self.my_slices

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.inputs_to_graph_inputs()
        #data_list = data_list[:2]
        #print(data_list)
        self.my_data, self.my_slices = self.collate(data_list)
        #print(self.slices)
        #print(self.__len__())
        #raise RuntimeError('dbg')

    def inputs_to_graph_inputs(self):
        data_list = []
        for edge_num, node_num,\
            state, edge_index, edge_attr in zip(self.edge_nums.unsqueeze(-1), self.node_nums.unsqueeze(-1),
                                                self.states, self.edge_indexes, self.edge_attrs):
            # states: array of tensor with shape [M_i]
            graph_data = input_to_graph_input(edge_num, node_num, state, edge_index, edge_attr)
            data_list.append(graph_data)
        return data_list