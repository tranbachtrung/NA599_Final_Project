import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import networkx as nx
import copy

import torch
from torch_geometric.data import Data, Dataset
import os.path as osp
from torch_geometric.loader import DataLoader
import plotly.express as px


def procure_dataset(params):
    # Parsing the parameters ------------------------------------------------------------------------
    data_name = params['data_name'] #'METR-LA' or 'PEMS-BAY' # The data set used
    n_nodes = params['n_nodes'] # 207 for full ('METR-LA'); ### for full ('PEMS-BAY') # Number of nodes used for GNN
    data_ratio = params['data_ratio'] # the ratio of total data used for training, validation, and testing
    kappa = params['kappa'] # cutoff tresh-hold for node connection
    train_ratio = params['train_ratio'] # ratio of the data for training
    val_ratio = params['val_ratio'] # ratio of the data for validation
    test_ratio = params['test_ratio'] # ratio of the data for testing
    T_past = params['T_past'] # number of time-steps for node-level features
    T_future = params['T_future'] # number of time-steps for node-level outputs
    include_t = params['include_t'] # include time of the day (normalized)

    # Get parent folder path ------------------------------------------------------------------------
    current_folder_path = os.getcwd() # Get the current folder path
    parent_folder_path = os.path.dirname(current_folder_path) # Get the parent folder path

    # Load the data ------------------------------------------------------------------------
    if data_name == 'METR-LA':
        h5_filename = 'metr-la.h5'
        dist_filename = 'distances_la_2012.csv'
        locations_filename = 'graph_sensor_locations.csv'
        adj_mx_filename = 'adj_mx.pkl'

    elif data_name == 'PEMS-BAY':
        h5_filename = 'pems-bay.h5'
        dist_filename = 'distances_bay_2017.csv'
        locations_filename = 'graph_sensor_locations_bay.csv'
        adj_mx_filename = 'adj_mx_bay.pkl'

    h5_file_path = os.path.join(parent_folder_path, 'data/'+h5_filename)
    dist_file_path = os.path.join(parent_folder_path, 'data/sensor_graph/'+dist_filename)
    locations_file_path = os.path.join(parent_folder_path, 'data/sensor_graph/'+locations_filename)
    adj_mx_file_path = os.path.join(parent_folder_path, 'data/sensor_graph/'+adj_mx_filename)

    # Open the HDF5 file and read it as a pandas DataFrame 
    h5_df = pd.read_hdf(h5_file_path)
    dist_df = pd.read_csv(dist_file_path)
    locations_df = pd.read_csv(locations_file_path)
    with open(adj_mx_file_path, 'rb') as f:
        adj_mx = pickle.load(f, encoding='latin1')

    # Obtain ordered_sensors by reordering adj_mx[1] by its values ------------------------------------------------------------------------
    ordered_sensors = dict(sorted(adj_mx[1].items(), key=lambda item: item[1]))
    ordered_sensors = pd.DataFrame(list(ordered_sensors.items()), columns=['Sensor_ID', 'Value']).astype('int64')

    # Data formating and data cleaning ------------------------------------------------------------------------
    # Data formating ---
    # Rename the columns of h5_df using the 'Value' from adj_mx[1]
    h5_df.columns = [adj_mx[1][col] for col in h5_df.columns]

    # Only choose a fraction of the data (good for debugging)
    h5_df = h5_df[h5_df.columns[0:n_nodes]]
    h5_df = h5_df[0:round(data_ratio*len(h5_df))]

    # Data cleaning ---
    # There is something wrong with the index...
    # Convert the index to string format
    index_as_string = h5_df.index.astype(str)

    # Convert the string-formatted index to datetime
    index_as_datetime = pd.to_datetime(index_as_string)

    # Reindex the DataFrame using the datetime-formatted index
    h5_df = h5_df.reindex(index_as_datetime)

    # Create a directed graph based on the given distance matrix ... ------------------------------------------------------------------------
    # ... this resulted in temporary sensors...
    temp_df = dist_df.pivot(index='from', columns='to', values='cost').fillna(0)
    temp_mat = temp_df.to_numpy()
    G = nx.from_numpy_array(temp_mat, create_using=nx.DiGraph)

    temp_sensor_list = np.array(temp_df.index.tolist())
    temp_sensor_value = np.arange(0, len(temp_sensor_list))
    temp_ordered_sensors = pd.DataFrame({'Sensor_ID': temp_sensor_list, 'Value': temp_sensor_value})

    # Extract the sensor IDs and their corresponding indices ------------------------------------------------------------------------
    num_sensors = len(ordered_sensors)

    # Initialize the distance matrix with infinity values
    distance_matrix = np.full((num_sensors, num_sensors), np.inf)

    # Fill the distance matrix with the actual distances
    for i in range(num_sensors):
        for j in range(num_sensors):
            source_id = int(ordered_sensors[ordered_sensors['Value']==i]['Sensor_ID'].to_numpy()[0])
            target_id = int(ordered_sensors[ordered_sensors['Value']==j]['Sensor_ID'].to_numpy()[0])

            source = int(temp_ordered_sensors[temp_ordered_sensors['Sensor_ID']==int(source_id)]['Value'].to_numpy()[0])
            target = int(temp_ordered_sensors[temp_ordered_sensors['Sensor_ID']==int(target_id)]['Value'].to_numpy()[0])

            shortest_distance = nx.shortest_path_length(G, source=source, target=target, weight='weight')
            distance_matrix[i, j] = shortest_distance

            if j==n_nodes:
                break
        if i==n_nodes:
            break

    distance_matrix = distance_matrix[0:n_nodes, 0:n_nodes] # Reduce data

    # Normalized (transform) the weight to be between [0, 1]
    std = np.std(distance_matrix)
    weighted_distance_matrix = np.exp(-(distance_matrix/std)**2)
    weighted_distance_matrix[weighted_distance_matrix<=kappa]=0

    # Obtain connectivity matrix
    connectivity_matrix = copy.deepcopy(weighted_distance_matrix)
    connectivity_matrix[connectivity_matrix>0] = 1

    # Obtain list of connectivity and weighted distance
    connectivity_list = []
    weight_distance_list = []

    for i in range(connectivity_matrix.shape[0]):
        for j in range(connectivity_matrix.shape[1]):
            if connectivity_matrix[i, j] != 0:
                connectivity_list.append((i, j))
                weight_distance_list.append([weighted_distance_matrix[i, j]])

    connectivity_list = torch.tensor(connectivity_list, dtype=torch.long).transpose(0, 1)
    weight_distance_list = torch.tensor(weight_distance_list, dtype=torch.float)

    # Visualization ------------------------------------------------------------------------
    # Visualize lat-lon on map
    locations_df = locations_df.merge(ordered_sensors, left_on='sensor_id', right_on='Sensor_ID')
    temp_location_df = locations_df[locations_df['Sensor_ID'].isin(ordered_sensors[0:n_nodes]['Sensor_ID'].to_list())]

    fig = px.scatter_mapbox(temp_location_df, 
                            lat="latitude", 
                            lon="longitude", 
                            hover_name="Value",
                            color_discrete_sequence=["blue"], 
                            zoom=10, 
                            height=600)

    fig.update_traces(marker=dict(size=7), 
                    textposition='top right',
                    text=temp_location_df['Value'])

    fig.update_layout(mapbox_style="carto-positron")

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

    # Visualize distance matrix 
    plt.figure(figsize=(10, 8))
    plt.imshow(weighted_distance_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Weight')
    plt.title('Weighted Distance Matrix')
    plt.xlabel('Sensor Index')
    plt.ylabel('Sensor Index')
    plt.show()

    # Data partition ------------------------------------------------------------------------
    # Create train, val, and test DataFrames from h5_df 
    train_len = round(train_ratio*len(h5_df))
    val_len = round(val_ratio*len(h5_df))
    test_len = len(h5_df) - train_len - val_len

    train_df = h5_df[0:train_len]
    val_df = h5_df[train_len:train_len+val_len]
    test_df = h5_df[train_len+val_len:]


    for i in range(len(T_past)):
        for j in range(len(T_future)):
            # Create graph list
            train_graphs = create_graph_list(train_df, T_past[i], T_future[j], connectivity_list, weight_distance_list, include_t=include_t)
            val_graphs = create_graph_list(val_df, T_past[i], T_future[j], connectivity_list, weight_distance_list, include_t=include_t)
            test_graphs = create_graph_list(test_df, T_past[i], T_future[j], connectivity_list, weight_distance_list, include_t=include_t)

            # Create the dataset
            train_dataset = CustomGraphDataset(
                graphs=train_graphs,
                root=parent_folder_path+'/data/' + data_name + f'/include_t_{include_t}_T_past_{T_past[i]}_T_future_{T_future[j]}/train' # Where to save processed files
            )

            val_dataset = CustomGraphDataset(
                graphs=val_graphs,
                root=parent_folder_path+'/data/' + data_name + f'/include_t_{include_t}_T_past_{T_past[i]}_T_future_{T_future[j]}/val' # Where to save processed files
            )

            test_dataset = CustomGraphDataset(
                graphs=test_graphs,
                root=parent_folder_path+'/data/' + data_name + f'/include_t_{include_t}_T_past_{T_past[i]}_T_future_{T_future[j]}/test' # Where to save processed files
            )
    return

# Custom dataset class
class CustomGraphDataset(Dataset):
    def __init__(self, graphs, root, transform=None, pre_transform=None):
        """
        Args:
            graphs: List of tuples (x, edge_index, edge_attr, y) or torch_geometric.data.Data objects
            root: Root directory where the dataset should be saved
            transform: Optional transform to be applied on each graph
            pre_transform: Optional transform to be applied on each graph before saving
        """
        self.graphs = graphs
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return []  # No raw files needed
    
    @property
    def processed_file_names(self):
        return [f'data_{idx}.pt' for idx in range(len(self.graphs))]
    
    def process(self):
        idx = 0
        for graph_data in self.graphs:
            # If the input is a tuple, convert to Data object
            if isinstance(graph_data, tuple):
                x, edge_index, edge_attr, y = graph_data
                data = Data(x=x, 
                          edge_index=edge_index,
                          edge_attr=edge_attr,
                          y=y)
            else:
                data = graph_data
                
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1
    
    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    

# Support Functions
def datetime_to_torch(input):
    return torch.tensor(input.astype('int64') // 10**9, dtype=torch.long)

def torch_to_datetime(input):
    return pd.to_datetime(input.numpy() * 10**9)

def create_graph_list(data_df, T_past, T_future, connectivity_list, weight_distance_list, include_t = False):
    T = T_past+T_future
    graph_list = []

    for i in range(0, len(data_df)-T+1):
        temp = data_df[i:i+T]
        
        x = temp[0:T_past]
        x_index = datetime_to_torch(x.index)
        temp_index = temp.index.map(lambda x: x.hour / 24 + x.minute / 1440)
        t = temp_index[-1]
        x = torch.tensor(x.to_numpy(), dtype=torch.float).transpose(0, 1)

        if include_t is True:
            t_tensor = torch.tensor([t] * x.shape[0], dtype=torch.float).unsqueeze(0).transpose(0, 1)
            x = torch.cat([t_tensor, x], dim=1) 

        y = temp[T_past:T_past+T_future]
        y_index = datetime_to_torch(y.index)
        y = torch.tensor(y.to_numpy(), dtype=torch.float).transpose(0, 1)

        data = Data(x=x,
                    x_index = x_index, 
                    edge_index=connectivity_list, 
                    edge_attr=weight_distance_list,
                    y=y,
                    y_index = y_index)
        
        graph_list.append(data)
    return graph_list