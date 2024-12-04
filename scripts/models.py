import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import networkx as nx
import copy

import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
import os.path as osp
from torch_geometric.loader import DataLoader

from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch.nn import ModuleList
import glob
import os

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=4, include_t=False):
        super(GNN, self).__init__()

        self.include_t = include_t

        # Create a list of GCN layers
        self.convs = ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=True))

        # Hidden layers
        for _ in range(num_layers - 3):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=True))

        # Aggregating the input and outputs of all previous layers
        temp_num_channel = hidden_channels*(num_layers - 2) + in_channels # Make sure that this is correct size...
        self.convs.append(GCNConv(temp_num_channel, temp_num_channel, normalize=True))
        
        # Final output layer
        self.convs.append(GCNConv(temp_num_channel, 1, normalize=True)) # 1 -- only one-time step prediction

    def forward(self, data):
        edge_index = data.edge_index
        edge_weight = data.edge_attr
        T_past = data.x.shape[1]
        T_future = data.y.shape[1]

        x_init = data.x
        y_tm = data.x[:, -1:] # Get the last segment that is closest to the output
        y_pred = []

        for i in range(T_future):
            layer_outputs = []
            layer_outputs.append(x_init)
            
            # Initial convolution
            x = self.convs[0](x_init, edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x)
            layer_outputs.append(x)

            # Subsequent convolutions
            for conv in self.convs[1:-2]:  # Apply ReLU to all but the last two layers
                x = conv(x, edge_index, edge_weight=edge_weight)
                x = F.leaky_relu(x)
                layer_outputs.append(x)

            # Second to last layer
            temp = torch.cat(layer_outputs, dim=1)
            x_temp = self.convs[-2](temp, edge_index, edge_weight=edge_weight)
            x_temp = F.leaky_relu(x_temp)

            # Final layer
            #y_pred_temp = y_tm + self.convs[-1](x_temp, edge_index, edge_weight=edge_weight)
            y_pred_temp = self.convs[-1](x_temp, edge_index, edge_weight=edge_weight)
            y_pred.append(y_pred_temp)

            # Shifting the input and concatenating the y_pred_temp to restart the prediction
            if self.include_t is False:
                if T_past == 1:
                    x_init = y_pred_temp
                else:
                    x_init = torch.cat([x_init[:, 1:], y_pred_temp], dim=1)
                y_tm = y_pred_temp
            else:
                t = (x_init[:, [0]] + (1.0 / 288)) % 1.0  # This will ensure that the values of t stay in [0, 1)
                if T_past == 1:
                    x_init = torch.cat([t, y_pred_temp], dim=1)
                else:
                    x_init = torch.cat([t, x_init[:, 2:], y_pred_temp], dim=1)
                y_tm = y_pred_temp


        y_pred = torch.cat(y_pred, dim=1)
        return y_pred
    
class GNN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=4, include_t = False):
        super(GNN2, self).__init__()

        self.include_t = include_t

        # Create a list of GCN layers
        self.convs = ModuleList()

        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 3):
            self.convs.append(GATConv(hidden_channels, hidden_channels))

        # Aggregating the input and outputs of all previous layers
        temp_num_channel = hidden_channels*(num_layers - 2) + in_channels # Make sure that this is correct size...

        #self.convs.append(GATConv(temp_num_channel, temp_num_channel))
        self.convs.append(GCNConv(temp_num_channel, temp_num_channel, normalize=True))
        
        # Final output layer
        self.convs.append(GATConv(temp_num_channel, 1)) # 1 -- only one-time step prediction


    def forward(self, data):
        edge_index = data.edge_index
        edge_weight = data.edge_attr
        T_past = data.x.shape[1]
        T_future = data.y.shape[1]

        x_init = data.x
        #y_tm = data.x[:, -1:] # Get the last segment that is closest to the output
        # moving average
        if self.include_t is False:
            y_tm = x_init.mean(dim=1, keepdim=True)
        else:
            y_tm = x_init[:, 1:].mean(dim=1, keepdim=True)

        y_pred = []

        for i in range(T_future):
            layer_outputs = []
            layer_outputs.append(x_init)
            
            # Initial convolution
            x = self.convs[0](x_init, edge_index, edge_attr = edge_weight)
            x = F.leaky_relu(x)
            layer_outputs.append(x)

            # Subsequent convolutions
            for conv in self.convs[1:-2]:  # Apply ReLU to all but the last two layers
                x = conv(x, edge_index, edge_attr = edge_weight)
                x = F.leaky_relu(x)
                layer_outputs.append(x)

            # Second to last layer
            temp = torch.cat(layer_outputs, dim=1)
            # x_temp = self.convs[-2](temp, edge_index, edge_attr = edge_weight)
            x_temp = self.convs[-2](temp, edge_index, edge_weight=edge_weight)
            x_temp = F.leaky_relu(x_temp)

            # Final layer
            y_pred_temp = y_tm + self.convs[-1](x_temp, edge_index, edge_attr = edge_weight)
            #y_pred_temp = self.convs[-1](x_temp, edge_index, edge_attr = edge_weight)
            y_pred.append(y_pred_temp)

            # Shifting the input and concatenating the y_pred_temp to restart the prediction
            if self.include_t is False:
                if T_past == 1:
                    x_init = y_pred_temp
                else:
                    x_init = torch.cat([x_init[:, 1:], y_pred_temp], dim=1)    
                #y_tm = y_pred_temp
                y_tm = x_init.mean(dim=1, keepdim=True)
            else:
                t = (x_init[:, [0]] + (1.0 / 288)) % 1.0  # This will ensure that the values of t stay in [0, 1)
                if T_past == 1:
                    x_init = torch.cat([t, y_pred_temp], dim=1)
                else:
                    x_init = torch.cat([t, x_init[:, 2:], y_pred_temp], dim=1)
                #y_tm = y_pred_temp
                y_tm = x_init[:, 1:].mean(dim=1, keepdim=True)


        y_pred = torch.cat(y_pred, dim=1)
        y_pred = torch.clamp(y_pred, min=0, max=90)
        return y_pred

class GNN3(torch.nn.Module):
    # Calculate using changes of traffic speed

    def __init__(self, in_channels, hidden_channels, num_layers=4, include_t=False):
        super(GNN3, self).__init__()

        self.include_t = include_t

        # Create a list of GCN layers
        self.convs = ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels-1, hidden_channels, normalize=True))

        # Hidden layers
        for _ in range(num_layers - 3):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=True))

        # Aggregating the input and outputs of all previous layers
        temp_num_channel = hidden_channels*(num_layers - 2) + in_channels-1 # Make sure that this is correct size...
        self.convs.append(GCNConv(temp_num_channel, temp_num_channel, normalize=True))
        
        # Final output layer
        self.convs.append(GCNConv(temp_num_channel, 1, normalize=True)) # 1 -- only one-time step prediction

    def forward(self, data):
        edge_index = data.edge_index
        edge_weight = data.edge_attr
        T_past = data.x.shape[1]
        T_future = data.y.shape[1]

        x_init = data.x
        y_tm = data.x[:, -1:] # Get the last segment that is closest to the output
        y_pred = []

        for i in range(T_future):

            if self.include_t is True:
                t = x_init[:, [0]]
                x_init_mod = torch.cat([t, x_init[:,2:] - x_init[:,1:-1]], dim=1)
            else:
                x_init_mod = x_init[:,1:] - x_init[:,0:-1]

            layer_outputs = []
            layer_outputs.append(x_init_mod)
            
            # Initial convolution
            x = self.convs[0](x_init_mod, edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x)
            layer_outputs.append(x)

            # Subsequent convolutions
            for conv in self.convs[1:-2]:  # Apply ReLU to all but the last two layers
                x = conv(x, edge_index, edge_weight=edge_weight)
                x = F.leaky_relu(x)
                layer_outputs.append(x)

            # Second to last layer
            temp = torch.cat(layer_outputs, dim=1)
            x_temp = self.convs[-2](temp, edge_index, edge_weight=edge_weight)
            x_temp = F.leaky_relu(x_temp)

            # Final layer
            y_pred_temp = y_tm + self.convs[-1](x_temp, edge_index, edge_weight=edge_weight)
            #y_pred_temp = self.convs[-1](x_temp, edge_index, edge_weight=edge_weight)
            y_pred.append(y_pred_temp)

            # Shifting the input and concatenating the y_pred_temp to restart the prediction
            if self.include_t is False:
                if T_past == 1:
                    x_init = y_pred_temp
                else:
                    x_init = torch.cat([x_init[:, 1:], y_pred_temp], dim=1)
                y_tm = y_pred_temp
            else:
                t = (x_init[:, [0]] + (1.0 / 288)) % 1.0  # This will ensure that the values of t stay in [0, 1)
                if T_past == 1:
                    x_init = torch.cat([t, y_pred_temp], dim=1)
                else:
                    x_init = torch.cat([t, x_init[:, 2:], y_pred_temp], dim=1)
                y_tm = y_pred_temp

        y_pred = torch.cat(y_pred, dim=1)
        return y_pred
    
class GNN4(torch.nn.Module):
    # Calculate using changes of traffic speed

    def __init__(self, in_channels, hidden_channels, num_layers=4, include_t=False):
        super(GNN4, self).__init__()

        self.include_t = include_t

        # Create a list of GCN layers
        self.convs = ModuleList()

        # Input layer
        self.convs.append(GATConv(in_channels-1, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 3):
            self.convs.append(GATConv(hidden_channels, hidden_channels))

        # Aggregating the input and outputs of all previous layers
        temp_num_channel = hidden_channels*(num_layers - 2) + in_channels-1 # Make sure that this is correct size...
        self.convs.append(GATConv(temp_num_channel, temp_num_channel))
        
        # Final output layer
        self.convs.append(GATConv(temp_num_channel, 1)) # 1 -- only one-time step prediction

    def forward(self, data):
        edge_index = data.edge_index
        edge_weight = data.edge_attr
        T_past = data.x.shape[1]
        T_future = data.y.shape[1]

        x_init = data.x
        y_tm = data.x[:, -1:] # Get the last segment that is closest to the output
        y_pred = []

        for i in range(T_future):

            if self.include_t is True:
                t = x_init[:, [0]]
                x_init_mod = torch.cat([t, x_init[:,2:] - x_init[:,1:-1]], dim=1)
            else:
                x_init_mod = x_init[:,1:] - x_init[:,0:-1]

            layer_outputs = []
            layer_outputs.append(x_init_mod)
            
            # Initial convolution
            x = self.convs[0](x_init_mod, edge_index, edge_attr = edge_weight)
            x = F.leaky_relu(x)
            layer_outputs.append(x)

            # Subsequent convolutions
            for conv in self.convs[1:-2]:  # Apply ReLU to all but the last two layers
                x = conv(x, edge_index, edge_attr = edge_weight)
                x = F.leaky_relu(x)
                layer_outputs.append(x)

            # Second to last layer
            temp = torch.cat(layer_outputs, dim=1)
            x_temp = self.convs[-2](temp, edge_index, edge_attr = edge_weight)
            x_temp = F.leaky_relu(x_temp)

            # Final layer
            y_pred_temp = y_tm + self.convs[-1](x_temp, edge_index, edge_attr = edge_weight)
            #y_pred_temp = self.convs[-1](x_temp, edge_index, edge_weight=edge_weight)
            y_pred.append(y_pred_temp)

            # Shifting the input and concatenating the y_pred_temp to restart the prediction
            if self.include_t is False:
                if T_past == 1:
                    x_init = y_pred_temp
                else:
                    x_init = torch.cat([x_init[:, 1:], y_pred_temp], dim=1)
                y_tm = y_pred_temp
            else:
                t = (x_init[:, [0]] + (1.0 / 288)) % 1.0  # This will ensure that the values of t stay in [0, 1)
                if T_past == 1:
                    x_init = torch.cat([t, y_pred_temp], dim=1)
                else:
                    x_init = torch.cat([t, x_init[:, 2:], y_pred_temp], dim=1)
                y_tm = y_pred_temp

        y_pred = torch.cat(y_pred, dim=1)
        return y_pred

class GNN5(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=4, include_t = False):
        super(GNN5, self).__init__()

        self.include_t = include_t

        # Create a list of GCN layers
        self.convs = ModuleList()

        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 3):
            self.convs.append(GATConv(hidden_channels, hidden_channels))

        # Aggregating the input and outputs of all previous layers
        temp_num_channel = hidden_channels*(num_layers - 2) + in_channels # Make sure that this is correct size...

        #self.convs.append(GATConv(temp_num_channel, temp_num_channel))
        self.convs.append(GCNConv(temp_num_channel, temp_num_channel, normalize=True))
        
        # Final output layer
        self.convs.append(GATConv(temp_num_channel, 1)) # 1 -- only one-time step prediction


    def forward(self, data):
        edge_index = data.edge_index
        edge_weight = data.edge_attr
        T_past = data.x.shape[1]
        T_future = data.y.shape[1]

        x_init = data.x
        #y_tm = data.x[:, -1:] # Get the last segment that is closest to the output
        # moving average
        if self.include_t is False:
            y_tm = x_init.mean(dim=1, keepdim=True)
        else:
            y_tm = x_init[:, 1:].mean(dim=1, keepdim=True)

        y_pred = []

        for i in range(T_future):
            layer_outputs = []
            layer_outputs.append(x_init)
            
            # Initial convolution
            x = self.convs[0](x_init, edge_index, edge_attr = edge_weight)
            #x = F.leaky_relu(x)
            layer_outputs.append(x)

            # Subsequent convolutions
            for conv in self.convs[1:-2]:  # Apply ReLU to all but the last two layers
                x = conv(x, edge_index, edge_attr = edge_weight)
                #x = F.leaky_relu(x)
                layer_outputs.append(x)

            # Second to last layer
            temp = torch.cat(layer_outputs, dim=1)
            # x_temp = self.convs[-2](temp, edge_index, edge_attr = edge_weight)
            x_temp = self.convs[-2](temp, edge_index, edge_weight=edge_weight)
            #x_temp = F.leaky_relu(x_temp)

            # Final layer
            y_pred_temp = self.convs[-1](x_temp, edge_index, edge_attr = edge_weight)
            #y_pred_temp = self.convs[-1](x_temp, edge_index, edge_attr = edge_weight)
            y_pred_temp = torch.clamp(y_pred_temp, min=0, max=90)

            y_pred.append(y_pred_temp)

            # Shifting the input and concatenating the y_pred_temp to restart the prediction
            if self.include_t is False:
                if T_past == 1:
                    x_init = y_pred_temp
                else:
                    x_init = torch.cat([x_init[:, 1:], y_pred_temp], dim=1)    
                #y_tm = y_pred_temp
                y_tm = x_init.mean(dim=1, keepdim=True)
            else:
                t = (x_init[:, [0]] + (1.0 / 288)) % 1.0  # This will ensure that the values of t stay in [0, 1)
                if T_past == 1:
                    x_init = torch.cat([t, y_pred_temp], dim=1)
                else:
                    x_init = torch.cat([t, x_init[:, 2:], y_pred_temp], dim=1)
                #y_tm = y_pred_temp
                y_tm = x_init[:, 1:].mean(dim=1, keepdim=True)


        y_pred = torch.cat(y_pred, dim=1)
        return y_pred

class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.file_list = [f for f in os.listdir(root) if f.startswith('data_') and f.endswith('.pt')]
        self.file_list.sort(key=lambda f: int(f.split('_')[1].split('.')[0])) # sort by increasing order

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        file_path = os.path.join(self.root, self.file_list[idx])
        data = torch.load(file_path, weights_only=False)
        return data
    

def train_val_test(params):

    # Parsing params
    data_name = params['data_name']
    T_past = params['T_past']
    T_future = params['T_future']
    include_t = params['include_t']

    preload = params['preload']
    preload_folder = params['preload_folder']
    preload_model_name = params['preload_model_name']

    GNN_class = params['GNN_class']
    hidden_channels = params['hidden_channels']
    num_layers = params['num_layers']
    patience = params['patience']
    overfit_ratio = params['overfit_ratio']

    lr = params['lr']
    scheduler_mode = params['scheduler_mode']
    scheduler_factor = params['scheduler_factor']
    scheduler_patience = params['scheduler_patience']
    max_epoch = params['max_epoch']

    # Setting up the model
    if GNN_class == "GNN":
        if include_t is True:
            model = GNN(in_channels = T_past + 1, 
                    hidden_channels = hidden_channels,
                    num_layers = num_layers,
                    include_t = include_t)
        else:
            model = GNN(in_channels = T_past, 
                    hidden_channels = hidden_channels,
                    num_layers = num_layers,
                    include_t = include_t)
            
        signal_extract = None
            
    elif GNN_class == "GNN2":
        if include_t is True:
            model = GNN2(in_channels = T_past + 1, 
                    hidden_channels = hidden_channels,
                    num_layers = num_layers,
                    include_t = include_t)
        else:
            model = GNN2(in_channels = T_past, 
                    hidden_channels = hidden_channels,
                    num_layers = num_layers,
                    include_t = include_t)
            
        signal_extract = None
            
    elif GNN_class == "GNN3":
        if include_t is True:
            model = GNN3(in_channels = T_past + 1, 
                    hidden_channels = hidden_channels,
                    num_layers = num_layers,
                    include_t = include_t)
        else:
            model = GNN3(in_channels = T_past, 
                    hidden_channels = hidden_channels,
                    num_layers = num_layers,
                    include_t = include_t)
            
        signal_extract = traffic_speed_differences
            
    elif GNN_class == "GNN4":
        if include_t is True:
            model = GNN4(in_channels = T_past + 1, 
                    hidden_channels = hidden_channels,
                    num_layers = num_layers,
                    include_t = include_t)
        else:
            model = GNN4(in_channels = T_past, 
                    hidden_channels = hidden_channels,
                    num_layers = num_layers,
                    include_t = include_t)
            
        signal_extract = traffic_speed_differences

    elif GNN_class == "GNN5":
        if include_t is True:
            model = GNN5(in_channels = T_past + 1, 
                    hidden_channels = hidden_channels,
                    num_layers = num_layers,
                    include_t = include_t)
        else:
            model = GNN5(in_channels = T_past, 
                    hidden_channels = hidden_channels,
                    num_layers = num_layers,
                    include_t = include_t)
            
        signal_extract = None
        
        
    criterion_1 = lambda y_pred, y_true: mse_loss(y_pred, y_true, signal_extract=signal_extract)
    criterion_2 = lambda y_pred, y_true: mse_loss(y_pred, y_true, signal_extract=None)

    # Get parent folder path
    current_folder_path = os.getcwd() # Get the current folder path
    parent_folder_path = os.path.dirname(current_folder_path) # Get the parent folder path

    # Load the parameters
    if preload is True:
        preload_model_path = parent_folder_path + f'/models/{data_name}/{preload_folder}/{preload_model_name}' 
        model.load_state_dict(torch.load(preload_model_path, weights_only=False))

    for j in range(len(T_future)):
        print('============================================================================================================================')
        print(f'Training with T_past={T_past} and T_future={T_future[j]}!')

        # Obtain data path
        train_path, train_dataset, train_loader, train_loader_viz = obtain_data_path(data_name, T_past, T_future[j], include_t, name='train')
        val_path, val_dataset, val_loader, val_loader_viz = obtain_data_path(data_name, T_past, T_future[j], include_t, name='val')
        test_path, test_dataset, test_loader, test_loader_viz = obtain_data_path(data_name, T_past, T_future[j], include_t, name='test')

        # Model save path
        model_save_path = parent_folder_path + f'/models/{data_name}' +  f'/include_t_{include_t}_T_past_{T_past}_T_future_{T_future[j]}'

        # Setting up model
        n_nodes = train_dataset[0].x.shape[0]

        # Reset the optimizer and scheduler for each warm-starting cycle
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=scheduler_factor, patience=scheduler_patience)
       
        # Initialize values
        train_loss_1 = validate(model, train_loader, criterion_1)
        best_val_loss_1 = np.inf # to always save the 1st epoch

        train_loss_2 = validate(model, train_loader, criterion_2)
        best_val_loss_2 = np.inf # to always save the 1st epoch
        
        val_loss_1 = validate(model, val_loader, criterion_1) # For early stopping
        val_loss_2 = validate(model, val_loader, criterion_2) # For early stopping
        
        overfit = best_val_loss_1/(np.finfo(float).eps + train_loss_1)
        best_model_state = copy.deepcopy(model.state_dict())
        counter = 0 # Counter for early stopping patience

        print(f'Epoch: {000}, Train RMS Error: {np.sqrt(train_loss_2):.4f} [miles/hr], Val RMS Error: {np.sqrt(val_loss_2):.4f} [miles/hr], Train Speed Diff RMS Error: {np.sqrt(train_loss_1):.4f} [miles/hr], Val Speed Diff RMS Error: {np.sqrt(val_loss_1):.4f} [miles/hr], Learning Rate = {lr:.4f}, Overfit Ratio = {overfit:.4f}')

        # Train the model
        for epoch in range(1, max_epoch):
            train(model, train_loader, optimizer, criterion_1)

            train_loss_1 = validate(model, train_loader, criterion_1)
            train_loss_2 = validate(model, train_loader, criterion_2)

            val_loss_1 = validate(model, val_loader, criterion_1)
            val_loss_2 = validate(model, val_loader, criterion_2)

            current_lr = optimizer.param_groups[0]['lr']
            overfit = val_loss_1/(np.finfo(float).eps + train_loss_1)

            # The print out is normalized by the number of future samples
            print(f'Epoch: {epoch:03d}, Train RMS Error: {np.sqrt(train_loss_2):.4f} [miles/hr], Val RMS Error: {np.sqrt(val_loss_2):.4f} [miles/hr], Train Speed Diff RMS Error: {np.sqrt(train_loss_1):.4f} [miles/hr], Val Speed Diff RMS Error: {np.sqrt(val_loss_1):.4f} [miles/hr], Learning Rate = {current_lr:.4f}, Overfit Ratio = {overfit:.4f}')
            

            # Check if (the current validation loss is the best) and (the model is not overfitting)
            if (val_loss_1 < best_val_loss_1) and (overfit_ratio[0]<=overfit) and (overfit <= overfit_ratio[1]):
                
                best_val_loss_1 = val_loss_1
                best_val_loss_2 = val_loss_2

                best_model_state = copy.deepcopy(model.state_dict())
                counter = 0  # Reset counter when improvement occurs
                print("(Validation loss decreased) and (not overfitting), saving the model...")
                
                # Saving the best model state
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                temp_path = model_save_path + f'/epoch_{epoch}.pt'
                torch.save(best_model_state, temp_path)
            else:
                counter += 1
                print(f"(No improvement in validation loss) or (overfitting, i.e., ratio={overfit_ratio}) for {counter} epoch(s).")
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

            # Step the scheduler
            scheduler.step(val_loss_1)

            # If the learning rate was reduced, load the best model state
            if optimizer.param_groups[0]['lr'] < current_lr:
                print("Learning rate reduced, loading the best model state...")
                model.load_state_dict(best_model_state)

        # After training, load the best model state for the next time horizon training
        model.load_state_dict(best_model_state)
        print('Loaded the best performing model!')
        
    return

# Obtain data path
def obtain_data_path(data_name, T_past, T_future, include_t, name='train'):
    # Get parent folder path
    current_folder_path = os.getcwd() # Get the current folder path
    parent_folder_path = os.path.dirname(current_folder_path) # Get the parent folder path

    # Obtaining data path
    path = parent_folder_path+'/data/' + data_name + f'/include_t_{include_t}_T_past_{T_past}_T_future_{T_future}/{name}/processed/'
    dataset = GraphDataset(root=path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    loader_viz = DataLoader(dataset, batch_size=144, shuffle=False) # time-ordered data (144 is one day)

    return path, dataset, loader, loader_viz


# Training method for regression
def train(model, train_loader, optimizer, criterion):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

    return

# Validation method for regression
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = len(data)*criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(loader.dataset)

# Criterion
def mae_loss(y_pred, y_true, signal_extract = None):
    if signal_extract is None:
        return torch.mean(torch.abs(y_pred - y_true))
    else:
        y_pred_extract = signal_extract(y_pred)
        y_true_extract = signal_extract(y_true)
        return torch.mean(torch.abs(y_pred_extract - y_true_extract))

def mape_loss(y_pred, y_true, signal_extract = None, eps=1e-8):
    if signal_extract is None:
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + eps))) * 100
    else:
        y_pred_extract = signal_extract(y_pred)
        y_true_extract = signal_extract(y_true)
        return torch.mean(torch.abs((y_true_extract - y_pred_extract) / (y_true_extract + eps))) * 100

def mse_loss(y_pred, y_true, signal_extract = None):
    if signal_extract is None:
        return torch.mean((y_pred - y_true)**2)
    else:
        y_pred_extract = signal_extract(y_pred)
        y_true_extract = signal_extract(y_true)
        return torch.mean((y_pred_extract - y_true_extract)**2)

# Signal extraction
def traffic_speed_differences(traffic_speed_mat):
    output_mat = traffic_speed_mat[:, 1:] - traffic_speed_mat[:, 0:-1]
    return output_mat

def traffic_speed(traffic_speed_mat):
    return traffic_speed_mat
