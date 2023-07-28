import torch
import torch.nn
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GCNConv , MessagePassing
from torch_geometric.data import Data , HeteroData
from torch_geometric.datasets import CoraFull
from torch_geometric.loader import DataLoader , NeighborLoader
from torch_geometric.utils import train_test_split_edges , degree
from sklearn.model_selection import train_test_split
import time



#encoder with random augmentation
class CCA_homo(torch.nn.Module):
    def __init__(this,channel_lst):
        super().__init__()
        this.GCN_module=torch.nn.ModuleList()
        for in_ch , out_ch in channel_lst:
            this.GCN_module.append(GCNConv(in_ch,out_ch))

    def augment(this,graph,pe,pf):
        output=Data()
        #dropping edges
        edge_mask=torch.full((graph.edge_index.shape[1],),1-pe,device=this.device)
        edge_mask=torch.bernoulli(edge_mask).bool()
        output.edge_index=torch.cat((graph.edge_index[0][edge_mask].unsqueeze(0),graph.edge_index[1][edge_mask].unsqueeze(0)),dim=0)
        #masking node features
        embd_mask=torch.full((graph.x.shape[0],graph.x.shape[1]),1-pf,device=this.device)
        embd_mask=torch.bernoulli(embd_mask)
        output.x=graph.x*embd_mask
        return output

    def forward(this, graph,aug_pe1=0.5,aug_pf1=0.5,aug_pe2=0.5,aug_pf2=0.5):
        this.device = next(this.parameters()).device
        if(not this.training):
            for conv in this.GCN_module:
                graph.x=conv(graph.x,graph.edge_index)
            return (graph.x-graph.x.mean(0))/graph.x.std(0)

        #augmenting graph
        graph1 = this.augment(graph,aug_pe1,aug_pf1)
        graph2 = this.augment(graph,aug_pe2,aug_pf2)

        #getting embeddings
        for conv in this.GCN_module:
            graph1.x=conv(graph1.x,graph1.edge_index)
        for conv in this.GCN_module:
            graph2.x=conv(graph2.x,graph2.edge_index)
        #normalizing node embeddings
        z1_norm=((graph1.x-graph1.x.mean(0))/graph1.x.std(0))/torch.sqrt(torch.tensor(graph1.x.shape[0]))
        z2_norm=((graph2.x-graph2.x.mean(0))/graph2.x.std(0))/torch.sqrt(torch.tensor(graph2.x.shape[0]))

        return z1_norm , z2_norm


#encoder with adaptive augmentation with degree centrality
class CCA_GCA_aug_homo(torch.nn.Module):
    def __init__(this,channel_lst):
        super().__init__()
        this.GCN_module=torch.nn.ModuleList()
        for in_ch , out_ch in channel_lst:
            this.GCN_module.append(GCNConv(in_ch,out_ch))

    def gca_augmentation_degree(this,graph,pe,pte,pf,ptf,epsilon=10**(-40)):
        output=Data()
        #computing degree centrality
        deg_cent=degree(graph.edge_index[1],graph.x.shape[0])
        deg_cent=deg_cent/graph.x.shape[0]
        #computing edge probability
        edge_cent=(deg_cent[graph.edge_index[0]]+deg_cent[graph.edge_index[1]])/2
        edge_cent=torch.log(edge_cent)
        max_edge_cent=torch.max(edge_cent)
        mean_edge_cent=edge_cent.mean()
        edge_prob=((max_edge_cent-edge_cent)/(max_edge_cent-mean_edge_cent))*pe
        edge_prob=torch.min(edge_prob,torch.full((edge_prob.shape[0],),pte,device=this.device))
        edge_prob=1-edge_prob
        #computing node feature probability
        fet_weights=(torch.abs(graph.x.t()+epsilon)*deg_cent).sum(dim=1)
        fet_prob=torch.log(fet_weights)
        fet_max=torch.max(fet_prob)
        fet_mean=fet_prob.mean()
        fet_prob=((fet_max-fet_prob)/(fet_max-fet_mean))*pf
        fet_prob=torch.min(fet_prob,torch.full((fet_prob.shape[0],),ptf,device=this.device))
        fet_prob=1-fet_prob
        #generating edge mask
        edge_mask=torch.bernoulli(edge_prob).bool()
        output.edge_index=torch.cat((graph.edge_index[0][edge_mask].unsqueeze(0),graph.edge_index[1][edge_mask].unsqueeze(0)),dim=0)
        #generating feature mask
        temp=torch.empty(graph.x.shape,device=this.device)
        temp[:]=fet_prob
        fet_mask=torch.bernoulli(temp)
        output.x=graph.x*fet_mask
        return output

    def forward(this, graph,pe1=0.5,pte1=0.5,pf1=0.5,ptf1=0.5,pe2=0.5,pte2=0.5,pf2=0.5,ptf2=0.5):
        this.device = next(this.parameters()).device
        if(not this.training):
            for conv in this.GCN_module:
                graph.x=conv(graph.x,graph.edge_index)
            return (graph.x-graph.x.mean(0))/graph.x.std(0)

        #augmenting graph
        graph1 = this.gca_augmentation_degree(graph,pe1,pte1,pf1,ptf1)
        graph2 = this.gca_augmentation_degree(graph,pe2,pte2,pf2,ptf2)

        #getting embeddings
        for conv in this.GCN_module:
            graph1.x=conv(graph1.x,graph1.edge_index)
        for conv in this.GCN_module:
            graph2.x=conv(graph2.x,graph2.edge_index)
        #normalizing node embeddings
        z1_norm=((graph1.x-graph1.x.mean(0))/graph1.x.std(0))/torch.sqrt(torch.tensor(graph1.x.shape[0]))
        z2_norm=((graph2.x-graph2.x.mean(0))/graph2.x.std(0))/torch.sqrt(torch.tensor(graph2.x.shape[0]))

        return z1_norm , z2_norm

#cca loss(prefered)
def loss_cca2(z1,z2,device,param=10**(-3)):
    temp1=((z1-z2)**2).sum()
    var1=torch.mm(z1.t(),z1)
    var2=torch.mm(z2.t(),z2)
    return temp1+((((var1-torch.eye(var1.shape[0],device=device))**2).sum()+((var2-torch.eye(var2.shape[0],device=device))**2).sum())*param)



#modified cca loss 
def loss_cca(z1,z2,device,param=10**(-3)):
    co_var=torch.mm(z1.t(),z2)
    var1=torch.mm(z1.t(),z1)
    var2=torch.mm(z2.t(),z2)
    return ((((var1-torch.eye(var1.shape[0],device=device))**2).sum()+((var2-torch.eye(var2.shape[0],device=device))**2).sum())*param)-torch.trace(co_var)


#function to split the data
def train_split(graph,train_ratio,val_ratio,test_ratio):
    train_data, test_data = train_test_split(torch.arange(graph.num_nodes), test_size=test_ratio)
    train_data, val_data = train_test_split(train_data, test_size=val_ratio)

    num_nodes = graph.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_data] = True
    test_mask[test_data] = True
    val_mask[val_data] = True

    return train_mask , val_mask , test_mask

#function to train cca model
def train_cca_model(model,loader_train,loader_val,optzr,loss_func,device,num_epochs=100,loss_param=10**(-3),pe1=0.5,pf1=0.5,pe2=0.5,pf2=0.5,save_path=None,name="model"):
    model.train()
    for epoch in range(num_epochs):
        print("________________")
        print("epoch:",epoch)
        st=time.time()
        tloss=0
        for graph in loader_train:
            optzr.zero_grad()
            z1 , z2 = model(graph,pe1,pf1,pe2,pf2)
            loss=loss_func(z1,z2,device,loss_param)
            tloss+=loss
            loss.backward()
            optzr.step()
        #validation
        vloss=0
        for graph in loader_val:
            z1 , z2 = model(graph,pe1,pf1,pe2,pf2)
            loss=loss_func(z1,z2,device,loss_param)
            vloss+=loss
        #saving weights
        if(save_path!=None):
             torch.save(model.state_dict(),save_path+"/CCA_model_"+name+".pth")
        et=time.time()
        print("train loss:",tloss.item())
        print("validation loss:",vloss.item())
        print("time taken:",et-st," s")
        print("________________\n\n")

#function to train encoder with adaptive augmentation
def train_cca_model_gca_aug(model,loader_train,loader_val,optzr,loss_func,device,num_epochs=100,loss_param=10**(-3),pe1=0.5,pte1=0.5,pf1=0.5,ptf1=0.5,pe2=0.5,pte2=0.5,pf2=0.5,ptf2=0.5,save_path=None,name="model"):
    model.train()
    for epoch in range(num_epochs):
        print("________________")
        print("epoch:",epoch)
        st=time.time()
        tloss=0
        for graph in loader_train:
            optzr.zero_grad()
            z1 , z2 = model(graph,pe1,pte1,pf1,ptf1,pe2,pte2,pf2,ptf2)
            loss=loss_func(z1,z2,device,loss_param)
            tloss+=loss
            loss.backward()
            optzr.step()
        #validation
        vloss=0
        for graph in loader_val:
            z1 , z2 = model(graph,pe1,pte1,pf1,ptf1,pe2,pte2,pf2,ptf2)
            loss=loss_func(z1,z2,device,loss_param)
            vloss+=loss
        #saving weights
        if(save_path!=None):
             torch.save(model.state_dict(),save_path+"/CCA_model_"+name+".pth")
        et=time.time()
        print("train loss:",tloss.item())
        print("validation loss:",vloss.item())
        print("time taken:",et-st," s")
        print("________________\n\n")

#decoder (MLP)
class MLP(torch.nn.Module):
    def __init__(this,channel_lst):
        super().__init__()
        this.layers=torch.nn.ModuleList()
        for in_ch , out_ch in channel_lst:
            this.layers.append(torch.nn.Linear(in_ch,out_ch))
    def forward(this,x):
        for layer in this.layers:
            x=layer(x)
            x=x.sigmoid()
        return x

#cross entropy loss
def cross_entropy_loss(x,labels):
    loss=0
    for x_i , label in zip(x,labels):
        loss+=(-torch.log(x_i[label]))
    return loss

#function to train decoder
def train_dec(model,encode_model,optzr,loader_train,loader_val,loss_func,num_epochs=100,save_path=None , name='model'):
    encode_model.eval()
    for epoch in range(num_epochs):
        print("________________")
        print("epoch:",epoch)
        st=time.time()
        model.train()
        tloss=0
        true_count=0
        total_count=0
        for graph in loader_train:
            x=encode_model(graph)
            optzr.zero_grad()
            x=model(x)
            loss=loss_func(x,graph.y)
            tloss+=loss
            loss.backward()
            optzr.step()

            temp=torch.max(x,dim=1).indices
            true_count+=(temp==graph.y).sum()
            total_count+=x.shape[0]
        if(save_path!=None):
            torch.save(model.state_dict(),save_path+"/logreg_model_"+name+".pth")
        print("Training acc:",(true_count/total_count).item())
        #validation
        model.eval()
        vloss=0
        true_count=0
        total_count=0
        for graph in loader_val:
            x=encode_model(graph)
            x=model(x)
            loss=loss_func(x,graph.y)
            vloss+=loss

            temp=torch.max(x,dim=1).indices
            true_count+=(temp==graph.y).sum()
            total_count+=x.shape[0]
        et=time.time()
        print("validation acc:",(true_count/total_count).item())
        print("train loss:",tloss.item())
        print("validation loss:",vloss.item())
        print("time taken:",et-st," s")
        print("________________\n\n")

#function to test cca model
def test_cca_model(enc_model,dec_model,loader_test):
    enc_model.eval()
    dec_model.eval()

    true_count=0
    total_count=0
    for graph in loader_test:
        z=enc_model(graph)
        op_dec=dec_model(z)
        temp=torch.max(op_dec,dim=1).indices
        true_count+=(temp==graph.y).sum()
        total_count+=op_dec.shape[0]
    print("accuracy:",(true_count/total_count).item())

