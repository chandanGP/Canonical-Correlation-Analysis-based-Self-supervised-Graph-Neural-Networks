import torch
import torch.nn
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GCNConv , MessagePassing
from torch_geometric.data import Data , HeteroData
from torch_geometric.datasets import CoraFull
from torch_geometric.loader import DataLoader , NeighborLoader
from torch_geometric.utils import train_test_split_edges , degree
from torch_geometric.datasets import Coauthor, Amazon, Planetoid
from sklearn.model_selection import train_test_split
import time

from Models.models import * 



if __name__ == "__main__":
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    #co-author physics dataset
    dataset=Coauthor("co_author",'Physics')
    coa_ph_dataset=dataset[0].to(device)
    coa_ph_dataset['train_mask'] , coa_ph_dataset['val_mask'] , coa_ph_dataset['test_mask'] = train_split(coa_ph_dataset,0.1,0.1,0.8)
    loader_train = NeighborLoader(
        coa_ph_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=coa_ph_dataset.train_mask
    )
    loader_val = NeighborLoader(
        coa_ph_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=coa_ph_dataset.val_mask
    )
    loader_test = NeighborLoader(
        coa_ph_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=coa_ph_dataset.test_mask
    )
    save_path=None # you can give path where trained weights should be saved
    model_lst=[(-1,512),(512,512)]
    model_coa_ph=CCA_GCA_aug_homo(model_lst).to(device)
    #you can also load the pretrained weights by providing path here
    weights_path="model_weights/CCA_model_Coa_ph.pth"
    model_coa_ph.load_state_dict(torch.load(weights_path))
    '''
    # if you want to train uncomment this part
    optimizer_coa_ph=torch.optim.Adam(model_coa_ph.parameters(), lr=0.001, weight_decay=0)
    train_cca_model_gca_aug(model_coa_ph,loader_train,loader_val,optimizer_coa_ph,loss_cca2,device,100,0.001,pe1=0.4,pte1=0.7,pf1=0.1,ptf1=0.7,pe2=0.1,pte2=0.7,pf2=0.4,ptf2=0.7,save_path=save_path,name='Coa_ph_gca_aug')
    '''

    #decoder
    channel_lst=[(512,256),(256,coa_ph_dataset.y.unique().size(0))]
    mlp_coa_ph=MLP(channel_lst).to(device)
    optimizer_mlp=torch.optim.Adam(mlp_coa_ph.parameters(),lr=0.005, weight_decay=0.0001)
    train_dec(mlp_coa_ph,model_coa_ph,optimizer_mlp,loader_train,loader_val,torch.nn.CrossEntropyLoss(),100,save_path,'Coa_ph')
    #testing 
    test_cca_model(model_coa_ph,mlp_coa_ph,loader_test)

    #co-author cs dataset
    dataset=Coauthor("co_author",'cs')
    coa_cs_dataset=dataset[0].to(device)
    coa_cs_dataset['train_mask'] , coa_cs_dataset['val_mask'] , coa_cs_dataset['test_mask'] = train_split(coa_cs_dataset,0.1,0.1,0.8)
    loader_train = NeighborLoader(
        coa_cs_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=coa_cs_dataset.train_mask
    )
    loader_val = NeighborLoader(
        coa_cs_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=coa_cs_dataset.val_mask
    )
    loader_test = NeighborLoader(
        coa_cs_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=coa_cs_dataset.test_mask
    )
    model_lst=[(-1,512),(512,512)]
    model_coa_cs=CCA_GCA_aug_homo(model_lst).to(device)
    optimizer_coa_cs=torch.optim.Adam(model_coa_cs.parameters(), lr=0.001, weight_decay=0)
    train_cca_model_gca_aug(model_coa_cs,loader_train,loader_val,optimizer_coa_cs,loss_cca2,device,50,0.001,pe1=0.3,pte1=0.7,pf1=0.3,ptf1=0.7,pe2=0.2,pte2=0.7,pf2=0.4,ptf2=0.7,save_path=save_path,name='Coa_cs_gca_aug')
    #decoder
    channel_lst=[(512,256),(256,coa_cs_dataset.y.unique().size(0))]
    mlp_coa_cs=MLP(channel_lst).to(device)
    optimizer_mlp=torch.optim.Adam(mlp_coa_cs.parameters(),lr=0.005, weight_decay=0.0001)
    train_dec(mlp_coa_cs,model_coa_cs,optimizer_mlp,loader_train,loader_val,torch.nn.CrossEntropyLoss(),100,save_path,'Coa_cs')
    test_cca_model(model_coa_cs,mlp_coa_cs,loader_test)

    #amazon computer dataset
    dataset=Amazon("amazon","Computers")
    azn_com_dataset=dataset[0].to(device)
    azn_com_dataset['train_mask'] , azn_com_dataset['val_mask'] , azn_com_dataset['test_mask'] = train_split(azn_com_dataset,0.1,0.1,0.8)
    loader_train = NeighborLoader(
        azn_com_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=azn_com_dataset.train_mask
    )
    loader_val = NeighborLoader(
        azn_com_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=azn_com_dataset.val_mask
    )
    loader_test = NeighborLoader(
        azn_com_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=azn_com_dataset.test_mask
    )
    model_lst=[(-1,512),(512,512)]
    model_azn_com=CCA_GCA_aug_homo(model_lst).to(device)
    optimizer_azn_com=torch.optim.Adam(model_azn_com.parameters(), lr=0.001, weight_decay=0)
    train_cca_model_gca_aug(model_azn_com,loader_train,loader_val,optimizer_azn_com,loss_cca2,device,50,0.0005,pe1=0.5,pte1=0.7,pf1=0.2,ptf1=0.7,pe2=0.5,pte2=0.7,pf2=0.1,ptf2=0.7,save_path=save_path,name='Azn_com_gca_aug')
    #decoder
    channel_lst=[(512,256),(256,azn_com_dataset.y.unique().size(0))]
    mlp_azn_com=MLP(channel_lst).to(device)
    optimizer_mlp=torch.optim.Adam(mlp_azn_com.parameters(),lr=0.005, weight_decay=0.0001)
    train_dec(mlp_azn_com,model_azn_com,optimizer_mlp,loader_train,loader_val,torch.nn.CrossEntropyLoss(),200,save_path,'Azn_com')
    #testing
    test_cca_model(model_azn_com,mlp_azn_com,loader_test)

    #amazon photos dataset
    dataset=Amazon("amazon","Photo")
    azn_ph_dataset=dataset[0].to(device)
    azn_ph_dataset['train_mask'] , azn_ph_dataset['val_mask'] , azn_ph_dataset['test_mask'] = train_split(azn_ph_dataset,0.1,0.1,0.8)
    loader_train = NeighborLoader(
        azn_ph_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=azn_ph_dataset.train_mask
    )
    loader_val = NeighborLoader(
        azn_ph_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=azn_ph_dataset.val_mask
    )
    loader_test = NeighborLoader(
        azn_ph_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=azn_ph_dataset.test_mask
    )
    model_lst=[(-1,512),(512,512)]
    model_azn_ph=CCA_GCA_aug_homo(model_lst).to(device)
    optimizer_azn_ph=torch.optim.Adam(model_azn_ph.parameters(), lr=0.001, weight_decay=0)
    train_cca_model_gca_aug(model_azn_ph,loader_train,loader_val,optimizer_azn_ph,loss_cca2,device,50,0.001,pe1=0.3,pte1=0.7,pf1=0.1,ptf1=0.7,pe2=0.5,pte2=0.7,pf2=0.1,ptf2=0.7,save_path=save_path,name='Azn_ph_gca_aug')
    #decoder
    channel_lst=[(512,256),(256,azn_ph_dataset.y.unique().size(0))]
    mlp_azn_ph=MLP(channel_lst).to(device)
    optimizer_mlp=torch.optim.Adam(mlp_azn_ph.parameters(),lr=0.005, weight_decay=0.0001)
    train_dec(mlp_azn_ph,model_azn_ph,optimizer_mlp,loader_train,loader_val,torch.nn.CrossEntropyLoss(),100,save_path,'Azn_ph')
    #testing
    test_cca_model(model_azn_ph,mlp_azn_ph,loader_test)

    # Cora dataset
    dataset=Planetoid('Planetoid','Cora')
    cora_dataset=dataset[0].to(device)
    cora_dataset['train_mask'] , cora_dataset['val_mask'] , cora_dataset['test_mask'] = train_split(cora_dataset,0.1,0.1,0.8)
    loader_train = NeighborLoader(
        cora_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=cora_dataset.train_mask
    )
    loader_val = NeighborLoader(
        cora_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=cora_dataset.val_mask
    )
    loader_test = NeighborLoader(
        cora_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=cora_dataset.test_mask
    )
    model_lst=[(-1,512),(512,512)]
    model_cora=CCA_GCA_aug_homo(model_lst).to(device)
    optimizer_cora=torch.optim.Adam(model_cora.parameters(), lr=0.001, weight_decay=0)
    train_cca_model_gca_aug(model_cora,loader_train,loader_val,optimizer_cora,loss_cca2,device,50,0.001,pe1=0.3,pte1=0.7,pf1=0.1,ptf1=0.7,pe2=0.5,pte2=0.7,pf2=0.1,ptf2=0.7,save_path=save_path,name='Cora_gca_aug')
    #decoder
    channel_lst=[(512,256),(256,cora_dataset.y.unique().size(0))]
    mlp_cora=MLP(channel_lst).to(device)
    optimizer_mlp=torch.optim.Adam(mlp_cora.parameters(),lr=0.01, weight_decay=0.0001)
    train_dec(mlp_cora,model_cora,optimizer_mlp,loader_train,loader_val,torch.nn.CrossEntropyLoss(),500,save_path,'Cora')
    #testing
    test_cca_model(model_cora,mlp_cora,loader_test)

    #Citeseer dataset
    dataset=Planetoid('Planetoid','Citeseer')
    citeseer_dataset=dataset[0]
    citeseer_dataset=citeseer_dataset.to(device)
    citeseer_dataset['train_mask'] , citeseer_dataset['val_mask'] , citeseer_dataset['test_mask'] = train_split(citeseer_dataset,0.1,0.1,0.8)
    loader_train = NeighborLoader(
        citeseer_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=citeseer_dataset.train_mask
    )
    loader_val = NeighborLoader(
        citeseer_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=citeseer_dataset.val_mask
    )
    loader_test = NeighborLoader(
        citeseer_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=citeseer_dataset.test_mask
    )
    model_lst=[(-1,512)]
    model_citeseer=CCA_GCA_aug_homo(model_lst).to(device)
    optimizer_citeseer=torch.optim.Adam(model_citeseer.parameters(), lr=0.001, weight_decay=0)
    train_cca_model_gca_aug(model_citeseer,loader_train,loader_val,optimizer_citeseer,loss_cca2,device,20,0.0005,pe1=0.3,pte1=0.7,pf1=0.1,ptf1=0.7,pe2=0.5,pte2=0.7,pf2=0.1,ptf2=0.7,save_path=save_path,name='Citeseer_gca_aug')
    #decoder
    channel_lst=[(512,256),(256,citeseer_dataset.y.unique().size(0))]
    mlp_citeseer=MLP(channel_lst).to(device)
    optimizer_mlp=torch.optim.Adam(mlp_citeseer.parameters(),lr=0.01, weight_decay=0.01)
    train_dec(mlp_citeseer,model_citeseer,optimizer_mlp,loader_train,loader_val,torch.nn.CrossEntropyLoss(),200,save_path,'Citeseer')
    #testing
    test_cca_model(model_citeseer,mlp_citeseer,loader_test)


    #Pubmed datset
    dataset=Planetoid('Planetoid','Pubmed')
    pbmd_dataset=dataset[0].to(device)
    pbmd_dataset['train_mask'] , pbmd_dataset['val_mask'] , pbmd_dataset['test_mask'] = train_split(pbmd_dataset,0.1,0.1,0.8)
    loader_train = NeighborLoader(
        pbmd_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=pbmd_dataset.train_mask
    )
    loader_val = NeighborLoader(
        pbmd_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=pbmd_dataset.val_mask
    )
    loader_test = NeighborLoader(
        pbmd_dataset,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=pbmd_dataset.test_mask
    )
    model_lst=[(-1,512),(512,512)]
    model_pbmd=CCA_GCA_aug_homo(model_lst).to(device)
    optimizer_pbmd=torch.optim.Adam(model_pbmd.parameters(), lr=0.001, weight_decay=0)
    train_cca_model_gca_aug(model_pbmd,loader_train,loader_val,optimizer_pbmd,loss_cca2,device,100,0.001,pe1=0.3,pte1=0.7,pf1=0.1,ptf1=0.7,pe2=0.5,pte2=0.7,pf2=0.1,ptf2=0.7,save_path=save_path,name='Pubmed_gca_aug')
    #decoder
    channel_lst=[(512,256),(256,pbmd_dataset.y.unique().size(0))]
    mlp_pbmd=MLP(channel_lst).to(device)
    optimizer_mlp=torch.optim.Adam(mlp_pbmd.parameters(),lr=0.01, weight_decay=0.0001)
    train_dec(mlp_pbmd,model_pbmd,optimizer_mlp,loader_train,loader_val,torch.nn.CrossEntropyLoss(),100,save_path,'Pubmed')
    #testing
    test_cca_model(model_pbmd,mlp_pbmd,loader_test)