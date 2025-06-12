python PnC_Protein.py --batch_size=16 --d_out=16 --num_layers 2 --dataset=bioinformatics --dataset_name=MUTAG --lr_dict=0.1 --lr_policy=0.001 --n_h_max_dict=10 

python PnC_Protein.py --batch_size=64 --d_out=16 --num_layers 2 --dataset=proteinshake --dataset_name=proteinshake --lr_dict=0.1 --lr_policy=0.001 --n_h_max_dict=10 --split=random --inds_to_visualise 1,2,3,4,5 --node_attr_encoding uniform --n_h_max 30 --n_h_min 5 --n_h_min_dict 5 


## Testing
 python PnC_Protein.py --batch_size=64 --d_out=16 --num_layers 2 --dataset=proteinshake --dataset_name=proteinshake --lr_dict=0.1 --lr_policy=0.001 --n_h_max_dict=10 --split=random --inds_to_visualise 1,2,3,4,5 --node_attr_encoding uniform --n_h_max 30 --n_h_min 5 --n_h_min_dict 5 --wandb_realtime False --wandb_name constraint_subgraph_size --wandb False --wandb_mode offline --num_epochs 1