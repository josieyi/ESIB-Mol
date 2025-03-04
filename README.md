# Expert-Guided Substructure Information Bottleneck for Molecular Property Prediction

This is Official Pytorch Implementation for the paper "Expert-Guided Substructure Information Bottleneck for Molecular Property Prediction".

## Environment:

We used the following Python packages for core development. We tested on `Python 3.7`.
```
- pytorch 1.13.1
- torch-geometric 2.3.1
```

## Datasets:

The datasets used for the experiments are provided in the `data` directory of this repository.

## Run  
```
python main.py --dataset {dataset name} --gnn_type {gnn name} --num_experts {expert name}
```