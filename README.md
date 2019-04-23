# Context Aggregation Network
This is the official Pytorch implementation code of paper `Context aggregation network for semantic labeling in aerial images`.
## Requirements
Please refer to this file [requirements.txt](https://github.com/Spritea/Context-Aggregation-Network/blob/master/requirements.txt).

## Getting started
1. Download ISPRS Vaihingen and Potsdam datasets on the [website](http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html) by following its instructions.
2. Put these datasets in corresponding `dataset` subfolder. Note that original colorful labels need to be converted to index-based (0,1,2,3,4,5) image using this [code](https://github.com/Spritea/Context-Aggregation-Network/blob/master/precode.py). 
3. Run this command to train CAN model on ISPRS Vaihingen dataset, or Potsdam dataset by replacing `isprs_vaihingen.yml` with `isprs_potsdam.yml`. You can set many customized parameters in the `.yml` file.:
```
python train.py --config configs/isprs_vaihingen.yml
```

## Dependency
This repo is heavily based on the framework provided by [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg). You can refer to that repo for more details.
