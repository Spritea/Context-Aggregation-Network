# Context Aggregation Network
This is the official implementation code of paper `Context aggregation network for semantic labeling in aerial images`.
## Requirements
* pytorch>=0.4
* [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)

## Getting started
1. Clone pytorch-semseg repo.
2. Put `CAN.py` in the folder `pytorch-semseg/ptsemseg/models`
3. Change `pytorch-semseg/ptsemseg/models/__init__.py` correspondingly.
4. Add your own config file to `pytorch-semseg/configs` folder.
5. Add your own dataloader file to `pytorch-semseg/ptsemseg/loader` folder. 
6. Change `pytorch-semseg/ptsemseg/loader/__init__.py` correspondingly.
7. Run `train.py` by following the specific orders in `pytorch-semseg` repo.
## Dependency
This repo is heavily based on the framework provided by [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).