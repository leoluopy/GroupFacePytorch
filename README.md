# GroupFacePytorch
none official  pytorch implementation of CVPR2020 GroupFace

> This repository give you a easy way to quick look through the all training process of Group Face


```
paper is avaliable at https://arxiv.org/abs/2005.10497
some code is ported from https://github.com/SeungyounShin/GroupFace
thanks for contributing
```

> python train.py to launch training after 5 epochs u will found the demo set top1 acc increase obivously

# TODO List
- [x] dataloader , sample ims,
- [x] implement loss  
- [x] main process of training a demo set
- [ ] image clustering
- [ ] learning DGN
--------------------------------------
- [ ] train at MSCeleb[cleaned] 
- [ ] eval at MegaFace LFW YTF CFP-TP AgeDB IJB-B IJB-C 
--------------------------------------
- [ ] hard or soft GroupFace probability study  
- [ ] group ablation study
- [ ] feature aggregation or concatenate study
- [ ] group aware similarity study



