TRAINING_PARAMS = {
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "../weights/darknet53_weights_pytorch.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
    },
    "lr": {
        "backbone_lr": 0.001,
        "other_lr": 0.1,
        "freeze_backbone": True,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.9,
        "decay_step": 4,           #  decay lr in every 2 epochs
    },
    "optimizer": {
        "type": "adabound",  # adabound, sgd
        "weight_decay": 4e-05,  # 4e-05
    },
    # 8
    "batch_size": 64,
    # 4
    "accumulation": 1,
    "train_path": "../data/coco/trainvalno5k.txt",
    "val_path": "../data/coco/5k.txt",
    "epochs": 4,
    "img_h": 416,
    "img_w": 416,
    # 2,0
    "parallels": [1],                         #  config GPU device
    "working_dir": "../models",                 #  replace with your working dir
    "pretrain_snapshot": "",                    #  load checkpoint
    "evaluate_type": "", 
    "try": 0,                                   #  index of experiment
    "export_onnx": False,                       #  export for other framework
}
