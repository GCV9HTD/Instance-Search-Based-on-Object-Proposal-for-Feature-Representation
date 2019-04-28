TRAINING_PARAMS = {
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 65,
    },
    "batch_size": 1,
    "confidence_threshold": 0.4,
    "images_path": "./images/",
    "classes_names_path": "../data/tiny_megvii/megvii.names",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    "pretrain_snapshot": "/media/data1/yz/ProposalYOLO/models/megvii/size416x416_try0/20190428151451/Ep0013-model.pth",
}