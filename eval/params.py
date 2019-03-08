TRAINING_PARAMS = {
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "", #  set empty to disable
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
    },

    "val_path": "../data/coco/5k.txt",
    # "val_path": "../data/voc/5k.txt",
    "img_h": 416,
    "img_w": 416,
    "parallels": [1],
    #
    "pretrain_snapshot": "../models/darknet_53/size416x416_try0/20190307164430/Ep0003-model.pth",
}
