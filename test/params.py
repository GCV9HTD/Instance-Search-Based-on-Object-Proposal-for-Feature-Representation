TRAINING_PARAMS = {
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
    },
    "batch_size": 1,

    "images_path": "./images",
    # "images_path": "../data/coco/pca",

    "qry_path": "../data/Instance/qry",
    "ref_path": "../data/Instance/ref",
    # "ref_path": "../data/Instance/dis100k",
    # "ref_path": "../data/Instance/dis1m",
    "ground_truth": "../data/Instance/bbox",

    # "qry_path": "../data/Instre/qry",
    # "ref_path": "../data/Instre/ref",
    # "ground_truth": "../data/Instre/bbox",

    # "qry_path": "../data/oxford/qry",
    # "ref_path": "../data/oxford/ref",
    # "ground_truth": "../data/oxford/bbox",

    # "qry_path": "../data/paris/qry",
    # "ref_path": "../data/paris/ref",
    # "ground_truth": "../data/paris/bbox",

    "visualize_path": "../data/test",

    "img_h": 416,
    "img_w": 416,
    "parallels": [4],
    #
    "pretrain_snapshot": "../models/darknet_53/size416x416_try0/20181225113954/Ep0003-model.pth",
    # "pretrain_snapshot": "",

    "pca":True,
    "pca_path": "../data/coco/pca",
    "mapping_fn": "/home/yz/cde/ProposalYOLO/feature/mapping/mapping-attnfrz.pkl",
}
