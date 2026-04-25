from Train import train, eval

modelConfig = {
    "state": "eval", # "train" or "eval"
    "epoch": 2000,
    "batch_size": 2,
    "T": 400,
    "channel": 64,
    "channel_mult": [1, 2, 3, 4],
    "attn": [2],
    "num_res_blocks": 1,
    "dropout": 0.15,
    "lr": 2e-5,
    "multiplier": 2,
    "beta_1": 1e-4,
    "beta_T": 0.04,
    "img_size": 512,
    "grad_clip": 1.,
    "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
    # Dataset paths
    "train_root": None,
    "val_root": None,
    "num_workers": 4,
    "pin_memory": True,
    # Training & Checkpoints
    "training_load_weight":"test_leak/20260321_194005/ckpt_1000epoch.pt",
    "save_weight_dir": "./",
    "test_load_weight":"weights/warmup_8000epoch.pt" ,
    # Sampling & Inference
    "sampled_dir": "./SampledImgs/",
    "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
    "sampledImgName": "SampledNoGuidenceImgs.png",
    "nrow": 8,
    "input_channel": 3
    }
def main(model_config = None):
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)
