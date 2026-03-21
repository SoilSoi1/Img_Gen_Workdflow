import os
import time
from Main import modelConfig
from Train import eval
from torchvision.utils import save_image

def sampling(num_pic, saved_dir):
    for i in range(num_pic):
        sampled_img = eval(modelConfig)
        save_image(sampled_img, f"{saved_dir}/sampled_img_{i}.png", nrow=modelConfig["nrow"])

if __name__ == '__main__':
    modelConfig["state"] = "eval"
    num_pic = 100
    weight = modelConfig["test_load_weight"].split("/")[-1][:-3]
    os.mkdir(f"./outfig/ddpm/{weight}", exist_ok=True)
    saved_dir = f"./outfig/ddpm/{weight}"

    s_t = time.time()
    sampling(num_pic, saved_dir)
    e_t = time.time()
    print(f"采样 {num_pic} 张图像总共用时: {e_t - s_t} 秒")