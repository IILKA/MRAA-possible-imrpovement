import os 
import numpy as np 
from PIL import Image
from tqdm import tqdm 
import glob
from PIL import Image
import numpy as np
#calculate the mean of the gif videos return a three channel mean 
def calculate_mean_gif(path):
    img_list = []
    for img in tqdm(os.listdir(path)):
        img_path = os.path.join(path, img)
        #open the 




        img = Image.open(img_path)
        img = np.array(img)
        print(img.shape)
    img_list = np.array(img_list)
    mean = np.mean(img_list, axis = 0)
    return mean


def calculate_mean_std(folder_path):
    # 创建用于存储所有帧像素值的列表
    pixel_values = []

    # 遍历文件夹中的所有GIF文件
    for gif_path in tqdm(glob.glob(f"{folder_path}/*.gif")):
        # 打开GIF文件
        with Image.open(gif_path) as im:
            # 遍历GIF文件的每一帧
            for frame in range(im.n_frames):
                im.seek(frame)
                # 将当前帧转换为NumPy数组
                frame_array = np.array(im.convert('RGB'))
                # 将当前帧的像素值添加到列表中
                pixel_values.extend(frame_array.reshape(-1, 3))

    # 将列表转换为NumPy数组
    pixel_array = np.array(pixel_values, dtype=np.float32)

    # 计算均值和标准差
    mean = pixel_array.mean(axis=0) / 255.0
    std = pixel_array.std(axis=0) / 255.0

    return mean, std

# 使用示例
folder_path = "/mnt/disk2/iilka/moving-gif-processed/moving-gif/test"
mean, std = calculate_mean_std(folder_path)
print(f"Mean: {mean}")
print(f"Std: {std}")