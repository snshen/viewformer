import numpy as np
import os
from PIL import Image


class ColmapDataLoader:
    
    def __init__(self, file_path, img_path, img_size=128, batch_size=30) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.img_size = img_size
        self.batch_size = batch_size
        self.file_path = os.path.join(file_path, 'images.txt')
        self.img_path = img_path
        self.img_names = []
        self.cameras = []

        file = open(self.file_path, 'r')
        Lines = file.readlines()

        count = 0
        for line in Lines:
            if count >= 5:
                data = line.strip().split(" ")
                self.img_names.append(int(data[-1].split(".")[0]))
                self.cameras.append(np.concatenate((data[5:8], data[1:5])).astype(float))
            count+=1
        idx = np.argsort(self.img_names)
        num_files = len(idx)
        self.img_names = np.array(self.img_names)[idx]
        self.cameras = np.array(self.cameras, dtype=np.float32)[idx]

    def crop_center(self, img):
        y, x, _ = img.shape
        scale = y/self.img_size
        img = np.array(Image.fromarray(img.astype(np.uint8)).resize((int(x/scale), self.img_size)))
        y, x, _ = img.shape
        startx = x//2-(self.img_size//2)
        starty = y//2-(self.img_size//2)    
        return img[starty:starty+self.img_size,startx:startx+self.img_size]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        output = dict()
        frames = []
        cameras = []
        for j in range(self.batch_size):
            if i+j >= len(self.img_names):
                break
            frame = np.array(Image.open(os.path.join(self.img_path, str(self.img_names[i+j])+".jpg")))
            frame = self.crop_center(frame)
            frames.append(frame)
            cameras.append(self.cameras[i+j])
        output['frames'] = np.array(frames)
        output['cameras'] = np.array(cameras)
        return output

if __name__ == "__main__":
    img_path = '/home/ec2-user/novel-cross-view-generation/viewformer/colmap/custom/images'
    file_path = '/home/ec2-user/novel-cross-view-generation/viewformer/colmap/custom/sparse/0'
    c = ColmapDataLoader(file_path, img_path)
    print(len(c))
    output = c[0]
    print(output['cameras'].shape)