import os.path

import h5py
import glob
from PIL import Image as pil_image
import numpy as np
import utils
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, filename, patch_size=33, batch=1024, scale=3):
        super(H5Dataset, self).__init__()

        self.filename=filename
        self.file=None
        self.patch_size=patch_size
        self.batch=batch
        self.scale=scale

        self.hr = []
        self.lr = []
        self.hr_names=[]
        self.lr_names=[]
        self.count=0
        self.ds_index=-1

        self._open()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        ds_index = index // self.batch
        dx_last_index = index % self.batch
        if self.ds_index!=ds_index:
            hr_name = "hr"+str(ds_index)
            lr_name = "lr"+str(ds_index)
            self.hr = self.file[hr_name]
            self.lr = self.file[lr_name]
            self.ds_index = ds_index

        hr_item = self.hr[dx_last_index]/255.0
        hr_item = np.expand_dims(hr_item, axis=0)
        #hr_item = np.transpose(hr_item, (2,0, 1))

        lr_item = self.lr[dx_last_index]/255.0
        lr_item = np.expand_dims(lr_item, axis=0)
        #lr_item = np.transpose(lr_item, (2, 0, 1))

        return lr_item, hr_item

    def prepare(self, folder, max=1000):
        file_count=0
        lr_patches = []
        hr_patches = []
        count=0
        for image_path in sorted(glob.glob('{}/*'.format(folder))):
            image_hr_patchs, image_lr_patchs = self._prepare_image(image_path)
            if image_hr_patchs is None or image_lr_patchs is None:
                continue
            #hr_patches.append(image_hr_patchs)
            #lr_patches.append(image_lr_patchs)
            hr_patches = hr_patches + image_hr_patchs
            lr_patches = lr_patches + image_lr_patchs

            if len(hr_patches)>self.batch:
                self.file.create_dataset("hr" + str(count), data=hr_patches[0:self.batch])
                hr_patches = hr_patches[self.batch+1:]

                self.file.create_dataset("lr" + str(count), data=lr_patches[0:self.batch])
                lr_patches = lr_patches[self.batch + 1:]

                count += 1

            file_count+=1
            if file_count%100 == 0:
                print(f"file_count={file_count}, dataset count={count}")

            if file_count>=max:
                break

        print(f"Done!!! file_count={file_count}, dataset count={count}")

    def _open(self):
        if os.path.exists(self.filename):
            self.file = h5py.File(self.filename, 'a')
        else:
            self.file = h5py.File(self.filename, 'a')

        index=0
        for dsname in iter(self.file):
            if self.batch==0:
                self.batch = len(self.file[dsname])
            if dsname.find("hr") == 0:
                self.hr_names.append(dsname)
                self.count += self.batch
            elif dsname.find("lr") == 0:
                self.lr_names.append(dsname)

            index += 1
            #if index%200 == 0:
            #    print(f"read dataset, index={index}, dataset name={dsname}")
        print(f"load all dataset names! hr+lr dataset count={index+1}")

    def _prepare_image(self, iamge_file):
        lr_patches = []
        hr_patches = []

        hr = pil_image.open(iamge_file).convert('RGB')
        hr_width = (hr.width // self.scale) * self.scale
        hr_height = (hr.height // self.scale) * self.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // self.scale, hr_height // self.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = utils.convert_rgb_to_y(hr)
        lr = utils.convert_rgb_to_y(lr)

        for i in range(0, lr.shape[0] - self.patch_size , self.patch_size):
            for j in range(0, lr.shape[1] - self.patch_size , self.patch_size):
                lr_patches.append(lr[i:i + self.patch_size, j:j + self.patch_size])
                hr_patches.append(hr[i * self.scale:i * self.scale + self.patch_size * self.scale,
                                  j * self.scale:j * self.scale + self.patch_size * self.scale])

        return hr_patches, lr_patches

if __name__ == "__main__":
    h5=H5Dataset("dataset/train_div2k_x3.h5", batch=1024)
    h5.prepare("d:/AI/dataset/DIV2K/images")