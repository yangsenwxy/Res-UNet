import numpy as np
import os
import glob
import nibabel as nib
import SimpleITK as sitk

class dataProcess(object):
    
    def __init__(self, out_rows, out_cols):
        self.out_rows = out_rows
        self.out_cols = out_cols 

    def create_train_data(self,total_img_paths,total_label_paths,save_dir):
        '''
        sample: boundingbox images, 512 x 512, value=[-3000,3000],mean=-2630
        dtype: int16
        labels: GT images, 512 x 512, value={0,1,2}, mean=0.07
        dtype: int16
        '''
        total_img_num = len(total_img_paths)
        total_label_num = len(total_label_paths)
        assert total_img_num == total_label_num, 'total_img_num != total_label_num'

        img_data = np.ndarray((total_img_num, self.out_rows, self.out_cols, 1), dtype=np.int16)
        label_data = np.ndarray((total_label_num, self.out_rows, self.out_cols, 1), dtype=np.int16)
        total_img_paths.sort()
        total_label_paths.sort()

        for i in range(total_img_num):
            img = sitk.ReadImage(total_img_paths[i])
            img = sitk.GetArrayFromImage(img)
            img = np.where(img<-200,-200,img)   #if pixel value < -200, let pixel value = -200
            img = np.where(img>250,250,img)     #if pixel value > 250, let pixel value = 250
            img = np.expand_dims(img,2)
            img_data[i] = img
            
            label = sitk.ReadImage(total_label_paths[i])
            label = sitk.GetArrayFromImage(label)
            label = np.where(label>1,1,label)            #convert 2 to 1
            label = np.expand_dims(label,2)
            label_data[i] = label
        print('loading done', label_data.shape)

        train_img_path = save_dir + 'train_img.npy'
        train_label_path = save_dir + 'train_label.npy'
        if(os.path.exists(train_img_path)):
            os.remove(train_img_path)
        if(os.path.exists(train_label_path)):
            os.remove(train_label_path)
        np.save(train_img_path, img_data) 
        np.save(train_label_path, label_data)
        print('Saving to .npy files done.')


    def load_train_data(self,npy_dir):

        print('load train images...')
        img_train = np.load(npy_dir + 'train_img.npy')
        label_train = np.load(npy_dir + 'train_label.npy')

        img_train = img_train.astype('float32')
        label_train = label_train.astype('float32')

        return img_train, label_train



if __name__ == "__main__":
    pass
    data_processor = dataProcess(512,512)
    data_dir = '/home/albelt/NoseData/'
    img_paths = glob.glob(data_dir + 'BBOX/**/*.nii',recursive=True)
    label_paths = glob.glob(data_dir + 'GT/**/*.nii',recursive=True)
    npy_save_dir = data_dir + 'NPY/'
    data_processor.create_train_data(img_paths,label_paths,npy_save_dir)


