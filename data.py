import numpy as np
import os
import glob
import nibabel as nib
import SimpleITK as sitk

class dataProcess(object):
    
    def __init__(self, out_rows, out_cols):
        self.out_rows = out_rows
        self.out_cols = out_cols 

    def create_data(self,total_img_paths,total_label_paths,save_dir,validation_ratio):
        '''
        create data for train and validation(test)
        sample: boundingbox images, 512 x 512, value=[-3000,3000],mean=-2630
        dtype: int16
        labels: GT images, 512 x 512, value={0,1,2}, mean=0.07
        dtype: int16
        '''

        total_img_num = len(total_img_paths)
        total_label_num = len(total_label_paths)
        assert total_img_num == total_label_num, 'total_img_num != total_label_num'

        total_img_paths.sort()
        total_label_paths.sort()
        sample_index = np.arange(total_img_num)
        np.random.shuffle(sample_index)
        train_sample_num = int( total_img_num * (1.0 - validation_ratio) )
        test_sample_num = total_img_num - train_sample_num

        #make .npy data for train
        img_train = np.ndarray((train_sample_num, self.out_rows, self.out_cols, 1), dtype=np.int16)
        labelt_train = np.ndarray((train_sample_num, self.out_rows, self.out_cols, 1), dtype=np.int16)
        for i,j in enumerate(sample_index[0:train_sample_num]):
            img = sitk.ReadImage(total_img_paths[j])
            img = sitk.GetArrayFromImage(img)
            img = np.where(img<-200,-200,img)   #if pixel value < -200, let pixel value = -200
            img = np.where(img>250,250,img)     #if pixel value > 250, let pixel value = 250
            img = np.expand_dims(img,2)
            img_train[i] = img
            
            label = sitk.ReadImage(total_label_paths[j])
            label = sitk.GetArrayFromImage(label)
            label = np.where(label>1,1,label)            #convert 2 to 1
            label = np.expand_dims(label,2)
            labelt_train[i] = label

        train_img_path = save_dir + 'train_img.npy'
        train_label_path = save_dir + 'train_label.npy'
        if(os.path.exists(train_img_path)):
            os.remove(train_img_path)
        if(os.path.exists(train_label_path)):
            os.remove(train_label_path)
        np.save(train_img_path, img_train) 
        np.save(train_label_path, labelt_train)

        #make .npy data for test and validation
        img_test = np.ndarray((test_sample_num, self.out_rows, self.out_cols, 1), dtype=np.int16)
        labelt_test = np.ndarray((test_sample_num, self.out_rows, self.out_cols, 1), dtype=np.int16)
        for i,j in enumerate(sample_index[train_sample_num:]):
            img = sitk.ReadImage(total_img_paths[j])
            img = sitk.GetArrayFromImage(img)
            img = np.where(img<-200,-200,img)   #if pixel value < -200, let pixel value = -200
            img = np.where(img>250,250,img)     #if pixel value > 250, let pixel value = 250
            img = np.expand_dims(img,2)
            img_test[i] = img
            
            label = sitk.ReadImage(total_label_paths[j])
            label = sitk.GetArrayFromImage(label)
            label = np.where(label>1,1,label)            #convert 2 to 1
            label = np.expand_dims(label,2)
            labelt_test[i] = label

        test_img_path = save_dir + 'test_img.npy'
        test_label_path = save_dir + 'test_label.npy'
        if(os.path.exists(test_img_path)):
            os.remove(test_img_path)
        if(os.path.exists(test_label_path)):
            os.remove(test_label_path)
        np.save(test_img_path, img_test) 
        np.save(test_label_path, labelt_test)


        print('Finished create data, {0} samples for train and {1} samples for test/validation, saved in {2}'.format(train_sample_num,test_sample_num,save_dir))


    def load_train_data(self,npy_dir):
        print('load train images...')
        img_train = np.load(npy_dir + 'train_img.npy')
        label_train = np.load(npy_dir + 'train_label.npy')

        img_train = img_train.astype('float32')
        label_train = label_train.astype('float32')

        return img_train, label_train

    def load_test_data(self,npy_dir):
        print('load test images...')
        img_test = np.load(npy_dir + 'test_img.npy')
        label_test = np.load(npy_dir + 'test_label.npy')

        img_test = img_test.astype('float32')
        label_test = label_test.astype('float32')

        return img_test,label_test


if __name__ == "__main__":
    pass
    data_processor = dataProcess(512,512)
    data_dir = '/home/albelt/NoseData/'
    img_paths = glob.glob(data_dir + 'BBOX/**/*.nii',recursive=True)
    label_paths = glob.glob(data_dir + 'GT/**/*.nii',recursive=True)
    npy_save_dir = data_dir + 'NPY/'
    data_processor.create_data(img_paths,label_paths,npy_save_dir,0.2)


