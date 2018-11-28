import numpy as np
import matplotlib.pyplot  as plt

if __name__ == '__main__':
    npy_dir = '/home/albelt/NoseData/NPY/'
    img = np.load(npy_dir+'train_img.npy')
    img = img[-40:,:,:,:]
    lab = np.load(npy_dir+'train_label.npy')
    lab = lab[-40:,:,:,:]
    pred = np.load(npy_dir+'predict.npy')
    result_save_dir = '/home/albelt/NoseData/RESULT/'
    for i in range(40):
        img_ = img[i,:,:,0]
        lab_ = lab[i,:,:,0]
        pred_ = pred[i,:,:,0]
        plt.imsave(result_save_dir+'img_'+str(i)+'.jpg',img_,cmap='gray')
        plt.imsave(result_save_dir+'lab_'+str(i)+'.jpg',lab_,cmap='gray') 
        plt.imsave(result_save_dir+'pred_'+str(i)+'.jpg',pred_,cmap='gray') 

