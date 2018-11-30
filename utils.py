import ptvsd
ptvsd.settrace(None,('0.0.0.0',8000))
ptvsd.wait_for_attach()

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

import numpy as np

def get_lab_pred(threshold):
    label = np.load('./label.npy')
    label = label.astype('uint8')
    result = np.load('./result.npy')
    result = np.where(result>threshold,1.0,result)
    result = np.where(result!=1.0,0.0,result)
    result = result.astype('uint8')

    return label[:10,:,:,0],result[:10,:,:,0]

# def cal():
#     lab, pred = get_lab_pred(0.8)
#     sample_num,row,column = lab.shape[0],lab.shape[1],lab.shape[2]
#     precisions,recalls,f1scores = [],[],[]
#     for i in range(sample_num):
#         TP,TN,FP,FN = 0.0,0.0,0.0,0.0
#         for j in range(row):
#             for k in range(column):
#                 if(lab[i,j,k]==1 and pred[i,j,k]==1):
#                     TP += 1
#                 elif(lab[i,j,k]==1 and pred[i,j,k]==0):
#                     FN += 1
#                 elif(lab[i,j,k]==0 and pred[i,j,k]==1):
#                     FP += 1
#                 else:
#                     TN += 1
#         precision = TP / (TP + FP)
#         recall = TP / (TP + FN)
#         f1score = 2 * (precision * recall) / (precision + recall)
#         precisions.append(precision)
#         recalls.append(recall)
#         f1scores.append(f1score)
#     return precisions,recalls,f1scores

if __name__ == '__main__':
    # precisions,recalls,f1scores = cal()
    # p_mean, p_var = np.mean(precisions), np.var(precisions)
    # r_mean, r_var = np.mean(recalls), np.var(recalls)
    # f_mean,f_var = np.mean(f1scores), np.var(recalls)
    # print(p_mean,p_var)
    # pass
    print('test1')
    print('test2')
    print('test3')
    pass
    