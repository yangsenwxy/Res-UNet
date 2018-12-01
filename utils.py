import numpy as np
import matplotlib.pyplot  as plt


def cal_metrics(pred_arr,lab_arr,threshold):
    
    sample_num,row,column = lab_arr.shape[0],lab_arr.shape[1],lab_arr.shape[2]
    precisions,recalls,f1scores = [],[],[]
    pred_arr = pred_arr.where(pred_arr>=threshold,1.0,0.0)
    pred_arr = pred_arr.astype('uint8')
    lab_arr = lab_arr.astype('uint8')
    for i in range(sample_num):
        TP,TN,FP,FN = 0.0,0.0,0.0,0.0
        for j in range(row):
            for k in range(column):
                if(pred_arr[i,j,k]==1 and lab_arr[i,j,k]==1):
                    TP += 1
                elif(pred_arr[i,j,k]==1 and lab_arr[i,j,k]==0):
                    FP += 1
                elif(pred_arr[i,j,k]==0 and lab_arr[i,j,k]==1):
                    FN += 1
                else:
                    TN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1score = 2 * (precision * recall) / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1scores.append(f1score)

    return np.array(precisions),np.array(recalls),np.array(f1scores)

