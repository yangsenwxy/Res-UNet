from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, Lambda, MaxPooling2D, BatchNormalization, Activation, add
from keras.layers.merge import concatenate
from keras.layers import *
from keras.layers import Activation
# from keras.optimizers import Adam
from keras import optimizers
from keras.initializers import he_normal
init = he_normal(seed=1)
from keras.models import load_model
from keras.callbacks import TensorBoard,ReduceLROnPlateau
# from keras.utils import plot_model
from keras.metrics import f1score
from keras import losses
from keras.utils.np_utils import to_categorical
# from keras.losses import binary_crossentropy

from ResNet import identity_block, conv_block
from data import dataProcess
from preprocess import random_enhance
import glob


train_log = TensorBoard(log_dir='/home/albelt/NoseData/LOG',histogram_freq=1,write_graph=False,
                        write_grads=False,batch_size=8,write_images=True)
lr_decay = ReduceLROnPlateau(monitor='f1score',factor=0.1,patience=1,verbose=1,mode='max')

def side_out(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same',
                        use_bias=False, activation=None, kernel_initializer=init)(x)
    return x


class myUnet(object):
    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self,npy_dir,is_train=True):
        mydata = dataProcess(self.img_rows, self.img_cols)
        if(is_train):
            img_train, label_train = mydata.load_train_data(npy_dir)
            return img_train,label_train
        else:
            img_val, label_val = mydata.load_test_data(npy_dir)
            return img_val,label_val

    def get_unet(self,gpu_count):
        inputs = Input((self.img_rows, self.img_cols,1))    
        x = Conv2D(16, (5, 5), strides=(1, 1), padding='same', name='conv1')(inputs)
        x = BatchNormalization(axis=-1, name='bn_conv1')(x)
        x = Activation('relu', name='act1')(x)  
        
        # Block 1
        c1 = conv_block(x, 3, (8, 8, 32), stage=1, block='a', strides=(1, 1))
        c1 = identity_block(c1, 3, (8, 8, 32), stage=1, block='b')      # 320, 480, 3
        # Block 2
        c2 = conv_block(c1, 3, (16, 16, 64), stage=2, block='a', strides=(2, 2))
        c2 = identity_block(c2, 3, (16, 16, 64), stage=2, block='b')    # 160, 240, 3
        # Block 3
        c3 = conv_block(c2, 3, (32, 32, 128), stage=3, block='a', strides=(2, 2))
        c3 = identity_block(c3, 3, (32, 32, 128), stage=3, block='b')   # 80, 120, 3
        # Block 4
        c4 = conv_block(c3, 3, (64, 64, 256), stage=4, block='a', strides=(2, 2))
        c4 = identity_block(c4, 3, (64, 64, 256), stage=4, block='b')   # 40, 60, 3
        s1=side_out(c4,8)
        # Block 5
        u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='upconv_6', kernel_initializer=init)(c4)
        u5 = concatenate([u5, c3], name='concat_6')        # 40, 60, 3
        c5 = conv_block(u5, 3, (32, 32, 128), stage=6, block='a', strides=(1, 1))
        c5 = identity_block(c5, 3, (32, 32, 128), stage=6, block='b')
        s2 = side_out(c5, 4)
        # Block 6
        u6 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='upconv_7', kernel_initializer=init)(c5)
        u6 = concatenate([u6, c2], name='concat_7')        # 80, 120, 3
        c6 = conv_block(u6, 3, (16, 16, 64), stage=7, block='a', strides=(1, 1))
        c6 = identity_block(c6, 3, (16, 16, 64), stage=7, block='b')
        s3 = side_out(c6, 2)
        # Block 7
        u7 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='upconv_8', kernel_initializer=init)(c6)
        u7 = concatenate([u7, c1], name='concat_8')        # 160, 240, 3
        c7 = conv_block(u7, 3, (8, 8, 32), stage=8, block='a', strides=(1, 1))
        c7 = identity_block(c7, 3, (8, 8, 32), stage=8, block='b')
        s4 = side_out(c7, 1)

        # fuse
        fuse = concatenate(inputs=[s1, s2, s3, s4], axis=-1)
        fuse = Conv2D(1, (1, 1), padding='same', activation=None)(fuse)       # 320 480 1
        # fuse = to_categorical(fuse,3)

        # outputs
        # o1    = Activation('sigmoid', name='o1')(s1)
        # o2    = Activation('sigmoid', name='o2')(s2)
        # o3    = Activation('sigmoid', name='o3')(s3)
        # o4    = Activation('sigmoid', name='o4')(s4)

        ofuse = Activation('sigmoid', name='ofuse')(fuse)
        
        model = Model(inputs=[inputs], outputs=[ofuse])
        # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        adam = optimizers.Adam(decay=1e-6)   #每个mini batch衰减一次

        if(gpu_count==0):
            model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[f1score])
            print('model compile')
            return model
        else:
            from keras.utils import multi_gpu_model
            parallel_model = multi_gpu_model(model,gpus=gpu_count)
            parallel_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[f1score])
            print('parallel model compile')
            return parallel_model


    def train(self,npy_dir,ckpt_dir,gpu_count,epochs,resume_from_lastest=True):
        model = None
        start_eopch = None
        end_epoch = None
        if(resume_from_lastest):
            model_list = glob.glob(ckpt_dir + 'model*.hdf5')
            model_list.sort()
            model = load_model(model_list[-1])
            last_epoch = int(model_list[-1].split('.')[0].split('_')[-1].split('epoch')[-1])
            start_eopch = last_epoch + 1
            end_epoch = start_eopch + epochs
            print('load model from eopch{0}'.format(last_epoch))
            print(model.summary())
        else:
            model = self.get_unet(gpu_count)
            start_eopch = 0
            end_epoch = epochs
            print("load a new model instance")
            print(model.summary())

        # # plot_model(model,to_file='./net.jpg')
        
        print("loading data")
        img_train, label_train = self.load_data(npy_dir)
        print("loading data done")

        print('Fitting model...')
        for epoch in range(start_eopch,end_epoch):
            # img_temp,label_temp = random_enhance(img_train,label_train)

            model.fit(img_train,label_train, 
                                batch_size=8, epochs=20, 
                                verbose=1,shuffle=True,
                                validation_split=0.05,
                                callbacks=[train_log,lr_decay])
            # model.save_weights(ckpt_dir + 'weight_epoch_' + str(epoch) + '.hdf5', True) #只保存模型参数
            model.save(ckpt_dir +  'model_epoch_' + str(epoch) +'.hdf5',True,True)      #保存网络结构、模型参数、optimizer的情况，用来直接加载执行预测
            print('\nepoch-{0} Finished'.format(epoch))

        print('\n\nTraining Finished')

    def test(self,npy_dir,ckpt_dir):
        img_test,label_test = self.load_data(npy_dir)
        img_test = img_test[:100,:,:,:]
        label_test = label_test[:100,:,:,:]
        print('test data load done,use 100 samples')
        model_list = glob.glob(ckpt_dir + 'model*.hdf5')
        model_list.sort()
        print('All model checkpoint avaible:')
        for i,model_name in enumerate(model_list):
            print('epoch-{0},\t{1}'.format(i,model_name))
        choice =  int(input('Chose one,input the index:')) 
        model = load_model(model_list[choice])
        model.save_weights(ckpt_dir + 'weight_epoch_' + str(epoch) + '.hdf5', True)
        # model = load_model('/home/albelt/NoseData/CKPT/model_epoch_0.hdf5')
        watch = model.evaluate(x=img_test,y=label_test,batch_size=8,verbose=1)
        print(watch)
        print('test finished')
    
    def predict(self,npy_dir,ckpt_dir):
        
        img_test,label_test = self.load_data(npy_dir)
        img_test = img_test[-40:,:,:,:]
        print('test data load done,use lastest 40 samples')
        model_list = glob.glob(ckpt_dir + 'model*.hdf5')
        model_list.sort()
        print('All model checkpoint avaible:')
        for i,model_name in enumerate(model_list):
            print('epoch-{0},\t{1}'.format(i,model_name))
        choice =  int(input('Chose one,input the index:')) 
        model = load_model(model_list[choice])
        # model.save_weights(ckpt_dir + 'weight_epoch_' + str(0) + '.hdf5', True)
        # watch = model.evaluate(x=img_test,y=label_test,batch_size=8,verbose=1)
        result = model.predict(x=img_test,batch_size=8,verbose=1)
        print(result.shape)
        result_save_path = npy_dir + 'predict.npy'
        np.save(result_save_path,result)
        print('predict result saved in {0}'.format(result_save_path))




if __name__ == '__main__':
    gpu_count = 0
    myunet = myUnet(512,512)
    # myunet.train('/home/albelt/NoseData/NPY/','/home/albelt/NoseData/CKPT/',
                #  gpu_count=gpu_count,epochs=1,resume_from_lastest=False)
    # myunet.test('/home/albelt/NoseData/NPY/','/home/albelt/NoseData/CKPT/')
    myunet.predict('/home/albelt/NoseData/NPY/','/home/albelt/NoseData/CKPT/')
