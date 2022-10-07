import numpy as np
import csv
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Activation,Dropout,Embedding,InputLayer,TimeDistributed,Input,Conv1D,Conv2D,Conv3D,MaxPooling2D,AveragePooling2D,AveragePooling3D,MaxPooling3D, Flatten,BatchNormalization,LocallyConnected2D, Permute,Reshape,Add,LayerNormalization,Layer,Lambda,GlobalAveragePooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

#Tensorflowの設定
gpu_id = 0
import tensorflow
physical_devices = tensorflow.config.list_physical_devices('GPU')
tensorflow.config.list_physical_devices('GPU')
tensorflow.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
tensorflow.config.experimental.set_memory_growth(physical_devices[gpu_id], True)

record_num = 0

np.random.seed(5)

number_emsensor = 6
number_ctmarker = 12

look_back = 18

#CT像の画素サイズ
ct_reso_x = 0.738000
ct_reso_y = 0.738000
ct_reso_z = 1.204745

#データファイル読み込み
def load_datafile(filepath):
    ctmarker = []
    emsensor_pos = []
    emsensor_dir = []
    emsensor_posinterpolate = []
    
    state = 'ctmarker'
    
    f = open(filepath, "r")
    csvreader = csv.reader(f, delimiter =' ')
    
    for row in csvreader:
        #ヘッダによる読み込みモード切替
        if row[0] == 'ctmarker':
            state = 'ctmarker'
            #next(csvreader)
            continue
        elif row[0] == 'sensorposanddir':
            state = 'sensorposanddir'
            #next(csvreader)
            continue
        elif row[0] == 'sensorinterpolate':
            state = 'sensorinterpolate'
            #next(csvreader)
            continue
            
        
        #各読み込みモードでの値読み込み
        if state == 'ctmarker':
            #print(row[0])
            ctmarker_oneframe = np.array([[0.0, 0.0, 0.0]]*(len(row)//3))
            #for i in range(0, len(row)-2, 2):
            for i in range(len(row)//3):
                ctmarker_oneframe[i][0] = row[i*3]
                ctmarker_oneframe[i][1] = row[i*3+1]
                ctmarker_oneframe[i][2] = row[i*3+2]
                
            ctmarker.append(ctmarker_oneframe)
            
        elif state == 'sensorposanddir':
            emsensorpos_oneframe = np.array([[0.0, 0.0, 0.0]]*(len(row)//6))
            emsensordir_oneframe = np.array([[0.0, 0.0, 0.0]]*(len(row)//6))
            for i in range(len(row)//6):
                emsensorpos_oneframe[i][0] = row[i*6]
                emsensorpos_oneframe[i][1] = row[i*6+1]
                emsensorpos_oneframe[i][2] = row[i*6+2]
                emsensordir_oneframe[i][0] = row[i*6+3]
                emsensordir_oneframe[i][1] = row[i*6+4]
                emsensordir_oneframe[i][2] = row[i*6+5]
                
            emsensor_pos.append(emsensorpos_oneframe)
            emsensor_dir.append(emsensordir_oneframe)
    
        elif state == 'sensorinterpolate':
            emsensorposint_oneframe = np.array([[0.0, 0.0, 0.0]]*(len(row)//3))
            for i in range(len(row)//3):
                emsensorposint_oneframe[i][0] = row[i*3]
                emsensorposint_oneframe[i][1] = row[i*3+1]
                emsensorposint_oneframe[i][2] = row[i*3+2]
            
            emsensor_posinterpolate.append(emsensorposint_oneframe)
            
    f.close()
    return ctmarker, emsensor_pos, emsensor_dir, emsensor_posinterpolate


#%%
#datax_pos_res, datax_dir_res = (データ数, look_back, number_emsensor*次元数3 + 1), datax_length_res = (データ数, look_back)に変換
#yはdatay_res=(データ数, number_ctmarker*次元数3)に変換
#dataxの時系列取得範囲を0...len(datax1)にした．範囲外は0で埋める
def create_rnndata6(datax1, datax2, datax3, datay, look_back, time_interval = 0):
    if time_interval < 0:
        print('Wrong value of time_interval')
    
    datax_pos_res, datax_dir_res, datax_length_res, datay_res = [], [], [], []
    datax_mask = []#時系列の範囲外なら0のマスク
    pos_max = [0.0 for i in range(3)]
    pos_min = [1000.0 for i in range(3)]
    dir_max = [0.0 for i in range(3)]
    dir_min = [1000.0 for i in range(3)]
    length_max = 0.0
    length_min = 1000.0
    
    blank_datavalue = 0.5

    for i_start in range(0, len(datax1)):
        datax_pos_current_i = []
        datax_dir_current_i = []
        datax_length_current_i = []
        datay_current_i = []
        datax_mask_i = []
        
        #i_start-look_back+1,...,i_start+1の範囲で取り出す
        #for i in range(i_start - look_back + 1, i_start + 1):
        for i in range(i_start - look_back*(time_interval+1) + 1, i_start + 1, time_interval + 1):
            datax_pos_current_j = []
            datax_dir_current_j = []
            datax_length_current_j = []
            
            #マスク
            if 0 <= i:
                datax_mask_i.append(1)
            else:
                datax_mask_i.append(0)
                
                #仮のデータを入れておく
                for j in range(number_emsensor):
                    for k in range(3):
                        datax_pos_current_j.append(blank_datavalue)
                        datax_dir_current_j.append(blank_datavalue)
                datax_length_current_j.append(blank_datavalue)

                datax_pos_current_i.append(datax_pos_current_j)
                datax_dir_current_i.append(datax_dir_current_j)
                datax_length_current_i.append(datax_length_current_j)
                continue
                
            #座標(dimension: number_emsensor*3)
            for j in range(number_emsensor):
                if j < len(datax1[i]):
                    for k in range(3):
                        datax_pos_current_j.append(datax1[i][j][k])
                        #max, min
                        if pos_max[k] < datax1[i][j][k]:
                            pos_max[k] = datax1[i][j][k]
                        if pos_min[k] > datax1[i][j][k]:
                            pos_min[k] = datax1[i][j][k]
                else:
                    for k in range(3):
                        datax_pos_current_j.append(datax1[i][ len(datax1[i]) - 1 ][k])#センサ数が6より小さい場合は，計測できたセンサの中で最後のインデックスの座標をコピーして利用

            #センサ方向(dimension: number_emsensor*3)
            for j in range(number_emsensor):
                if j < len(datax2[i]):
                    for k in range(3):
                        #datax_current_j.append( (datax2[i][j][k] + 1.0)*np.max(datax1[i]) ) #+1.0)*np.maxは値を正にして座標とスケールをそろえるため
                        datax_dir_current_j.append(datax2[i][j][k])
                        #max,min
                        if dir_max[k] < datax2[i][j][k]:
                            dir_max[k] = datax2[i][j][k]
                        if dir_min[k] > datax2[i][j][k]:
                            dir_min[k] = datax2[i][j][k]
                else:
                    for k in range(3):
                        #datax_current_j.append( (datax2[i][ len(datax2[i]) - 1 ][k] + 1.0)*np.max(datax1[i])) #+1.0)*np.maxは値を正にして座標とスケールをそろえるため
                        datax_dir_current_j.append(datax2[i][ len(datax2[i]) - 1 ][k])
                        
            #挿入長(dimension: 1)
            length = 0.0
            for j in range(1, len(datax3[i])):
                length_temp = (datax3[i][j][0] - datax3[i][j-1][0]) * (datax3[i][j][0] - datax3[i][j-1][0]) * ct_reso_x * ct_reso_x + (datax3[i][j][1] - datax3[i][j-1][1]) * (datax3[i][j][1] - datax3[i][j-1][1]) * ct_reso_y * ct_reso_y + (datax3[i][j][2] - datax3[i][j-1][2]) * (datax3[i][j][2] - datax3[i][j-1][2]) * ct_reso_z * ct_reso_z
                length_temp = np.sqrt(length_temp)
                length += length_temp
            datax_length_current_j.append(length)
            #max, min
            if length_max < length:
                length_max = length
            if length_min > length:
                length_min = length
                                                    
            datax_pos_current_i.append(datax_pos_current_j)
            datax_dir_current_i.append(datax_dir_current_j)
            datax_length_current_i.append(datax_length_current_j)

        datax_pos_res.append(datax_pos_current_i)
        datax_dir_res.append(datax_dir_current_i)
        datax_length_res.append(datax_length_current_i)
        datax_mask.append(datax_mask_i)
        
        #datayをi_start+look_backから取り出す
        for j in range(number_ctmarker):
            if j < len(datay[i_start]):
                for k in range(3):
                    datay_current_i.append(datay[i_start][j][k])
            else:
                for k in range(3):
                    datay_current_i.append(datay[i_start][ len(datay[i_start]) - 1 ][k])#マーカ数が12より小さい場合は，計測できたマーカの中で最後のインデックスの座標をコピーして利用
        datay_res.append(datay_current_i)

    #datax_res normalize
    for i in range(len(datax_pos_res)):
        for j in range(len(datax_pos_res[i])):
            if datax_mask[i][j] == 1:
                for k in range(number_emsensor):
                    datax_pos_res[i][j][k*3+0] = (datax_pos_res[i][j][k*3+0]-pos_min[0]) / (pos_max[0]-pos_min[0])#posx
                    datax_pos_res[i][j][k*3+1] = (datax_pos_res[i][j][k*3+1]-pos_min[1]) / (pos_max[1]-pos_min[1])#posy
                    datax_pos_res[i][j][k*3+2] = (datax_pos_res[i][j][k*3+2]-pos_min[2]) / (pos_max[2]-pos_min[2])#posz
                for k in range(number_emsensor):
                    datax_dir_res[i][j][k*3+0] = (datax_dir_res[i][j][k*3+0]-dir_min[0]) / (dir_max[0]-dir_min[0])#dirx
                    datax_dir_res[i][j][k*3+1] = (datax_dir_res[i][j][k*3+1]-dir_min[1]) / (dir_max[1]-dir_min[1])#diry
                    datax_dir_res[i][j][k*3+2] = (datax_dir_res[i][j][k*3+2]-dir_min[2]) / (dir_max[2]-dir_min[2])#dirz
                datax_length_res[i][j][ 0 ] = (datax_length_res[i][j][ 0 ] - length_min) / (length_max - length_min)#length
            else:
                #マスク0なのでデータなしを示す値を入れる
                for k in range(number_emsensor):
                    datax_pos_res[i][j][k*3+0] = blank_datavalue#posx
                    datax_pos_res[i][j][k*3+1] = blank_datavalue#posy
                    datax_pos_res[i][j][k*3+2] = blank_datavalue#posz
                for k in range(number_emsensor):
                    datax_dir_res[i][j][k*3+0] = blank_datavalue#dirx
                    datax_dir_res[i][j][k*3+1] = blank_datavalue#diry
                    datax_dir_res[i][j][k*3+2] = blank_datavalue#dirz
                datax_length_res[i][j][ 0 ] = blank_datavalue#length
        
    return datax_pos_res, datax_dir_res, datax_length_res, datay_res

#推定結果ファイル書き出し
def save_datafile(filepath, ctmarker_result):
    f = open(filepath, "w")
    csvwriter = csv.writer(f, delimiter =' ', lineterminator='\n')
    for i in range(len(ctmarker_result)):
        row = [0]
        count = 0   #現在のフレームの点数
        for j in range(len(ctmarker_result[i])):
            x = ctmarker_result[i][j][0]
            y = ctmarker_result[i][j][1]
            z = ctmarker_result[i][j][2]
            if x == 0 and y == 0 and z == 0:
                break
            row.append(x)
            row.append(y)
            row.append(z)
            count = count+1
            
        row[0] = count
        csvwriter.writerow(row)
    f.close()
    

#学習と推定実行
#index_testcase: 0...6
def mainprocess(index_testcase):
    #data preparation
    ctmarker1, emsensor_pos1, emsensor_dir1, emsensor_posinterpolate1 = load_datafile('./data/colonoscope1_modified_markerordered.dat')
    ctmarker2, emsensor_pos2, emsensor_dir2, emsensor_posinterpolate2 = load_datafile('./data/colonoscope2_modified2.dat')
    ctmarker3, emsensor_pos3, emsensor_dir3, emsensor_posinterpolate3 = load_datafile('./data/colonoscope3_modified2.dat')
    ctmarker4, emsensor_pos4, emsensor_dir4, emsensor_posinterpolate4 = load_datafile('./data/colonoscope4_modified_markerordered.dat')
    ctmarker5, emsensor_pos5, emsensor_dir5, emsensor_posinterpolate5 = load_datafile('./data/colonoscope5_modified2.dat')
    ctmarker6, emsensor_pos6, emsensor_dir6, emsensor_posinterpolate6 = load_datafile('./data/colonoscope6_modified_markerordered.dat')
    ctmarker8, emsensor_pos8, emsensor_dir8, emsensor_posinterpolate8 = load_datafile('./data/colonoscope8_modified2.dat')
    ctmarker11, emsensor_pos11, emsensor_dir11, emsensor_posinterpolate11 = load_datafile('./data/colonoscope11_modified_markerordered.dat')
    
    emsensor_pos_list = []
    emsensor_pos_list.append(emsensor_pos1)
    emsensor_pos_list.append(emsensor_pos2)
    emsensor_pos_list.append(emsensor_pos3)
    emsensor_pos_list.append(emsensor_pos4)
    emsensor_pos_list.append(emsensor_pos5)
    emsensor_pos_list.append(emsensor_pos6)
    emsensor_pos_list.append(emsensor_pos8)
    emsensor_pos_list.append(emsensor_pos11)
    
    emsensor_dir_list = []
    emsensor_dir_list.append(emsensor_dir1)
    emsensor_dir_list.append(emsensor_dir2)
    emsensor_dir_list.append(emsensor_dir3)
    emsensor_dir_list.append(emsensor_dir4)
    emsensor_dir_list.append(emsensor_dir5)
    emsensor_dir_list.append(emsensor_dir6)
    emsensor_dir_list.append(emsensor_dir8)
    emsensor_dir_list.append(emsensor_dir11)
    
    emsensor_posinterpolate_list = []
    emsensor_posinterpolate_list.append(emsensor_posinterpolate1)
    emsensor_posinterpolate_list.append(emsensor_posinterpolate2)
    emsensor_posinterpolate_list.append(emsensor_posinterpolate3)
    emsensor_posinterpolate_list.append(emsensor_posinterpolate4)
    emsensor_posinterpolate_list.append(emsensor_posinterpolate5)
    emsensor_posinterpolate_list.append(emsensor_posinterpolate6)
    emsensor_posinterpolate_list.append(emsensor_posinterpolate8)
    emsensor_posinterpolate_list.append(emsensor_posinterpolate11)
    
    ctmarker_list = []
    ctmarker_list.append(ctmarker1)
    ctmarker_list.append(ctmarker2)
    ctmarker_list.append(ctmarker3)
    ctmarker_list.append(ctmarker4)
    ctmarker_list.append(ctmarker5)
    ctmarker_list.append(ctmarker6)
    ctmarker_list.append(ctmarker8)
    ctmarker_list.append(ctmarker11)
    
    #方向ベクトルの単位ベクトル化
    for data in range(len(emsensor_dir_list)):
        for frame in range(len(emsensor_dir_list[data])):
            for sensor in range(len(emsensor_dir_list[data][frame])):
                length = emsensor_dir_list[data][frame][sensor][0] * emsensor_dir_list[data][frame][sensor][0] + emsensor_dir_list[data][frame][sensor][1] * emsensor_dir_list[data][frame][sensor][1] + emsensor_dir_list[data][frame][sensor][2] * emsensor_dir_list[data][frame][sensor][2]
                length = np.sqrt(length)
                emsensor_dir_list[data][frame][sensor][0] /= length
                emsensor_dir_list[data][frame][sensor][1] /= length
                emsensor_dir_list[data][frame][sensor][2] /= length
    
    
    datax_pos_list = []
    datax_dir_list = []
    datax_length_list = []
    datay_list = []
    for i in range(len(emsensor_dir_list)):
        #interval0
        datax_pos_temp, datax_dir_temp, datax_length_temp, datay_temp = create_rnndata6(emsensor_pos_list[i], emsensor_dir_list[i], emsensor_posinterpolate_list[i], ctmarker_list[i], look_back, 0)

        datax_pos_temp = np.array(datax_pos_temp).reshape(len(datax_pos_temp), look_back, number_emsensor*3) #number_emsensor*次元数3 + ...
        datax_dir_temp = np.array(datax_dir_temp).reshape(len(datax_dir_temp), look_back, number_emsensor*3) #number_emsensor*次元数3 + ...
        datax_length_temp = np.array(datax_length_temp).reshape(len(datax_length_temp), look_back)
        datay_temp = np.array(datay_temp).reshape(len(datay_temp), number_ctmarker*3) #number_ctmarker*次元数3
        datax_pos_list.append(datax_pos_temp)
        datax_dir_list.append(datax_dir_temp)
        datax_length_list.append(datax_length_temp)
        datay_list.append(datay_temp)
    
    
    #datayを相対位置に変換
    #肛門位置を基準にする
    datay_basepos = [ emsensor_posinterpolate_list[index_testcase][0][len(emsensor_posinterpolate_list[index_testcase][0])-1][0], emsensor_posinterpolate_list[index_testcase][0][len(emsensor_posinterpolate_list[index_testcase][0])-1][1], emsensor_posinterpolate_list[index_testcase][0][len(emsensor_posinterpolate_list[index_testcase][0])-1][2] ]
    
    for i in range(len(datay_list)):
        for j in range(len(datay_list[i])):
            for k in range(number_ctmarker):
                for l in range(3):
                    datay_list[i][j][k*3+l] -= datay_basepos[l]
    
    #datayを0-1に正規化
    datay_max_list = []
    datay_min_list = []
    
    for i in range(len(datay_list)):
        datay_max = [0.0, 0.0, 0.0]
        datay_min = [1000.0, 1000.0, 1000.0]
        
        for j in range(len(datay_list[i])):
            for k in range(number_ctmarker):
                for l in range(3):
                    if datay_max[l] < datay_list[i][j][k*3+l]:
                        datay_max[l] = datay_list[i][j][k*3+l]
                    if datay_min[l] > datay_list[i][j][k*3+l]:
                        datay_min[l] = datay_list[i][j][k*3+l]
    
        for j in range(len(datay_list[i])):
            for k in range(number_ctmarker):
                for l in range(3):
                    datay_list[i][j][k*3+l] = (datay_list[i][j][k*3+l]-datay_min[l]) / (datay_max[l]-datay_min[l])
                    
        datay_max_list.append(datay_max)
        datay_min_list.append(datay_min)
        
    
    #train
    index_initialcase = 0
    index_range_from = 1
    index_range_to = len(emsensor_dir_list)
    
    if index_testcase == 0:
        index_initialcase = 1
        index_range_from = 2
        index_range_to = len(emsensor_dir_list)
        
    trainx_pos = datax_pos_list[index_initialcase]
    trainx_dir = datax_dir_list[index_initialcase]
    trainx_length = datax_length_list[index_initialcase]
    trainy = datay_list[index_initialcase]
    
    for i in range(index_range_from, index_range_to):
        if i != index_testcase:
            trainx_pos = np.concatenate([trainx_pos, datax_pos_list[i]], axis=0)
            trainx_dir = np.concatenate([trainx_dir, datax_dir_list[i]], axis=0)
            trainx_length = np.concatenate([trainx_length, datax_length_list[i]], axis=0)
            trainy = np.concatenate([trainy, datay_list[i]], axis=0)
    
    #test
    testx_pos = datax_pos_list[index_testcase]
    testx_dir = datax_dir_list[index_testcase]
    testx_length = datax_length_list[index_testcase]
    testy = datay_list[index_testcase]
    
    
    #trainx_pos, trainx_dir表示
    n = 20
    plt.figure(figsize=(40, 4))
    for i in range(n):
       #distance
       ax = plt.subplot(2, n, i+1)
       plt.imshow(trainx_pos[50].reshape(look_back, number_emsensor*3))
       plt.gray()
       ax.get_xaxis().set_visible(False)
       ax.get_yaxis().set_visible(False)
       
       #direction
       ax = plt.subplot(2, n, i+1+n)
       plt.imshow(trainx_dir[50].reshape(look_back, number_emsensor*3))
       plt.gray()
       ax.get_xaxis().set_visible(False)
       ax.get_yaxis().set_visible(False)
    plt.show()
    
    
    #model definition
    #参考
    #https://qiita.com/T-STAR/items/dcaa7873a6d193912ed1
    dense_kwargs = {
        'kernel_initializer':'glorot_normal',
        'bias_initializer': tensorflow.keras.initializers.RandomNormal(stddev=1e-2),
        'kernel_regularizer':tensorflow.keras.regularizers.l2(0.0)
        }
    
    class STMix(Layer):
        def __init__(
            self,
            num_patches: int,
            channel_dim: int,
            patch_mix_hidden_dim: int,
            channel_mix_hidden_dim: int,
            dropout_rate: float = 0.0,
            stoch_depth: float = 0.1,
            **kwargs
        ):
            super(STMix, self).__init__(**kwargs)
            
            self.num_patches = num_patches
            self.channel_dim = channel_dim
            self.patch_mix_hidden_dim = patch_mix_hidden_dim
            self.channel_mix_hidden_dim = channel_mix_hidden_dim
            self.dropout_rate = dropout_rate
            self.stoch_depth = stoch_depth
            
            self.norm = LayerNormalization()
            self.permute = Permute((2, 1))
            self.activataion = Activation(tensorflow.keras.activations.gelu)
            self.dense1 = Dense(self.patch_mix_hidden_dim, **dense_kwargs)
            self.dense2 = Dense(self.num_patches, **dense_kwargs)
            self.dense3 = Dense(self.channel_mix_hidden_dim, **dense_kwargs)
            self.dense4 = Dense(self.channel_dim, **dense_kwargs)
            self.drop = Dropout(rate=self.dropout_rate)
            self.drop_stoch = Dropout(rate=self.stoch_depth, noise_shape=(None,1,1))
            self.add = Add()
            
        def call(self, inputs):
            x = inputs
            x_skip = x
            x = self.norm(x)
            x = self.permute(x)
            x = self.dense1(x)
            x = self.activataion(x)
            x = self.drop(x)
            x = self.dense2(x)
            x = self.permute(x)
            x = self.drop_stoch(x)
            x = self.add([x, x_skip])
            
            x_skip = x
            x = self.norm(x)
            x = self.dense3(x)
            x = self.activataion(x)
            x = self.drop(x)
            x = self.dense4(x)
            x = self.drop_stoch(x)
            x = self.add([x, x_skip])
            
            return x
        
        def compute_output_shape(self, input_shape):
            return input_shape
        
        def get_config(self):
            config = super(STMix, self).get_config()
            config.update({
                'num_patches': self.num_patches,
                'channel_dim': self.channel_dim,
                'patch_mix_hidden_dim': self.patch_mix_hidden_dim,
                'channel_mix_hidden_dim': self.channel_mix_hidden_dim,
                'dropout_rate': self.dropout_rate,
                'stoch_depth': self.stoch_depth,
            })
            return config
    
    
    #STMixによる推定モデル
    def model_colonshapeestimate6(number_emsensor, number_ctmarker):
        patch_size = 6
        patch_size_time = 3
        num_blocks = 7
        hidden_dim = 9
        patch_dim = 64
        channel_dim = 128
        
        input_pos_shape = (look_back, number_emsensor*3, 1)
        input_pos_height, input_pos_width, _ = input_pos_shape #Permute((2,1,3))で入れ替えるからh,wの順
        pos_num_patches = (input_pos_width*input_pos_height)//(patch_size*patch_size_time)
        
        input_dir_shape = (look_back, number_emsensor*3, 1)
        input_dir_height, input_dir_width, _ = input_dir_shape #Permute((2,1,3))で入れ替えるからh,wの順
        dir_num_patches = (input_dir_width*input_dir_height)//(patch_size*patch_size_time)
        
        ##pos
        input_pos = Input(input_pos_shape, name='input_pos')
        x_pos = Permute((2, 1, 3))(input_pos)
        
        x_pos = Conv2D(hidden_dim, kernel_size=(patch_size, patch_size_time), strides=(patch_size, patch_size_time), padding='same')(x_pos)
        x_pos = Reshape([-1, hidden_dim])(x_pos)
        
        ##dir
        input_dir = Input(input_dir_shape, name='input_dir')
        x_dir = Permute((2, 1, 3))(input_dir)

        x_dir = Conv2D(hidden_dim, kernel_size=(patch_size, patch_size_time), strides=(patch_size, patch_size_time), padding='same')(x_dir)
        x_dir = Reshape([-1, hidden_dim])(x_dir)
        
        
        x = concatenate([x_pos, x_dir], axis=-2)
        
        for _ in range(num_blocks):
            x = STMix(num_patches=pos_num_patches*2, channel_dim=hidden_dim, 
                      patch_mix_hidden_dim=patch_dim, 
                      channel_mix_hidden_dim=channel_dim,
                      dropout_rate=0.1, stoch_depth=0.1)(x)
        
        #length
        input_length = Input(shape=(look_back), name='input_length')
        x_length = input_length
        x_length = Dense(look_back//2)(x_length)
        x_length = Dense(4)(x_length)
        

        x = Flatten()(x)
        x = concatenate([x, x_length], axis=-1)
        x = Dense(256)(x)
        x = Dropout(0.3)(x)
        x = Dense(128)(x)
        x = Dropout(0.3)(x)
        
        x = Dense(number_ctmarker*3, activation='linear', name='output_main')(x)
        
        model = Model(inputs=[input_pos, input_dir, input_length], outputs=x)
        
        return model

            
    
    model = model_colonshapeestimate6(number_emsensor, number_ctmarker)
    
    #model.summary()
    
    
    model.compile(loss={'output_main': 'mean_squared_error'}, optimizer='adam')
    
    #train
    training = model.fit({'input_pos': trainx_pos, 'input_dir': trainx_dir, 'input_length': trainx_length}, 
                         {'output_main': trainy}, 
                         epochs=200, batch_size=50, 
                         validation_data=({'input_pos': testx_pos, 'input_dir': testx_dir, 'input_length': testx_length}, 
                                          {'output_main': testy}
                                          ), 
                         shuffle=True, verbose=1)
    
    
    #test
    start_time = time.perf_counter()
    estimated = model.predict({'input_pos': testx_pos, 'input_dir': testx_dir, 'input_length': testx_length})
    end_time = time.perf_counter()
    
    
    #学習履歴表示
    def plot_history(history):
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.legend(['loss', 'val_loss'])
        plt.show()
        
    plot_history(training.history)
    
    
    #推定座標表示
    plt.figure()
    plt.plot(range(0, len(estimated)), estimated[0:len(estimated), 1], label="estimated")
    plt.plot(range(0, len(testy)), testy[0:len(testy), 1], label="original")
    plt.legend()
    plt.show()
    
    
    #%%
    #推定結果の座標の0-1正規化を戻す
    for i in range(len(estimated)):
        for j in range(number_ctmarker):
            for k in range(3):
                estimated[i][j*3+k] = estimated[i][j*3+k] * (datay_max_list[index_testcase][k]-datay_min_list[index_testcase][k]) + datay_min_list[index_testcase][k]
    
    #推定結果の座標を相対位置から絶対位置に戻す
    for i in range(len(estimated)):
        for j in range(number_ctmarker):
            for k in range(3):
                estimated[i][j*3+k] += datay_basepos[k]
    
    
    #推定結果をファイル書き出し用に並べ替え
    ctmarker_result = []
    for i in range(len(estimated)):
        ctmarker_oneframe = np.array([[0.0, 0.0, 0.0]]*number_ctmarker)
        for j in range(number_ctmarker):
            ctmarker_oneframe[j][0] = estimated[i][j*3 + 0]
            ctmarker_oneframe[j][1] = estimated[i][j*3 + 1]
            ctmarker_oneframe[j][2] = estimated[i][j*3 + 2]
    
        ctmarker_result.append(ctmarker_oneframe)
    
    if index_testcase == 0:
        save_datafile('./result/colonoscope1_estimate_miccai2022_%d.dat'%(record_num), ctmarker_result)
    elif index_testcase == 1:
        save_datafile('./result/colonoscope2_estimate_miccai2022_%d.dat'%(record_num), ctmarker_result)
    elif index_testcase == 2:
        save_datafile('./result/colonoscope3_estimate_miccai2022_%d.dat'%(record_num), ctmarker_result)
    elif index_testcase == 3:
        save_datafile('./result/colonoscope4_estimate_miccai2022_%d.dat'%(record_num), ctmarker_result)
    elif index_testcase == 4:
        save_datafile('./result/colonoscope5_estimate_miccai2022_%d.dat'%(record_num), ctmarker_result)
    elif index_testcase == 5:
        save_datafile('./result/colonoscope6_estimate_miccai2022_%d.dat'%(record_num), ctmarker_result)
    elif index_testcase == 6:
        save_datafile('./result/colonoscope8_estimate_miccai2022_%d.dat'%(record_num), ctmarker_result)
    elif index_testcase == 7:
        save_datafile('./result/colonoscope11_estimate_miccai2022_%d.dat'%(record_num), ctmarker_result)
    
    
    # 推定に要した時間表示(秒)
    elapsed_time = end_time - start_time
    print("Time of prediction from " + repr(len(testx_pos)) + " data was " + repr(elapsed_time))
    
    #誤差算出
    error = 0.0
    errorcount = 0
    for frame in range(len(ctmarker_list[index_testcase])):
        for marker in range(number_ctmarker):
            if len(ctmarker_list[index_testcase][frame]) <= marker:
                continue
            truth_x = ctmarker_list[index_testcase][frame][marker][0]
            truth_y = ctmarker_list[index_testcase][frame][marker][1]
            truth_z = ctmarker_list[index_testcase][frame][marker][2]
            #print("frame" + repr(frame) + " marker" + repr(marker))
            estimate_x = ctmarker_result[frame][marker][0]
            estimate_y = ctmarker_result[frame][marker][1]
            estimate_z = ctmarker_result[frame][marker][2]
            if estimate_x == 0 and estimate_y == 0 and estimate_z == 0:
                continue
            
            error_temp = (truth_x - estimate_x)*(truth_x - estimate_x)*ct_reso_x*ct_reso_x + (truth_y - estimate_y)*(truth_y - estimate_y)*ct_reso_y*ct_reso_y + (truth_z - estimate_z)*(truth_z - estimate_z)*ct_reso_z*ct_reso_z
            error_temp = np.sqrt(error_temp)
            error += error_temp
            errorcount += 1
    
    if errorcount > 0:
        print("Marker estimation error (mm): " + repr(error/errorcount))
    else:
        print("Marker estimation error: no marker pair of truth and estimated found.")
        
    return error/errorcount


errorlist = [0.0 for i in range(8)]

for testcase in range(8):
    errorlist[testcase] = mainprocess(testcase)

print("Errors: " + repr(errorlist))
average = 0.0
averagecount = 0
for testcase in range(8):
    average += errorlist[testcase]
    averagecount += 1
print("Average: " + repr(average/averagecount))
