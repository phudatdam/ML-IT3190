import os
import json
import sys
import glob
from sklearn.model_selection import train_test_split
import random
# change your data path
data_dir = '/home/adminn/theanh28/SSDG-CVPR2020/data/origin/'


def msu_process():
    test_list = []
    # data_label for msu
    for line in open('/home/adminn/theanh28/SSDG-CVPR2020/data/origin/MSU-MFSD/test_sub_list.txt', 'r'):
        test_list.append(line[0:9])
    train_list = []
    for line in open('/home/adminn/theanh28/SSDG-CVPR2020/data/origin/MSU-MFSD/train_sub_list.txt', 'r'):
        train_list.append(line[0:9])
    print(test_list)
    print(train_list)
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './msu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    
    dataset_path = '/home/adminn/theanh28/SSDG-CVPR2020/data/origin/MSU-MFSD/pics/'
    
    print(f'\nDataset path: {dataset_path}')
    # Quét tất cả file .png đệ quy và sắp xếp
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    path_list.sort()
    
    print(f'Number of MSU images: {len(path_list)}')
    
    for i in range(len(path_list)):
        flag = path_list[i].find('/real/')
        if(flag != -1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        video_num = path_list[i].split('/')[-1].split('_')[1]
        if (video_num in train_list):
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)
        all_final_json.append(dict)
        if(label == 1):
            real_final_json.append(dict)
        else:
            fake_final_json.append(dict)
    print('\nMSU: ', len(path_list))
    print('MSU(train): ', len(train_final_json))
    print('MSU(test): ', len(test_final_json))
    print('MSU(all): ', len(all_final_json))
    print('MSU(real): ', len(real_final_json))
    print('MSU(fake): ', len(fake_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()


def casia_process():
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './casia/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    dataset_path = data_dir + 'CASIA-FASD/'
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].split('/')[-1].split('_')[3]
        #breakpoint()
        if (flag == 'real.jpg'):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        flag = path_list[i].split('/')[-3].split('_')[0]
        if (flag == 'train'):
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)
        all_final_json.append(dict)
        if (label == 1):
            real_final_json.append(dict)
        else:
            fake_final_json.append(dict)
    print('\nCasia: ', len(path_list))
    print('Casia(train): ', len(train_final_json))
    print('Casia(test): ', len(test_final_json))
    print('Casia(all): ', len(all_final_json))
    print('Casia(real): ', len(real_final_json))
    print('Casia(fake): ', len(fake_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def replay_process():
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './replay/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    dataset_path = data_dir + 'replay_256/'
    path_list = glob.glob(dataset_path + '**/*.png', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].find('/real/')
        if (flag != -1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        if (path_list[i].find('/replayattack-train/') != -1):
            train_final_json.append(dict)
        elif(path_list[i].find('/replayattack-devel/') != -1):
            valid_final_json.append(dict)
        else:
            test_final_json.append(dict)
        if(path_list[i].find('/replayattack-devel/') != -1):
            continue
        else:
            all_final_json.append(dict)
            if (label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
    print('\nReplay: ', len(path_list))
    print('Replay(train): ', len(train_final_json))
    print('Replay(valid): ', len(valid_final_json))
    print('Replay(test): ', len(test_final_json))
    print('Replay(all): ', len(all_final_json))
    print('Replay(real): ', len(real_final_json))
    print('Replay(fake): ', len(fake_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def oulu_process():
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './oulu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    dataset_path = data_dir + 'OULU-NPU/'
    path_list = glob.glob(dataset_path + '**/*.png', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].split('/')[-2]
        if (flag == 'real'):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        if (path_list[i].split('/')[-3]=='Session1' or path_list[i].split('/')[-3]=='Session2'):
            train_final_json.append(dict)
        elif(path_list[i].split('/')[-3]=='Session3'):
            valid_final_json.append(dict)
        else:
            test_final_json.append(dict)
        if(path_list[i].split('/')[-3]=='Session3'):
            continue
        else:
            all_final_json.append(dict)
            if (label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
    val_size = 0.7 

        # Sử dụng train_test_split
    valid_final_json, test_final_json = train_test_split(
            valid_final_json,           # Danh sách dữ liệu đầu vào
            test_size=val_size,  # Tỷ lệ dữ liệu dành cho tập Test (hoặc Val)
            random_state=42      # Seed ngẫu nhiên (để đảm bảo kết quả chia nhất quán)
        )
    print('\nOulu: ', len(path_list))
    print('Oulu(train): ', len(train_final_json))
    print('Oulu(valid): ', len(valid_final_json))
    print('Oulu(test): ', len(test_final_json))
    print('Oulu(all): ', len(all_final_json))
    print('Oulu(real): ', len(real_final_json))
    print('Oulu(fake): ', len(fake_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()


if __name__=="__main__":
    #msu_process()
    #casia_process()
    #replay_process()
    oulu_process()