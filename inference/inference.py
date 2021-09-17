import os
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import joblib
import sys
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import xgboost as xgb
import pickle
from PIL import Image
#sys.path.append('.')
from inference import clip_cycle
from inference import stft_code
from inference import wavelet_code

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


'''
parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=200)
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--size_m', type=int, default=128)
parser.add_argument('--mixup', type=eval, default=True, choices=[True, False])
parser.add_argument('--nonLocal', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=18)
parser.add_argument('--step', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--save', type=str, default='./experiment2')
parser.add_argument('--comment', type=str, default='./experiment1')
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=1)
'''
#parser.add_argument('--input', '-i', default='C:/Users/dell/Desktop/train/analysis/mixup/wavelet_stft_train.p', type=str, help='path to directory with input data archives')
#parser.add_argument('--test', default='C:/Users/dell/Desktop/train/analysis/mixup/wavelet_stft_test.p', type=str, help='path to directory with test data archives')
#parser.add_argument('--input', '-i', default='./inference_data.p', type=str, help='path to directory with input data archives')
#parser.add_argument('--test', default='C:/Users/dell/Desktop/split0.8/analysis/pack/wavelet_stft_location_test.p', type=str, help='path to directory with test data archives')
#args = parser.parse_args()

#%%

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class nonLocal(nn.Module):
    
    def __init__(self, inplanes, planes):
        super(nonLocal, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.theta = conv1x1(inplanes, planes, stride=1)
        self.phi = conv1x1(inplanes, planes, stride=1)
        self.g = conv1x1(inplanes, planes, stride=1)
        self.final = conv1x1(planes, inplanes, stride=1)
        
    def forward(self,x):
        old = x
        H = list(old.size())[-1]
        batch = list(old.size())[0]
        mid_theta = self.theta(x) #[batch_number,channel,H,W]
        mid_phi = self.phi(x)
        mid_g = self.g(x)
        paste = torch.empty(1,self.planes,H,H,device='cuda')
        for i in range(batch):
            i_mid_theta = mid_theta[i].reshape(self.planes,-1)#[channel,HW]
            i_mid_phi = mid_phi[i].reshape(self.planes,-1).t()#[HW,channel]
            i_mid_g = mid_g[i].reshape(self.planes,-1).t()
        
            mid_tp = torch.mm(i_mid_phi,i_mid_theta)
            HW = list(mid_tp.size())[0]
            mid_tp = mid_tp.view(-1)
            output_tp = torch.nn.functional.softmax(mid_tp)
            
            output_tp = output_tp.reshape(HW,HW)
            output = torch.mm(output_tp,i_mid_g).t()
            
            cat_output = output[0].reshape([H,H]).t()
            for i in range(1, len(output)):
                output_mid = output[i].reshape([H,H]).t()
                cat_output = torch.cat([cat_output,output_mid])
                
        
            cat_output = cat_output.reshape([1,-1,H,H])
            paste = torch.cat([paste,cat_output])
        paste = paste[1:]
        paste = self.final(paste)
        paste = paste+old
        return paste

class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.droupout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x
#        print("input:"+str(x.size()))
        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.droupout(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.droupout(out)
#        print("output:"+str(out.size()))
        return out + shortcut


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class BiResNet(nn.Module):
    
    def __init__(self,dim):
        super(BiResNet, self).__init__()
        self.conv0 = nn.Conv2d(1, 64, 3, 1)
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.ResNet_0_0 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_0_1 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_1_0 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_1_1 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_0 = ResBlock(64, 64)
        self.ResNet_1 = ResBlock(64, 64)
        self.ResNet_2 = ResBlock(64, 64)
        self.ResNet_3 = ResBlock(64, 64)
        self.ResNet_4 = ResBlock(64, 64)
        self.ResNet_5 = ResBlock(64, 64)
        self.ResNet_6 = ResBlock(64, 64)
        self.ResNet_7 = ResBlock(64, 64)
        self.ResNet_8 = ResBlock(64, 64)
        self.ResNet_9 = ResBlock(64, 64)
        self.ResNet_10 = ResBlock(64, 64)
        self.ResNet_11 = ResBlock(64, 64)
        self.ResNet_12 = ResBlock(64, 64)
        self.ResNet_13 = ResBlock(64, 64)
   
        self.nonLocal1 = nonLocal(64, 32)
        self.nonLocal2 = nonLocal(64, 32)
        self.nonLocal3 = nonLocal(64, 32)
        self.nonLocal4 = nonLocal(64, 32)

        self.norm0 = norm(dim)
        self.norm1 = norm(dim)
        self.relu0 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool0 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64*2+7, 4)
        self.dropout = nn.Dropout(0.1)
        self.flat = Flatten()
        
        
    def forward(self,stft,mfcc,location):
        
        out_s = self.conv0(stft)
        out_s = self.ResNet_0_0(out_s)
        out_s = self.ResNet_0_1(out_s)
        out_s = self.ResNet_0(out_s)
        out_s = self.ResNet_2(out_s)
        out_s = self.ResNet_4(out_s)
        out_s = self.ResNet_6(out_s)

        out_s = self.ResNet_8(out_s)

        if args.nonLocal:
            out_s = self.nonLocal1(out_s)
            out_s = self.relu0(out_s)
        out_s = self.ResNet_10(out_s)
        out_s = self.ResNet_12(out_s)

        out_s = self.norm0(out_s)
        out_s = self.relu2(out_s)
        out_s = self.pool0(out_s)

        
        out_m = self.conv1(mfcc)
        out_m = self.ResNet_1_0(out_m)
        out_m = self.ResNet_1_1(out_m)
        out_m = self.ResNet_1(out_m)
        out_m = self.ResNet_3(out_m)
        out_m = self.ResNet_5(out_m)
        out_m = self.ResNet_7(out_m)
        out_m = self.ResNet_9(out_m)
        
        if args.nonLocal:
            out_m = self.nonLocal2(out_m)
            out_m = self.relu1(out_m)
            
        out_m = self.ResNet_11(out_m)
        out_m = self.ResNet_13(out_m)

        out_m = self.norm1(out_m)
        out_m = self.relu3(out_m)
        out_m = self.pool1(out_m)
        #print (out_m.shape)
        #print (out_s.shape)
        
        out = torch.cat((out_s,out_m),1)


        out = self.flat(out)
        out=  torch.cat((out,location),1)
        out = self.linear(out)
        out = self.dropout(out)
           
        return out
        
    def get_code(self,stft,mfcc,location):
        
        out_s = self.conv0(stft)
        out_s = self.ResNet_0_0(out_s)
        out_s = self.ResNet_0_1(out_s)
        out_s = self.ResNet_0(out_s)
        out_s = self.ResNet_2(out_s)
        out_s = self.ResNet_4(out_s)
        out_s = self.ResNet_6(out_s)

        out_s = self.ResNet_8(out_s)

        #if args.nonLocal:
        out_s = self.nonLocal1(out_s)
        out_s = self.relu0(out_s)
        
        out_s = self.ResNet_10(out_s)
        out_s = self.ResNet_12(out_s)

        out_s = self.norm0(out_s)
        out_s = self.relu2(out_s)
        out_s = self.pool0(out_s)

        
        out_m = self.conv1(mfcc)
        out_m = self.ResNet_1_0(out_m)
        out_m = self.ResNet_1_1(out_m)
        out_m = self.ResNet_1(out_m)
        out_m = self.ResNet_3(out_m)
        out_m = self.ResNet_5(out_m)
        out_m = self.ResNet_7(out_m)
        out_m = self.ResNet_9(out_m)
        
        #if args.nonLocal:
        out_m = self.nonLocal2(out_m)
        out_m = self.relu1(out_m)
            
        out_m = self.ResNet_11(out_m)
        out_m = self.ResNet_13(out_m)

        out_m = self.norm1(out_m)
        out_m = self.relu3(out_m)
        out_m = self.pool1(out_m)
        #print (out_m.shape)
        #print (out_s.shape)
        
        out = torch.cat((out_s,out_m),1)


        out = self.flat(out)
        out=  torch.cat((out,location),1)
        # out = self.linear(out)
        # out = self.dropout(out)
       
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class myDataset(data.Dataset):
    def __init__(self, stft, mfcc,location):
        self.stft = stft
        self.mfcc = mfcc
        self.location=location
        #self.targets = targets

    def __getitem__(self, index):
               
        sample_stft = self.stft[index]
        sample_mfcc = self.mfcc[index]
        sample_location= self.location[index]
        #target = self.targets[index]
        #target = torch.from_numpy(target)
        
        min_s = np.min(sample_stft)
        max_s = np.max(sample_stft)
        sample_stft = (sample_stft-min_s)/(max_s-min_s) 
        min_m = np.min(sample_mfcc)
        max_m = np.max(sample_mfcc)
        sample_mfcc = (sample_mfcc-min_m)/(max_m-min_m) 
        
        output_stft = torch.FloatTensor([sample_stft])
        crop_s = transforms.Resize([128,128])
        img_s = transforms.ToPILImage()(output_stft)
        croped_img=crop_s(img_s)
        output_stft = transforms.ToTensor()(croped_img)
        
        output_mfcc = torch.FloatTensor([sample_mfcc])
        crop_m = transforms.Resize([128,128])
        img_m = transforms.ToPILImage()(output_mfcc)
        croped_img_m=crop_m(img_m)
        output_mfcc = transforms.ToTensor()(croped_img_m)
        sample_location=torch.tensor(sample_location)
#        print(output.size())

        return output_stft,output_mfcc,sample_location
    def __len__(self):
        return len(self.mfcc)

def turn_numpy(labels):
    mid = np.array([labels[0]]) 
    #print (mid)    
    new_labels = one_hot(mid, 4)
    for i in range(1,len(labels)):
        mid = np.array([labels[i]])        
        new_labels = np.append(new_labels, one_hot(mid, 4), axis=0)
    new_labels = list(new_labels)
    #new_labels = [[0,0,0,1],[0,0,0,1],[0,0,0,1]...]    
    return new_labels

def shuffle(stft,labels):
    bond = random.shuffle(list(zip(stft,turn_numpy(labels))))
    stft = []
    new_labels = []
    for i in bond:
        stft.append(i[0])
        new_labels.append(i[1])    
    return stft, new_labels

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

GLOBAL_SEED=1
GLOBAL_WORKER_ID = None

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

        
def get_loaders( batch_size=1, perc=1.0):
    stft, mfcc,location= joblib.load(open('../inference/inference_data.p', mode='rb'))
    #print (stft)
    #print (labels)
    #print (type(location[-1]))
    #labels = turn_numpy(labels)
#    stft, labels = shuffle(stft,labels)
        

    data_loader = DataLoader(
        myDataset(stft,mfcc,location), batch_size=batch_size,
        shuffle=False, num_workers=2, drop_last=True, worker_init_fn=worker_init_fn
    )


    #print (len(data_loader))

    return data_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()



def one_hot(x, K):
    #x is a array from np
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


    


def confusion_matrix_tree(model, dataset_loader,xgb):
    targets = []
    outputs = []

    for stft,mfcc,location, y in dataset_loader:
        location=location.to(torch.float32)
        stft, mfcc,location= stft.cuda(), mfcc.cuda(),location.cuda()
        target_class = np.argmax(y, axis=1)
        targets = np.append(targets,target_class)
        with torch.no_grad():
            feature = model.get_code(stft,mfcc,location)
        feature=feature.detach().cpu().numpy() 
        predicted_class=xgb.predict(feature)         
        #predicted_class = np.argmax(logits.cpu().detach().numpy(), axis=1)
        outputs = np.append(outputs,predicted_class)
    #print (targets.tolist())   
    Confusion_matrix=sk_confusion_matrix(targets.tolist(), outputs.tolist())
    print('Confusion_matrix:')
    print(Confusion_matrix)
    Se = Confusion_matrix[0][0]/(sum(Confusion_matrix[0]))
    Sq = (Confusion_matrix[1][1]+Confusion_matrix[2][2]+Confusion_matrix[3][3])/(sum(Confusion_matrix[1])+sum(Confusion_matrix[2])+sum(Confusion_matrix[3]))
    return Se, Sq, (Se+Sq)/2
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def inference_forward(model, dataset_loader,xgb):
    #total_correct = 0
    test_features=[]
    predicted_class_list=[]
    #test_labels=[]
    #print (dataset_loader)
    for stft, mfcc,location in dataset_loader:
        #print (location)
        location=location.to(torch.float32)
        stft, mfcc,location= stft.cuda(), mfcc.cuda(),location.cuda()
        #target_class = np.argmax(y.numpy(), axis=1)
        with torch.no_grad():
            feature = model.get_code(stft,mfcc,location)
        feature=feature.detach().cpu().numpy() 
        test_features.append(feature)
        #test_labels.append(target_class)
        predicted_class=xgb.predict(feature)        
        predicted_class_list.append(int(predicted_class[0]))

    return predicted_class_list


def pack(dir_stft,dir_wavelet):       
    feature_stft_list=[]
    feature_wavelet_list=[]
    #label_list=[]
    locations=[]
    for file in os.listdir(dir_stft):
        I_stft = Image.open(dir_stft+file).convert('L')
        I_wavelet = Image.open(dir_wavelet+file).convert('L')
        location= file[8:10]
        #print(location)
        if location=="Tc":
            location=[1,0,0,0,0,0,0]
        elif location=="Al":
            location=[0,1,0,0,0,0,0]
        elif location=="Ar":
            location=[0,0,1,0,0,0,0]
        elif location=="Pl":
            location=[0,0,0,1,0,0,0]
        elif location=="Pr":
            location=[0,0,0,0,1,0,0]
        elif location=="Ll":
            location=[0,0,0,0,0,1,0]
        else:
            location=[0,0,0,0,0,0,1]
        #print(location)
        I_stft = np.array(I_stft)
        I_wavelet = np.array(I_wavelet)
        locations.append(location)
        feature_wavelet_list.append(I_wavelet)
        feature_stft_list.append(I_stft)
        #label_list.append(label)
    return feature_stft_list,feature_wavelet_list,locations
    

def process_rawdata(file):
    #print (file)
    clip_cycle.clip_cycle(file,"../inference/cycle/")
    
    stft_code.main("../inference/cycle/")
    #print (111)
    wavelet_code.main("../inference/cycle/")
    
    stft,wavelet,location  = pack("../inference/stft/",'../inference/wavelet/')

    joblib.dump((stft,wavelet,location), open('../inference/inference_data.p', 'wb'))


def main(file):

    process_rawdata(file)
    
    RESUME=True
    

    use_cuda = torch.cuda.is_available()

    batch_size = 1
    net = BiResNet(64)
    
    if use_cuda:
    # data parallel
        n_gpu = torch.cuda.device_count()
        batch_size *= n_gpu
        net = torch.nn.DataParallel(net)
        net.cuda()
        
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')
    

    pos_weight = torch.tensor([1.254,1,2.16,2.215]).cuda()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)
    test_loader = get_loaders(batch_size)
    #print (1111111111)
    start_epoch=-1
    
    if RESUME:
        path_checkpoint = '../inference/model.pth' # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        #optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
    


    if isinstance(net,torch.nn.DataParallel):
        net = net.module
    net.cuda()
    net.eval()       
    xgb=pickle.load(open("../inference/xgboost.dat",'rb'))
    #xgb=train_tree(net, train_loader)
    #train_acc = accuracy_tree(net, train_loader,xgb)
    #train_loss = Loss(net, train_loader,xgb)
    pred_class = inference_forward(net, test_loader,xgb)
    #print (pred_class)
    labels=['normal','crackles', 'wheezes', 'both']
    final_labels=[]
    for item in pred_class:
        final_labels.append(labels[item])
    
    return final_labels
    #print (final_labels)
    #val_loss = Loss(net, test_loader,xgb)
    #scheduler.step()
###############################################################################
        
    #writer.add_scalar('train/loss',train_loss,epoch)
    #writer.add_scalar('test/loss',val_loss,epoch)
    #writer.add_scalar('train/acc',train_acc,epoch)
    #writer.add_scalar('test/acc',val_acc,epoch)
    #test_Se,test_Sq,test_Score = confusion_matrix_tree(net, test_loader,xgb)
    #train_Se,train_Sq,train_Score = confusion_matrix_tree(net, train_loader,xgb)

