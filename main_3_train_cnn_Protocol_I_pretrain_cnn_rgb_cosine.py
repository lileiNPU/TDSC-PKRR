import numpy as np
import os
#gpus = [0]
# 设置CUDA_VISIBLE_DEVICES
#os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(f) for f in gpus)
import torch
import torch.nn as nn
import torch.optim
from torchvision import transforms
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import math
import scipy.io as scio
import cv2
import sys
sys.path.append('core')
from raft import Basic_timesformer_mse_multiscale_cnn
from utils import flow_viz
from utils.utils import InputPadder
import argparse
import skimage.feature
import skimage.segmentation
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_txtlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]


class Data_myself(Dataset):

    def __init__(self, listroot=None, labelroot=None, shuffle=True):
        self.listroot = listroot
        self.labelroot = labelroot
        self.transform = transforms.ToTensor()
        listfile_root = self.listroot#os.path.join(self.listroot, 'train_img_label.txt')

        with open(listfile_root, 'r') as file:
            self.lines = file.readlines()
        if shuffle:
            random.shuffle(self.lines)
        # self.nSamples = len(self.lines[:30]) if debug else len(self.lines)
        self.nSamples = len(self.lines)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        imgpath_labelpath = self.lines[index].rstrip()
        img_rgb_all, label_texture, label_pixel = self.load_data_label(imgpath_labelpath)
        return (img_rgb_all, label_texture, label_pixel)

    def load_data_label(self, imgpath):
        img_path = imgpath
        f = open(img_path)
        sequence_path = f.read()
        sequence_path = sequence_path.split()
        # img read
        img_rgb_all = []
        list_len = len(sequence_path)
        for i in range(list_len - 1):
            img_name = sequence_path[i]
            img_rgb = cv2.imread(img_name, cv2.IMREAD_COLOR)
            # norm scale
            img_rgb = cv2.resize(img_rgb, (224, 224))
            #img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
            img_rgb = self.transform(img_rgb).float()
            img_rgb = img_rgb.unsqueeze(0)
            if i == 0:
                img_rgb_all = img_rgb
            else:
                img_rgb_all = np.concatenate((img_rgb_all, img_rgb), axis=0)

        label_texture = int(sequence_path[-1])
        if label_texture == 1:
            label_pixel = torch.ones((14, 14)).float()
        else:
            label_pixel = 0*torch.ones((14, 14)).float()
        label_pixel = label_pixel.unsqueeze(0)

        return img_rgb_all, label_texture, label_pixel#, label_color

img_transforms = transforms.ToTensor()
# load image and label
batch_size = 1
epoch_num = 5

# printed attack
############################RGB HSV YCBCR#########################
# set network
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='raft', help="name your experiment")
parser.add_argument('--stage', help="determines which dataset to use for training")
parser.add_argument('--restore_ckpt', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--validation', type=str, nargs='+')

parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--frame', type=float, default=5)
parser.add_argument('--num_steps', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

parser.add_argument('--iters', type=int, default=12)
parser.add_argument('--wdecay', type=float, default=.00005)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
parser.add_argument('--add_noise', action='store_true')
args = parser.parse_args()
model = Basic_timesformer_mse_multiscale_cnn(args)
print("Parameter Count: %d" % count_parameters(model))


#net = AttentionNet(img_channel=37)
# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion_CrossEntropyLoss = torch.nn.CrossEntropyLoss()
criterion_MSELoss = torch.nn.MSELoss()
criterion_HingeLoss = torch.nn.HingeEmbeddingLoss()
writer = SummaryWriter('runs/exp')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model_save_path = "./results/model_texture_pulse_overall_basic_timesformer_mse_cnn_rgb_cosine_SelectionMode3"
#model_save_path = "./results/model_texture_pulse_overall_basic_resnet"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

list_path = "./data/TxtList/train_deep_flow_overall.txt"

train_loss = open(model_save_path + "/train_loss.txt", "w")
dis_fake = 1
counter_index = 0
counter_real_index = 0
counter_fake_index = 0

real_counter = 0
fake_counter = 0
train_data = Data_myself(listroot=list_path)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

for batch_L, batch_label_texture, batch_label_pixel in train_loader:
    print(real_counter + fake_counter)
    if batch_label_texture == 0:
        real_counter = real_counter + 1
    else:
        fake_counter = fake_counter + 1

ratio_real = real_counter / (real_counter + fake_counter)
ratio_fake = fake_counter / (real_counter + fake_counter)

for epoch in range(epoch_num):

    train_data = Data_myself(listroot=list_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    b_counter = 0
    real_counter = 0
    fake_0 = 0
    outputs_real_fc = []
    for batch_L, batch_label_texture, batch_label_pixel in train_loader:
        b_counter = b_counter + 1
        counter_index = counter_index + 1
        # zero the parameter gradients
        optimizer.zero_grad()

        grid1 = []
        grid2 = []
        grid3 = []
        grid4 = []
        grid5 = []
        grid6 = []
        grid7 = []
        # tensor board
        grid1 = vutils.make_grid(batch_L, normalize=True, scale_each=True)
        grid1 = grid1.numpy()

        if batch_label_texture == 1:
            batch_label_pixel_temp = 255 * batch_label_pixel
        else:
            batch_label_pixel_temp = 0 * batch_label_pixel
        grid2 = vutils.make_grid(batch_label_pixel_temp, normalize=True, scale_each=True)
        grid2 = grid2.numpy()

        ################################################################################
        # two classification
        # forward + backward + optimize
        batch_L = batch_L.to(device)

        batch_label_texture = batch_label_texture.type(torch.LongTensor)
        batch_label_texture = batch_label_texture.view(1)
        batch_label_texture = batch_label_texture.to(device)
        batch_label_pixel = batch_label_pixel.to(device)

        if batch_label_texture == 0:
            similarity_label = torch.ones(1).float()
        else:
            similarity_label = 0 * torch.ones(1).float()
        similarity_label = similarity_label.to(device)


        #batch_label_color = batch_label_color.type(torch.LongTensor)
        #batch_label_color = batch_label_color.view(1)
        #batch_label_color = batch_label_color.to(device)

        outputs, img_sub_original, similarity = model(batch_L)
        #outputs_3, outputs_6, outputs_9, outputs_12, img_sub_original = model(batch_L)

        grid3 = vutils.make_grid(outputs, normalize=False, scale_each=False)
        grid3 = grid3.cpu()
        grid3 = grid3.detach().numpy()

        grid5 = vutils.make_grid(img_sub_original, normalize=False, scale_each=False)
        grid5 = grid5.cpu()
        grid5 = grid5.detach().numpy()

        CrossEntropyLoss_0 = criterion_MSELoss(outputs, batch_label_pixel)
        #CrossEntropyLoss_1 = criterion_CrossEntropyLoss(outputs_2, batch_label_texture)

        CrossEntropyLoss_2 = criterion_MSELoss(similarity, similarity_label)

        #CrossEntropyLoss_0 = criterion_MSELoss(outputs_3, batch_label_pixel)
        #CrossEntropyLoss_1 = criterion_MSELoss(outputs_6, batch_label_pixel)
        #CrossEntropyLoss_2 = criterion_MSELoss(outputs_9, batch_label_pixel)
        #CrossEntropyLoss_3 = criterion_MSELoss(outputs_12, batch_label_pixel)

        # update parame
        '''
        if epoch < 10:
            loss = CrossEntropyLoss_2
        else:
            loss = CrossEntropyLoss_2 + (1 - batch_label_texture) * (1 - ratio_real) * (CrossEntropyLoss_0 + CrossEntropyLoss_1) \
                                      + batch_label_texture * (1 - ratio_fake) * (CrossEntropyLoss_0 + CrossEntropyLoss_1)
        '''
        loss = (1 - batch_label_texture) * (1 - ratio_real) * (0*CrossEntropyLoss_0 + 1*CrossEntropyLoss_2) \
              + batch_label_texture * (1 - ratio_fake) * (0*CrossEntropyLoss_0 + 1*CrossEntropyLoss_2)
        loss.backward()
        optimizer.step()

        running_crossentropy0 = CrossEntropyLoss_0.item()
        #running_crossentropy1 = CrossEntropyLoss_1.item()
        running_crossentropy2 = CrossEntropyLoss_2.item()


        #writer.add_image('input', grid1, global_step=counter_index)
        #writer.add_image('label', grid2, global_step=counter_index)
        writer.add_image('map_1414', grid3, global_step=counter_index)
        #writer.add_image('map_3', grid3, global_step=counter_index)
        #writer.add_image('map_6', grid4, global_step=counter_index)
        #writer.add_image('map_9', grid5, global_step=counter_index)
        writer.add_image('input_original', grid5, global_step=counter_index)
        writer.add_scalar('loss_mse_1414', running_crossentropy0, global_step=counter_index)
        #writer.add_scalar('loss_cross', running_crossentropy1, global_step=counter_index)
        writer.add_scalar('loss_mse_similarity', running_crossentropy2, global_step=counter_index)

        print('Protocol I [%d, %d] mse_1414: %.7f mse_similarity: %.7f' %
              (epoch + 1, b_counter, running_crossentropy0, running_crossentropy2))


    torch.save(model.state_dict(), (model_save_path + "/" + "net_epoch_" + str(epoch) + ".pkl"))

#train_loss.close()
print('Finished Training')

