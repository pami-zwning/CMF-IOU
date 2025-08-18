import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.spconv_utils import spconv
from torch.nn.functional import normalize
import numpy as np

class PointNet(nn.Module):
    def __init__(self, in_channel=9, out_channels=32):
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channel, out_channels, 1)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.transpose(1,2)

        return x

class CPConvs(nn.Module):
    def __init__(self):
        super(CPConvs, self).__init__()
        self.pointnet1_fea = PointNet(  6,12)
        self.pointnet1_wgt = PointNet(  6,12)
        self.pointnet1_fus = PointNet(108,12)

        self.pointnet2_fea = PointNet( 12,24)
        self.pointnet2_wgt = PointNet(  6,24)
        self.pointnet2_fus = PointNet(216,24)

        self.pointnet3_fea = PointNet( 24,48)
        self.pointnet3_wgt = PointNet(  6,48)
        self.pointnet3_fus = PointNet(432,48)

    def forward(self, points_features, points_neighbor):
        if points_features.shape[0] == 0:
            return points_features

        N, F = points_features.shape
        N, M = points_neighbor.shape
        point_empty = (points_neighbor == 0).nonzero()
        points_neighbor[point_empty[:,0], point_empty[:,1]] = point_empty[:,0]

        pointnet_in_xiyiziuiviri = torch.index_select(points_features[:,[0,1,2,6,7,8]],0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet_in_x0y0z0u0v0r0 = points_features[:,[0,1,2,6,7,8]].unsqueeze(dim=1).repeat([1,M,1])
        pointnet_in_xyzuvr       = pointnet_in_xiyiziuiviri - pointnet_in_x0y0z0u0v0r0
        points_features[:, 3:6] /= 255.0
        points_features[:,:3] = normalize(points_features[:,:3],dim=0)
        points_features[:,6:] = normalize(points_features[:,6:],dim=0)
        
        pointnet1_in_fea        = points_features[:,:6].view(N,1,-1)
        pointnet1_out_fea       = self.pointnet1_fea(pointnet1_in_fea).view(N,-1)
        pointnet1_out_fea       = torch.index_select(pointnet1_out_fea,0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet1_out_wgt       = self.pointnet1_wgt(pointnet_in_xyzuvr)
        pointnet1_feas          = pointnet1_out_fea * pointnet1_out_wgt
        pointnet1_feas          = self.pointnet1_fus(pointnet1_feas.reshape(N,1,-1)).view(N,-1)   

        pointnet2_in_fea        = pointnet1_feas.view(N,1,-1)
        pointnet2_out_fea       = self.pointnet2_fea(pointnet2_in_fea).view(N,-1)
        pointnet2_out_fea       = torch.index_select(pointnet2_out_fea,0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet2_out_wgt       = self.pointnet2_wgt(pointnet_in_xyzuvr)
        pointnet2_feas           = pointnet2_out_fea * pointnet2_out_wgt
        pointnet2_feas          = self.pointnet2_fus(pointnet2_feas.reshape(N,1,-1)).view(N,-1)

        pointnet3_in_fea        = pointnet2_feas.view(N,1,-1)
        pointnet3_out_fea       = self.pointnet3_fea(pointnet3_in_fea).view(N,-1)
        pointnet3_out_fea       = torch.index_select(pointnet3_out_fea,0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet3_out_wgt       = self.pointnet3_wgt(pointnet_in_xyzuvr)
        pointnet3_feas           = pointnet3_out_fea * pointnet3_out_wgt
        pointnet3_feas          = self.pointnet3_fus(pointnet3_feas.reshape(N,1,-1)).view(N,-1)
 
        pointnet_feas     = torch.cat([pointnet3_feas, pointnet2_feas, pointnet1_feas, points_features[:,:6]], dim=-1)
        return pointnet_feas

class Attention(nn.Module):
    def __init__(self, channels, output_channels, num):
        super(Attention, self).__init__()
        self.input_channel = channels
        self.num = num
        self.output_channels = output_channels
        middle = self.input_channel // 4
        self.fc_list, self.conv_list = [],[]
        for i in range(num):
            self.fc_list.append(nn.Linear(self.input_channel, middle))
            self.conv_list.append(nn.Sequential(nn.Conv1d(self.input_channel, self.output_channels, 1),
                                    nn.BatchNorm1d(self.output_channels),
                                    nn.ReLU()))
        self.fc_layers = nn.Sequential(*self.fc_list)
        self.proj_fc = nn.Linear(num*middle, num)
        self.conv_layers = nn.Sequential(*self.conv_list)


    def forward(self, total_feas):
        batch = total_feas[0].size(0)

        cur_feas_f = []
        for i,cur_feas in enumerate(total_feas):
            cur_feas_f.append(self.fc_list[i](cur_feas))

        total_feas_f = torch.cat(cur_feas_f,dim=-1)
        weight = torch.sigmoid(self.proj_fc(total_feas_f))

        features_attn_list = []
        for i in range(self.num):
            cur_weight = weight[:,i].squeeze()
            cur_weight = cur_weight.view(batch, 1, -1)
            features_attn_list.append(self.conv_layers[i](total_feas[i].unsqueeze(-1))  * cur_weight)

        return features_attn_list

class PSCT(nn.Module):
    def __init__(self, input_channel, output_channel, input_num):
        super(PSCT, self).__init__()
        self.attention = Attention(input_channel, output_channel, input_num)
        self.conv1 = torch.nn.Conv1d( output_channel*input_num, output_channel, 1)
        self.bn1 = torch.nn.BatchNorm1d(output_channel)

    def forward(self, total_features):
        total_features_att_list=  self.attention(total_features)
        fusion_features = torch.cat(total_features_att_list, dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features.squeeze(-1)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, demb=256):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    # pos_seq =  pos_seq = torch.arange(seq_len-1, -1, -1.0)
    def forward(self, pos_seq, batch_size=2):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if batch_size is not None:
            return pos_emb[:, None, :].expand(-1, batch_size, -1)
        else:
            return pos_emb[:, None, :]

class CrossAttention(nn.Module):

    def __init__(self, hidden_dim, pos = True, head = 4):
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.pos_dim = 8
        self.pos = pos

        if self.pos:
            self.pos_en = PositionalEmbedding(self.pos_dim)

            self.Q_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
        else:

            self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.att = nn.MultiheadAttention(hidden_dim, head)


    def forward(self, inputs, Q_in): # N,B,C

        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]

        if self.pos:
            pos_input = torch.from_numpy(np.arange(seq_len)+1).cuda()
            pos_input = self.pos_en(pos_input, batch_size)
            inputs_pos = torch.cat([inputs, pos_input], -1)
            pos_Q = torch.from_numpy(np.array([seq_len])).cuda()
            pos_Q = self.pos_en(pos_Q, batch_size)
            Q_in_pos = torch.cat([Q_in, pos_Q], -1)
        else:
            inputs_pos = inputs
            Q_in_pos = Q_in

        Q = self.Q_linear(Q_in_pos)
        K = self.K_linear(inputs_pos)
        V = self.V_linear(inputs_pos)

        out = self.att(Q, K, V)

        return out[0]

class IMG_CNN(nn.Module):
    def __init__(self,img_size, input_channel, output_channel):
        super(IMG_CNN,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear( (img_size[0] // 8 + 1) * (img_size[1] // 8 + 1) * 128, output_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, h, w, input_channel = x.size()


        x = x.permute(0,3,1,2).to(torch.float32)
        img_feat = self.conv1(x)
        img_feat = self.conv2(img_feat)
        img_feat = self.conv3(img_feat)
        img_feat = img_feat.reshape(batch_size, -1)
        img_feat = self.relu(self.fc(img_feat))
        return img_feat

class IMG_SPCNN(nn.Module):
    def __init__(self,img_size, input_channel, output_channel):
        super(IMG_SPCNN,self).__init__()

        self.spconv1 = nn.Sequential(
            spconv.SparseConv2d(in_channels=input_channel, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.spconv2 = nn.Sequential(
            spconv.SparseConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.spconv3 = nn.Sequential(
            spconv.SparseConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear( (img_size[0] // 8) * (img_size[1] // 8 ) * 128, output_channel)
        self.relu = nn.ReLU()

    def forward(self, input_feat, spconv_coords):
        batch_size, h, w, input_channel = input_feat.size()
        newimg_coords = spconv_coords.clone().to(torch.long)
        newimg_features = input_feat[newimg_coords]
        sparse_shape = torch.tensor([batch_size,h,w])

        newimg_sp_tensor = spconv.SparseConvTensor(
            features = newimg_features,
            indices = newimg_coords.int(),
            spatical_shape = sparse_shape,
            batch_size = batch_size
        )

        img_feat = self.spconv1(newimg_sp_tensor)
        img_feat = self.spconv2(img_feat)
        img_feat = self.spconv3(img_feat)
        img_feat = img_feat.dense()
        img_feat = img_feat.reshape(batch_size, -1)
        img_feat = self.relu(self.fc(img_feat))
        return img_feat
