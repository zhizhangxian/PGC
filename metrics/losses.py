import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss as loss
import numpy as np
from math import floor


class ssp_loss(nn.Module):
    def __init__(self, exclusive=True, ignore_index=255):
        super(ssp_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.mse_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()
        self.ignore_index = ignore_index

    def forward(self, outputs, overlap, flips, labels):
        len_img = outputs.shape[0] // 2
        mse = 0
        ce_1_2 = 0
        ce_2_1 = 0
        ex_labels = labels.detach().clone()
        for i in range(len_img):
            shape_1 = (overlap[i][0][1][0] - overlap[i][0][0][0], overlap[i][0][1][1] - overlap[i][0][0][1])
            shape_2 = (overlap[i][1][1][0] - overlap[i][1][0][0], overlap[i][1][1][1] - overlap[i][1][0][1])
            img_1 = outputs[2 * i, :, overlap[i][0][0][0]:overlap[i][0][1][0], overlap[i][0][0][1]:overlap[i][0][1][1]]
            img_2 = outputs[2 * i + 1, :, overlap[i][1][0][0]:overlap[i][1][1][0], overlap[i][1][0][1]:overlap[i][1][1][1]]

            ex_labels[2 * i, overlap[i][0][0][0]:overlap[i][0][1][0], overlap[i][0][0][1]:overlap[i][0][1][1]] = self.ignore_index
            ex_labels[2 * i + 1, overlap[i][1][0][0]:overlap[i][1][1][0], overlap[i][1][0][1]:overlap[i][1][1][1]] = self.ignore_index

            if flips[i] == -1:
                img_2 = torch.flip(img_2, [2])

            if ((shape_1[0] < 1) or (shape_1[1] < 1) or (shape_2[0] < 1) or (shape_2[1] < 1)):
                mse_loss = 0
                ce_loss_1_2 = 0
                ce_loss_2_1 = 0
            else:
                img_1_label = img_1.argmax(dim=0).unsqueeze(0).requires_grad_(False)
                img_2_label = img_2.argmax(dim=0).unsqueeze(0).requires_grad_(False)
                mse_loss = self.mse_loss(img_1, img_2)
                ce_loss_1_2 = self.ce_loss(img_1.unsqueeze(0), img_2_label)
                ce_loss_2_1 = self.ce_loss(img_2.unsqueeze(0), img_1_label)
                sym_ce_loss = 0.5 * ce_loss_1_2 + 0.5 * ce_loss_2_1
            mse += mse_loss
            ce_1_2 += ce_loss_1_2
            ce_2_1 += ce_loss_2_1

        mse /= len_img
        ce_1_2 /= len_img
        ce_2_1 /= len_img
        sym_ce = 0.5 * (ce_1_2 + ce_2_1)
        ce = self.ce_loss(outputs, labels)
        ex_ce = self.ce_loss(outputs, ex_labels)

        return mse, ce_1_2, ce_2_1, sym_ce, ce, ex_ce





def negcos(p, z):
    # z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z.detach()).sum(dim=1).mean()
    # return - nn.functional.cosine_similarity(p, z.detach(), dim=-1).mean()


class Mixed_Loss(ssp_loss):
    def __init__(self, exclusive=True, ignore_index=255):
        super(Mixed_Loss, self).__init__(exclusive, ignore_index)

    def forward(self, outputs, overlap, flips, labels):
        seg = outputs['seg']
        z, p = outputs['embedding']
        z1, z2 = z[1::2, :], z[::2, :]
        p1, p2 = p[1::2, :], p[::2, :]
        mse, ce_1_2, ce_2_1, sym_ce, ce, ex_ce = ssp_loss.forward(self, seg, overlap, flips, labels)
        Cosine = negcos(p1, z2) / 2 + negcos(p2, z1) / 2
        return mse, ce_1_2, ce_2_1, sym_ce, ce, ex_ce, Cosine


class new_ssp_loss(nn.Module):
    def __init__(self, exclusive=True, ignore_index=255):
        super(new_ssp_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.mse_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()
        self.ignore_index = ignore_index

    def forward(self, outputs, overlap, flips, labels):
        output1, output2 = outputs
        N = output1.shape[0]
        mse = 0
        ce_1_2 = 0
        ce_2_1 = 0
        ex_labels = labels.detach().clone()

        # for i in range(N):
        for i in range(N):
            shape_1 = (overlap[i][0][1][0] - overlap[i][0][0][0], overlap[i][0][1][1] - overlap[i][0][0][1])
            shape_2 = (overlap[i][1][1][0] - overlap[i][1][0][0], overlap[i][1][1][1] - overlap[i][1][0][1])
            img_1 = output1[i, :, overlap[i][0][0][0]:overlap[i][0][1][0], overlap[i][0][0][1]:overlap[i][0][1][1]]
            img_2 = output2[i, :, overlap[i][1][0][0]:overlap[i][1][1][0], overlap[i][1][0][1]:overlap[i][1][1][1]]

            ex_labels[2 * i, overlap[i][0][0][0]:overlap[i][0][1][0], overlap[i][0][0][1]:overlap[i][0][1][1]] = self.ignore_index
            ex_labels[2 * i + 1, overlap[i][1][0][0]:overlap[i][1][1][0], overlap[i][1][0][1]:overlap[i][1][1][1]] = self.ignore_index

            if flips[i] == -1:
                img_2 = torch.flip(img_2, [2])

            if ((shape_1[0] < 1) or (shape_1[1] < 1) or (shape_2[0] < 1) or (shape_2[1] < 1)):
                mse_loss = 0
                ce_loss_1_2 = 0
                ce_loss_2_1 = 0
            else:
                img_1_label = img_1.argmax(dim=0).unsqueeze(0).requires_grad_(False)
                img_2_label = img_2.argmax(dim=0).unsqueeze(0).requires_grad_(False)
                mse_loss = self.mse_loss(img_1, img_2)
                ce_loss_1_2 = self.ce_loss(img_1.unsqueeze(0), img_2_label)
                ce_loss_2_1 = self.ce_loss(img_2.unsqueeze(0), img_1_label)
                sym_ce_loss = 0.5 * ce_loss_1_2 + 0.5 * ce_loss_2_1
            mse += mse_loss
            ce_1_2 += ce_loss_1_2
            ce_2_1 += ce_loss_2_1

        mse /= N
        ce_1_2 /= N
        ce_2_1 /= N
        sym_ce = 0.5 * (ce_1_2 + ce_2_1)
        label1 = labels[::2]
        label2 = labels[1::2]
        exlabel1 = ex_labels[::2]
        exlabel2 = ex_labels[1::2]
        Labels = torch.cat([label1, label2], dim=0).detach()
        ex_labels = torch.cat([exlabel1, exlabel2], dim=0).detach()
        Output = torch.cat([outputs[0], outputs[1]], dim=0)
        ce = self.ce_loss(Output, Labels)
        #
        ex_ce = self.ce_loss(Output, ex_labels)
        return mse, ce_1_2, ce_2_1, sym_ce, ce


class ssp_loss_inner(new_ssp_loss):
    def __init__(self) -> None:
        super(ssp_loss_inner, self).__init__(exclusive=True, ignore_index=255)

    def forward(self, outputs, overlap, flips, downsamples=1):
        len_img = outputs[0].shape[0]
        mse = 0
        l1 = 0
        ce_1_2 = 0
        ce_2_1 = 0

        # overlap_new = overlap.copy()
        overlap_new = np.zeros((len_img, 2, 2, 2), dtype=np.int)


        if downsamples != 1:
            for i in range(8):
                for j in range(2):
                    if (j == 0):
                        for k in range(2):
                            for l in range(2):
                                overlap_new[i][j][k][l] = floor(overlap[i][j][k][l] / downsamples)
                        h = overlap_new[i][j][1][0] - overlap_new[i][j][0][0]
                        w = overlap_new[i][j][1][1] - overlap_new[i][j][0][1]
                        size = (h, w)
                 #       print('h:', h, 'w:', w)

                    elif (j == 1):
                        for k in range(2):
                            for l in range(2):
                                if k == 0:
                                    overlap_new[i][j][k][l] = (overlap[i][j][k][l] // downsamples)
                                else:
                                    overlap_new[i][j][k][l] = overlap_new[i][j][0][l] + size[l]

        for i in range(len_img):
            shape_1 = (overlap_new[i][0][1][0] - overlap_new[i][0][0][0], overlap_new[i][0][1][1] - overlap_new[i][0][0][1])
            shape_2 = (overlap_new[i][1][1][0] - overlap_new[i][1][0][0], overlap_new[i][1][1][1] - overlap_new[i][1][0][1])
            img_1 = outputs[0][:, overlap_new[i][0][0][0]:overlap_new[i][0][1][0], overlap_new[i][0][0][1]:overlap_new[i][0][1][1]]
            img_2 = outputs[1][:, overlap_new[i][1][0][0]:overlap_new[i][1][1][0], overlap_new[i][1][0][1]:overlap_new[i][1][1][1]]

            if flips[i] == -1:
                img_2 = torch.flip(img_2, [2])

            if ((shape_1[0] < 1) or (shape_1[1] < 1) or (shape_2[0] < 1) or (shape_2[1] < 1)):
                mse_loss = 0
                ce_loss_1_2 = 0
                ce_loss_2_1 = 0
                l1_loss = 0
            else:
                img_1_label = img_1.argmax(dim=0).unsqueeze(0).requires_grad_(False)
                img_2_label = img_2.argmax(dim=0).unsqueeze(0).requires_grad_(False)
                mse_loss = self.mse_loss(img_1, img_2)
                l1_loss = self.L1_loss(img_1, img_2)

                ce_loss_1_2 = self.ce_loss(img_1.unsqueeze(0), img_2_label)
                ce_loss_2_1 = self.ce_loss(img_2.unsqueeze(0), img_1_label)
                mse += mse_loss

            mse += mse_loss
            l1 += l1_loss
            ce_1_2 += ce_loss_1_2
            ce_2_1 += ce_loss_2_1

        mse /= len_img
        ce_1_2 /= len_img
        ce_2_1 /= len_img
        l1 /= len_img
        sym_ce = 0.5 * (ce_1_2 + ce_2_1)
        return mse, sym_ce, l1


class PGC_loss(ssp_loss_inner):
    def __init__(self, exclusive=True, ignore_index=255, use_pgc=[0, 1, 2], down_rate=[16, 16, 4]):
        super(PGC_loss, self).__init__()
        self.use_pgc = use_pgc
        self.down_rate = dict(zip(use_pgc, down_rate))

    def forward(self, outputs, overlap, flips, labels):
        mse, _, _, sym_ce, ce = new_ssp_loss.forward(self, outputs[-1], overlap, flips, labels)
        mid_mse = []
        mid_ce = []
        mid_l1 = []
        for i in self.use_pgc:
            down_rate = self.down_rate[i]
            mse1, sym_ce1, l11 = ssp_loss_inner.forward(self, outputs[i], overlap, flips, down_rate)
            mid_mse.append(mse1)
            mid_ce.append(sym_ce1)
            mid_l1.append(l11)

        return mse, sym_ce, mid_mse, mid_ce, mid_l1, ce
