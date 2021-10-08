import torch as tr
import torch.nn as nn
import component as cp


class BackBone(nn.Module):
    def __init__(self, in_channel, out_channel, csp_nums=[1, 2, 8, 8, 4]):
        super(BackBone, self).__init__()
        self.act = cp.Mish
        self.fCBM = cp.CBa(self.act, in_channel, out_channel, 3, 1)
        res_kernel = [1, 3]
        kernels = [3, res_kernel, 1, 1]
        frs_strides = [2, 1, 1]
        strides = [2, 1, 1]
        res_channel0 = [out_channel, out_channel*2]
        out_channels0 = [out_channel*2, res_channel0, out_channel*2, out_channel*2]
        self.CSP0 = cp.CSPx(self.act, csp_nums[0], self.fCBM.out_channel, out_channels0, kernels, frs_strides)
        self.CBM0 = cp.CBa(self.act, self.CSP0.out_channel, out_channels0[-1], 1, 1)

        res_channel1 = [out_channel*2, out_channel*2]
        out_channels1 = [out_channel*4, res_channel1, out_channel*2, out_channel*2]
        self.CSP1 = cp.CSPx(self.act, csp_nums[1], self.CBM0.out_channel, out_channels1, kernels, strides)
        self.CBM1 = cp.CBa(self.act, self.CSP1.out_channel, out_channels1[-1]+out_channels1[-2], 3, 1)

        out_channels2 = self.double_channels(out_channels1)
        self.CSP2 = cp.CSPx(self.act, csp_nums[2], self.CBM1.out_channel, out_channels2, kernels, strides)
        self.CBM2 = cp.CBa(self.act, self.CSP2.out_channel, out_channels2[-1]*2, 1, 1)

        out_channels3 = self.double_channels(out_channels2)
        self.CSP3 = cp.CSPx(self.act, csp_nums[3], self.CBM2.out_channel, out_channels3, kernels, strides)
        self.CBM3 = cp.CBa(self.act, self.CSP3.out_channel, out_channels3[-1]*2, 1, 1)

        out_channels4 = self.double_channels(out_channels3)
        self.CSP4 = cp.CSPx(self.act, csp_nums[4], self.CBM3.out_channel, out_channels4, kernels, strides)
        self.CBM4 = cp.CBa(self.act, self.CSP4.out_channel, out_channels4[-1]*2, 1, 1)

    def double_channels(self, channels):
        lens = len(channels)
        doubled_channels = []
        for i in range(lens):
            if type(channels[i]) == list:
                temp = len(channels[i])
                inner_channels = []
                for j in range(temp):
                    inner_channels.append(channels[i][j] * 2)
                doubled_channels.append(inner_channels)
            else:
                doubled_channels.append(channels[i] * 2)
        return doubled_channels

    def forward(self, x):
        x = self.fCBM(x)
        x = self.CSP0(x)
        x = self.CBM0(x)
        x = self.CSP1(x)
        x = self.CBM1(x)
        x = self.CSP2(x)
        first_b = self.CBM2(x) #00000000000000000000
        x = self.CSP3(first_b)
        second_b = self.CBM3(x)
        x = self.CSP4(second_b)
        x = self.CBM4(x)
        return first_b, second_b, x


class YOLOV4(nn.Module):
    def __init__(self, clas, in_channel, out_channel, n_fold, csp_nums=[1, 2, 8, 8, 4]):
        super(YOLOV4, self).__init__()
        self.n_f = n_fold
        in_channel = in_channel * self.n_f
        out_channel_holder = out_channel
        self.leaky_param = {"negative_slope":0.1, "inplace":False}
        self.act = nn.LeakyReLU
        self.backbone = BackBone(in_channel, out_channel, csp_nums)
        out_channel = out_channel*32
        self.CBL0 = cp.CBa(self.act, out_channel, out_channel//2, 1, extra=self.leaky_param)
        self.CBL1 = cp.CBa(self.act, self.CBL0.out_channel, out_channel, 3, extra=self.leaky_param)
        self.CBL2 = cp.CBa(self.act, self.CBL1.out_channel, out_channel//2, 1, extra=self.leaky_param)
        self.spp = cp.SPP([5, 9, 13])
        self.CBL3 = cp.CBa(self.act, self.CBL2.out_channel*4, out_channel//2, 1, extra=self.leaky_param)
        self.CBL4 = cp.CBa(self.act, self.CBL3.out_channel, out_channel, 3, extra=self.leaky_param)
        self.CBL5 = cp.CBa(self.act, self.CBL4.out_channel, out_channel//2, 1, extra=self.leaky_param)
        self.CBL6 = cp.CBa(self.act, self.CBL5.out_channel, out_channel//4, 1, extra=self.leaky_param)
        self.upsam = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        out_channel = out_channel_holder
        self.fst_b(clas, out_channel*8, out_channel*4)
        self.sec_b(clas, out_channel*16, out_channel*8)
        self.trd_b(clas, out_channel*32, out_channel*16)


    def fst_b(self, clas, in_channel, out_channel):
        self.fst_CBL0 = cp.CBa(self.act, in_channel, out_channel, 1, extra=self.leaky_param)
        self.fst_CBL1 = cp.CBa(self.act, out_channel*2, out_channel, 1, extra=self.leaky_param)
        self.fst_CBL2 = cp.CBa(self.act, self.fst_CBL1.out_channel, out_channel*2, 3, extra=self.leaky_param)
        self.fst_CBL3 = cp.CBa(self.act, self.fst_CBL2.out_channel, out_channel, 1, extra=self.leaky_param)
        self.fst_CBL4 = cp.CBa(self.act, self.fst_CBL3.out_channel, out_channel*2, 3, extra=self.leaky_param)
        self.fst_CBL5 = cp.CBa(self.act, self.fst_CBL4.out_channel, out_channel, 1, extra=self.leaky_param)
        self.fst_CBL6 = cp.CBa(self.act, self.fst_CBL5.out_channel, out_channel*2, 3, extra=self.leaky_param)
        self.fst_out = cp.Conv(out_channel*2, 3*(5+clas), 1)

        self.fst_CBL7 = cp.CBa(self.act, self.fst_CBL6.out_channel, out_channel*2, 3, 2, extra=self.leaky_param)


    def sec_b(self, clas, in_channel, out_channel):
        self.sec_CBL0 = cp.CBa(self.act, in_channel, out_channel, 1, extra=self.leaky_param)
        self.sec_CBL1 = cp.CBa(self.act, out_channel*2, out_channel, 1, extra=self.leaky_param)
        self.sec_CBL2 = cp.CBa(self.act, self.sec_CBL1.out_channel, out_channel*2, 3, extra=self.leaky_param)
        self.sec_CBL3 = cp.CBa(self.act, self.sec_CBL2.out_channel, out_channel, 1, extra=self.leaky_param)
        self.sec_CBL4 = cp.CBa(self.act, self.sec_CBL3.out_channel, out_channel*2, 3, extra=self.leaky_param)
        self.sec_CBL5 = cp.CBa(self.act, self.sec_CBL4.out_channel, out_channel, 1, extra=self.leaky_param)
        self.sec_CBL6 = cp.CBa(self.act, self.sec_CBL5.out_channel, out_channel//2, 1, extra=self.leaky_param)
        self.sec_upsam = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.sec_CBL7 = cp.CBa(self.act, out_channel*2, out_channel, 1, extra=self.leaky_param)
        self.sec_CBL8 = cp.CBa(self.act, self.sec_CBL7.out_channel, out_channel*2, 3, extra=self.leaky_param)
        self.sec_CBL9 = cp.CBa(self.act, self.sec_CBL8.out_channel, out_channel, 1, extra=self.leaky_param)
        self.sec_CBL10 = cp.CBa(self.act, self.sec_CBL9.out_channel, out_channel*2, 3, extra=self.leaky_param)
        self.sec_CBL11 = cp.CBa(self.act, self.sec_CBL10.out_channel, out_channel, 1, extra=self.leaky_param)
        self.sec_CBL12 = cp.CBa(self.act, self.sec_CBL11.out_channel, out_channel*2, 3, extra=self.leaky_param)
        self.sec_out = cp.Conv(self.sec_CBL12.out_channel, 3*(5+clas), 1)
        
        self.sec_CBL13 = cp.CBa(self.act, self.sec_CBL12.out_channel, out_channel*2, 3, 2, extra=self.leaky_param)


    def trd_b(self, clas, in_channel, out_channel):
        self.trd_CBL0 = cp.CBa(self.act, in_channel, out_channel, 1, extra=self.leaky_param)
        self.trd_CBL1 = cp.CBa(self.act, self.trd_CBL0.out_channel, out_channel*2, 3, extra=self.leaky_param)
        self.trd_CBL2 = cp.CBa(self.act, self.trd_CBL1.out_channel, out_channel, 1, extra=self.leaky_param)
        self.trd_CBL3 = cp.CBa(self.act, self.trd_CBL2.out_channel, out_channel*2, 3, extra=self.leaky_param)
        self.trd_CBL4 = cp.CBa(self.act, self.trd_CBL3.out_channel, out_channel, 1, extra=self.leaky_param)
        self.trd_CBL5 = cp.CBa(self.act, self.trd_CBL4.out_channel, out_channel*2, 3, extra=self.leaky_param)
        self.trd_out = cp.Conv(self.trd_CBL5.out_channel, 3*(5+clas), 1)


    def forward(self, x):
        # b, c, h, w = x.shape
        # new_im = []
        # nh, nw = h//self.n_f if h%self.n_f==0 else h//self.n_f + 1, \
        #          w//self.n_f if w%self.n_f==0 else w//self.n_f + 1
        # start = 0
        # for r in range(self.n_f):
        #     for i in range(c):
        #         temp = x[:, i:(i+1), start::self.n_f, start::self.n_f]
        #         _, _, j, i = temp.shape
        #         im = tr.zeros((b, 1, nh, nw), device=x.device)
        #         im[:, :, 0:j, 0:i] = temp
        #         # del temp
        #         start = start + 1
        #         # print(im.shape)
        #         new_im.append(im[:])
        # x = tr.cat(new_im, dim=1)
        f_b, s_b, t_b = self.backbone(x)
        t_b = self.CBL0(t_b)
        t_b = self.CBL1(t_b)
        t_b = self.CBL2(t_b)
        t_b = self.spp(t_b)
        t_b = self.CBL3(t_b)
        t_b = self.CBL4(t_b)
        t_b = self.CBL5(t_b)
        t_s_b = self.CBL6(t_b)

        t_s_b = self.upsam(t_s_b)
        # print(s_b.shape, t_s_b.shape)
        s_b = self.sec_CBL0(s_b)
        s_b = tr.cat((s_b, t_s_b), dim=1)
        s_b = self.sec_CBL1(s_b)
        s_b = self.sec_CBL2(s_b)
        s_b = self.sec_CBL3(s_b)
        s_b = self.sec_CBL4(s_b)
        s_b = self.sec_CBL5(s_b)
        s_f_b = self.sec_CBL6(s_b)

        f_b = self.fst_CBL0(f_b)
        s_f_b = self.upsam(s_f_b)
        f_b = tr.cat((s_f_b, f_b), dim=1)
        f_b = self.fst_CBL1(f_b)
        f_b = self.fst_CBL2(f_b)
        f_b = self.fst_CBL3(f_b)
        f_b = self.fst_CBL4(f_b)
        f_b = self.fst_CBL5(f_b)
        f_b = self.fst_CBL6(f_b)

        f_s_b = self.fst_CBL7(f_b)
        s_b = tr.cat((s_b, f_s_b), dim=1)
        s_b = self.sec_CBL7(s_b)
        s_b = self.sec_CBL8(s_b)
        s_b = self.sec_CBL9(s_b)
        s_b = self.sec_CBL10(s_b)
        s_b = self.sec_CBL11(s_b)
        s_b = self.sec_CBL12(s_b)
        s_t_b = self.sec_CBL13(s_b)
        
        t_b = tr.cat((s_t_b, t_b), dim=1)
        t_b = self.trd_CBL0(t_b)
        t_b = self.trd_CBL1(t_b)
        t_b = self.trd_CBL2(t_b)
        t_b = self.trd_CBL3(t_b)
        t_b = self.trd_CBL4(t_b)
        t_b = self.trd_CBL5(t_b)

        return [self.fst_out(f_b), self.sec_out(s_b), self.trd_out(t_b)]