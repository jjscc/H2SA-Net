import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import math

def tanh(x):
    return x.clamp(-15, 15).tanh()

def atanh(x):
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return 0.5 * torch.log((1 + x) / (1 - x))

class PoincareMath:
    @staticmethod
    def exp_map_0(x: torch.Tensor, c=1.0, epsilon=1e-5) -> torch.Tensor:
        if not torch.is_tensor(c):
            c = torch.tensor(c, dtype=x.dtype, device=x.device)
        sqrt_c = torch.sqrt(c)
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(epsilon)
        gamma = tanh(sqrt_c * x_norm) / (sqrt_c * x_norm)
        return PoincareMath.project_hyp_vecs(gamma * x, c)

    @staticmethod
    def log_map_0(y: torch.Tensor, c=1.0, epsilon=1e-5) -> torch.Tensor:
        if not torch.is_tensor(c):
            c = torch.tensor(c, dtype=y.dtype, device=y.device)  
        sqrt_c = torch.sqrt(c)
        y_norm = torch.norm(y, dim=-1, keepdim=True).clamp_min(epsilon)
        scale = (1.0 / sqrt_c) * atanh(sqrt_c * y_norm) / y_norm
        return scale * y

    @staticmethod
    def project_hyp_vecs(x: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
        max_norm = (1.0 - 1e-5) / torch.sqrt(c)
        x_norm = torch.norm(x, dim=dim, keepdim=True)
        cond = x_norm > max_norm
        projected = x / x_norm * max_norm
        return torch.where(cond, projected, x)

    @staticmethod
    def sqnorm(x, dim=1, keepdim=True):
        return torch.sum(x ** 2, dim=dim, keepdim=keepdim)

    @staticmethod
    def dist_sq(x: torch.Tensor, y: torch.Tensor, c=1.0, epsilon=1e-5) -> torch.Tensor:

        if not torch.is_tensor(c):
            c = torch.tensor(c, dtype=x.dtype, device=x.device)
        x2 = torch.sum(x.pow(2), dim=-1, keepdim=True)
        y2 = torch.sum(y.pow(2), dim=-1, keepdim=True).transpose(1, 2)
        xy = torch.matmul(x, y.transpose(1, 2))
        eucl_dist_sq = (x2 + y2 - 2 * xy).clamp_min(0.0)
        denom = (1 - c * x2) * (1 - c * y2)
        denom = torch.clamp(denom, min=epsilon) 
        arg = 1 + 2 * c * eucl_dist_sq / denom
        arg = torch.clamp(arg, min=1.0 + epsilon) 
        dist = torch.acosh(arg) / torch.sqrt(c)
        return dist


    @staticmethod
    def poincare_to_klein(x: torch.Tensor, c=1.0) -> torch.Tensor:
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        denom = 1 + c * x_norm_sq
        return 2 * x / torch.clamp(denom, min=1e-5)

    @staticmethod
    def klein_to_poincare(x: torch.Tensor, c=1.0) -> torch.Tensor:
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        denom = 1 + torch.sqrt(torch.clamp(1 - c * x_norm_sq, min=1e-5))
        return x / torch.clamp(denom, min=1e-5)

    @staticmethod
    def einstein_midpoint(x: torch.Tensor, weights: torch.Tensor, c=1.0, epsilon=1e-5) -> torch.Tensor:

        x_klein = PoincareMath.poincare_to_klein(x, c)
        x_klein_norm_sq = torch.sum(x_klein ** 2, dim=-1, keepdim=True)
        gamma = 1.0 / torch.sqrt(torch.clamp(1 - c * x_klein_norm_sq, min=epsilon)) 
        weighted_gamma = weights.unsqueeze(-1) * gamma.unsqueeze(1) 
        denominator = torch.sum(weighted_gamma, dim=2) 
        vals = gamma * x_klein
        numerator = torch.matmul(weights, vals)
        m_klein = numerator / torch.clamp(denominator, min=epsilon)
        m_norm = torch.norm(m_klein, dim=-1, keepdim=True)
        sqrt_c = torch.sqrt(c) if torch.is_tensor(c) else c**0.5
        max_norm = (1.0 - 1e-3) / sqrt_c
        cond = m_norm > max_norm
        m_klein = torch.where(cond, m_klein / m_norm.clamp_min(epsilon) * max_norm, m_klein)
        m_poincare = PoincareMath.klein_to_poincare(m_klein, c)
        return m_poincare

    @staticmethod
    def hyp_mlr(inputs: torch.Tensor, c: torch.Tensor, P_mlr: torch.Tensor, A_mlr: torch.Tensor) -> torch.Tensor:
        EPS = 1e-15
        PROJ_EPS = 1e-5
        B, D, H, W = inputs.shape
        M = P_mlr.shape[0]
        xx = PoincareMath.sqnorm(inputs, dim=1, keepdim=True)
        pp = PoincareMath.sqnorm(-P_mlr, dim=1, keepdim=False)
        P_kernel = (-P_mlr).unsqueeze(2).unsqueeze(3)
        px = F.conv2d(inputs, P_kernel)
        sqsq = (c * xx) * (c * pp.view(1, -1, 1, 1))
        A_ = 1.0 + 2 * c * px + sqsq
        B_ = (1.0 - c * pp).view(1, -1, 1, 1)
        D_ = torch.clamp(1.0 + 2 * c * px + sqsq, min=EPS)
        alpha = A_ / D_
        beta  = B_ / D_
        normed_A = F.normalize(A_mlr, p=2, dim=1)
        pdota = (-P_mlr * normed_A).sum(dim=1).view(1, -1, 1, 1) 
        A_kernel = normed_A.unsqueeze(2).unsqueeze(3)
        xa = F.conv2d(inputs, A_kernel)
        mobdota = beta * xa + alpha * pdota
        mobaddnorm = alpha**2 * pp.view(1, -1, 1, 1) + beta**2 * xx + 2 * alpha * beta * px
        sqrt_c = torch.sqrt(c) if torch.is_tensor(c) else c**0.5
        maxnorm = (1.0 - PROJ_EPS) / sqrt_c
        mobaddnorm_sqrt = torch.sqrt(torch.clamp(mobaddnorm, min=0.0))
        cond = mobaddnorm_sqrt > maxnorm
        proj_factor = torch.where(
            cond, 
            maxnorm / torch.clamp(mobaddnorm_sqrt, min=EPS), 
            torch.ones_like(mobaddnorm)
        )
        mobaddnormproj = torch.where(cond, maxnorm**2, mobaddnorm)
        lamb_px = 2.0 / torch.clamp(1.0 - c * mobaddnormproj, min=EPS) 
        sineterm = sqrt_c * mobdota * lamb_px * proj_factor
        A_norm = torch.norm(A_mlr, p=2, dim=1).view(1, -1, 1, 1)
        return (2.0 / sqrt_c) * A_norm * torch.asinh(sineterm)

class HSA_pixel(nn.Module):
    def __init__(self, channels):
        super(HSA_pixel, self).__init__()
        self.channels = channels

        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.pre_scale = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, c):

        B, C, H, W = x.shape
        N = H * W
        residual = x
        q_euc = self.q_proj(x).flatten(2)  
        k_euc = self.k_proj(x).flatten(2)  
        v_euc = self.v_proj(x).flatten(2) 
        q_euc = F.normalize(q_euc, p=2, dim=-1) * self.pre_scale
        k_euc = F.normalize(k_euc, p=2, dim=-1) * self.pre_scale
        v_euc = F.normalize(v_euc, p=2, dim=-1) * self.pre_scale
        q = PoincareMath.exp_map_0(q_euc, c=c)
        k = PoincareMath.exp_map_0(k_euc, c=c)  
        v = PoincareMath.exp_map_0(v_euc, c=c)  
        dist_matrix = PoincareMath.dist_sq(q, k, c=c)
        scores = -self.beta * dist_matrix + self.bias
        attn_weights = F.softmax(scores, dim=-1)  
        out_hyp = PoincareMath.einstein_midpoint(v, attn_weights, c=c)  
        out_euc = PoincareMath.log_map_0(out_hyp, c=c) 
        out_img = out_euc.reshape(B, C, H, W)          
        out_img = self.out_proj(out_img) + residual    
        out = out_img.permute(0, 2, 3, 1)              
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)                

        return out


class HSA_channel(nn.Module):
    def __init__(self, channels):
        super(HSA_channel, self).__init__()
        self.channels = channels
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, c):

        b, cc, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2) 
        residual = x_flat
        q = PoincareMath.exp_map_0(self.q_proj(x_flat), c=c)
        k = PoincareMath.exp_map_0(self.k_proj(x_flat), c=c)
        v = PoincareMath.exp_map_0(self.v_proj(x_flat), c=c)
        dist_matrix = PoincareMath.dist_sq(q, k, c=c)
        scores = -self.beta * dist_matrix + self.bias
        attn_weights = F.softmax(scores, dim=-1)
        out_hyp = PoincareMath.einstein_midpoint(v, attn_weights, c=c)
        out_euc = PoincareMath.log_map_0(out_hyp, c=c)
        out = self.out_proj(out_euc)
        out = self.norm(out + residual) 

        return out.transpose(1, 2).reshape(b, cc, h, w)


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
    def forward(self, x):
        return torch.mul(self.weight, x)

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1'):
        super(WTConv2d, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])
        self.wavelet_convs = nn.ModuleList([nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)])
        self.wavelet_scale = nn.ModuleList([_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)])
        if self.stride > 1: self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else: self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:,:,0,:,:]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])
        next_x_ll = 0
        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
        x = self.base_scale(self.base_conv(x)) + next_x_ll
        if self.do_stride is not None: x = self.do_stride(x)
        return x


class HyperbolicSegHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(HyperbolicSegHead, self).__init__()
        self.num_classes = num_classes    
        self.adjust_layer = nn.Conv2d(in_channels, in_channels, 1)
        self.pre_scale = nn.Parameter(torch.tensor(0.1))
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        self.p_vals = nn.Parameter(torch.randn(num_classes, in_channels))
        self.a_vals = nn.Parameter(torch.randn(num_classes, in_channels))
        self.b_vals = nn.Parameter(torch.zeros(num_classes))   
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.a_vals, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.p_vals, a=math.sqrt(5))

    def forward(self, x, c):

        B, C, H, W = x.shape 
        x = self.adjust_layer(x) 
        x_normalized = F.normalize(x, p=2, dim=1) 
        x_scaled = x_normalized * self.pre_scale
        z_query = PoincareMath.exp_map_0(x_scaled, c=c)
        p_hyp = PoincareMath.exp_map_0(self.p_vals, c=c)   
        p_len_sq = torch.sum(p_hyp.pow(2), dim=1, keepdim=True)
        conformal_factor = 1.0 - c * p_len_sq
        a_hyp = self.a_vals * conformal_factor
        logits = PoincareMath.hyp_mlr(z_query, c, p_hyp, a_hyp)
        logits = logits * self.logit_scale
        logits = logits + self.b_vals.view(1, -1, 1, 1) 

        return logits


class H2SANet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, wave_level=2, model_size='mid', init_c=1.0):
        super(H2SANet, self).__init__()
        if model_size == 'small': num_channels = [32, 64, 128, 256, 512]
        elif model_size == 'mid': num_channels = [64, 128, 256, 512, 1024]
        elif model_size == 'large': num_channels = [128, 256, 512, 1024, 2048]
        else: raise ValueError(f"Unsupported model size: {model_size}")
            
        self.in_conv = nn.Conv2d(in_channels, num_channels[0], kernel_size=1)
        
        self.encoder1 = self.conv_block(num_channels[0], num_channels[1], wave_level)
        self.encoder2 = self.conv_block(num_channels[1], num_channels[2], wave_level)
        self.encoder3 = self.conv_block(num_channels[2], num_channels[3], wave_level)
        self.encoder4 = self.conv_block(num_channels[3], num_channels[4], wave_level)
        
        self.middle = self.midconv_block(num_channels[4], num_channels[4])

        self.HSA_2 = HSA_pixel(channels=num_channels[2])
        self.HSA_3 = HSA_channel(num_channels[3])
        self.HSA_4 = HSA_channel(num_channels[4])

        self.decoder4 = self.dconv_block(num_channels[4]*2, num_channels[3], wave_level)
        self.decoder3 = self.dconv_block(num_channels[3]*2, num_channels[2], wave_level)
        self.decoder2 = self.dconv_block(num_channels[2]*2, num_channels[1], wave_level)
        self.decoder1 = self.dconv_block(num_channels[1]*2, num_channels[0], wave_level)

        self.seg_head = HyperbolicSegHead(in_channels=num_channels[0], num_classes=out_channels)
        
        self.sigmoid = nn.Sigmoid()
 
        inv_softplus_c = math.log(math.exp(init_c) - 1)
        self.c_raw = nn.Parameter(torch.tensor(inv_softplus_c, dtype=torch.float32))


    def midconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),  
        )

    def conv_block(self, in_channels, out_channels, wave_level):
        return nn.Sequential(
            WTConv2d(in_channels, in_channels, wt_levels=wave_level),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
    
    def dconv_block(self, in_channels, out_channels, wave_level):
        return nn.Sequential(
            WTConv2d(in_channels, in_channels, wt_levels=wave_level),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )


    def forward(self, x):
        cur_c = F.softplus(self.c_raw) + 1e-5
        x = self.in_conv(x)
        enc1 = F.leaky_relu(self.encoder1(x))
        enc2 = F.leaky_relu(self.encoder2(enc1))
        enc2 = self.HSA_2(enc2, cur_c)
        enc3 = F.leaky_relu(self.encoder3(enc2))
        enc3 = self.HSA_3(enc3, cur_c)  
        enc4 = F.leaky_relu(self.encoder4(enc3))
        enc4 = self.HSA_4(enc4, cur_c)

        middle = self.middle(enc4)

        dec4 = self.decoder4(torch.cat([middle, enc4], 1))
        dec4 = F.leaky_relu(F.interpolate(dec4, scale_factor=(2,2),mode ='bilinear'))
        dec3 = self.decoder3(torch.cat([dec4, enc3], 1))
        dec3 = F.leaky_relu(F.interpolate(dec3, scale_factor=(2,2),mode ='bilinear'))
        dec2 = self.decoder2(torch.cat([dec3, enc2], 1))
        dec2 = F.leaky_relu(F.interpolate(dec2, scale_factor=(2,2),mode ='bilinear'))
        dec1 = self.decoder1(torch.cat([dec2, enc1], 1))
        dec1 = F.leaky_relu(F.interpolate(dec1, scale_factor=(2,2),mode ='bilinear'))

        output = self.seg_head(dec1, c=cur_c)
        
        return output

