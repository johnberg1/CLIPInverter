from torch import nn
from models.stylegan2.model import PixelNorm
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential, Module, Conv2d, GroupNorm

class TextModulationModule(Module):
    def __init__(self, in_channels):
        super(TextModulationModule, self).__init__()
        self.conv = Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False)
        self.norm = GroupNorm(32, in_channels)
        self.gamma_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, in_channels))
        self.beta_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, in_channels))
        self.leakyrelu = LeakyReLU()
        
    def forward(self, x, embedding):
        x = self.conv(x)
        x = self.norm(x)
        log_gamma = self.gamma_function(embedding.float())
        gamma = log_gamma.exp().unsqueeze(2).unsqueeze(3)
        beta = self.beta_function(embedding.float()).unsqueeze(2).unsqueeze(3)
        out = x * (1 + gamma) + beta
        out = self.leakyrelu(out)
        return out
        
class SubTextMapper(Module):
    def __init__(self, opts, in_channels):
        super(SubTextMapper, self).__init__()
        self.opts = opts
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([TextModulationModule(in_channels) for _ in range(1)])
        
    def forward(self, x, embedding):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
            x = modulation_module(x, embedding)
        return x

class CLIPAdapter(Module): 
    def __init__(self, opts):
        super(CLIPAdapter, self).__init__()
        self.opts = opts

        if not opts.no_coarse_mapper: 
            self.coarse_mapping = SubTextMapper(opts, 512)
        if not opts.no_medium_mapper:
            self.medium_mapping = SubTextMapper(opts, 256)
        if not opts.no_fine_mapper:
            self.fine_mapping = SubTextMapper(opts, 128)


    def forward(self, features, txt_embed):
        txt_embed = txt_embed.detach()
        c1, c2, c3 = features

        if not self.opts.no_coarse_mapper:
            c3 = self.coarse_mapping(c3, txt_embed)
        if not self.opts.no_medium_mapper:
            c2 = self.medium_mapping(c2, txt_embed)
        if not self.opts.no_fine_mapper:
            c1 = self.fine_mapping(c1, txt_embed)
        return (c1,c2,c3)