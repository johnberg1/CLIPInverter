import torch.nn as nn
import torch   

class CLIPLoss(nn.Module):    
    def __init__(self, model, StyleGAN_size=256):
        super(CLIPLoss, self).__init__()
        self.model = model
        self.upsample = nn.Upsample(scale_factor=7)
        self.avg_pool = nn.AvgPool2d(kernel_size=StyleGAN_size // 32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image,text)[0] / 100
        return similarity
    
class CLIPImageLoss(nn.Module):    
    def __init__(self, model, StyleGAN_size=256):
        super(CLIPImageLoss, self).__init__()
        self.model = model
        self.upsample = nn.Upsample(scale_factor=7)
        self.avg_pool = nn.AvgPool2d(kernel_size=StyleGAN_size // 32)

    def forward(self, image1, image2):
        image1 = self.avg_pool(self.upsample(image1))
        image2 = self.avg_pool(self.upsample(image2))
        image1_features = self.model.encode_image(image1)
        image2_features = self.model.encode_image(image2)
        
        image1_features = image1_features / image1_features.norm(dim=1, keepdim=True)
        image2_features = image2_features / image2_features.norm(dim=1, keepdim=True)
        
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image1_features @ image2_features.t()
        
        similarity = 1 - logits_per_image / 100
        return similarity
		
class DirectionalCLIPLoss(nn.Module):    
    def __init__(self, model, StyleGAN_size=256):
        super(DirectionalCLIPLoss, self).__init__()
        self.model = model
        self.upsample = nn.Upsample(scale_factor=7)
        self.avg_pool = nn.AvgPool2d(kernel_size=StyleGAN_size // 32)
        self.loss_func = nn.CosineSimilarity()
		
    def compute_text_direction(self, source, target):
        text_direction = (target - source)
        text_direction = text_direction / (text_direction.norm(dim=-1, keepdim=True) + 1e-6)
        return text_direction
		
    def compute_image_direction(self, source, target):
        image_direction = (target - source)
        image_direction = image_direction / (image_direction.clone().norm(dim=-1, keepdim=True) + 1e-6)
        return image_direction

    def forward(self, source_image, target_image, source_text, target_text):
        source_image = self.avg_pool(self.upsample(source_image))
        target_image = self.avg_pool(self.upsample(target_image))
        
        source_image_embed = self.model.encode_image(source_image)
        target_image_embed = self.model.encode_image(target_image)
        source_image_embed = source_image_embed / (source_image_embed.clone().norm(dim=-1, keepdim=True) + 1e-6)
        target_image_embed = target_image_embed / (target_image_embed.clone().norm(dim=-1, keepdim=True) + 1e-6)
        
        source_text_embed = self.model.encode_text(source_text)
        target_text_embed = self.model.encode_text(target_text)
        source_text_embed = source_text_embed / (source_text_embed.norm(dim=-1, keepdim=True) + 1e-6)
        target_text_embed = target_text_embed / (target_text_embed.norm(dim=-1, keepdim=True) + 1e-6)
        
        text_direction = self.compute_text_direction(source_text_embed, target_text_embed)
        image_direction = self.compute_image_direction(source_image_embed, target_image_embed)
        directional_loss = 1. - torch.sum(image_direction * text_direction, dim=-1) 
        return directional_loss