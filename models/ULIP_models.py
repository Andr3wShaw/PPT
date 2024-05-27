'''
 * Adapted from ULIP (https://github.com/salesforce/ULIP)
 * By Hongyu Sun
'''
from collections import OrderedDict

import os
import numpy as np
import torch
from torch import nn
from models.pointnet2.pointnet2 import Pointnet2_Ssg, Pointnet2_Msg
from data.dataset_3d import  *

from models import losses
from torch.nn.parameter import Parameter
from easydict import EasyDict

from utils.tokenizer import SimpleTokenizer


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class PromptLearner(nn.Module):
    def __init__(self, token_embedding, kwargs):
        super().__init__()

        self.class_name_position = kwargs.class_name_position
        self.classnames = kwargs.classnames
        self.num_learnable_prompt_tokens = kwargs.num_learnable_prompt_tokens
        self.transformer_width = kwargs.transformer_width
        self.device = kwargs.device

        if kwargs.template_init != '':
            template_init = kwargs.template_init.replace("_", " ")  
            self.num_learnable_prompt_tokens = len(template_init.split(' '))
            prompt_prefix = template_init
        else:
            self.num_learnable_prompt_tokens = kwargs.num_learnable_prompt_tokens
            prompt_prefix = " ".join(["X"] * self.num_learnable_prompt_tokens)

        self.learnable_tokens = nn.Parameter(torch.empty(self.num_learnable_prompt_tokens, self.transformer_width))

        classnames = [name.replace("_", " ") for name in self.classnames]  
        tokenizer = SimpleTokenizer()
        self.name_lengths = [len(tokenizer.encode(name)) for name in classnames]  
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        self.tokenized_prompts = torch.stack([tokenizer(p) for p in prompts])
        self.embedding = token_embedding(self.tokenized_prompts).to(self.device)

    def forward(self):
        num_classes = self.embedding.shape[0]
        if self.learnable_tokens.dim() == 2:
            learnable_tokens = self.learnable_tokens.unsqueeze(0).repeat(num_classes, 1, 1).to(self.device)

        prefix = self.embedding[:, :1, :]   
        suffix = self.embedding[:, 1+self.num_learnable_prompt_tokens:, :]  

        if self.class_name_position == "front":
            prompts = []
            for i in range(num_classes):
                shape_name_len = self.name_lengths[i]

                prefix_i = prefix[i : i+1, :1, :]
                class_i = suffix[i : i+1, :shape_name_len, :]
                learnable_tokens_i = learnable_tokens[i : i+1, :, :]
                suffix_i = suffix[i : i+1, shape_name_len:, :]

                prompt = torch.cat([prefix_i, class_i, learnable_tokens_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_name_position == "middle":
            prompts = []
            half_len = self.num_learnable_prompt_tokens // 2
            for i in range(num_classes):
                shape_name_len = self.name_lengths[i]

                prefix_i = prefix[i : i+1, :, :]
                learnable_tokens_i_half1 = learnable_tokens[i : i+1, :half_len, :]
                class_i = suffix[i : i+1, :shape_name_len, :]
                learnable_tokens_i_half2 = learnable_tokens[i : i+1, half_len:, :]
                suffix_i = suffix[i : i+1, shape_name_len:, :]

                prompt = torch.cat([prefix_i, learnable_tokens_i_half1, class_i, 
                                    learnable_tokens_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_name_position == "end":
            prompts = torch.cat([prefix, learnable_tokens, suffix], dim=1)

        else:
            raise ValueError(f'`class_name_position`: {self.class_name_position} not in supported modes ["front", "middle", "end"]')

        return prompts.to(self.device)

class PromptLearner3D(nn.Module):
    def __init__(self, point_encoder, kwargs):
        super(PromptLearner3D, self).__init__()
        self.point_encoder = point_encoder
        self.context_length = kwargs.context_length
        self.token_embedding = nn.Parameter(torch.randn(self.context_length, kwargs.pc_feat_dims))
        self.learnable_tokens = nn.Parameter(torch.randn(self.context_length, kwargs.pc_feat_dims))

    def forward(self, point_cloud, cls_label=None):
        batch_size = point_cloud.size(0)
        context = self.learnable_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Ensure point_cloud has the shape [B, N, C]
        print(f"Initial point_cloud shape: {point_cloud.shape}")
        if point_cloud.dim() == 2:  # Assuming point_cloud has shape [batch_size, feature_dim]
            point_cloud = point_cloud.view(batch_size, -1, 3)  # Change shape to [batch_size, N, 3]
        print(f"Reshaped point_cloud shape: {point_cloud.shape}")

        pc_feat = self.point_encoder(point_cloud)  # Encode the point cloud to get features
        print(f"pc_feat shape after point_encoder: {pc_feat.shape}")

        # Reshape pc_feat to match context dimensions if necessary
        if pc_feat.dim() == 2:  # Assuming pc_feat has shape [batch_size, feature_dim]
            pc_feat = pc_feat.unsqueeze(1)  # Change shape to [batch_size, 1, feature_dim]
        print(f"Reshaped pc_feat shape: {pc_feat.shape}")

        combined = torch.cat([context, pc_feat], dim=1)  # Concatenate learnable tokens with point cloud features
        print(f"combined shape: {combined.shape}")
        return combined

class ULIP_WITH_IMAGE(nn.Module):
    def __init__(self, point_encoder, **kwargs):
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.task = kwargs.task
        self.context_length = kwargs.context_length
        self.device = kwargs.device

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, kwargs.embed_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.point_encoder = point_encoder

        # --- learning to prompt --- 
        self.prompt_learner = PromptLearner(self.token_embedding, kwargs)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # 3D prompt learner
        self.prompt_learner_3d = PromptLearner3D(self.point_encoder, kwargs)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.prompt_learner.learnable_tokens, std=0.02)
        nn.init.normal_(self.prompt_learner_3d.learnable_tokens, std=0.02)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def encode_text(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.unsqueeze(dim=0).repeat(prompts.shape[0], 1, 1)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_pc(self, pc, pc_prompts, cls_label=None):
        # Use the original point cloud data with the point_encoder
        if self.task == 'partseg':
            pc_feat = self.point_encoder(pc, cls_label)
        else:
            pc_feat = self.point_encoder(pc)

        # Project the point cloud features
        pc_embed = pc_feat @ self.pc_projection

        # Combine the point cloud embeddings with the context tokens
        context_combined = pc_prompts @ self.pc_projection

        # Sum the context_combined and pc_embed
        combined_embed = context_combined + pc_embed.unsqueeze(1)

        return combined_embed

    def forward(self, pc, cls_label=None):
        if self.task == 'partseg':
            pc_prompts = self.prompt_learner_3d(pc, cls_label)
        else:
            pc_prompts = self.prompt_learner_3d(pc)

        # Use the original point cloud data for point_encoder
        pc_embed = self.encode_pc(pc, pc_prompts)

        # Average the combined embeddings across the token dimension
        pc_embed = pc_embed.mean(dim=1)

        prompts = self.prompt_learner()  # forward pass
        tokenized_prompts = self.tokenized_prompts

        text_embed = self.encode_text(prompts, tokenized_prompts)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * pc_embed @ text_embed.t()

        return logits

def get_loss():
    return losses.ULIPWithImageLoss()


def get_metric_names():
    return ['loss', 'acc']


def ULIP_PN_SSG(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    point_encoder = Pointnet2_Ssg()
    pc_feat_dims = 256
    # =====================================================================

    # 前向传播返回的是一个字典，包含文本特征、点云特征、图像特征，外加一个归一化因子
    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77, 
            vocab_size=49408, classnames=args.classnames, template_init=args.template_init, class_name_position=args.class_name_position, 
            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12, 
            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_point_model = torch.load('./data/pretrained_models/pointnet2_ssg.pt', map_location=torch.device(f'cpu'))
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_point_model_params.items()}

        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device(f'cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        # 首先这里用的是 named_parameters() 而不是 parameters，表明 model 的参数都是有对应名称的
        for name, param in model.named_parameters():
            if name == 'prompt_learner.learnable_tokens': 
                continue

            if name in pretrain_point_model_params:
                if isinstance(pretrain_point_model_params[name], Parameter):
                    param_new = pretrain_point_model_params[name].data
                else:
                    param_new = pretrain_point_model_params[name]
            else:
                if isinstance(pretrain_slip_model_params[name], Parameter):
                    param_new = pretrain_slip_model_params[name].data
                else:
                    param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model


def ULIP_PN_MSG(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    point_encoder = Pointnet2_Msg()
    pc_feat_dims = 256
    # =====================================================================

    # 前向传播返回的是一个字典，包含文本特征、点云特征、图像特征，外加一个归一化因子
    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77, 
            vocab_size=49408, classnames=args.classnames, template_init=args.template_init, class_name_position=args.class_name_position, 
            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12, 
            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_point_model = torch.load('./data/pretrained_models/pointnet2_msg_1kpts.pt', map_location=torch.device('cpu'))
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_point_model_params.items()}

        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        # 首先这里用的是 named_parameters() 而不是 parameters，表明 model 的参数都是有对应名称的
        for name, param in model.named_parameters():
            if name == 'prompt_learner.learnable_tokens': 
                continue

            if name in pretrain_point_model_params:
                if isinstance(pretrain_point_model_params[name], Parameter):
                    param_new = pretrain_point_model_params[name].data
                else:
                    param_new = pretrain_point_model_params[name]
            else:
                if isinstance(pretrain_slip_model_params[name], Parameter):
                    param_new = pretrain_slip_model_params[name].data
                else:
                    param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model


def ULIP_PN_MLP(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointmlp.pointMLP import pointMLP
    point_encoder = pointMLP()
    pc_feat_dims = 256
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77, 
            vocab_size=49408, classnames=args.classnames, template_init=args.template_init, class_name_position=args.class_name_position, 
            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12, 
            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_point_model = torch.load('./data/pretrained_models/pointmlp.pt', map_location=torch.device('cpu'))
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_point_model_params.items()}
        
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name == 'prompt_learner.learnable_tokens': 
                continue

            if name in pretrain_point_model_params:
                if isinstance(pretrain_point_model_params[name], Parameter):
                    param_new = pretrain_point_model_params[name].data
                else:
                    param_new = pretrain_point_model_params[name]
            else:
                if isinstance(pretrain_slip_model_params[name], Parameter):
                    param_new = pretrain_slip_model_params[name].data
                else:
                    param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model

def ULIP_PointBERT(args):
    from models.pointbert.point_encoder import PointTransformer
    config_addr = './models/pointbert/PointTransformer_8192point.yaml'
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer(config.model, args=args)
    pc_feat_dims = 768

    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77, 
            vocab_size=49408, classnames=args.classnames, template_init=args.template_init, class_name_position=args.class_name_position, 
            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12, 
            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    unfreeze_modules = []  # prompt_only
    if args.head_type > 0:  # linear
        unfreeze_modules += ['point_encoder.blocks.blocks.11.norm2.weight', 'point_encoder.blocks.blocks.11.norm2.bias',
                             'point_encoder.blocks.blocks.11.mlp.fc2.weight', 'point_encoder.blocks.blocks.11.mlp.fc2.bias']
    if args.head_type > 1:  # mlp
        unfreeze_modules += ['point_encoder.blocks.blocks.11.norm1.weight', 'point_encoder.blocks.blocks.11.norm1.bias',
                             'point_encoder.blocks.blocks.11.mlp.fc1.weight', 'point_encoder.blocks.blocks.11.mlp.fc1.bias']
    if args.head_type > 2:  # atten_block
        unfreeze_modules += ['point_encoder.blocks.blocks.11.attn.qkv.weight', 'point_encoder.blocks.blocks.11.attn.proj.weight',
                             'point_encoder.blocks.blocks.11.attn.proj.bias']

    if not args.evaluate_3d:
        if args.ulip2:
            pretrain_point_model = torch.load('./data/pretrained_models/pointbert_ulip2.pt', map_location='cpu')
        else:
            pretrain_point_model = torch.load('./data/pretrained_models/pointbert.pt', map_location='cpu')
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                       pretrain_point_model_params.items()}

        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location='cpu')
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name in ['prompt_learner.learnable_tokens', 'prompt_learner_3d.learnable_tokens', 'point_encoder.cls_head']:
                continue

            if name in unfreeze_modules:  # Allow the last block of transformer to update
                continue

            if name in pretrain_point_model_params:
                param_new = pretrain_point_model_params[name].data if isinstance(pretrain_point_model_params[name], torch.nn.Parameter) else pretrain_point_model_params[name]
                param.requires_grad = False
                param.data.copy_(param_new)
                print('load {} from pretrain_point_model_params and freeze'.format(name))
            elif name in pretrain_slip_model_params:
                param_new = pretrain_slip_model_params[name].data if isinstance(pretrain_slip_model_params[name], torch.nn.Parameter) else pretrain_slip_model_params[name]
                param.requires_grad = False
                param.data.copy_(param_new)
                print('load {} from pretrain_slip_model_params and freeze'.format(name))
            else:
                print('parameter {} not found in pretrained models, initializing randomly'.format(name))
                param.requires_grad = True  # Allow updating this parameter

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n====================\n\tNumber of learnable params:', params, '\n====================\n')

    return model

def ULIP_PointBERT_partseg(args):
    CUR_DIR = os.path.dirname(os.path.abspath(__file__)) # current dir
    PROJ_DIR = os.path.dirname(CUR_DIR) # project dir
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointbert.point_encoder import PointTransformer_partseg
    config_addr = os.path.join(PROJ_DIR, 'models/pointbert/PointTransformer_8192point.yaml')
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer_partseg(config.model, args=args)
    pc_feat_dims = 128
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77, 
            vocab_size=49408, classnames=args.classnames, template_init=args.template_init, class_name_position=args.class_name_position, 
            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12, 
            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    count = 0
    if not args.evaluate_3d:
        # load the pretrained model
        if args.ulip2:
            pretrain_point_model = torch.load(os.path.join(PROJ_DIR, 'data/pretrained_models/pointbert_ulip2.pt'), map_location=torch.device('cpu'))
        else:
            pretrain_point_model = torch.load(os.path.join(PROJ_DIR, 'data/pretrained_models/pointbert.pt'), map_location=torch.device('cpu'))
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_point_model_params.items()}
        
        pretrain_slip_model = torch.load(os.path.join(PROJ_DIR, 'data/initialize_models/slip_base_100ep.pt'), map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name.startswith('prompt_learner'):
                count += 1
                print('------ prompt_learner params:', name)

            elif name.startswith('point_encoder.'):
                if name in pretrain_point_model_params:
                    param.requires_grad = False
                    print('load {} and freeze'.format(name))
                else:
                    count += 1
                    print('------ pointbert_partseg params:', name)

            else:   # image and text encoder
                param.requires_grad = False
                print('load {} and freeze'.format(name))

        print(f'>>>>>> {count}')

    return model


def ULIP_PN_NEXT(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointnext.pointnext import PointNEXT
    point_encoder = PointNEXT()
    pc_feat_dims = 256
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77, 
            vocab_size=49408, classnames=args.classnames, template_init=args.template_init, class_name_position=args.class_name_position, 
            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12, 
            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_point_model = torch.load('./data/pretrained_models/pointnext.pt', map_location=torch.device('cpu'))
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_point_model_params.items()}
        
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name == 'prompt_learner.learnable_tokens': 
                continue

            if name in pretrain_point_model_params:
                if isinstance(pretrain_point_model_params[name], Parameter):
                    param_new = pretrain_point_model_params[name].data
                else:
                    param_new = pretrain_point_model_params[name]
            else:
                if isinstance(pretrain_slip_model_params[name], Parameter):
                    param_new = pretrain_slip_model_params[name].data
                else:
                    param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model


def ULIP_CUSTOMIZED(args):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # This is a sample template to pre-train your customized 3D backbones, please modify this part accordingly!
    from models.customized_backbone.customized_backbone import CUSTOMIZED_BACKBONE
    point_encoder = CUSTOMIZED_BACKBONE()
    # We assume you might have different point cloud output feature dimension,
    # we added a projecting layer to unify the point cloud output dimension before doing the multimodal alignment,
    # please change the output feature dimension here.
    pc_feat_dims = 512
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, 
                            context_length=77, vocab_size=49408, template_init=args.template_init, 
                            class_name_position=args.class_name_position, num_learnable_prompt_tokens=args.num_learnable_prompt_tokens,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model