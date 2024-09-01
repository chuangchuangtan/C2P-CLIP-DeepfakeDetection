
# conda -n create clip-text-decoder python=3.8.5
import argparse
import os
from torch import nn
import numpy as np
import torch # 2.3.1+cu118 conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
import torch.nn.functional as nnf 
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # 4.25.0
import skimage.io as io
import PIL.Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import time
import warnings

warnings.filterwarnings('ignore')

N = type(None)
T = torch.Tensor
D = torch.device

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)



class ClipCaptionModel(nn.Module):

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = MLP(
            (
                prefix_size,
                (self.gpt_embedding_size * prefix_length) // 2,
                self.gpt_embedding_size * prefix_length,
            )
        )

def generate2( model, tokenizer, tokens=None, prompt=None, embed=None, entry_count=1, entry_length=67, top_p=0.8, temperature=1.0, stop_token: str = ".",):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]



def parse_args():
    parser = argparse.ArgumentParser(description='decode detection feature to text')
    parser.add_argument('--prefix_length'     ,  type=int  , default=10                                                                                               )
    parser.add_argument('--model_path'        ,  type=str  , default='https://www.now61.com/f/Xljmi0/coco_prefix_latest.pt', help='model_path'                        )
    parser.add_argument('--image_path'        ,  type=str  , default=''                                                    , help='image_path'         , required=True)
    parser.add_argument('--fc_path'           ,  type=str  , default='https://www.now61.com/f/qwvoH5/fc_parameters.pth'    , help='fc_path'                           )
    parser.add_argument('--cal_detection_feat',              action="store_true"                                           , help='cal_detection_feat'                )
    parser.add_argument('--device'            ,  type=str  , default='cuda:0'                                              , help='cuda:n or cpu'                     )
    args = parser.parse_args()
    
    def print_options(parser, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
    print_options(parser, args)
    return args

def get_text(image_features, tokenizer, model, fc_path, cal_detection_feat=True, prefix_length=10, device='cpu'):
    mod = torch.hub.load_state_dict_from_url( fc_path, map_location="cpu", progress=True ) if fc_path.startswith("http") else torch.load(fc_path, map_location="cpu")
    weight, bias =  mod['fc.weight'].to(device), mod['fc.bias'].to(device)
    with torch.no_grad():
        prob = nnf.linear(image_features, weight, bias).sigmoid().cpu().numpy()[0][0]
        dict_prob = {False: 'Fake', True: 'Real'}
        # print( f'\nPredicted prob: {prob}, {dict_prob[prob<0.5]}' )
        # tmp=image_features;print(f'image_features: {tmp.shape}, max: {tmp.max()}, min: {tmp.min()}, mean: {tmp.mean()}')
        if cal_detection_feat:  image_features = torch.mul(image_features, weight) + bias
        image_features /= image_features.norm(2, dim=-1, keepdim=True)
        prefix_embed = model.clip_project(image_features).reshape(1, prefix_length, -1)
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
    return generated_text_prefix

def get_clip_model(clip_name = 'openai/clip-vit-large-patch14', device = 'cpu'):
    clipmodel        = CLIPModel.from_pretrained(clip_name)
    processor    = CLIPProcessor.from_pretrained(clip_name)
    del clipmodel.text_model 
    del clipmodel.text_projection 
    del clipmodel.logit_scale
    clipmodel = clipmodel.to(device)
    return clipmodel, processor

def get_clipcap_model(model_path, prefix_length=10, device='cpu'):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = ClipCaptionModel(prefix_length, prefix_size=768)
    pretrained = torch.hub.load_state_dict_from_url( model_path, map_location = "cpu", progress = True ) if model_path.startswith("http") else torch.load(model_path, map_location="cpu")
    model.load_state_dict(pretrained) 
    assert pretrained.keys() == model.state_dict().keys()
    model = model.eval() 
    model = model.to(device)
    return model, tokenizer
    
def get_image_features(image_path, clipmodel, processor, device='cpu'):
    with torch.no_grad():
        image          = PIL.Image.fromarray( io.imread(image_path) )
        inputs         = processor(images=image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
        image_features = clipmodel.get_image_features(**inputs)
    return image_features

if __name__ == '__main__':
    opt = parse_args()
    device = torch.device(opt.device)
    
    clipmodel, processor = get_clip_model(clip_name='openai/clip-vit-large-patch14', device = device)
    model, tokenizer = get_clipcap_model(opt.model_path, device=device)
    
    image_features = get_image_features(opt.image_path, clipmodel, processor, device = device)
    text = get_text(image_features, tokenizer, model, opt.fc_path, opt.cal_detection_feat, device=device)
    print(text)





