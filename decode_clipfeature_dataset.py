import argparse
import time
import os
import torch
from tqdm import tqdm
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
from decode_clipfeature_oneImage import get_clip_model, get_clipcap_model, get_image_features, get_text

def parse_args():
    parser = argparse.ArgumentParser(description='decode detection feature to text')
    parser.add_argument('--prefix_length'     ,  type=int  , default=10                                                                                )
    parser.add_argument('--model_path'        ,  type=str  , default='https://www.now61.com/f/Xljmi0/coco_prefix_latest.pt', help='model_path'         )
    parser.add_argument('--images_root'       ,  type=str  , default=''                                                    , help='image_path'         , required=True)
    parser.add_argument('--save_path'         ,  type=str  , default=''                                                    , help='image_path'         , required=True)
    parser.add_argument('--fc_path'           ,  type=str  , default='https://www.now61.com/f/qwvoH5/fc_parameters.pth'    , help='fc_path'            )
    parser.add_argument('--cal_detection_feat',              action="store_true"                                           , help='cal_detection_feat' )
    parser.add_argument('--device'            ,  type=str  , default='cuda:0'                                              , help='cuda:n or cpu'      )
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


def get_image_files_in_directory(directory):
    image_extensions = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]
    image_files = []
    for root, dirs, files in os.walk(directory, followlinks=True):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                absolute_path = os.path.join(root, file)
                image_files.append(absolute_path)
    return image_files

if __name__ == '__main__':
    opt = parse_args()
    device = torch.device(opt.device)
    assert os.path.exists(opt.images_root)
    
    opt.images_root = os.path.abspath(opt.images_root)
    opt.save_path   = os.path.abspath(opt.save_path)
    os.makedirs(opt.save_path, mode = 0o777, exist_ok = True) 
    
    image_files = get_image_files_in_directory(opt.images_root)
    print(f'len(image_files): {len(image_files)}')

    clipmodel, processor = get_clip_model(clip_name='openai/clip-vit-large-patch14', device = device)
    model, tokenizer = get_clipcap_model(opt.model_path, device=device)
    
    for image_file in tqdm(image_files):
        image_features = get_image_features(image_file, clipmodel, processor, device = device)
        text = get_text(image_features, tokenizer, model, opt.fc_path, opt.cal_detection_feat, device=device)
        text_save_path = os.path.splitext( image_file.replace(opt.images_root, opt.save_path) )[0]
        os.makedirs(os.path.dirname(text_save_path), mode=0o777, exist_ok=True)
        with open(f'{text_save_path}.txt', 'w') as file:
            file.write(text)