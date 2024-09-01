import glob
from tqdm import tqdm
import matplotlib.pyplot as plt  
import string  
import argparse  
import os  
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from collections import OrderedDict
import numpy as np
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['HTTP_PROXY'] = "http://*:7890"
# os.environ['HTTPS_PROXY'] = "http://*:7890"
# os.environ['ALL_PROXY'] = "socks://*:7891"
nltk.download('stopwords')


def parse_args():
    parser = argparse.ArgumentParser(description='Word_Frequency_Analysis')
    parser.add_argument('--root_path', type=str, help='')
    parser.add_argument('--save_path', type=str, help='save_path', default='')
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

def get_list(folder_path):
    image_paths_real = [] 
    image_paths_fake = [] 
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.txt')):
                abspath_tmp = os.path.abspath(os.path.join(root, file))
                if '/0_real/' in  abspath_tmp: image_paths_real.append(abspath_tmp)
                if '/1_fake/' in  abspath_tmp: image_paths_fake.append(abspath_tmp)
    return image_paths_real, image_paths_fake

def get_words_counts(image_paths):
    all_text = []
    for tpath in image_paths:
        with open(tpath, 'r') as file:
            all_text.append( file.read() )
    content = ' '.join(all_text)
    words = re.findall(r'\b\w+\b', content.lower())

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    word_counts = Counter(filtered_words)

    common_words = word_counts.most_common(20)
    common_words_dict = {}
    for common_word in common_words:
        common_words_dict[ common_word[0] ] = common_word[1]
    return common_words_dict, dict(word_counts)

if __name__ == '__main__':
    opt = parse_args()
    os.makedirs(os.path.dirname(opt.save_path), mode = 0o777, exist_ok = True)
    if opt.root_path[-1]=='/': opt.root_path = opt.root_path[:-1]
    if isinstance(opt.root_path, list):
        image_paths_real, image_paths_fake = [], []
        for textp in opt.root_path:
            tmp_real, tmp_fake = get_list(textp)
            image_paths_real.extend( tmp_real )
            image_paths_fake.extend( tmp_fake )
    else:
        image_paths_real, image_paths_fake = get_list(opt.root_path)

    words_counts_real, words_counts_real_all = get_words_counts(image_paths_real)
    words_counts_fake, words_counts_fake_all = get_words_counts(image_paths_fake)
    
    all_words = set( list(words_counts_real.keys()) + list(words_counts_fake.keys()) )
    for word in all_words:
        if word not in words_counts_real.keys(): 
            words_counts_real[word] = words_counts_real_all[word] if word in list(words_counts_real_all.keys()) else 0
        if word not in words_counts_fake.keys(): 
            words_counts_fake[word] = words_counts_fake_all[word] if word in list(words_counts_fake_all.keys()) else 0
        
    words_counts_fake_sorted = OrderedDict((key, words_counts_fake[key]) for key in words_counts_real if key in words_counts_fake)
    
    words_counts_real        = [(k, v) for k, v in words_counts_real.items()       ]
    words_counts_fake_sorted = [(k, v) for k, v in words_counts_fake_sorted.items()]
    
    words_real, counts_real = zip(*words_counts_real)
    words_fake, counts_fake = zip(*words_counts_fake_sorted)
    assert words_real == words_fake
    
    print(f'words_real: {words_real}')
    print(f'counts_real: {counts_real}')
    
    print(f'words_fake: {words_fake}')
    print(f'counts_fake: {counts_fake}')
    # exit()
    words_real, counts_real = words_real[:15], counts_real[:15]
    words_fake, counts_fake = words_fake[:15], counts_fake[:15]
    plt.figure(figsize=(16, 8))
    width = 0.35
    x = np.arange(len(words_real)) 
    plt.bar(x - width/2, counts_real, width, label=f'{" ".join(opt.root_path.split("/")[-2:])} real')
    plt.bar(x + width/2, counts_fake, width, label=f'{" ".join(opt.root_path.split("/")[-2:])} fake')
    # plt.ylabel('Frequency')
    # plt.title('Top 15 Most Common Words')
    plt.xticks(x, labels= words_fake, rotation=90, fontsize=35)
    plt.legend(prop = {'size':20})
    plt.show()
    plt.savefig(f'{opt.save_path}', bbox_inches='tight', pad_inches=0.1)
