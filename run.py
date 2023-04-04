import torch
import numpy as np
import cv2
import numpy
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.refcoco import RefcocoTask
from models.ofa import OFAModel
from PIL import Image
import pandas as pd
import requests
from tqdm.auto import tqdm
# Register refcoco task
tasks.register_task('refcoco', RefcocoTask)
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--csv_test', type=str,  default='/home/komleva/WSDMCup2023/train_sample.csv')
parser.add_argument('--imgs_test', type=str,  default='/home/komleva/WSDMCup2023/imgs')
parser.add_argument('--out', type=str,  default='./answer.csv')
args = parser.parse_args()
# turn on cuda if GPU is available
use_cuda = True#torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = True
f = open('./out.txt', 'w')
# Load pretrained ckpt & config
overrides={"bpe_dir":"utils/BPE"}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        #utils.split_paths('/home/komleva/mdetr/ofa/OFA/run_scripts/refcoco/refcoco_checkpoints_my/{10,}_{3e-5,}_{512,}/checkpoint.best_score_0.8970.pt'),#('/home/komleva/mdetr/ofa/refcocog_large_best.pt'),
        #utils.split_paths('/home/komleva/mdetr/ofa/OFA/run_scripts/refcoco/refcoco_checkpoints_my/{10,}_{3e-5,}_{512,}/checkpoint.best_score_0.9840.pt'),
        #utils.split_paths('/home/komleva/mdetr/ofa/OFA/run_scripts/refcoco/refcoco_checkpoints_5emy_ref_my/{10,}_{3e-5,}_{512,}/checkpoint_best.pt'),
        #utils.split_paths('/home/komleva/mdetr/ofa/OFA/run_scripts/refcoco/checkpoints_gqa/{10,}_{3e-5,}_{512,}/checkpoint_best.pt'),
        utils.split_paths('./checkpoint.best_score_0.9840.pt'),
        #task = RefcocoTask#'refcoco'
        arg_overrides=overrides
    )

cfg.common.seed = 7
cfg.generation.beam = 5
cfg.generation.min_len = 4
cfg.generation.max_len_a = 0
cfg.generation.max_len_b = 4
cfg.generation.no_repeat_ngram_size = 3

# Fix seed for stochastic decoding
if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

# Image transform
from torchvision import transforms
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()
def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text.lower()),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Construct input for refcoco task
patch_image_size = cfg.task.patch_image_size
def construct_sample(image: Image, text: str):
    w, h = image.size
    w_resize_ratio = torch.tensor(patch_image_size / w).unsqueeze(0)
    h_resize_ratio = torch.tensor(patch_image_size / h).unsqueeze(0)
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True, append_eos=True).unsqueeze(0)
    #src_text = encode_text(text, append_bos=True, append_eos=True).unsqueeze(0)
    
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        },
        "w_resize_ratios": w_resize_ratio,
        "h_resize_ratios": h_resize_ratio,
        "region_coords": torch.randn(1, 4)
    }
    return sample
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

predictions = []
#image = Image.open('./test.jpg')
#text = "a blue turtle-like pokemon with round head"


train = pd.read_csv(args.csv_test)

progress = tqdm(train.iterrows(), total=len(train))
for _, row in progress:
    try:
        img_url = row['image']
        text = row['question']#.replace('What', '').replace('Which', '').replace('Where','').replace('Who','').replace('what','')
        
        #text = row['question']
        #print(img_url.split('/')[-1])
        image = Image.open(args.imgs_test+img_url.split('/')[-1])

        # Construct input sample & preprocess for GPU if cuda available
        sample = construct_sample(image, text)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

        # Run eval step for refcoco
        with torch.no_grad():
            result, scores = eval_step(task, generator, models, sample)
        left = int(result[0]["box"][0])
        right = int(result[0]["box"][2])
        top =  int(result[0]["box"][1])
        bottom=  int(result[0]["box"][3])
        if left >right:
            t = left
            left = right
            rights = t
        if bottom <top:
            t = top
            top = bottom
            bottom = t
        if left == right:
            left = left -10
        predictions.append([img_url, left,  top,right, bottom])
        #f.write(img_url)
        #f.write('|')
        #f.write(str( int(result[0]["box"][0])))
        #f.write('|')
        #f.write(str( int(result[0]["box"][1])))
        #f.write('|')
        #f.write(str( int(result[0]["box"][2])))
        #f.write('|')
        #f.write(str( int(result[0]["box"][3])))
        #f.write('|')
        #.write('\n')
        #print([img_url, int(result[0]["box"][0]),  int(result[0]["box"][1]), int(result[0]["box"][2]), int(result[0]["box"][3])])
        #img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        #cv2.rectangle(
        #    img,
        #    (int(result[0]["box"][0]), int(result[0]["box"][1])),
        #    (int(result[0]["box"][2]), int(result[0]["box"][3])),
        #    (0, 255, 0),
        #    3
        #)
        #path = img_url.split('/')[-1]
        #cv2.imwrite(f'out_img/{path}.jpg',img)
    except KeyboardInterrupt:
        break
    except Exception as e: print(e)
print("predictions",predictions)
predictions = pd.DataFrame(predictions, columns=['image', 'left', 'top', 'right', 'bottom'])
predictions.to_csv('answer.csv', index=None)