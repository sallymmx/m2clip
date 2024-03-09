# modified from https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/misc.py#L170

import os
import builtins
import datetime
from collections import defaultdict, deque
import time
from simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch
import torch.distributed as dist
import pandas as pd
from typing import Union, List
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pdb
import random
from typing import Union, List

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def replace_masked_tokens(token_ids, candidate_pred_positions, num_mlm_preds,all_mlm_id,masked_token_rate=1,context_length=77):
    pred_positions = []
    
    mlm_input_tokens_id = [token_id for token_id in token_ids]
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions) >= num_mlm_preds:
            break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
        masked_token_id = None
        if random.random() < masked_token_rate:  # 0.8
            masked_token_id = 'mask'
        mlm_input_tokens_id[mlm_pred_position] = masked_token_id
        pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
    
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    len_words = []
    words_tokens = []
    for word in mlm_input_tokens_id:
        word_token = _tokenizer.encode(word)
        words_tokens.extend(word_token)
        # pdb.set_trace()
        len_words.append(len(word_token))
    
    all_tokens = [sot_token] + words_tokens + [eot_token]
    mlm_label = [[-100]*len_words[idx] if idx not in pred_positions
                 else all_mlm_id[idx] for idx in range(len(token_ids))]
    flattened_mlm_label = []
    for item in mlm_label:
        if isinstance(item, list):
            flattened_mlm_label.extend(item)
        else:
            flattened_mlm_label.append(item)
    mlm_label = [-100] + flattened_mlm_label + [-100]
    results = torch.zeros(context_length, dtype=torch.long)
    results_labels = -100 * torch.ones(context_length, dtype=torch.long)
    # pdb.set_trace()
    
    if len(all_tokens) > context_length:
        raise RuntimeError(f"Input is too long for context length {context_length}")
    results[:len(all_tokens)] = torch.tensor(all_tokens)
    results_labels[:len(all_tokens)] = torch.tensor(mlm_label)
    # pdb.set_trace()
    # all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in [' '.join(mlm_input_tokens_id)]]
    
    return results.tolist(), results_labels.tolist()
    
def get_masked_sample(text,mlm_label_dict,masked_rate=0.15):
    candidate_pred_positions = []  # 候选预测位置的索引
    pdb.set_trace()
    
    all_mlm_id = [mlm_label_dict.get(word.lower(), 0) for word in text.split()]
    # pdb.set_trace()
    text_list = list(text.lower().split())
    for i, ids in enumerate(text_list):
        candidate_pred_positions.append(i)
    random.shuffle(candidate_pred_positions)  
    num_mlm_preds = max(1, round(len(candidate_pred_positions) * masked_rate))
    # pdb.set_trace()
    mlm_input_tokens_id, mlm_label = replace_masked_tokens(
        text_list, candidate_pred_positions, num_mlm_preds,all_mlm_id)
    return mlm_input_tokens_id, mlm_label


def load_word_index_mapping(txt_file):
    """
    Load the word-to-index mapping from the given txt file.

    Parameters
    ----------
    txt_file : str
        The path to the txt file containing word and index pairs.

    Returns
    -------
    A dictionary with word-to-index mapping.
    """
    word_to_index = {}
    with open(txt_file, "r") as f:
        for line in f:
            word, index = line.strip().split()
            word_to_index[word.lower()] = int(index)
    return word_to_index

_tokenizer = _Tokenizer()
def tokenize(texts: Union[str, List[str]], txt_file: str, context_length: int = 77) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]
    
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    # pdb.set_trace()
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    # pdb.set_trace()
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def get_classes(path):
  classes_all = pd.read_csv(path)
  return classes_all.values.tolist()

def text_prompt(path_classes,path_mlm):
    text_aug = [f"A video of {{}}"]
    text_dict = {}
    num_text_aug = len(text_aug)
    data_classes = get_classes(path_classes)
    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([tokenize(txt.format(c),path_mlm) for i, c in data_classes])
    # pdb.set_trace()
    classes = torch.cat([v for k, v in text_dict.items()])
    classes_dict = {num: "A video of " + string for num, string in data_classes}
    return classes, num_text_aug,text_dict,classes_dict

def create_logits(x1, x2, logit_scale):
    # pdb.set_trace()
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        # print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0 
        self.count = 0 
        self.fmt = fmt 

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n 

    def synchronize_between_processes(self):
        """ 
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )   
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0 
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd' 
        log_msg = [ 
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]   
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj 
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def load_model(args, model_without_ddp, optimizer, lr_sched, loss_scaler):
  if args.resume is None and args.auto_resume:
    print('trying to auto resume from save directory')
    if os.path.isdir(args.save_dir):
      ckpts = [x for x in os.listdir(args.save_dir) if x.startswith('checkpoint-') and x.endswith('.pth')]
    else:
      ckpts = []
    ckpt_epochs = [int(x[len('checkpoint-'):-len('.pth')]) for x in ckpts]
    ckpt_epochs.sort()
    print(f'{len(ckpt_epochs)} candidate checkpoint(s) found.')
    for epoch in ckpt_epochs[::-1]:
      ckpt_path = os.path.join(args.save_dir, 'checkpoint-%d.pth' % epoch)
      try:
        torch.load(ckpt_path, map_location='cpu')
      except Exception as e:
        print(f'loading checkpoint {ckpt_path} failed with error:', e)
        continue
      print('found valid checkpoint:', ckpt_path)
      args.resume = ckpt_path
      break
    if args.resume is None:
      print('did not find any valid checkpoint to resume from.')

  if args.resume:
    print('resuming from:', args.resume)
    ckpt = torch.load(args.resume, map_location='cpu')
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(ckpt['model'], strict=False)
    # strict loading but only for params with grad
    assert len(unexpected_keys) == 0, unexpected_keys
    unexpected_keys = set(unexpected_keys)
    for n, p in model_without_ddp.named_parameters():
      if p.requires_grad:
        assert n not in missing_keys
      else:
        assert n in missing_keys, n

    if optimizer is not None:
      optimizer.load_state_dict(ckpt['optimizer'])
    if lr_sched is not None:
      lr_sched.load_state_dict(ckpt['lr_sched'])
    if loss_scaler is not None:
      loss_scaler.load_state_dict(ckpt['loss_scaler'])
    return ckpt['next_epoch']

  elif args.pretrain:
    print('using pretrained model:', args.pretrain)
    ckpt = torch.load(args.pretrain, map_location='cpu')
    ckpt = ckpt['model']
    for n, p in model_without_ddp.named_parameters():
      if not p.requires_grad and n in ckpt:
        del ckpt[n]
    print(model_without_ddp.load_state_dict(ckpt, strict=False))

  return 0

def save_model(args, epoch, model_without_ddp, optimizer, lr_sched, loss_scaler):
  if dist.get_rank() == 0 and ((epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs):
    os.makedirs(args.save_dir, exist_ok=True)
    state_dict = model_without_ddp.state_dict()
    for n, p in model_without_ddp.named_parameters():
      if not p.requires_grad:
        del state_dict[n]
    torch.save({
      'model': state_dict,
      'optimizer': optimizer.state_dict(),
      'lr_sched': lr_sched.state_dict(),
      'loss_scaler': loss_scaler.state_dict(),
      'next_epoch': epoch + 1,
      }, os.path.join(args.save_dir, 'checkpoint-%d.pth' % epoch))
    if 'k400' in args.dataset and epoch >= 9:
        args.auto_remove=False
    if 'ssv2' in args.dataset and epoch >= 40:
        args.auto_remove=False
    if args.auto_remove:
      for ckpt in os.listdir(args.save_dir):
        try:
          if not (ckpt.startswith('checkpoint-') and ckpt.endswith('.pth')):
            raise ValueError()
          ckpt_epoch = int(ckpt[len('checkpoint-'):-len('.pth')])
        except ValueError:
          continue

        if ckpt_epoch < epoch:
          ckpt_path = os.path.join(args.save_dir, ckpt)
          print('removing old checkpoint:', ckpt_path)
          os.remove(ckpt_path)
    
