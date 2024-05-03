import difflib
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_polynomial_decay_schedule_with_warmup

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import chain

import torch
import os, sys
import numpy as np
import argparse
import copy
import math
import random

class Arguments:
    def __init__(self):
        self.seed = 0 
        self.mode="test" 
        self.data_dir="data" 
        self.model_type="gpt2" 
        self.bos_token="<bos>" 
        self.sp1_token="<sp1>" 
        self.sp2_token="<sp2>" 
        self.gpu="0" 
        self.max_len=1024 
        self.max_turns=5 
        self.top_p=0.8 
        self.ckpt_dir="saved_models" 
        self.ckpt_name="best_ckpt_epoch=3_valid_loss=2.6631" 
        self.end_command="Abort!"

#원래 shell로 들어가는 파라미터를 정의합니다.

class Manager():
    def __init__(self, args):
        self.args = args
        
        if torch.cuda.is_available():
            self.args.device = torch.device(f"cuda:{self.args.gpu}")
        else:
            self.args.device = torch.device("cpu")
        
        # Tokenizer & Vocab
        print("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_type)
        special_tokens = {
            'bos_token': self.args.bos_token,
            'additional_special_tokens': [self.args.sp1_token, self.args.sp2_token]
        }
        self.args.eos_token = self.tokenizer.eos_token
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.args.bos_id = vocab[self.args.bos_token]
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.sp1_id = vocab[self.args.sp1_token]
        self.args.sp2_id = vocab[self.args.sp2_token]
        
        # Load model    
        print("Loading the model...")
        self.fix_seed(self.args.seed)
        self.model = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
        self.model.resize_token_embeddings(self.args.vocab_size)
        
        self.args.max_len = min(self.args.max_len, self.model.config.n_ctx)
            
        
        
        if self.args.ckpt_name is not None:
            ckpt_path = f"{self.args.ckpt_dir}/{self.args.ckpt_name}.ckpt"
            if os.path.exists(ckpt_path):
                print("Loading the trained checkpoint...")
                ckpt = torch.load(ckpt_path, map_location=self.args.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                
                if self.args.mode == 'train':
                    print(f"The training restarts with the specified checkpoint: {self.args.ckpt_name}.ckpt.")
                    self.optim.load_state_dict(ckpt['optim_state_dict'])
                    self.sched.load_state_dict(ckpt['sched_state_dict'])
                    self.best_loss = ckpt['loss']
                    self.last_epoch = ckpt['epoch']
                else:
                    print("The inference will start with the specified checkpoint.")
            else:
                print(f"Cannot fine the specified checkpoint {ckpt_path}.")
                if self.args.mode == 'train':
                    print("Training will start with the initialized model.")
                else:
                    print("Cannot inference.")
                    exit()
              
        print("Setting finished.")
        
    def nucleus_sampling(self, input_ids, token_type_ids, input_len):
        output_ids = []
        for pos in range(input_len, self.args.max_len):
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)[0][:, pos-1]  # (1, V)
            output = F.softmax(output, dim=-1)  # (1, V)
            
            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (1, V)
            idx_remove = cumsum_probs > self.args.top_p
            idx_remove[:, 1:] = idx_remove[:, :-1].clone()
            idx_remove[:, 0] = False
            sorted_probs[idx_remove] = 0.0
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)  # (1, V)
            
            probs = torch.zeros(output.shape, device=self.args.device).scatter_(-1, sorted_idxs, sorted_probs)  # (1, V)
            idx = torch.multinomial(probs, 1)  # (1, 1)
            
            idx_item = idx.squeeze(-1).squeeze(-1).item()
            output_ids.append(idx_item)
            
            if idx_item == self.args.eos_id:
                break
                
            input_ids = torch.cat((input_ids, idx), dim=-1)
            next_type_id = torch.LongTensor([[self.args.sp2_id]]).to(self.args.device)
            token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
            assert input_ids.shape == token_type_ids.shape
            
        return output_ids

    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)  

    def test(self):
        print("Let's start!")
        self.model.eval()
        self.fix_seed(self.args.seed)
        
        import json
        if not os.path.isdir(args.data_dir):
            os.makedirs(args.data_dir)
        with open(f"{args.data_dir}/valid_utters.json") as f:
            json_obj = json.load(f)

        #valid set을 사용하겠습니다.

        summarized_utter = []
        user = []
        ground_truth = []
        for utters in json_obj:
            summarized_utter.append(utters[0]) #요약문장
            user.append(utters[1]) #user utter
            ground_truth.append(utters[2]) #ground truth 진짜 정답.

        
        with torch.no_grad():
            input_hists = []
            ex_cnt = -1
            similarity = 0
            while True:
                #utter = input("You: ")
                ex_cnt += 1
                if ex_cnt == len(json_obj):
                    break


                utter = user[ex_cnt]
                

                if utter == self.args.end_command:
                    print("Bot: Good bye.")
                    break
                
                ## summarized input ##
                input_hists = []
                sumrz_input = summarized_utter[ex_cnt]
                sumrz_input_ids = [self.args.sp1_id] + self.tokenizer.encode(sumrz_input) # sp1 sumar.. utter
                input_hists.append(sumrz_input_ids)
                ## sumarized utter를 하나씩 꺼냅니다. ##

                input_ids = [self.args.sp2_id] + self.tokenizer.encode(utter) # sp2 user utter
                input_hists.append(input_ids)
                
                #if len(input_hists) >= self.args.max_turns:
                    #num_exceeded = len(input_hists) - self.args.max_turns + 1
                    #input_hists = input_hists[num_exceeded:]
                # 역시 턴 개념은 사용하지 않습니다.
                    
                input_ids = [self.args.bos_id] + list(chain.from_iterable(input_hists)) + [self.args.sp1_id] # 2 -> 1
                #start_sp_id = input_hists[0][0]
                start_sp_id = self.args.sp1_id
                


                #next_sp_id = self.args.sp1_id if start_sp_id == self.args.sp2_id else self.args.sp2_id
                next_sp_id = self.args.sp2_id

                assert start_sp_id != next_sp_id
                token_type_ids = [[start_sp_id] * len(hist) if h % 2 == 0 else [next_sp_id] * len(hist) for h, hist in enumerate(input_hists)] 
                assert len(token_type_ids) == len(input_hists)
                token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [self.args.sp1_id] # 2 -> 1
                assert len(input_ids) == len(token_type_ids)
                input_len = len(input_ids)
                
                input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.args.device)
                token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(self.args.device)
                
                output_ids = self.nucleus_sampling(input_ids, token_type_ids, input_len)
           
                # output_ids = self.model.generate(
                #     input_ids=input_ids, token_type_ids=token_type_ids, pad_token_id=self.args.eos_id,
                #     do_sample=True, top_p=self.args.top_p, max_length=self.args.max_len,
                #     output_hidden_states=True, output_scores=True, return_dict_in_generate=True,
                # ).sequences
                # output_ids = output_ids[0].tolist()[input_len:]
                res = self.tokenizer.decode(output_ids, skip_special_tokens=True)

                similarity += compute_similarity(res, ground_truth[ex_cnt])

#                 print(f"summarized : {summarized_utter[ex_cnt]}")
#                 print(f"user : {user[ex_cnt]}")
#                 print(f"res : {res}\n gt : {ground_truth[ex_cnt]}")

                # 예측한 문장과 ground truth를 비교할 수 있습니다.
                # 아직 눈으로 밖에 비교할 방법이 없음.

                #print(f"Bot: {res}")
                #input_hists.append([self.args.sp2_id] + self.tokenizer.encode(res))
        print(f"문자열 간에 유사도 : {similarity / len(json_obj)}")


def compute_similarity(string1, string2):

    matcher = difflib.SequenceMatcher(None, string1, string2)
    return matcher.ratio()

# 두 문자열을 비교하는 함수이나...
# 큰 의미는 없는것 같다.


args = Arguments()
args.data_dir = f"{args.data_dir}/{args.model_type}"
args.ckpt_dir = f"{args.ckpt_dir}/{args.model_type}"
assert args.ckpt_name is not None, "Please specify the trained model checkpoint."
manager = Manager(args)
manager.test()


from datasets import *
from tqdm import tqdm


# For all
space = 'Ġ'
pre_quote = '’'
end_marks = ['.', ',', '?', '!', '...']
quotes = ['"', '\'']
abbreviations = ['s', 'd', 't', 'm', 're', 'll', 've', 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']

# For empathetic dialogues
exclude_symbol = "_conv"
comma_symbol = "_comma_"

# For persona chat
persona_chat_url = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
silence_symbol = "__ SILENCE __"


def load_daily(tokenizer, train_frac):
    dataset = load_dataset('daily_dialog')
    train_dialogues = dataset['train']['dialog']
    valid_dialogues = dataset['validation']['dialog']
    test_dialogues = dataset['test']['dialog']
    
    total_dialogues = train_dialogues + valid_dialogues + test_dialogues
    
    for i, dialogue in enumerate(tqdm(total_dialogues)):
        new_dialogue = []
        for utter in dialogue:
            token_list = tokenizer.tokenize(utter.strip().replace(pre_quote, quotes[1]))
            token_list = process_token_list(token_list)
            text = tokenizer.convert_tokens_to_string(token_list)
            new_dialogue.append(text)
            
        total_dialogues[i] = new_dialogue
    
    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = total_dialogues[:int(len(total_dialogues)*train_frac)]
    valid_dialogues = total_dialogues[int(len(total_dialogues)*train_frac):]
    
    for dialogue in train_dialogues:
        train_utter_num += len(dialogue)
        
    for dialogue in valid_dialogues:
        valid_utter_num += len(dialogue)
    
    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num
    
    
def load_empathetic(tokenizer, train_frac):
    dataset = load_dataset('empathetic_dialogues')
    train_data = dataset['train']
    valid_data = dataset['validation']
    test_data = dataset['test']
    
    total_utters = train_data['utterance'] + valid_data['utterance'] + test_data['utterance']
    total_conv_ids = train_data['conv_id'] + valid_data['conv_id'] + test_data['conv_id']
    total_speaker_ids = train_data['speaker_idx'] + valid_data['speaker_idx'] + test_data['speaker_idx']
    
    assert len(total_utters) == len(total_conv_ids) and len(total_conv_ids) == len(total_speaker_ids)
    
    num = 0
    
    conv_dict = {}
    cur_speaker_idx = -1
    for i, utter in enumerate(tqdm(total_utters)):
        conv_id = total_conv_ids[i]
        speaker_idx = total_speaker_ids[i]
        
        utter_modified = utter.strip().replace(comma_symbol, ',')
        new_token_list = process_token_list(tokenizer.tokenize(utter_modified))
        text = tokenizer.convert_tokens_to_string(new_token_list)
        
        if exclude_symbol in utter:
            continue
        
        if conv_id not in conv_dict:
            conv_dict[conv_id] = []
            cur_speaker_idx = -1

        if cur_speaker_idx != speaker_idx:
            conv_dict[conv_id].append(text)
            cur_speaker_idx = speaker_idx
        else:
            conv_dict[conv_id][-1] += f" {text}"
    
    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = []
    valid_dialogues = []
    
    train_dialogue_num = int(len(conv_dict) * train_frac)
    for i, (conv_id, utter_list) in enumerate(conv_dict.items()):
        if i < train_dialogue_num:
            train_utter_num += len(utter_list)
            train_dialogues.append(utter_list)
        else:
            valid_utter_num += len(utter_list)
            valid_dialogues.append(utter_list)
            
    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num


def load_persona(tokenizer, train_frac):
    import urllib.request, json
    with urllib.request.urlopen(persona_chat_url) as f:
        dataset = json.loads(f.read().decode())
        
    train_data = dataset['train']
    valid_data = dataset['valid']
    total_data = train_data + valid_data
    total_dialogues = []
    
    for obj in tqdm(total_data):
        dialogue = obj['utterances'][-1]['history']
        new_dialogue = []
        
        for i, utter in enumerate(dialogue):
            if utter.strip() != silence_symbol:
                token_list = tokenizer.tokenize(utter.strip())
                new_token_list = process_token_list(token_list)
                text = tokenizer.convert_tokens_to_string(new_token_list)
                new_dialogue.append(text)
        
        total_dialogues.append(new_dialogue)
        
    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = total_dialogues[:int(len(total_dialogues)*train_frac)]
    valid_dialogues = total_dialogues[int(len(total_dialogues)*train_frac):]
    
    for dialogue in train_dialogues:
        train_utter_num += len(dialogue)
        
    for dialogue in valid_dialogues:
        valid_utter_num += len(dialogue)
    
    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num


def load_blended(tokenizer, train_frac):
    dataset = load_dataset('blended_skill_talk')
    data_train = dataset['train']
    data_valid = dataset['validation']
    data_test = dataset['test']
    
    total_previous_utterance = data_train['previous_utterance'] + data_valid['previous_utterance'] + data_test['previous_utterance']
    total_free_messages = data_train['free_messages'] + data_valid['free_messages'] + data_test['free_messages']
    total_guided_messages = data_train['guided_messages'] + data_valid['guided_messages'] + data_test['guided_messages']

    total_dialogues = []
    for i, free_message in enumerate(tqdm(total_free_messages)):
        free_message_list = [utter.strip() for utter in free_message if len(utter.strip())>0]
        guided_message_list = [utter.strip() for utter in total_guided_messages[i] if len(utter.strip())>0]
        dialogue = total_previous_utterance[i]
        
        for j in range(len(free_message_list)):
            token_list = process_token_list(tokenizer.tokenize(free_message_list[j]))
            text = tokenizer.convert_tokens_to_string(token_list)
            dialogue.append(text)
            
            if j < len(guided_message_list):
                token_list = process_token_list(tokenizer.tokenize(guided_message_list[j]))
                text = tokenizer.convert_tokens_to_string(token_list)
                dialogue.append(text)
            
        total_dialogues.append(dialogue)
        
    train_utter_num = 0
    valid_utter_num = 0
    train_dialogues = total_dialogues[:int(len(total_dialogues)*train_frac)]
    valid_dialogues = total_dialogues[int(len(total_dialogues)*train_frac):]
    
    for dialogue in train_dialogues:
        train_utter_num += len(dialogue)
        
    for dialogue in valid_dialogues:
        valid_utter_num += len(dialogue)
    
    return train_dialogues, valid_dialogues, train_utter_num, valid_utter_num
    

def process_token_list(token_list):
    token_list[0] = token_list[0].capitalize()
    
    quote_count = 0
    for i, token in enumerate(token_list):
        if space in token:
            if token[1:] in end_marks or token[1:] in abbreviations:
                token_list[i] = token[1:]
                
            if token[1:] == quotes[1]:
                if i<len(token_list)-1:
                    if token_list[i+1] in abbreviations or (token_list[i+1][0] == space and token_list[i+1][1:] in abbreviations):
                        token_list[i] = token[1:]
                        
        if token[0] == space and token[1:] in quotes:
            if quote_count % 2 == 1:
                token_list[i] = token[1:]
                quote_count = 0
            else:
                if i<len(token_list)-1 and token_list[i+1][0] == space:
                    token_list[i+1] = token_list[i+1][1:]
                quote_count += 1
                
        if token in end_marks or token[1:] in end_marks:
            if i<len(token_list)-1:
                if token_list[i+1][0] != space:
                    token_list[i+1] = space + token_list[i+1].capitalize()
                else:
                    token_list[i+1] = space + token_list[i+1][1:].capitalize()
                
    new_token_list = [token for token in token_list if token != space and len(token)>0]
    if new_token_list[-1] not in end_marks:
        new_token_list.append(end_marks[0])
        
    return new_token_list


import difflib
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_polynomial_decay_schedule_with_warmup

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import chain

import torch
import os, sys
import numpy as np
import argparse
import copy
import math
import random

class Arguments:
    def __init__(self):
        self.seed = 0 
        self.mode="test" 
        self.data_dir="data" 
        self.model_type="gpt2" 
        self.bos_token="<bos>" 
        self.sp1_token="<sp1>" 
        self.sp2_token="<sp2>" 
        self.gpu="0" 
        self.max_len=1024 
        self.max_turns=5 
        self.top_p=0.8 
        self.ckpt_dir="saved_models" 
        self.ckpt_name="best_ckpt_epoch=3_valid_loss=2.6631" 
        self.end_command="Abort!"

#원래 shell로 들어가는 파라미터를 정의합니다.

class Manager():
    def __init__(self, args):
        self.args = args
        
        if torch.cuda.is_available():
            self.args.device = torch.device(f"cuda:{self.args.gpu}")
        else:
            self.args.device = torch.device("cpu")
        
        # Tokenizer & Vocab
        print("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model_type)
        special_tokens = {
            'bos_token': self.args.bos_token,
            'additional_special_tokens': [self.args.sp1_token, self.args.sp2_token]
        }
        self.args.eos_token = self.tokenizer.eos_token
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.args.bos_id = vocab[self.args.bos_token]
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.sp1_id = vocab[self.args.sp1_token]
        self.args.sp2_id = vocab[self.args.sp2_token]
        
        # Load model    
        print("Loading the model...")
        self.fix_seed(self.args.seed)
        self.model = GPT2LMHeadModel.from_pretrained(self.args.model_type).to(self.args.device)
        self.model.resize_token_embeddings(self.args.vocab_size)
        
        self.args.max_len = min(self.args.max_len, self.model.config.n_ctx)
            
        
        
        if self.args.ckpt_name is not None:
            ckpt_path = f"{self.args.ckpt_dir}/{self.args.ckpt_name}.ckpt"
            if os.path.exists(ckpt_path):
                print("Loading the trained checkpoint...")
                ckpt = torch.load(ckpt_path, map_location=self.args.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                
                if self.args.mode == 'train':
                    print(f"The training restarts with the specified checkpoint: {self.args.ckpt_name}.ckpt.")
                    self.optim.load_state_dict(ckpt['optim_state_dict'])
                    self.sched.load_state_dict(ckpt['sched_state_dict'])
                    self.best_loss = ckpt['loss']
                    self.last_epoch = ckpt['epoch']
                else:
                    print("The inference will start with the specified checkpoint.")
            else:
                print(f"Cannot fine the specified checkpoint {ckpt_path}.")
                if self.args.mode == 'train':
                    print("Training will start with the initialized model.")
                else:
                    print("Cannot inference.")
                    exit()
              
        print("Setting finished.")
        
    def nucleus_sampling(self, input_ids, token_type_ids, input_len):
        output_ids = []
        for pos in range(input_len, self.args.max_len):
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)[0][:, pos-1]  # (1, V)
            output = F.softmax(output, dim=-1)  # (1, V)
            
            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (1, V)
            idx_remove = cumsum_probs > self.args.top_p
            idx_remove[:, 1:] = idx_remove[:, :-1].clone()
            idx_remove[:, 0] = False
            sorted_probs[idx_remove] = 0.0
            sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)  # (1, V)
            
            probs = torch.zeros(output.shape, device=self.args.device).scatter_(-1, sorted_idxs, sorted_probs)  # (1, V)
            idx = torch.multinomial(probs, 1)  # (1, 1)
            
            idx_item = idx.squeeze(-1).squeeze(-1).item()
            output_ids.append(idx_item)
            
            if idx_item == self.args.eos_id:
                break
                
            input_ids = torch.cat((input_ids, idx), dim=-1)
            next_type_id = torch.LongTensor([[self.args.sp2_id]]).to(self.args.device)
            token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
            assert input_ids.shape == token_type_ids.shape
            
        return output_ids

    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)  

    def test(self):
        print("Let's start!")
        self.model.eval()
        self.fix_seed(self.args.seed)
        
        import json
        if not os.path.isdir(args.data_dir):
            os.makedirs(args.data_dir)
        with open(f"{args.data_dir}/valid_utters.json") as f:
            json_obj = json.load(f)

        #valid set을 사용하겠습니다.

        summarized_utter = []
        user = []
        ground_truth = []
        for utters in json_obj:
            summarized_utter.append(utters[0]) #요약문장
            user.append(utters[1]) #user utter
            ground_truth.append(utters[2]) #ground truth 진짜 정답.

        
        with torch.no_grad():
            input_hists = []
            ex_cnt = -1
            similarity = 0
            while True:
                #utter = input("You: ")
                ex_cnt += 1
                if ex_cnt == len(json_obj):
                    break


                utter = user[ex_cnt]
                

                if utter == self.args.end_command:
                    print("Bot: Good bye.")
                    break
                
                ## summarized input ##
                input_hists = []
                sumrz_input = summarized_utter[ex_cnt]
                sumrz_input_ids = [self.args.sp1_id] + self.tokenizer.encode(sumrz_input) # sp1 sumar.. utter
                input_hists.append(sumrz_input_ids)
                ## sumarized utter를 하나씩 꺼냅니다. ##

                input_ids = [self.args.sp2_id] + self.tokenizer.encode(utter) # sp2 user utter
                input_hists.append(input_ids)
                
                #if len(input_hists) >= self.args.max_turns:
                    #num_exceeded = len(input_hists) - self.args.max_turns + 1
                    #input_hists = input_hists[num_exceeded:]
                # 역시 턴 개념은 사용하지 않습니다.
                    
                input_ids = [self.args.bos_id] + list(chain.from_iterable(input_hists)) + [self.args.sp1_id] # 2 -> 1
                #start_sp_id = input_hists[0][0]
                start_sp_id = self.args.sp1_id
                


                #next_sp_id = self.args.sp1_id if start_sp_id == self.args.sp2_id else self.args.sp2_id
                next_sp_id = self.args.sp2_id

                assert start_sp_id != next_sp_id
                token_type_ids = [[start_sp_id] * len(hist) if h % 2 == 0 else [next_sp_id] * len(hist) for h, hist in enumerate(input_hists)] 
                assert len(token_type_ids) == len(input_hists)
                token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [self.args.sp1_id] # 2 -> 1
                assert len(input_ids) == len(token_type_ids)
                input_len = len(input_ids)
                
                input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.args.device)
                token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(self.args.device)
                
                output_ids = self.nucleus_sampling(input_ids, token_type_ids, input_len)
           
                # output_ids = self.model.generate(
                #     input_ids=input_ids, token_type_ids=token_type_ids, pad_token_id=self.args.eos_id,
                #     do_sample=True, top_p=self.args.top_p, max_length=self.args.max_len,
                #     output_hidden_states=True, output_scores=True, return_dict_in_generate=True,
                # ).sequences
                # output_ids = output_ids[0].tolist()[input_len:]
                res = self.tokenizer.decode(output_ids, skip_special_tokens=True)

                similarity += compute_similarity(res, ground_truth[ex_cnt])

                print(f"summarized : {summarized_utter[ex_cnt]}")
                print(f"user : {user[ex_cnt]}")
                print(f"res : {res}\n gt : {ground_truth[ex_cnt]}")

                # 예측한 문장과 ground truth를 비교할 수 있습니다.
                # 아직 눈으로 밖에 비교할 방법이 없음.

                #print(f"Bot: {res}")
                #input_hists.append([self.args.sp2_id] + self.tokenizer.encode(res))
        print(f"문자열 간에 유사도 : {similarity / len(json_obj)}")


def compute_similarity(string1, string2):

    matcher = difflib.SequenceMatcher(None, string1, string2)
    return matcher.ratio()

# 두 문자열을 비교하는 함수이나...
# 큰 의미는 없는것 같다.

      
    
    
args = Arguments()
args.data_dir = f"{args.data_dir}/{args.model_type}"
args.ckpt_dir = f"{args.ckpt_dir}/{args.model_type}"
assert args.ckpt_name is not None, "Please specify the trained model checkpoint."
manager = Manager(args)
manager.test()

      