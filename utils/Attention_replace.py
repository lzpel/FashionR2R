"""
Util functions based on prompt-to-prompt.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import  abc
from einops import rearrange
from typing import Optional, Union, Tuple, List, Callable, Dict


from torchvision.utils import save_image


class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, q, k, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, q, k, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = q.shape[0]
            q[h // 2:], k[h // 2:] = self.forward(q[h // 2:], k[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # self.between_steps()
        return q, k
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.LOW_RESOURCE = False

        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, q, k, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
        #     self.step_store[key].append(attn)
        return q, k

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):

                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
       
class SelfAttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
        
    def replace_self_attention(self, q, k, place_in_unet):
        q_base, q_replace = q[0], q[1:]
        k_base, k_replace = k[0], k[1:]
        # print(q_replace.shape[2])
        if q_replace.shape[2] >= self.flag: #<= self.flag <: deep layers; >: shallow layers
            q_base = q_base.unsqueeze(0).expand(q_replace.shape[0], *q_base.shape)
            k_base = k_base.unsqueeze(0).expand(k_replace.shape[0], *k_base.shape)
            return q_base, k_base
        else:
            return q_replace, k_replace
    
    # modified
    def forward(self, q, k, is_cross: bool, place_in_unet: str):
        super(SelfAttentionControlEdit, self).forward(q, k, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = q.shape[0] // self.batch_size

            q = q.reshape(self.batch_size, h, *q.shape[1:])
            k = k.reshape(self.batch_size, h, *k.shape[1:])


            if is_cross:
                pass
            else:
                q[1:], k[1:] = self.replace_self_attention(q, k, place_in_unet)
            q = q.reshape(self.batch_size * h, *q.shape[2:])
            k = k.reshape(self.batch_size * h, *k.shape[2:])
        return q, k
    
    def __init__(self, prompts, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]], flag):
        super(SelfAttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        if self_replace_steps == 0:
            self.num_self_replace = 0, 0
        elif type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
            self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.flag = flag