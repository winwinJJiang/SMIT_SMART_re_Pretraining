# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

def AttMask(attention, masking_prob, masking_mode, masking_ratio, show_ratio, show_max):
    
    # Get AttMask (High, Hints or Low)
    masks = get_mask(attention,
                     masking_prob,
                     masking_mode,
                     masking_ratio
                     )
    
    # For AttMask-Hints, randomly reveal some of the most highly attended tokens
    if masking_mode == 'attmask_hint':
        
        # Get a mask of the top show(%) most attended tokens
        top_masks = get_mask(attention,
                             1,
                             masking_mode,
                             show_max
                             )
    
        # Reveal some of the most attended tokens
        masks = show_hints(top_masks, masks, show_ratio)
    
    return masks


def get_mask(attention, masking_prob, masking_mode, masking_ratio):
    
    # Token masking
    token_mask = attention_masking(attention, masking_mode, masking_ratio)

    # Mask a subset based on masking_prob threshold
    generator = torch.rand(attention.shape[0], device=attention.device)
    token_mask[generator > masking_prob] = 0

    return token_mask


def attention_masking(attention, masking_mode, masking_ratio):

    #size [batch, N_Token]
    N = int(attention.shape[1]*masking_ratio)
    attn_mask = torch.zeros(attention.shape, dtype=torch.int, device = attention.device)
    #print ('N is ',N)
    if masking_mode in ['attmask_high', 'attmask_hint']:
        idx = torch.argsort(attention, descending=True)[:,:N]
        #print ('idx ',idx)
    elif masking_mode == 'attmask_low':
        idx = torch.argsort(attention, descending=False)[:,:N]
    else:
        raise('Use attmask_high, attmask_hint or attmask_low')
    
    attn_mask.scatter_(1, idx, 1)
    #print ('attn_mask size ',attn_mask.shape)
    
    return attn_mask


def show_hints(top_masks, masks, show_ratio):

    _, n_tokens = masks.shape
    reveal_tokens = int(show_ratio*n_tokens)

    selected_high = torch.multinomial(top_masks.float(), reveal_tokens)

    masks.scatter_(1, selected_high, 0)

    return masks



def AttMask_Debug(attention, masking_prob, masking_mode, masking_ratio, show_ratio, show_max):
    
    # Get AttMask (High, Hints or Low)
    masks = get_mask_debug(attention,
                     masking_prob,
                     masking_mode,
                     masking_ratio
                     )
    print ('info: the debug mask High max is ',masks.max())
    # For AttMask-Hints, randomly reveal some of the most highly attended tokens
    if masking_mode == 'attmask_hint':
        
        # Get a mask of the top show(%) most attended tokens
        top_masks = get_mask_debug(attention,
                             1,
                             masking_mode,
                             show_max
                             )
    
        # Reveal some of the most attended tokens
        masks = show_hints_debug(top_masks, masks, show_ratio)
    
    print ('info: the debug mask after hint max is ',masks.max())
    return masks


def get_mask_debug(attention, masking_prob, masking_mode, masking_ratio):
    
    # Token masking
    token_mask = attention_masking_debug(attention, masking_mode, masking_ratio)

    print (' before pro mask max is ',token_mask.max())
    # Mask a subset based on masking_prob threshold
    generator = torch.rand(attention.shape[0], device=attention.device)
    token_mask[generator > masking_prob] = 0

    print (' after pro mask max is ',token_mask.max())
    return token_mask


def attention_masking_debug(attention, masking_mode, masking_ratio):

    #size [batch, N_Token]
    N = int(attention.shape[1]*masking_ratio)
    attn_mask = torch.zeros(attention.shape, dtype=torch.int, device = attention.device)
    print ('attention shape ',attention.shape)
    print ('masking_ratio is ',masking_ratio)
    print ('N is ',N)
    print ('attention max is ',attention.max())
    if masking_mode in ['attmask_high', 'attmask_hint']:
        idx = torch.argsort(attention, descending=True)[:,:N]
        print ('idx size ',idx.shape)
    elif masking_mode == 'attmask_low':
        idx = torch.argsort(attention, descending=False)[:,:N]
    else:
        raise('Use attmask_high, attmask_hint or attmask_low')
    
    attn_mask.scatter_(1, idx, 1)
    print ('attn_mask size ',attn_mask.shape)
    print ('attn_mask.attn_mask max ',attn_mask.max())
    
    return attn_mask


def show_hints_debug(top_masks, masks, show_ratio):

    _, n_tokens = masks.shape
    reveal_tokens = int(show_ratio*n_tokens)

    selected_high = torch.multinomial(top_masks.float(), reveal_tokens)

    masks.scatter_(1, selected_high, 0)

    return masks