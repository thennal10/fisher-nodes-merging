import torch
import operator
from copy import deepcopy
from einops import rearrange, reduce
from collections import OrderedDict


class Mask:
    def __init__(self, neuron_mask, head_mask):
        self.neuron_mask = neuron_mask
        self.head_mask = head_mask
    
    def __add__(self, other):
        neuron_mask = self.neuron_mask + other.neuron_mask
        head_mask = self.head_mask + other.head_mask
        return Mask(neuron_mask, head_mask)


class FisherNodeWrapper():
    def __init__(self, pretrained=None, finetuned=None, neuron_mask=None,
                 head_mask=None, vector=None, model_config=None, use_universal=True, device='cpu'):
        """
        The wrapper can be initialized as either a task vector from a pretrained and a finetuned checkpoints,
        or a wrapper over the model parameters.

        Either a vector of parameters or the finetuned model needs to be given. 
        If a pretrained model is given, the vector is 
        the difference between the finetuned and pretrained model, 
        otherwise it is the finetuned model.

        `use_universal` controls whether parameters other than the ones associated with the mask nodes are weighted.
        If true, a universal weight is generated from the mean of the mask node Fishers and applied to all parameters.
        Otherwise, they are averaged uniformly.
        """
        neuron_mask = neuron_mask.to(device)
        head_mask = head_mask.to(device)

        self.mask = Mask(neuron_mask, head_mask)

        if vector:
            self.vector = vector
        else:    
            if pretrained: pretrained.to(device) 
            finetuned.to(device)

            finetuned_state_dict = finetuned.base_model.state_dict()
            pretrained_state_dict = pretrained.base_model.state_dict() if pretrained else None
            
            self.vector = {}
            for key in finetuned_state_dict:
                if finetuned_state_dict[key].dtype in [torch.int64, torch.uint8]:
                    # for the position ids and token type ids
                    continue
                if key not in finetuned_state_dict:
                    print(
                        f'Warning: key {key} is present in the pretrained state dict but not in the finetuned state dict')
                    continue
                if pretrained_state_dict:
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
                else:
                    self.vector[key] = finetuned_state_dict[key]
        
        self.model_config = model_config if model_config else finetuned.config 
        # if pretrained is not given, the vector is the finetuned model 
        # and will overwrite any model that it is applied to
        self.overwrite = not bool(pretrained)
        self.use_universal = use_universal

    def __repr__(self):
        """Representation of the wrapper."""
        return f'FisherNodeWrapper({self.vector})'

    def __add__(self, other):
        """Merge two sets of parameters together."""
        # make sure if both mask values are 0, they are set to 1 (50/50 split)
        zeroes_on_both = torch.logical_and(self.mask.neuron_mask==0, other.mask.neuron_mask==0)
        self.mask.neuron_mask[zeroes_on_both] = 1
        other.mask.neuron_mask[zeroes_on_both] = 1

        zeroes_on_both = torch.logical_and(self.mask.head_mask==0, other.mask.head_mask==0)
        self.mask.head_mask[zeroes_on_both] = 1
        other.mask.head_mask[zeroes_on_both] = 1

        # vector = a.vector * a.mask + b.vector * b.mask
        masked_vector_self = self.apply()
        masked_vector_other = other.apply()
        n_vector = OrderedDict()
        for k, v in self.vector.items(): 
            n_vector[k] = masked_vector_self[k] + masked_vector_other[k]
        # param = param / a.mask + b.mask, for normalization
        mask = self.mask + other.mask
        n_vector = self.apply(vector=n_vector, mask=mask, op=operator.truediv)
        
        return FisherNodeWrapper(vector=n_vector, neuron_mask=mask.neuron_mask, head_mask=mask.head_mask, model_config=self.model_config)

    def apply(self, vector=None, mask=None, op=operator.mul):
        # Apply the operation to the vector and mask
        # By default this is elementwise multiplication
        if vector is None:
            vector = self.vector
        if mask is None:
            mask = self.mask
        
        # averaged attention mask for other attention-related params
        head_sum_mask = reduce(mask.head_mask, 'l h -> l', 'mean')
        # averaged neuron mask for ffwd params 
        neuron_sum_mask = reduce(mask.neuron_mask, 'l d -> l', 'mean')
        # summed mask for other misc params
        if self.use_universal:
            universal_mask = reduce(head_sum_mask + neuron_sum_mask, 'l -> 1', 'mean')
        else:
            universal_mask = torch.ones(1, device=mask.neuron_mask.device)

        n_vector = OrderedDict()
        for k, v in vector.items():
            if 'embeddings' in k:
                nv = op(v, universal_mask)
            elif 'layer' in k:
                splitk = k.split('.')
                # for some reason there's always a copy of the entire encoder under 'layers' as well as 'layer'
                layername = 'layer' if 'layer' in splitk else 'layers'
                layer = int(splitk[splitk.index(layername) + 1])
                
                if 'attention' in k:
                    # uses averaged for value and output
                    if ('query' in k) or ('key' in k):
                        if 'weight' in k:
                            layer_head_mask = rearrange(mask.head_mask[layer], 'h -> h 1 1')
                            nv = op(rearrange(v, '(h d) hd -> h d hd', h=self.model_config.num_attention_heads), layer_head_mask)
                            nv = rearrange(nv, 'h d hd -> (h d) hd')
                        elif 'bias' in k:
                            layer_head_mask = rearrange(mask.head_mask[layer], 'h -> h 1 ')
                            nv = op(rearrange(v, '(h d) -> h d', h=self.model_config.num_attention_heads), layer_head_mask)
                            nv = rearrange(nv, 'h d -> (h d)')
                        else:
                            raise ValueError(f'Unexpected key in attention sublayer {k}')
                    elif 'value' in k:
                        if 'weight' in k:
                            nv = op(v, head_sum_mask[layer])
                        elif 'bias' in k:
                            nv = op(v, head_sum_mask[layer])
                        else:
                            raise ValueError(f'Unexpected key in attention sublayer {k}')
                    elif 'output' in k:
                        if 'weight' in k:
                            nv = op(v, head_sum_mask[layer])
                        elif 'bias' in k:
                            nv = op(v, head_sum_mask[layer])
                        else:
                            raise ValueError(f'Unexpected key in attention sublayer {k}')
                elif 'intermediate' in k:
                    if 'weight' in k:
                        nv = op(v, rearrange(mask.neuron_mask[layer], 'd -> d 1'))
                    elif 'bias' in k:
                        nv = op(v, mask.neuron_mask[layer])
                    else:
                        raise ValueError(f'Unexpected key in layer {k}')
                # uses average for output
                elif 'output' in k:
                    if 'weight' in k:
                        nv = op(v, neuron_sum_mask[layer])
                    elif 'bias' in k:
                        nv = op(v, neuron_sum_mask[layer])
                    else:
                        raise ValueError(f'Unexpected key in output sublayer: {k}')
                else:
                    raise ValueError(f'Unexpected key in layer: {k}')
            elif 'pooler' in k:
                nv = op(v, universal_mask)
            else:
                raise ValueError(f'Unexpected key in vector: {k}')
            
            # Just in case
            try:
                n_vector[k] = nv
                del nv
            except UnboundLocalError as e:
                raise ValueError(f'Unexpected key in vector: {k}') 
        
        return n_vector

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate the parameters."""
        new_vector = {}
        for key in self.vector:
            new_vector[key] = - self.vector[key]
        return FisherNodeWrapper(vector=new_vector, neuron_mask=self.mask.neuron_mask, head_mask=self.mask.head_mask)

    def to(self, device):
        """Move the parameters to a device."""
        new_vector = {}
        self.mask.neuron_mask = self.mask.neuron_mask.to(device)
        self.mask.head_mask = self.mask.head_mask.to(device)
        for key in self.vector:
            new_vector[key] = self.vector[key].to(device)
        self.vector = new_vector
        return self

    def apply_to(self, pretrained_checkpoint, in_place=False):
        """
        Apply the parameters to a pretrained model. If in_place is True, the 
        pretrained model is modified in place. 
        Otherwise, a new model is returned.
        """
        if in_place:
            pretrained_model = pretrained_checkpoint
        else:
            pretrained_model = deepcopy(pretrained_checkpoint)
        
        # take the base model and work w/ that
        pretrained_base = pretrained_model.base_model
        new_state_dict = {}
        pretrained_state_dict = pretrained_base.state_dict()
        for key in pretrained_state_dict:
            if key not in self.vector:
                # Print warning if it isn't just a layer/layers issue
                if key.replace('layers', 'layer') not in self.vector:
                    print(
                        f'Warning: key {key} is present in the pretrained state dict but not in the wrapper parameters')
                continue
            if self.overwrite:
                new_state_dict[key] = self.vector[key]
            else:
                new_state_dict[key] = pretrained_state_dict[key] + self.vector[key]
        
        pretrained_base.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
