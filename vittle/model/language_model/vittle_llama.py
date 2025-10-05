
from dataclasses import dataclass
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

import pdb

from typing import List, Optional, Tuple, Union
try:
    Unpack = typing.Unpack
except:
    import typing_extensions
    Unpack = typing_extensions.Unpack
from transformers.cache_utils import DynamicCache, Cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from ..vittle_arch import VittleMetaModel, VittleMetaForCausalLM

from ...constants import IGNORE_INDEX

import transformers

logger = transformers.logging.get_logger(__name__)



@dataclass
class CausalIBLMOutputWithPast(ModelOutput):
    tot_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    ib_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(VittleMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, VittleMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        ib_fadein_coef: Optional[float] = 0.5,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    
























class VittleConfig(LlamaConfig):
    model_type = "vittle_llama"


class VittleLlamaModel(VittleMetaModel, LlamaModel):
    config_class = VittleConfig

    def __init__(self, config: LlamaConfig):
        super(VittleLlamaModel, self).__init__(config)


class VittleLlamaForCausalLM(LlamaForCausalLM, VittleMetaForCausalLM):
    config_class = VittleConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VittleLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.hidden_size = config.hidden_size
        self.post_init()
        


    def get_model(self):
        return self.model

    def ib_layer_index_parse_t(self, n_layers):
        layer_str_split = self.model.config.bottleneck_layeridx_t.split('-')
        if len(layer_str_split) > 1:
            idx = int(layer_str_split[1])
        else:
            idx = n_layers - 1
        return idx

    def ib_layer_index_parse_v(self, n_layers):
        layer_str_split = self.model.config.bottleneck_layeridx_v.split('-')
        if len(layer_str_split) > 1:
            idx = int(layer_str_split[1])
        else:
            idx = n_layers - 1
        return idx

    def reparameterize(self, mu, std, n_samples, dtype):
        batch_size, seq_length, hidden_dim = mu.shape
        z = torch.randn(n_samples, batch_size, seq_length, hidden_dim, dtype=dtype).cuda().mean(0)
        return mu + std * z


    def vib_kld_loss_lp(self, mu, logsigma_sq, mask=None, modality='v'):
        if mask is not None:
            mu = mu[mask.bool()]
            logsigma_sq = logsigma_sq[mask.bool()]
        logsigma = logsigma_sq / 2
        
        if modality == 'v':
            mu_prior = self.model.learnable_prior['mean_v'] if self.model.learnable_prior['mean_v'] is not None else torch.tensor([[0.0]]).to('cuda')
            logsigma_sq_prior = self.model.learnable_prior['logsigma_sq_v'] if self.model.learnable_prior['logsigma_sq_v'] is not None else torch.tensor([[0.0]]).to('cuda')
        else:
            mu_prior = self.model.learnable_prior['mean_t'] if self.model.learnable_prior['mean_t'] is not None else torch.tensor([[0.0]]).to('cuda')
            logsigma_sq_prior = self.model.learnable_prior['logsigma_sq_t'] if self.model.learnable_prior['logsigma_sq_t'] is not None else torch.tensor([[0.0]]).to('cuda')
        
        if len(mu.shape) == 2:
            mu_prior = mu_prior.squeeze(0)
        if len(logsigma.shape) == 2: 
            logsigma_sq_prior = logsigma_sq_prior.squeeze(0)
        
        logsigma_sq_prior.data.clamp_(min=-2.3025, max=2.3025)
        logsigma_prior = logsigma_sq_prior / 2

        kl_loss = -0.5 * (1 + (logsigma - logsigma_prior) - (mu-mu_prior).pow(2)/(2*logsigma_prior).exp() - (2*logsigma).exp()/(2*logsigma_prior).exp()).mean() #.sum(-1).mean() 

        if kl_loss > 10000:
            import warnings
            warnings.warn(f'Detected a VIB loss explosion ({kl_loss=} > 10000). Clampping it to 10000 for stability.')
            return torch.clamp(kl_loss, max=10000.0)
        return kl_loss

    
    def vib_kld_loss(self, mu, logsigma_sq, mask=None):
        if mask is not None:
            mu = mu[mask.bool()]
            logsigma_sq = logsigma_sq[mask.bool()]
        
        logsigma = logsigma_sq / 2
        kl_loss = -0.5 * (1 + logsigma - mu.pow(2) - logsigma_sq.exp()).mean()
  
        if kl_loss > 10000:
            import warnings
            warnings.warn(f'Detected a VIB loss explosion ({kl_loss=} > 10000). Clampping it to 10000 for stability.')
            return torch.clamp(kl_loss, max=10000.0)
        return kl_loss
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        ib_fadein_coef: Optional[float] = 0.5,
    ) -> Union[Tuple, CausalIBLMOutputWithPast]:
        text_tok_length = input_ids.shape[1]
        

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        
        #! --------------------------------------------------
        #! adapted from the transformers 4.36 
        #! --------------------------------------------------
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = True #return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.model.gradient_checkpointing and self.model.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if self.model._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        
        #! --------------------------------------------------
        #! IB specification
        #! --------------------------------------------------
        if self.model.config.learnable_prior_flag == 'L':
            self.kld_func = self.vib_kld_loss_lp
        else:
            self.kld_func = self.vib_kld_loss
        n_layers = len(self.model.layers)
        for layer_idx, decoder_layer in enumerate(self.model.layers):
            ib_target_layer_t = self.ib_layer_index_parse_t(n_layers)
            ib_target_layer_v = self.ib_layer_index_parse_v(n_layers)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.model.gradient_checkpointing and self.model.training:
                layer_outputs = self.model._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            image_seq_len = hidden_states.size(1) - text_tok_length
            
            
            if layer_idx == ib_target_layer_v:    
                
                
                hidden_states_mean_v, hidden_states_logsigma_sq_v = torch.chunk(self.model.bottleneck_layer_v(hidden_states[:,:image_seq_len,:]),2,dim=2)
            
                hidden_states_logsigma_sq_v.data.clamp_(min=-2.3025,max=2.3025)

                if image_seq_len > 0:
                    if self.training:
                        std_v = (hidden_states_logsigma_sq_v / 2).exp()
                        hidden_states_v_ = self.reparameterize(hidden_states_mean_v, std_v, 1, hidden_states.dtype)#.mean(0)
                        hidden_states_v = (1.0 - ib_fadein_coef) * hidden_states[:,:image_seq_len,:] + ib_fadein_coef * hidden_states_v_
                        hidden_states_t = hidden_states[:,image_seq_len:,:] 
                        hidden_states = torch.cat((hidden_states_v, hidden_states_t), dim=1)
                    else:
                        hidden_states_v = (1.0 - ib_fadein_coef) * hidden_states[:,:image_seq_len,:] + ib_fadein_coef * hidden_states_mean_v
                        hidden_states_t = hidden_states[:,image_seq_len:,:]
                        hidden_states = torch.cat((hidden_states_v, hidden_states_t), dim=1)
                else:
                    pass

        
            if layer_idx == ib_target_layer_t:
                if ib_target_layer_t == ib_target_layer_v:
                    pass
                else:
                    if output_hidden_states:
                        all_hidden_states += (hidden_states,)

                hidden_states_mean_t, hidden_states_logsigma_sq_t = torch.chunk(self.model.bottleneck_layer_t(hidden_states[:,image_seq_len:,:]), 2, dim=2)
                hidden_states_logsigma_sq_t.data.clamp_(min=-2.3025,max=2.3025)

            
                if image_seq_len > 0:
                    if self.training:
                        std_t= (hidden_states_logsigma_sq_t / 2).exp()
                        hidden_states_t_ = self.reparameterize(hidden_states_mean_t, std_t, 1, hidden_states.dtype)#.mean(0)
                        hidden_states_v = hidden_states[:,:image_seq_len,:]
                        hidden_states_t = (1.0 - ib_fadein_coef) * hidden_states[:,image_seq_len:,:] + ib_fadein_coef * hidden_states_t_
                        hidden_states = torch.cat((hidden_states_v, hidden_states_t), dim=1)
                    else:
                        hidden_states_v = hidden_states[:,:image_seq_len,:] 
                        hidden_states_t = (1.0 - ib_fadein_coef) * hidden_states[:,image_seq_len:,:] + ib_fadein_coef * hidden_states_mean_t
                        hidden_states = torch.cat((hidden_states_v, hidden_states_t), dim=1)
                else:
                    hidden_states_mean = hidden_states_mean_t
                    hidden_states_logsigma_sq = hidden_states_logsigma_sq_t
                    if self.training:
                        std = (hidden_states_logsigma_sq / 2).exp()
                        hidden_states_ = self.reparameterize(hidden_states_mean, std, 1, hidden_states.dtype)#.mean(0)
                        hidden_states = (1.0 - ib_fadein_coef) * hidden_states + ib_fadein_coef * hidden_states_
                    else:
                        hidden_states = (1.0 - ib_fadein_coef) * hidden_states + ib_fadein_coef * hidden_states_mean

            

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        tot_loss, loss, kld_loss = torch.tensor([0.0]).cuda(), torch.tensor([0.0]), torch.tensor([0.0])

        
        if labels is not None:
            #loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            
            loss = loss_fct(shift_logits, shift_labels)

            kld_loss = torch.tensor(0.0).cuda()
            kld_loss_scaled = torch.tensor(0.0).cuda()
            if self.training:
                response_masking = ((labels == IGNORE_INDEX) & attention_mask).int()
                
                if image_seq_len > 0:
                    kld_loss_v = self.kld_func(hidden_states_mean_v, hidden_states_logsigma_sq_v)
                else:
                    kld_loss_v = self.kld_func(hidden_states_mean_v, hidden_states_logsigma_sq_v) * 0.0
            
                kld_loss_t = self.kld_func(hidden_states_mean_t, hidden_states_logsigma_sq_t, response_masking[:,image_seq_len:])

                kld_loss = kld_loss_v + kld_loss_t
                kld_loss_scaled = self.model.config.ib_strength_v * kld_loss_v + self.model.config.ib_strength_t * kld_loss_t

            tot_loss = loss + kld_loss_scaled


        return CausalIBLMOutputWithPast(
            tot_loss=tot_loss,
            loss=loss,
            ib_loss=kld_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
AutoConfig.register("vittle_llama", VittleConfig)
AutoModelForCausalLM.register(VittleConfig, VittleLlamaForCausalLM)