# Visual Instruction Bottleneck Tuning (NeurIPS 2025)

By [Changdae Oh](https://changdaeoh.github.io/), [Jiatong Li](https://cslijt.github.io/), [Shawn Im](https://shawn-im.github.io/), and [Yixuan Li](https://pages.cs.wisc.edu/~sharonli/).

[![Paper](https://img.shields.io/badge/arXiv-2505.13946-orange)](https://arxiv.org/abs/2505.13946)
[![Paper](https://img.shields.io/badge/Summary-Post-royalblue)](https://www.linkedin.com/posts/changdae-oh-440587215_thrilled-to-share-that-our-work-%F0%9D%97%A9%F0%9D%97%B6%F0%9D%98%80%F0%9D%98%82-activity-7377550663188111360-ZHSN?utm_source=share&utm_medium=member_desktop&rcm=ACoAADZeRbABjEs_QZU6TVighnglDGRBetnDGi8)


## A Quick Walkthrough on Vittle
Vittle is a new instruction tuning framework for developing VLMs that are robust to distribution shifts. It inserts simple two-layer MLPs inside the LLM backbone that model the latent posterior distributions for each modality, and the bottleneck-inserted entire model is then trained to maximize the lower bound of the variational information bottleneck.

**The bottleneck layer for each modality** is just a two-layer MLP as below,
```python
bottleneck_layer_v = nn.Sequential(
        nn.Linear(config.hidden_size, config.hidden_size),
        nn.GELU(),
        nn.Linear(config.hidden_size, config.hidden_size*2),
    )
```
* It models the mean and std parameters of latent representation distributions, each of which has the same dimension as the hidden state.
* We couldn't search the architectural variants actively due to limited budget, but the design can be advanced further to achieve a better performance-efficiency trade-off, e.g., [GLU](https://arxiv.org/abs/2002.05202). 
* We've inserted this bottleneck layer over only one intermediate LLM layer, but one could try to attach it to multiple layers. Besides, inserting the bottleneck in different locations for each modality is also an option worth exploring.


**These bottleneck layers produce posterior (Gaussian) representations** with the reparameterization technique, and are interpolated with the pre-bottleneck representations as follows,
```python
def reparameterize(self, mu, std, n_samples=1):
    batch_size, seq_length, h_dim = mu.shape
    z = torch.randn(n_samples, batch_size, seq_length, h_dim).cuda().mean(0)
    return mu + std * z

h_mean_v, h_logsigma_sq_v = torch.chunk(self.model.bottleneck_layer_v(h[:,:img_seq_len,:]),2,dim=2)
h_mean_t, h_logsigma_sq_t = torch.chunk(self.model.bottleneck_layer_t(h[:,img_seq_len:,:]),2,dim=2)
std_v, std_t = (h_logsigma_sq_v / 2).exp(), (h_logsigma_sq_t / 2).exp()

h_v_, h_t_ = self.reparameterize(h_mean_v, std_v), self.reparameterize(h_mean_t, std_t)
h_v = (1.0 - alpha) * h[:,:img_seq_len,:] + alpha * h_v_
h_t = (1.0 - alpha) * h[:,img_seq_len:,:] + alpha * h_t_
h = torch.cat((h_v, h_t), dim=1)
```
* The outcome representation may enjoy the balance between invariance to low-level superficial features and sensitivity to high-level abstract features.
* One can adopt multiple posterior samples to approximate the expectation more precisely, though we here just do the [single sample approximation](https://arxiv.org/pdf/1312.6114), which can be sufficient given enough batch size.
* After the training is done, we use the posterior mean directly (without sampling) to construct the post-bottleneck representation.

As we assumed Gaussian prior and posterior distributions for the post-bottleneck representations, **the KL divergence term in the variational lower bound** can be computed in closed form, and then contributes to shape the empirical lower bound of the information bottleneck objective as follows,
```python
def vib_kld_loss(self, mu, logsigma_sq, response_mask=None):
    if response_mask is not None:
        mu = mu[attention_mask.bool()]
        logsigma_sq = logsigma_sq[attention_mask.bool()]
    
    kl_loss = -0.5 * (1 + logsigma_sq - mu.pow(2) - logsigma_sq.exp()).mean() # dim-normalized
    return kl_loss

kld_loss_v = self.vib_kld_loss(h_mean_v, h_logsigma_sq_v)
kld_loss_t = self.vib_kld_loss(h_mean_t, h_logsigma_sq_t, response_mask)
kld_loss_scaled = beta * (kld_loss_v + kld_loss_t)

tot_loss = ce_loss + kld_loss_scaled
```
where the `ce_loss` is just the standard token-aggregated cross-entropy loss by the next token prediction task.
* Although we assumed simple diagonal covariance Gaussians, one can explore more sophisticated alternatives, and even design a conditional prior, e.g., a visual-conditional textual prior, depending on the developer's knowledge of the [data-generating process of the instruction tuning dataset](https://fuxiaoliu.github.io/LRV/), e.g., a case where the visual content affects the generation of the textual query or not.


---

## Install
```linux
conda create -n vittle python=3.10 -y
conda activate vittle
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn=2.5.5 --no-build-isolation
```

## Pretraining Phase
is exactly the same as LLaVA
* Download the 558K subset of the LAION-CC-SBU dataset with BLIP captions [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).
  * Place it to `./playground/data/LLaVA-Pretrain`.
* Check out the training script with DeepSpeed ZeRO-2: `scripts/_pretrain.sh`.

## Instruction Tuning Phase
### 1. Prepare data (llava-v1.5 mix)
* Download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from ingredient datasets:
  - COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
  - GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
  - OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing) (in `.jpg` format)
  - TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
  - VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
* After downloading all of them, organize the data as follows in `./playground/data/LLaVA-Instruct`,
    ```
    ├── coco
    │   └── train2017
    ├── gqa
    │   └── images
    ├── ocr_vqa
    │   └── images
    ├── textvqa
    │   └── train_images
    └── vg
        ├── VG_100K
        └── VG_100K_2
    ```

### 2. Training

Checkout the training script with DeepSpeed ZeRO-3, `scripts/_finetune_vittle.sh`, and check some important arguments:
- `--bottleneck_layeridx_v`: target layer that IB will be applied on for visual tokens (default = 24)
- `--bottleneck_layeridx_t`: target layer that IB will be applied on for textual tokens (default = 24)
- `--ib_strength_v`: strength for the KLD regularization $\beta$ for visual tokens (default = 0.1)
- `--ib_strength_t`: strength for the KLD regularization $\beta$ for textual tokens (default = 0.1)
- `--ib_fadein_end`: the pre/post-bottelneck interpolation coefficient $\alpha$ (default = 0.5)
- `--learnable_prior_flag`: set to 'F' for reproducing Vittle (F) or 'L' for Vittle (L)

Note: we've fixed some unnecessary inefficiency in our training loop, and the expected walk-clock runtime for the full training of the 7B scale model is about **12.5 hours** (far fater than the one in our initial draft) with 8 * A100 GPUs under the default training setup.


Pretrained model weights will be available soon.

## Evaluation (under construction)
* There are three types of downstream tasks: object hallucination detection, closed-form QA, and open-ended QA.
* Evaluation for the first two tasks are intuitive, and you can check the `scripts/_eval_cqa.sh` and `scripts/_eval_pope.sh` for these.
* Evaluation for the open-ended QA requirean additional step: LLM-as-a-Judge evaluation phase for model outputs. 
    * We adopted `AzureOpenAI` module, which requires `API_KEY` and `MODEL_ENDPOINT` specification. See `vittle/eval/eval_gpt_review_visual.py` for details.
    * After filling that information, you can evaluate models on open-ended QA tasks through `scripts/_eval_oqa.sh`
* Note: the current scripts only contain the evaluation command on **clean** datasets. The instructions for the evaluation on **perturbation scenarios** will be provided soon.
* The collection of perturbation variants of COCO dataset is our main benchmark suite. We mainly adopt [MM_Robustness](https://github.com/Jielin-Qiu/MM_Robustness) repository to generate all the nine visual perturbations and six (out of nine) char/word-level textual perturbations, and the remaining three sentence-level perturbations were generated by prompting the GPT-4o for translation (with prompt: `You are a helpful translator who translates individual sentences provided by the user. Please translate the English sentence provided by the user into {TARGET_LANGUAGE}`).
    * [TODO] HuggingFace datasets release

Please refer to the [instruction written by LLaVA authors](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) for additional tips on the evaluation.


---

## Citation
```
@inproceedings{oh2025vittle,
  title={Visual Instruction Bottleneck Tuning},
  author={Oh, Changdae and Li, Jiatong and Im, Shawn and Li, Yixuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License
This work is released under the MIT License.

## Acknolwedgement
This project was built on top of the [LLaVA codebase](https://github.com/haotian-liu/LLaVA). We thank the authors for their awesome work and for sharing. We also thank the authors of [MM_Robustness](https://github.com/Jielin-Qiu/MM_Robustness) for providing a comprehensive toolkit for perturbation generation.
