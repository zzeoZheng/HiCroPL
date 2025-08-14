import json
import os.path as osp
import random
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import QuickGELU
from clip.model import convert_weights
from .imagenet_templates import IMAGENET_TEMPLATES
from collections import OrderedDict
import math

_tokenizer = _Tokenizer()

CoPrompt_dataset_name_mapping = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def load_clip_to_cpu_teacher(cfg, zero_shot_model=False):
    backbone_name = cfg.TRAINER.HICROPL.TEACHER_NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    print(f"CLIP Teacher name is {backbone_name}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # Return original CLIP model for generating frozen VL features
    design_details = {"trainer": 'IVLP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model

def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'HiCroPL',
                          "vision_depth": cfg.TRAINER.HICROPL.PROMPT_DEPTH,
                          "language_depth": cfg.TRAINER.HICROPL.PROMPT_DEPTH,
                          "vision_ctx": cfg.TRAINER.HICROPL.N_CTX,
                          "language_ctx": cfg.TRAINER.HICROPL.N_CTX}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, cross_prompts_text_deeper):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        combined = [x, cross_prompts_text_deeper]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

# LKP
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(AttentionPooling, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, token_query, sequence_key, sequence_value):
        token_query = token_query + self.attn(self.ln_1(token_query), self.ln_1(sequence_key), self.ln_1(sequence_value), need_weights=False)[0]
        token_query = self.ln_2(token_query)
        return token_query

# Multi-scale Knowledge Mapper
class CrossPromptAttention(nn.Module):
    def __init__(self, hidden_size, encoder_hidden_size, num_attention_heads):
        super(CrossPromptAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        # hidden_size is Q's dimension, encoder_hidden_size is K, V's dimension
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(encoder_hidden_size, hidden_size)
        self.linear_v = nn.Linear(encoder_hidden_size, hidden_size)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(hidden_size, hidden_size * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hidden_size * 4, hidden_size))
        ]))
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v):
        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)
        v_proj = self.linear_v(v)
        q_proj = q_proj + self.attn(self.ln_1(q_proj), self.ln_1(k_proj), self.ln_1(v_proj), need_weights=False)[0]
        q_proj = q_proj + self.ffn(self.ln_2(q_proj))
        return q_proj


class CrossModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.HICROPL.PROMPT_DEPTH >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        n_ctx = cfg.TRAINER.HICROPL.N_CTX
        ctx_init = cfg.TRAINER.HICROPL.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        v_dim = 768
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.cross_prompts_depth = cfg.TRAINER.HICROPL.PROMPT_DEPTH
        self.cross_layer = cfg.TRAINER.HICROPL.CROSS_LAYER
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ######## cross-modal text token initialization ########
        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"HiCroPL design: Hierarchical Cross-modal Prompt Learning")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of HiCroPL context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)
        # create deeper prompts by nn.ParameterList
        cross_prompts_text = nn.ParameterList([self.ctx] + [nn.Parameter(torch.empty(n_ctx, 512, dtype=dtype)) for _ in range(self.cross_prompts_depth - 1)])
        for single_para in cross_prompts_text[1:]:
            nn.init.normal_(single_para, std=0.02)
        self.cross_prompts_text = cross_prompts_text
        ######## cross-modal text token initialization end ########

        ######## cross-modal visual token initialization ########
        visual_vectors = torch.empty(n_ctx, v_dim, dtype=dtype)
        nn.init.normal_(visual_vectors, std=0.02)
        cross_prompts_visual = nn.ParameterList([nn.Parameter(visual_vectors) for _ in range(self.cross_prompts_depth)])
        self.cross_prompts_visual = cross_prompts_visual
        ######## cross-modal visual token initialization end ########

        ######## knowledge mapper network and LKP network initialization ########
        self.text2visual_net = CrossPromptAttention(hidden_size=v_dim, encoder_hidden_size=ctx_dim, num_attention_heads=8)
        self.visual2text_net = CrossPromptAttention(hidden_size=ctx_dim, encoder_hidden_size=v_dim, num_attention_heads=8)
        if cfg.TRAINER.HICROPL.PREC == "fp16":
            self.text2visual_net, self.visual2text_net = self.text2visual_net.half(), self.visual2text_net.half()

        attn_pooling_text = AttentionPooling(hidden_size=ctx_dim, num_attention_heads=8)
        self.attn_pooling_text_nets = _get_clones(attn_pooling_text, self.cross_layer)
        attn_pooling_visual = AttentionPooling(hidden_size=v_dim, num_attention_heads=8)
        self.attn_pooling_visual_nets = _get_clones(attn_pooling_visual, self.cross_prompts_depth - self.cross_layer)
        text_token_query = torch.randn(1, ctx_dim, dtype=dtype)
        self.text_token_query = nn.ParameterList([nn.Parameter(text_token_query) for _ in range(self.cross_layer)])
        img_token_query = torch.randn(1, v_dim, dtype=dtype)
        self.img_token_query = nn.ParameterList([nn.Parameter(img_token_query) for _ in range(self.cross_layer, self.cross_prompts_depth)])
        if cfg.TRAINER.HICROPL.PREC == "fp16":
            self.attn_pooling_text_nets, self.attn_pooling_visual_nets = self.attn_pooling_text_nets.half(), self.attn_pooling_visual_nets.half()
        ######## knowledge mapper network and LKP network initialization end ########

        ######## preparation for distillation ########
        # visual
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu_teacher(cfg, True)
        with torch.no_grad():
            self.ZS_image_encoder = clip_model_temp_image.visual
        # text
        with open(f"gpt_file/{CoPrompt_dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json") as f:
            gpt3_prompt = json.load(f)
        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = gpt_clip_classifier(
            classnames, gpt3_prompt, clip_model_temp, cfg.DATASET.NAME
        )
        self.fixed_embeddings = clip_weights
        ######## preparation for distillation end ########

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # construct the text, a photo of a <class>.

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn): [n_cls, 77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # [n_cls, n_tkn, n_dim]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names 
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor, [n_cls, 77]
        self.name_lens = name_lens


    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        return prompts

    def forward(self):
        # first layer text token
        ctx = self.cross_prompts_text[0]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, 4, 512]
        prefix = self.token_prefix
        suffix = self.token_suffix
        # construct first layer text input
        text_input = self.construct_prompts(ctx, prefix, suffix)  # [n_cls, 77, 512]

        ######## T->I mapping ########
        visual_prompts = torch.cat([self.cross_prompts_visual[i].unsqueeze(0) for i in range(self.cross_layer)], dim=0)  # [self.cross_layer, n_ctx, 768]
        text_prompts = torch.cat([self.cross_prompts_text[i].unsqueeze(0) for i in range(self.cross_layer)], dim=0)  # [self.cross_layer, n_ctx, 512]
        # LKP's work
        layer_proxy_text_tokens = []
        for i in range(self.cross_layer):
            # For T->I mapping, the text prompts should be compressed, text_token_query as Q, cross_prompts_text[i] as K, V.
            text_token = self.attn_pooling_text_nets[i](
                token_query=self.text_token_query[i],  # [1, ctx_dim]
                sequence_key=self.cross_prompts_text[i],  # [n_ctx, ctx_dim]
                sequence_value=self.cross_prompts_text[i]  # [n_ctx, ctx_dim]
            )
            layer_proxy_text_tokens.append(text_token)
        proxy_text_prompts = torch.cat(layer_proxy_text_tokens, dim=0)  # [self.cross_layer, 1, ctx_dim]
        visual_prompts = visual_prompts.view(-1, visual_prompts.shape[-1])  # [self.cross_layer * n_ctx, 768]
        proxy_text_prompts = proxy_text_prompts.view(-1, proxy_text_prompts.shape[-1])  # [self.cross_layer, 512]
        # cross modal action for [0: self.cross_layer]: T->I
        updated_visual_prompts = self.text2visual_net(visual_prompts, proxy_text_prompts, proxy_text_prompts)  # [self.cross_layer * n_ctx, 768]
        updated_visual_prompts = updated_visual_prompts.view(self.cross_layer, -1, updated_visual_prompts.shape[-1])  # [self.cross_layer, n_ctx, 768]
        for i in range(self.cross_layer):
            self.cross_prompts_visual[i].data.copy_(updated_visual_prompts[i])
        ######## T->I mapping end ########

        ######## I->T mapping ########
        text_prompts = torch.cat([self.cross_prompts_text[i].unsqueeze(0) for i in range(self.cross_layer, self.cross_prompts_depth)], dim=0)  # [all_layer - self.cross_layer, n_ctx, 512]
        visual_prompts = torch.cat([self.cross_prompts_visual[i].unsqueeze(0) for i in range(self.cross_layer, self.cross_prompts_depth)], dim=0)  # [all_layer - self.cross_layer, n_ctx, 768]
        # LKP's work
        layer_proxy_visual_tokens = []
        for i in range(self.cross_layer, self.cross_prompts_depth):
            # For I->T mapping, the visual prompts should be compressed, img_token_query as Q, cross_prompts_visual[i] as K, V.
            visual_token = self.attn_pooling_visual_nets[i - self.cross_layer](
                token_query=self.img_token_query[i - self.cross_layer],  # [1, v_dim]
                sequence_key=self.cross_prompts_visual[i],  # [n_ctx, v_dim]
                sequence_value=self.cross_prompts_visual[i]  # [n_ctx, v_dim]
            )
            layer_proxy_visual_tokens.append(visual_token)
            proxy_visual_prompts = torch.cat(layer_proxy_visual_tokens, dim=0)  # [self.cross_prompts_depth - self.cross_layer, 1, v_dim]
        text_prompts = text_prompts.view(-1, text_prompts.shape[-1])  # [(all_layer - self.cross_layer) * n_ctx, 512]
        proxy_visual_prompts = proxy_visual_prompts.view(-1, proxy_visual_prompts.shape[-1])  # [(all_layer - self.cross_layer) * n_ctx, 768]
        # cross modal action for [0: self.cross_layer]: I->T
        updated_text_prompts = self.visual2text_net(text_prompts, proxy_visual_prompts, proxy_visual_prompts)  # [(all_layer - self.cross_layer) * n_ctx, 512]
        updated_text_prompts = updated_text_prompts.view(self.cross_prompts_depth - self.cross_layer, -1, updated_text_prompts.shape[-1])  # [self.cross_prompts_depth - self.cross_layer, n_ctx, 512]
        for i in range(self.cross_layer, self.cross_prompts_depth):
            self.cross_prompts_text[i].data.copy_(updated_text_prompts[i - self.cross_layer])
        ######## I->T mapping end ########

        # extract deeper prompts
        cross_prompts_text_deeper = [self.cross_prompts_text[i] for i in range(1, len(self.cross_prompts_text))]
        cross_prompts_visual_deeper = [self.cross_prompts_visual[i] for i in range(1, len(self.cross_prompts_visual))]
        return text_input, self.cross_prompts_visual[0], cross_prompts_text_deeper, cross_prompts_visual_deeper


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = CrossModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.lambd = cfg.TRAINER.HICROPL.LAMBD

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        with torch.no_grad():
            image_features_fixed = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
            image_features_fixed = image_features_fixed / image_features_fixed.norm(dim=-1, keepdim=True)

        # Compute the prompted image and text features
        text_input, visual_ctx, cross_prompts_text_deeper, cross_prompts_visual_deeper = self.prompt_learner()
        text_features = self.text_encoder(text_input, tokenized_prompts, cross_prompts_text_deeper)
        image_features = self.image_encoder(image.type(self.dtype), visual_ctx, cross_prompts_visual_deeper)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features + image_features_fixed
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features + self.prompt_learner.fixed_embeddings.half()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # prompted logits
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            loss_cls = F.cross_entropy(logits, label)
            text_features_fixed = self.prompt_learner.fixed_embeddings
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
            score = cos(text_features, text_features_fixed)
            loss_distill_text = 1.0 - torch.mean(score)
            score = cos(image_features, image_features_fixed)
            loss_distill_image = 1.0 - torch.mean(score)
            loss_distill = loss_distill_text + loss_distill_image
            return loss_cls + self.lambd * loss_distill
        return logits


def gpt_clip_classifier(classnames, gpt_prompts, clip_model, dataset_name):
    import os
    os.makedirs("cache/", exist_ok=True)

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace("_", " ")
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts)
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
                texts = texts.cuda()
            # prompt ensemble
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings /= class_embeddings.norm()
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=0)
        if torch.cuda.is_available():
            clip_weights = clip_weights.cuda()
        torch.save(clip_weights, f"cache/{dataset_name}_clip_weights_random.pt")
    return clip_weights

@TRAINER_REGISTRY.register()
class HiCroPL(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.IVLP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.HICROPL.PREC == "fp32" or cfg.TRAINER.HICROPL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)


        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.IVLP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.HICROPL.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)