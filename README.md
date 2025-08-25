# Hierarchical Cross-modal Prompt Learning for Vision-Language Models [ICCV 2025]

> This is the official implementation of the paper " [Hierarchical Cross-modal Prompt Learning for Vision-Language Models](https://arxiv.org/pdf/2507.14976)".
>
> Authors: Hao Zheng, Shunzhi Yang, Zhuoxin He, Jinfeng Yang, Zhenhua Huang

------

## Highlights

![motivation](docs/motivation.png)

*Figure 1. Comparison of HiCroPL with existing prompting approaches. (a) Most existing methods adopt uni-modal adaptation or isolated multi-modal solutions to fine-tune CLIP. (b) Multi-modal Prompt Learning (MaPLe) proposes a one way (i.e. text-to-vision) coupling function to bridge the two modalities, but visual concepts lack pathways to guide textual semantics. (c) HiCroPL introduces a bidirectional knowledge flow mechanism between the two modalities, enabling them to refine their semantics mutually for deep alignment. Besides, the representation used for downstream decisions contains rich intermediate features for improved generalization.*

![framework_final](docs/framework_final.png)

*Figure 2. (a) Overview of the proposed HiCroPL framework. (b) Detailed illustration of the Bidirectional Knowledge flow mechanism. From Layer 1 to k, the LKP first initializes layer-specific proxy tokens to encapsulate the key information relevant to the current layer, which then guide visual prompt refinement via the mapper M. The reverse flow from Layer k+1 to L follows an identical process.*

> **Abstract:** Pre-trained Vision-Language Models (VLMs) such as CLIP have shown excellent generalization abilities. However, adapting these large-scale models to downstream tasks while preserving their generalization capabilities remains challenging. Although prompt learning methods have shown promise, they suffer from two fundamental bottlenecks that limit generalization: (a) modality isolation, and (b) hierarchical semantic decay. To address these limitations, we propose HiCroPL, a Hierarchical Crossmodal Prompt Learning framework that establishes bidirectional knowledge flow between text and vision modalities, enabling them to refine their semantics mutually. HiCroPL routes knowledge flows by leveraging the complementary strengths of text and vision. In early layers, text prompts inject relatively clear semantics into visual prompts through a hierarchical knowledge mapper, enhancing the representation of low-level visual semantics. In later layers, visual prompts encoding specific task-relevant objects flow back to refine text prompts, enabling deeper alignment. Crucially, our hierarchical knowledge mapper allows representations at multi-scales to be fused, ensuring that deeper representations retain transferable shallow semantics thereby enhancing generalization. We further introduce a lightweight layer-specific knowledge proxy to enable efficient cross-modal interactions. Extensive evaluations across four tasks demonstrate HiCroPL’s superior performance, achieving state-of-the-art results on 11 benchmarks with significant improvements.

## Contributions

• **We propose a novel hierarchical prompt learning framework** that effectively adapts VLMs to downstream tasks while preserving their inherent generalization capability. 

• The bidirectional knowledge flow **establishes reciprocal pathways between text and vision modalities**, enabling mutual refinement of cross-modal semantics

• The design of the hierarchical knowledge mapper **facilitates information transfer between modalities at multiple scales**, mitigates semantic decay, and improves generalization performance. 

• Comprehensive experiments across 4 tasks and 11 benchmarks validate HiCroPL’s effectiveness and robustness.

## Results on B2N

| Method                                                       | Base  | Novel | HM    |
| ------------------------------------------------------------ | ----- | ----- | ----- |
| [CLIP](https://arxiv.org/abs/2103.00020)                     | 69.34 | 74.22 | 71.70 |
| [CoOp](https://arxiv.org/abs/2109.01134)                     | 82.69 | 63.22 | 71.66 |
| [CoCoOp](https://arxiv.org/pdf/2203.05557)                   | 80.47 | 71.69 | 75.83 |
| [KgCoOp](https://arxiv.org/pdf/2303.13283)                   | 80.73 | 73.60 | 77.00 |
| [MaPLe](https://arxiv.org/abs/2210.03117)                    | 82.28 | 75.14 | 78.55 |
| [PromptSRC](https://arxiv.org/abs/2307.06948)                | 84.26 | 76.10 | 79.97 |
| [TCP](https://arxiv.org/abs/2311.18231)                      | 84.13 | 75.36 | 79.50 |
| [MMA](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.pdf) | 83.20 | 76.80 | 79.87 |
| [CoPrompt](https://arxiv.org/abs/2306.01195)                 | 84.00 | 77.23 | 80.47 |
| [HiCroPL](https://arxiv.org/pdf/2507.14976)                  | 85.89 | 77.99 | 81.75 |

------

## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```
@article{zheng2025hierarchical,
  title={Hierarchical Cross-modal Prompt Learning for Vision-Language Models},
  author={Zheng, Hao and Yang, Shunzhi and He, Zhuoxin and Yang, Jinfeng and Huang, Zhenhua},
  journal={arXiv preprint arXiv:2507.14976},
  year={2025}
}
```

## Acknowledgements

Our code is based on [Co-CoOp, CoOp](https://github.com/KaiyangZhou/CoOp) and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning). We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.

