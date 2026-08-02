# HiCroPL: Aprendizaje Jerárquico de Prompts Intermodales para Modelos de Lenguaje y Visión [ICCV 2025]

> Esta es la implementación oficial del artículo "[Hierarchical Cross-modal Prompt Learning for Vision-Language Models](https://arxiv.org/pdf/2507.14976)".
>
> Autores: Hao Zheng, Shunzhi Yang, Zhuoxin He, Jinfeng Yang, Zhenhua Huang

------

## Visión General

![motivation](docs/motivation.png)

**HiCroPL** es un marco de aprendizaje de prompts intermodales jerárquico para adaptar modelos de lenguaje y visión como CLIP.

A diferencia de métodos de prompts unimodales o diseños de acoplamiento unidireccional, HiCroPL construye un **flujo bidireccional de conocimiento** entre las ramas textuales y visuales. La idea central es simple:

- En las capas tempranas, los prompts textuales transfieren priors semánticos relativamente claros a la rama visual.
- En las capas posteriores, los prompts anclados visualmente refinen la rama textual y mejoran la alineación intermodal.
- Un mapeador jerárquico de conocimiento y tokens proxy específicos por capa permiten la interacción entre prompts a través de capas, preservando la semántica superficial transferible.

## Aspectos Destacados

- **Aprendizaje de prompts intermodales bidireccional.** HiCroPL permite la interacción entre prompts en direcciones de texto-a-visión y visión-a-texto, en lugar de depender de adaptaciones aisladas o unidireccionales.
- **Flujo jerárquico de conocimiento entre capas.** La interacción entre prompts se distribuye a través del codificador, permitiendo que la semántica superficial transferible y las pistas más profundas relevantes para la tarea colaboren.
- **Tokens proxy específicos por capa.** Los tokens proxy ligeros facilitan la interacción intermodal de manera eficiente sin introducir módulos adicionales pesados.
- **Rendimiento sólido en tareas posteriores.** HiCroPL ofrece una generalización competitiva y es especialmente fuerte en configuraciones de adaptación de bajo disparo.

## Metodología

HiCroPL se basa en tres componentes clave:

| Componente | Rol |
| --- | --- |
| Aprendizador de prompts intermodales | Mantiene tokens de prompts textuales y visuales a través de capas. |
| Proxy de conocimiento específico por capa | Resume la información de los prompts en cada capa para una interacción eficiente. |
| Mapeador jerárquico de conocimiento | Transfiere información de los prompts entre modalidades de manera bidireccional y consciente de capas. |

Una representación general del pipeline es:

1. Inicializar prompts textuales y visuales a través de múltiples capas.
2. Refinar los prompts visuales en capas tempranas con flujo de conocimiento de texto-a-visión.
3. Refinar los prompts textuales en capas posteriores con flujo de conocimiento de visión-a-texto.
4. Inyectar los prompts resultantes en los codificadores de CLIP para la predicción.

## Ejecución de HiCroPL

### Entorno de Ejecución

```bash
conda create -n hicropl python=3.10 -y
conda activate hicropl
pip install -r requirements.txt
```
Versiones recomendadas de PyTorch: `1.13.0` o `2.2.0`.

### Datos

Prepare los conjuntos de datos siguiendo la configuración estándar [CoOp dataset setup](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) y actualice la ruta de los datos en los scripts ubicados en `scripts/hicropl/`:

```bash
DATA="/ruta/a/carpeta/de/conjunto/de/datos"
```

### Inicio Rápido

Entrenamiento base-a-novedad:

```bash
sh scripts/hicropl/base2new_train_hicropl.sh imagenet 1
```

Evaluación base-a-novedad:

```bash
sh scripts/hicropl/base2new_test_hicropl.sh imagenet 1
```

Entrenamiento con pocos disparos:

```bash
sh scripts/hicropl/few_shot.sh oxford_pets 16
```

Entrenamiento inter-conjunto:

```bash
sh scripts/hicropl/xd_train.sh caltech101 1
```

Evaluación inter-conjunto:

```bash
sh scripts/hicropl/xd_test.sh caltech101 1
```

## Resultados

Resultados promedio base-a-novedad en 11 conjuntos de datos:

| Método                                                       | Base | Novedad | HM    |
| ------------------------------------------------------------ | ----- | ----- | ----- |
| [CLIP](https://arxiv.org/abs/2103.00020)                     | 69.34 | 74.22 | 71.70 |
| [CoOp](https://arxiv.org/abs/2109.01134)                     | 82.69 | 63.22 | 71.66 |
| [CoCoOp](https://arxiv.org/pdf/2203.05557)                   | 80.47 | 71.69 | 75.83 |
| [KgCoOp](https://arxiv.org/pdf/2303.13283)                   | 80.73 | 73.60 | 77.00 |
| [MaPLe](https://arxiv.org/abs/2210.03117)                    | 82.28 | 75.14 | 78.55 |
| [PromptSRC](https://arxiv.org/abs/2307.06948)                | 84.26 | 76.10 | 79.97 |
| [TCP](https://arxiv.org/abs/2311.18231)                      | 84.13 | 75.36 | 79.50 |
| [MMA](https://openaccess.thecvf.com/content/CVPR2024/paper/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.pdf) | 83.20 | 76.80 | 79.87 |
| [CoPrompt](https://arxiv.org/abs/2306.01195)                 | 84.00 | 77.23 | 80.47 |
| [HiCroPL](https://arxiv.org/pdf/2507.14976)                  | 85.89 | 77.99 | 81.75 |


HiCroPL logra la mejor media armónica entre los métodos comparados, mostrando un equilibrio sólido entre adaptarse a clases base y preservar la transferibilidad a clases novedosas.

## Cita

Si encuentra útil nuestro trabajo para su investigación, por favor cite la siguiente entrada BibTeX.

``` BibTeX
@inproceedings{zheng2025hierarchical,
  title={Hierarchical cross-modal prompt learning for vision-language models},
  author={Zheng, Hao and Yang, Shunzhi and He, Zhuoxin and Yang, Jinfeng and Huang, Zhenhua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1891--1901},
  year={2025}
}
```

## Agradecimientos

Nuestro código se basa en [Co-CoOp, CoOp](https://github.com/KaiyangZhou/CoOp) y [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning). Agradecemos a los autores por compartir su código. Si utiliza nuestro código, le agradeceríamos citar también estos trabajos.
