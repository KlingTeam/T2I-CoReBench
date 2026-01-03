<div align="center">
  <img src="assets/title.png" alt="title" width="90%">
</div>

<div align="center">

  <h1>
  Easier Painting Than Thinking: Can Text-to-Image Models <br>
  Set the Stage, but Not Direct the Play?
  </h1>

  <p align="center">
    <a href='https://t2i-corebench.github.io/'>
      <img src='https://img.shields.io/badge/Project Page-0065D3?logo=rocket&logoColor=white'>
    </a>
    <a href='https://t2i-corebench.github.io/#leaderboard'>
      <img src='https://img.shields.io/badge/Leaderboard-7B2CBF?logo=instatus&logoColor=white'>
    </a>
    <a href='https://arxiv.org/abs/2509.03516'>
      <img src='https://img.shields.io/badge/Arxiv-2509.03516-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
    </a>
    <a href='https://huggingface.co/datasets/lioooox/T2I-CoReBench'>
      <img src='https://img.shields.io/badge/HF-Dataset-FFB000?style=flat&logo=huggingface&logoColor=white'>
    </a>
    <a href='https://huggingface.co/datasets/lioooox/T2I-CoReBench-Images'>
      <img src='https://img.shields.io/badge/HF-Images-FFB000?style=flat&logo=huggingface&logoColor=white'>
    </a>
    <a href='https://github.com/KwaiVGI/T2I-CoReBench'>
      <img src='https://img.shields.io/badge/GitHub-Code-181717?style=flat&logo=github&logoColor=white'>
    </a>
  </p>

  [**Ouxiang Li**](https://scholar.google.com/citations?user=g2oUt1AAAAAJ&hl)<sup>1*</sup>, [**Yuan Wang**](https://scholar.google.com/citations?user=jCmA4IoAAAAJ&hl)<sup>1</sup>, [**Xinting Hu**](https://scholar.google.com/citations?user=o6h6sVMAAAAJ&hl)<sup>†</sup>, [**Huijuan Huang**](https://scholar.google.com/citations?user=BMPobCoAAAAJ)<sup>2‡</sup>, [**Rui Chen**](https://scholar.google.com/citations?user=bJzPwcsAAAAJ)<sup>2</sup>, [**Jiarong Ou**](https://scholar.google.com/citations?user=DQLWdVUAAAAJ&hl)<sup>2</sup>, <br>
  [**Xin Tao**](https://scholar.google.com/citations?user=sQ30WyUAAAAJ&hl)<sup>2†</sup>, [**Pengfei Wan**](https://scholar.google.com/citations?user=P6MraaYAAAAJ&hl)<sup>2</sup>, [**Xiaojuan Qi**](https://scholar.google.com/citations?user=bGn0uacAAAAJ)<sup>1</sup>, [**Fuli Feng**](https://scholar.google.com/citations?user=QePM4u8AAAAJ&hl)<sup>1</sup>

  <sup>1</sup>University of Science and Technology of China, <sup>2</sup>Kling Team, Kuaishou Technology, <sup>3</sup>The University of Hong Kong
  <br>
  <sup>*</sup>Work done during internship at Kling Team, Kuaishou Technology. <sup>†</sup>Corresponding authors. <sup>†</sup>Project lead.

</div>

![teaser](assets/teaser.jpeg)

**Overview of our T2I-CoReBench.** (a) Our benchmark comprehensively covers two fundamental T2I capabilities (i.e., *composition* and *reasoning*), further refined into 12 dimensions. (b–e) Our benchmark poses greater challenges to advanced T2I models, with higher compositional density than [DPG-Bench](https://arxiv.org/abs/2403.05135) and greater reasoning intensity than [R2I-Bench](https://arxiv.org/abs/2505.23493), enabling clearer performance differentiation across models under real-world complexities. Each image is scored based on the ratio of correctly generated elements.

## 📣 News
- `2026/01` 🌟 We have optimized `evaluate.py` to improve evaluation efficiency for open-source evaluators and updated the human alignment study results (see [📏 Run Evaluation](#run-evaluation)).
- `2026/01` 🌟 We have updated the evaluation results of [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512).
- `2025/12` 🌟 We have updated the evaluation results of [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) and [LongCat-Image](https://huggingface.co/meituan-longcat/LongCat-Image).
- `2025/12` 🌟 We have updated the evaluation results of [HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0) and [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).
- `2025/11` 🌟 We have updated the evaluation results of [🍌 Nano Banana Pro](https://ai.google.dev/gemini-api/docs/image-generation#gemini-3-capabilities), which achieves a new SOTA across all 12 dimensions by a substantial margin (see our [🏆 Leaderboard](https://t2i-corebench.github.io/#leaderboard) for more details).
- `2025/10` 🌟 We have integrated the [Qwen3-VL series](https://github.com/QwenLM/Qwen3-VL) MLLMs into `evaluate.py`.
- `2025/09` 🌟 We have updated the evaluation results of [Seedream 4.0](https://seed.bytedance.com/en/seedream4_0).
- `2025/09` 🌟 We have released our benchmark dataset and code.

## Benchmark Comparison

![benchmark_comparison](assets/benchmark_comparison.jpeg)

T2I-CoReBench comprehensively covers 12 evaluation dimensions spanning both *composition* and *reasoning* scenarios. The symbols indicate different coverage levels: <span style="font-size:32px; vertical-align: -5px; line-height:1;">●</span> means coverage with high compositional (visual elements > 5) or reasoning (one-to-many or many-to-one inference) complexity. <span style="font-size:16px; line-height:1;">◐</span> means coverage under simple settings (visual elements ≤ 5 or one-to-one inference). <span style="font-size:32px; vertical-align: -5px; line-height:1;">○</span> means this dimension is not covered.

## 🚀 Quick Start

To evaluate text-to-image models on our T2I-CoReBench, follow these steps:

### 🖼️ Generate Images

Use the provided script to generate images from the benchmark prompts in `./data`. You can customize the T2I models by editing `MODELS` and adjust GPU usage by setting `GPUS`. Here, we take [Qwen-Image](https://github.com/QwenLM/Qwen-Image) as an example, and the corresponding Python environment can be referred to its [official repository](https://github.com/QwenLM/Qwen-Image).

  ```bash
  bash sample.sh
  ```

If you wish to sample with your own model, simply modify the sampling code in `sample.py`, i.e., the model loading part in `lines 44–72` and the sampling part in `line 94`; no other changes are required.

### 📏 Run Evaluation

We provide evaluation code supporting various MLLMs, including **Gemini 2.5 Flash** (used in our main paper) and the **Qwen series** (complementary open-source evaluators), both of which are used to assess the generated images in our benchmark.

> [!NOTE]
> If Gemini 2.5 Flash is not available due to closed-source API costs, we recommend using [Qwen3-VL-32B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking) or [Qwen3-VL-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking) as alternatives. Both models offer a strong balance between human consistency and computational cost in open-source MLLMs (see [Table](#table-human-alignment) below). **Qwen3-VL-30B-A3B-Thinking** is more efficient due to its MoE architecture, making it a more cost-effective choice. Comprehensive evaluation results for different MLLM evaluators are available in our [🏆 Leaderboard](https://t2i-corebench.github.io/#leaderboard).

For the **Gemini series**, please refer to the [Gemini documentation](https://ai.google.dev/gemini-api/docs) for environment setup. An official API key `GEMINI_API_KEY` should be set as an environment variable in `evaluate.py`.  For the **Qwen series**, please follow the [vLLM User Guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#installation) and consult their [official repository](https://github.com/QwenLM/Qwen3-VL) for environment setup.

  ```bash
  bash eval.sh
  ```

The evaluation process will automatically assess the generated images across all 12 dimensions of our benchmark and provide a `mean_score` for each dimension in an individual `json` file.

<div id="table-human-alignment">
  <table>
    <caption>Table: Human alignment study and GPU requirement for different MLLMs, evaluated with <em>balanced accuracy</em> (%).</caption>
    <thead>
      <tr><th>MLLM</th><th>#GPUs (80GB)</th><th>MI</th><th>MA</th><th>MR</th><th>TR</th><th>Mean</th></tr>
    </thead>
    <tbody>
      <tr><td>Qwen2.5-VL-72B-Instruct</td><td>4</td><td>81.3</td><td>63.1</td><td>64.2</td><td>73.7</td><td>70.6</td></tr>
      <tr><td>InternVL3-78B</td><td>4</td><td>70.8</td><td>56.8</td><td>56.5</td><td>67.7</td><td>62.9</td></tr>
      <tr><td>GLM4.5V-106B</td><td>4</td><td>78.0</td><td>61.3</td><td>60.3</td><td>71.8</td><td>67.8</td></tr>
      <tr><td>Qwen3-VL-8B-Instruct</td><td>1</td><td>72.0</td><td>56.2</td><td>56.6</td><td>65.4</td><td>62.5</td></tr>
      <tr><td>Qwen3-VL-8B-Thinking</td><td>1</td><td>79.6</td><td>68.9</td><td>70.7</td><td>76.2</td><td>73.8</td></tr>
      <tr><td>Qwen3-VL-32B-Instruct</td><td>2</td><td>80.8</td><td>63.4</td><td>60.6</td><td>73.3</td><td>69.5</td></tr>
      <tr><td>Qwen3-VL-32B-Thinking</td><td>2</td><td>81.9</td><td>72.9</td><td>75.4</td><td>79.8</td><td>77.5</td></tr>
      <tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>2</td><td>83.1</td><td>61.9</td><td>59.1</td><td>74.2</td><td>69.6</td></tr>
      <tr><td>Qwen3-VL-30B-A3B-Thinking</td><td>2</td><td>82.5</td><td>73.9</td><td>75.4</td><td>77.7</td><td>77.4</td></tr>
      <tr><td>GPT-4o</td><td>-</td><td>78.3</td><td>67.5</td><td>63.6</td><td>72.0</td><td>70.3</td></tr>
      <tr><td>OpenAI o3</td><td>-</td><td>83.5</td><td>77.8</td><td>80.4</td><td>86.8</td><td>82.1</td></tr>
      <tr><td>OpenAI o4 mini</td><td>-</td><td>81.9</td><td>74.7</td><td>77.0</td><td>83.0</td><td>79.1</td></tr>
      <tr><td>Gemini 2.5 Pro</td><td>-</td><td>83.4</td><td>76.5</td><td>82.2</td><td>88.4</td><td>82.6</td></tr>
      <tr><td>Gemini 2.5 Flash</td><td>-</td><td>83.8</td><td>76.9</td><td>78.0</td><td>85.7</td><td>81.1</td></tr>
      <tr><td>Gemini 2.5 Flash Lite</td><td>-</td><td>69.1</td><td>60.1</td><td>58.0</td><td>74.5</td><td>65.4</td></tr>
      <tr><td>Gemini 2.0 Flash</td><td>-</td><td>73.5</td><td>61.0</td><td>67.7</td><td>77.1</td><td>69.8</td></tr>
    </tbody>
  </table>
</div>

## 📊 Examples of Each Dimension

<p align="center">
  <img src="assets/1-C-MI.jpeg" width="95%"><br>
  <!-- <em></em> -->
</p>

<p align="center">
  <img src="assets/2-C-MA.jpeg" width="95%"><br>
  <!-- <em></em> -->
</p>

<p align="center">
  <img src="assets/3-C-MR.jpeg" width="95%"><br>
  <!-- <em></em> -->
</p>

<p align="center">
  <img src="assets/4-C-TR.jpeg" width="95%"><br>
  <!-- <em></em> -->
</p>

<p align="center">
  <img src="assets/5-R-LR.jpeg" width="95%"><br>
  <!-- <em></em> -->
</p>

<p align="center">
  <img src="assets/6-R-BR.jpeg" width="95%"><br>
  <!-- <em></em> -->
</p>

<p align="center">
  <img src="assets/7-R-HR.jpeg" width="95%"><br>
  <!-- <em></em> -->
</p>

<p align="center">
  <img src="assets/8-R-PR.jpeg" width="95%"><br>
  <!-- <em></em> -->
</p>

<p align="center">
  <img src="assets/9-R-GR.jpeg" width="95%"><br>
  <!-- <em></em> -->
</p>

<p align="center">
  <img src="assets/10-R-AR.jpeg" width="95%"><br>
  <!-- <em></em> -->
</p>

<p align="center">
  <img src="assets/11-R-CR.jpeg" width="95%"><br>
  <!-- <em></em> -->
</p>

<p align="center">
  <img src="assets/12-R-RR.jpeg" width="95%"><br>
  <!-- <em></em> -->
</p>

## ✍️ Citation
If you find the repo useful, please consider citing.
```
@article{li2025easier,
  title={Easier Painting Than Thinking: Can Text-to-Image Models Set the Stage, but Not Direct the Play?},
  author={Li, Ouxiang and Wang, Yuan and Hu, Xinting and Huang, Huijuan and Chen, Rui and Ou, Jiarong and Tao, Xin and Wan, Pengfei and Qi, Xiaojuan and Feng, Fuli},
  journal={arXiv preprint arXiv:2509.03516},
  year={2025}
}
```
