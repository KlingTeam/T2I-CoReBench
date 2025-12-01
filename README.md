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
    <a href='https://arxiv.org/abs/2509.03516'>
      <img src='https://img.shields.io/badge/Arxiv-2509.03516-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
    </a>
    <a href='https://huggingface.co/datasets/lioooox/T2I-CoReBench'>
      <img src='https://img.shields.io/badge/HuggingFace-Dataset-FFB000?style=flat&logo=huggingface&logoColor=white'>
    </a>
    <a href='https://huggingface.co/datasets/lioooox/T2I-CoReBench-Images'>
      <img src='https://img.shields.io/badge/HuggingFace-Images-FFB000?style=flat&logo=huggingface&logoColor=white'>
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
- `2025/12` 🌟 We have updated the evaluation results of [HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0) and [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).
- `2025/11` 🌟 We have updated the evaluation results of [🍌 Nano Banana Pro](https://ai.google.dev/gemini-api/docs/image-generation#gemini-3-capabilities), which achieves a new SOTA across all 12 dimensions by a substantial margin (see our [🚀 Project Page](https://t2i-corebench.github.io/) for more details).
- `2025/10` 🌟 We have integrated the [Qwen3-VL series](https://github.com/QwenLM/Qwen3-VL) MLLMs into `evaluate.py`.
- `2025/09` 🌟 We have updated the evaluation results of [Seedream 4.0](https://seed.bytedance.com/en/seedream4_0).
- `2025/09` 🌟 We have released our benchmark dataset and code.

## Benchmark Comparison

![benchmark_comparison](assets/benchmark_comparison.jpeg)

T2I-CoReBench comprehensively covers 12 evaluation dimensions spanning both *composition* and *reasoning* scenarios. The symbols indicate different coverage levels: <span style="font-size:32px; vertical-align: -5px; line-height:1;">●</span> means coverage with high compositional (visual elements > 5) or reasoning (one-to-many or many-to-one inference) complexity. <span style="font-size:16px; line-height:1;">◐</span> means coverage under simple settings (visual elements ≤ 5 or one-to-one inference). <span style="font-size:32px; vertical-align: -5px; line-height:1;">○</span> means this dimension is not covered.

## 🚀 Quick Start

To evaluate text-to-image models on our T2I-CoReBench, follow these steps:

### 🖼️ Generate Images

Use the provided script to generate images from the benchmark prompts in `./data`. You can customize the T2I models by editing `MODELS` and adjust GPU usage by setting `GPUS`. Here, we take *Qwen-Image* as an example, and the corresponding Python environment can be referred to in its [official repository](https://github.com/QwenLM/Qwen-Image).

  ```bash
  bash sample.sh
  ```

If you wish to sample with your own model, simply modify the sampling code in `sample.py`, i.e., the model loading part in `lines 44–72` and the sampling part in `line 94`; no other changes are required.

### 📏 Run Evaluation

We provide evaluation code supporting various MLLMs, including **Gemini 2.5 Flash** (used in our main paper) and the **Qwen series** (complementary open-source evaluators), both of which are used to assess the generated images in our benchmark.

For the **Gemini series**, please refer to the [Gemini documentation](https://ai.google.dev/gemini-api/docs) for environment setup. An official API key is required and should be specified in `evaluate.py` at `line 375`.  For the **Qwen series**, please follow the [vLLM User Guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#installation) and consult their [official repository](https://github.com/QwenLM/Qwen3-VL) for environment setup. In our experiments, all Qwen evaluators are implemented using 8 × A800 GPUs (80GB each). 

  ```bash
  bash eval.sh
  ```

The evaluation process will automatically assess the generated images across all 12 dimensions of our benchmark and provide a `mean_score` for each dimension in an individual `json` file.


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
  author={Li, Ouxiang and Wang, Yuan and Hu, Xinting and Huang, Huijuan and Chen, Rui and Ou, Jiarong and Tao, Xin and Wan, Pengfei and Feng, Fuli},
  journal={arXiv preprint arXiv:2509.03516},
  year={2025}
}
```