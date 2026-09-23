# 🌊 BioWater

### A Domain-Specialized Small Language Model for Scientific Research and Sustainable Intelligent Wastewater Treatment

**Benchmarking scientific capability · Human–AI collaboration · Sustainable local deployment**

## 📢 Paper Now Available on arXiv

Our paper **"Scientific capabilities and deployment sustainability of small-scale LLMs in biological wastewater treatment"** is now publicly available on **arXiv**.

📄 **Read the paper:**  
https://arxiv.org/abs/2609.25774


**Runze Xu¹²**, **Chu-Kuan Jiang¹**, **Dylan Ming-Han Li¹**, **Hong-Xiao Guo¹**, **Jia-shun Cao²**, **Guang-Hao Chen¹***

¹ Department of Civil and Environmental Engineering, Water Technology Center, Hong Kong Branch of Chinese National Engineering Research Center for Control & Treatment of Heavy Metal Pollution, The Hong Kong University of Science and Technology, Hong Kong, China
  
² College of Environment, Hohai University, Nanjing 210098, China
  
**\*Contact email:** runzexu@ust.hk; runzexu@hhu.edu.cn
**\* Correspondence:** redjiang@ust.hk; ceghchen@ust.hk

**arXiv:** 2609.25774  
**Primary category:** Computational Engineering, Finance, and Science (cs.CE)

> If you are interested in our work, please consider citing the paper and ⭐ starring this repository.

</div>

---

## Overview

Large language models are emerging as scientific assistants, but their computational
demands, dependence on cloud infrastructure, and limited domain specialization pose
challenges for their practical deployment in environmental engineering.

**BioWater** investigates whether a domain-specialized small language model can combine
scientific capability with computationally efficient and sustainable deployment in
biological wastewater treatment.

We introduce a scientific benchmark that evaluates LLMs across three complementary
dimensions:

🧠 **Retrospective cognition** — retrieval and reasoning over established domain knowledge

🔬 **Comprehension fidelity** — faithful interpretation of specialized scientific information

🔭 **Prospective extrapolation** — generation and evaluation of scientifically plausible
research hypotheses

Beyond benchmarking, we evaluate **human–BioWater collaboration through laboratory
experiments** and assess the **economic and environmental sustainability of LLM
deployment across wastewater treatment plants worldwide**.

## ✨ Key Findings

### 1. Small model, competitive scientific capability

BioWater contains only **8 billion parameters**, yet achieved higher
comprehension-fidelity scores than participating human experts and performance
comparable to a **397-billion-parameter general-purpose LLM** in retrospective
cognition and prospective extrapolation.

### 2. From scientific reasoning to experimental validation

Human–BioWater collaboration extended beyond benchmark performance to
**hypothesis generation and laboratory validation**, demonstrating the potential
of domain-specialized small LLMs to contribute to prospective scientific research.

### 3. Sustainability depends on inference demand

Global deployment analysis revealed **frequency-dependent crossover points**
between cloud-based large-scale LLMs and locally deployed BioWater.

Local deployment was estimated to achieve lower:

| Indicator | Approximate crossover |
|---|---:|
| 💧 Water footprint | **123 inferences WWTP⁻¹ d⁻¹** |
| 🌱 Carbon emissions | **174 inferences WWTP⁻¹ d⁻¹** |
| ⚡ Electricity consumption | **186 inferences WWTP⁻¹ d⁻¹** |
| 💰 Cost | **253 inferences WWTP⁻¹ d⁻¹** |

These values are **scenario-dependent rather than prescriptive operational
thresholds**. The central finding is the underlying trend: as inference demand
increases, the environmental and economic advantages of locally deployed
small-scale LLMs become increasingly pronounced.

## 🧪 Scientific Benchmark

Our benchmark was designed to evaluate scientific capability beyond conventional
knowledge-based question answering.

| Dimension | Core question |
|---|---|
| 🧠 **Retrospective Cognition** | Can the model reason from established scientific knowledge? |
| 🔬 **Comprehension Fidelity** | Can the model faithfully understand specialized scientific information? |
| 🔭 **Prospective Extrapolation** | Can the model reason beyond existing observations towards scientifically meaningful hypotheses? |

Together, these dimensions evaluate the transition from **understanding existing
science** to **supporting future scientific discovery**.

## 📂 Repository Contents

This repository provides the datasets, evaluation results, prompts, and analysis scripts used to benchmark the scientific capabilities of BioWater and to assess the environmental and economic sustainability of LLM deployment in wastewater treatment plants.

```text
Scientific_benchmark_BioWater/
│
├── Evaluation results/
│   ├── Results of comprehension fidelity task.xlsx
│   ├── Results of environmental and economic sustainability evaluation.xlsx
│   ├── Results of prospective extrapolation task.xlsx
│   └── Results of retrospective cognition task.xlsx
│
├── Materials for BioWater data analysis/
│   ├── Data analysis by BioWater.py
│   └── prompt_template.txt
│
└── Materials for comprehension fidelity task/
    │
    ├── prompt_templates/
    │   └── prompt template.txt
    │
    ├── utils/
    │   ├── data_utils.py
    │   ├── general_utils.py
    │   └── model_utils.py
    │
    ├── Abstract-test for LLMs.py
    ├── Abstract-test for human experts.html
    └── Abstractdata.csv
