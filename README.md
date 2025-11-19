# 超效率 SBM 模型实现与应用 / Super-Efficiency SBM Models

> 中文介绍在前，后附英文简介；内容对齐，方便中英文读者共同使用本仓库。

---

## 项目简介（ZH）

本项目实现了含/不含非期望产出的 SBM（Slack-Based Measure）和超效率 SBM 模型，用于评价地区或决策单元（DMU）的绿色发展效率。  
代码以 Python 实现为主，并支持从 Excel 表中批量读取投入、期望产出与非期望产出指标。R 版本仅作为历史参考实现，已不再维护。

模型主要参考：

- Tone, K. (2001). *A slacks-based measure of efficiency in data envelopment analysis.*
- Tone, K. (2002). *A slacks-based measure of super-efficiency in data envelopment analysis.*

典型应用场景：

- 区域绿色发展 / 碳排放效率的学术实证研究；
- 多地区、多年份的绩效评价与排序；
- 在 Python 或 R 环境中进行二次开发、批量实验与可视化分析。

---

## Project Overview (EN)

This repository implements standard and super-efficiency SBM (Slack-Based Measure) models,  
with and without undesirable outputs, to evaluate the “green” efficiency of regions or DMUs.  
The Python implementation is the canonical and maintained version; the R script is kept only as a historical reference.

The implementation is based on:

- Tone, K. (2001). *A slacks-based measure of efficiency in data envelopment analysis.*
- Tone, K. (2002). *A slacks-based measure of super-efficiency in data envelopment analysis.*

Typical use cases:

- Empirical studies on regional green development or carbon-emission efficiency;
- Performance evaluation and ranking across regions and years;
- Further development, experimentation, and visualization in Python or R.

---

## 仓库结构 / Repository Layout

- `Code/`
  - `sbm_run.py`  
    主分析脚本：读取 Excel 数据，构造投入 / 期望产出 / 非期望产出矩阵，调用标准 SBM 与超效率 SBM 模型并导出结果。
    默认配置为：标准 SBM 使用 VRS/BCC，超效率 SBM 使用 CRS/CCR，与 Tone (2001, 2002) 文献和模块化 Python 实现对齐。
  - `sbm_super_efficiency.py`  
    **已弃用**的模块化 Python 实现（含非期望产出），保留用于公式核查和与 R 版本的历史对照；推荐直接使用 `sbm_run.py` 作为主入口。
  - `sbm_super_efficiency.R`  
    **已弃用**的 R 版 SBM / 超效率 SBM 参考实现，使用 `lpSolve`、`readxl` 等包，仅供历史对照与公式核查。
  - `sbm_mapping.json`  
    预留的字段映射配置文件（当前为空），可用于自定义 Excel 列名与模型变量映射。
- `InputData/`  
  输入数据（Excel/CSV），例如各地区的投入、期望产出和非期望产出指标。
- `Result/`  
  模型运行后的输出结果（如效率值、超效率值），为派生文件，可随时通过代码重新生成。
- `ModelRef/`  
  SBM 与超效率 SBM 相关的经典英文文献 PDF。
- `Ref/`  
  与本课题相关的中文论文和参考资料。
- `SPSSres/`  
  使用 SPSS 得到的中间分析结果。

> 说明：早期文档中出现的 `Data/`、`Model/` 等目录名，在当前仓库中已统一为 `InputData/`、`ModelRef/` 等，请以上述结构为准。

---

## 环境与依赖 / Environment & Dependencies

### Python

- Python 3.10+
- 核心依赖（在 `pyproject.toml` 中维护）：
  - `numpy`
  - `pandas`
  - `scipy`
  - `openpyxl`（用于读取 `.xlsx` 文件）

本仓库推荐使用 [uv](https://github.com/astral-sh/uv) 管理依赖和虚拟环境（根目录下已提供 `pyproject.toml` 与 `uv.lock`）。

**使用 uv 初始化环境：**

```bash
uv sync                 # 创建/更新虚拟环境并安装依赖
uv run python Code/sbm_run.py
```

**不使用 uv 时，可直接用 pip：**

```bash
pip install numpy pandas scipy openpyxl
python Code/sbm_run.py
```

### R（已弃用，仅供参考 / Deprecated, Reference Only）

- 仓库中保留了 `Code/sbm_super_efficiency.R` 作为早期实现和公式对照用的参考脚本。
- 该 R 版本不再维护，后续工作建议全部基于 Python：
  - 脚本入口：`Code/sbm_run.py`
  - 模块化接口：`Code/sbm_super_efficiency.py`

若确有需要在 R 中阅读或复现，可参考脚本头部注释，自行配置 `readxl`、`dplyr`、`tibble`、`lpSolve` 等依赖，但建议不要再在 R 版本上做新功能开发。

---

## 主要脚本与模型说明 / Main Scripts & Models

### `Code/sbm_run.py`

功能要点：

- 从指定 Excel 文件（默认工作表为“格式化数据”）读取 DMU 的投入、期望产出与非期望产出；
- 使用固定列名映射（例如：`建设用地面积`、`耕地面积`、`单位GDP碳排放` 等）构造数据矩阵；
- 实现 4 类模型的调用：
  1. 标准 SBM（不含非期望产出）`sbmeff`
  2. 超效率 SBM（不含非期望产出）`sup_sbmeff`
  3. 标准 SBM（含非期望产出）`un_sbmeff`
  4. 超效率 SBM（含非期望产出）`sup_un_sbmeff`
- 在 `main()` 中：
  - 先计算所有 DMU 的标准 SBM 效率；
  - 对接近有效前沿（效率≈1）的 DMU 计算超效率；
  - 输出带有标准效率与超效率结果的 Excel 文件（保存在 `Result/` 或脚本配置的路径下）。

关键参数（位于 `main()` 顶部）包括：

- `rts_standard`：标准 SBM 的规模报酬，固定为 VRS/BCC（1）；
- `rts_super`：超效率 SBM 的规模报酬，固定为 CRS/CCR（0）；
- `sup`：是否计算超效率（`1` 计算，`0` 只算标准 SBM）；
- `undesirable`：是否考虑非期望产出；
- `excel_path`：输入数据文件路径。

### `Code/sbm_super_efficiency.py`

面向 Python 开发者的模块化实现，主要包含：

- `load_excel(path, sheet=None)`：按约定格式加载 Excel；
- `mapping(df)`：根据列名构建 `dmu_id`、`inputs`、`good_outputs`、`bad_outputs` 映射；
- `build_matrices(df, mapping)`：转换为数值矩阵 `X`、`Yg`、`Yb` 以及 DMU ID 列表；
- `sbm_super_efficiency_with_undesirable(...)`：含非期望产出的超效率 SBM；
- `sbm_standard_with_undesirable(...)`：含非期望产出的标准 SBM。

示例（Example）：

```python
from sbm_super_efficiency import (
    load_excel,
    mapping,
    build_matrices,
    sbm_super_efficiency_with_undesirable,
)

df = load_excel("your_data.xlsx", sheet="Sheet1")
mp = mapping(df)
X, Yg, Yb, dmu_ids = build_matrices(df, mp)

res_sup = sbm_super_efficiency_with_undesirable(X, Yg, Yb, rts=0)
print(res_sup["rho"])
```

### `Code/sbm_super_efficiency.R`（Deprecated）

- 保留为与 Python 实现相对应的历史 R 版本，用于核对约束形式、目标函数和结果结构。
- 不再作为推荐入口或维护对象，后续如需扩展或复现文献结果，请优先使用 Python 版本。

---

## 快速开始 / Quick Start

1. 将原始数据整理为与脚本中列名一致的 Excel 表，并放入 `InputData/` 或自定义路径。
2. 在项目根目录下初始化 Python 环境：
   - 推荐：`uv sync`
3. 运行主脚本：
   - `uv run python Code/sbm_run.py`
4. 在 `Result/`（或脚本配置的输出目录）中查看生成的效率结果表。

For English users:

1. Prepare an Excel file whose column names match those used in the scripts (inputs, good outputs, undesirable outputs).
2. Run `uv sync` (or install dependencies via `pip`).
3. Execute `uv run python Code/sbm_run.py` (or `python Code/sbm_run.py`).
4. Inspect the generated efficiency results in the `Result/` folder.

---

## 结果与复现建议 / Outputs & Reproducibility

- 所有模型输出建议统一写入 `Result/` 目录，以便与原始数据区分；
- 由于结果完全由代码和 `InputData/` 中的数据生成，只需保留原始数据和脚本，即可在任意支持 Python 3.10+ 的环境中复现计算过程；
- 若对模型公式或实现细节有进一步需求，可结合：
  - `Code/sbm_super_efficiency.py` / `Code/sbm_super_efficiency.R` 源码；
  - `ModelRef/` 中收录的 Tone (2001, 2002) 等原始文献；
  对照约束条件和目标函数进行核查。
