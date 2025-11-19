# 超效率 SBM 模型实现与应用

本项目实现了含/不含非期望产出的 SBM（Slack-Based Measure）和超效率 SBM 模型，用于评价地区或决策单元（DMU）的绿色发展效率。代码提供了 Python 与 R 两种实现形式，并支持从 Excel 数据表中读取指标进行批量计算。

## 目录结构概览

- `Code/`
  - `sbm_run.py`：主分析脚本（原 `new.py`），完成数据读取、标准 SBM 和超效率 SBM 计算，并导出结果。
  - `sbm_super_efficiency.py`：Python 版本的标准 SBM 与超效率 SBM 核心函数实现（含非期望产出），接口更灵活，便于二次开发或集成。
  - `sbm_super_efficiency.R`：R 版本的 SBM / 超效率 SBM 参考实现，使用 `lpSolve` 与 `readxl` 等包。
  - `sbm_mapping.json`：预留的字段映射配置文件（目前为空），可用于自定义 Excel 列名与模型变量的映射。
- `Data/`：输入数据（Excel/CSV），例如各地区的投入、期望产出和非期望产出指标。
- `Result/`：模型运行后的输出结果（如效率值与超效率值），为派生文件，可随时通过代码重新生成。
- 其他目录（`Model/`, `Ref/`, `SPSSres/` 等）：模型原始文献、参考资料和中间分析结果。

---

## 环境要求

- Python 3.10+
- 已安装依赖：
  - `numpy`
  - `pandas`
  - `scipy`
- （可选）R 环境，且安装：
  - `readxl`
  - `dplyr`
  - `tibble`
  - `lpSolve`（若在 R 中求解线性规划）

建议在虚拟环境中安装依赖，例如：

```bash
pip install numpy pandas scipy
```

---

## 各代码文件说明

### 1. `Code/sbm_run.py` —— 主分析脚本（含超效率）

**功能概览：**

- 从指定 Excel 文件（默认读取工作表“格式化数据”）导入 DMU 的投入、期望产出与非期望产出数据。
- 内置字段映射（列名）：
  - 投入：
    - `建设用地面积`
    - `耕地面积`
    - `生态用地面积`
    - `年末从业人数`
    - `固定资产投资额`
    - `用电量`
    - `研究与试验发展（R&D）经费支出`
  - 期望产出：
    - `城乡居民可支配收入`
    - `粮食总产量`
    - `城乡绿化水平`
  - 非期望产出：
    - `单位GDP碳排放`
- 实现 4 类模型：
  1. 不含非期望产出的标准 SBM：`sbmeff`
  2. 不含非期望产出的超效率 SBM：`sup_sbmeff`
  3. 含非期望产出的标准 SBM：`un_sbmeff`
  4. 含非期望产出的超效率 SBM：`sup_un_sbmeff`
- 在 `main()` 中：
  - 先计算所有 DMU 的标准 SBM 效率值；
  - 再对接近有效前沿（效率≈1）的 DMU 计算超效率；
  - 将结果导出为 Excel 文件 `effres.xlsx`（包含 DMU 编码、CCR 标准效率和超效率等）。

**关键配置（`main()` 顶部）：**

```python
rts = 0          # 0为CRS（CCR）规模报酬不变；1为VRS(BCC)规模报酬可变
sup = 1          # 1：计算超效率；0：只算标准SBM
undesirable = 1  # 1：存在非期望产出；0：只考虑期望产出
tol = 1e-3       # 判定“效率≈1”的容差，用于筛选超效率计算对象
excel_path = r"d:\Desk\MyDeskFiles\Compe&Event\Feng\Data\SupeffSbmData(2).xlsx"
```

如需切换 CCR/BCC 或只计算标准 SBM，可修改上述参数。

**运行方式：**

在项目根目录下：

```bash
python Code/sbm_run.py
```

运行完成后，会在当前目录生成 `effres.xlsx`，包含每个 DMU 的：

- `标准效率_CCR`：标准 SBM 效率（当前配置为 CCR，如将 `rts=1` 则为 BCC）。
- `超效率_CCR`：对有效 DMU 计算得到的超效率值。
- `sup_status`：超效率模型求解状态（Optimal / Infeasible / Skipped 等）。

---

### 2. `Code/sbm_super_efficiency.py` —— Python SBM 核心实现（含超效率）

**用途：**

- 提供更模块化、灵活的 SBM + 超效率实现，适合：
  - 在 Python 中编写自定义分析脚本；
  - 做批量实验或与其他模型集成；
  - 对比不同规模报酬假设（CRS / VRS）下的效率结果。

**主要函数与结构：**

- `load_excel(path, sheet=None)`  
  从 Excel 中读取数据，假定：
  - 前两行为标签；
  - 第二行为表头；
  - 数据从第三行开始。

- `mapping(df)`  
  根据表头构建字段映射字典，包括：
  - `dmu_id`：DMU 标识列（例如 `代码`）
  - `inputs`：投入列名列表
  - `good_outputs`：期望产出列名列表
  - `bad_outputs`：非期望产出列名列表  
  若缺少必要列名，会抛出异常。

- `build_matrices(df, mapping)`  
  将数据框转换为：
  - `X`：投入矩阵（DMU × 投入）
  - `Yg`：期望产出矩阵
  - `Yb`：非期望产出矩阵
  - `dmu_ids`：DMU ID 列表

- `sbm_super_efficiency_with_undesirable(X, Yg, Yb, rts=0, returns_to_scale=None, eps=1e-8)`  
  - 实现含非期望产出的超效率 SBM。
  - 使用 SciPy `linprog` 求解。
  - 关键特征：
    - 参考集在变量层面排除被评价的 DMU；
    - 归一化：  
      `t − (1/m) Σ s_in'/x_o = 1`
    - 目标：  
      `min D = t + (1/(s1+s2)) [ Σ s_g'/y_g_o + Σ s_b'/y_b_o ]`
    - 支持 `rts` 或 `returns_to_scale` 指定 CRS/VRS。
  - 返回字典，包含每个 DMU 的：
    - `rho`（模型目标值/超效率指标）
    - `status`（Optimal / Infeasible / Unbounded 等）
    - `t`
    - `s_input`, `s_good`, `s_bad`（归一化后的松弛变量）
    - `lambda`（权重）

- `sbm_standard_with_undesirable(X, Yg, Yb, ...)`  
  标准含非期望产出的 SBM（效率 ≤ 1），与原始 Tone 模型对应。

**典型用法（示例）：**

```python
from sbm_super_efficiency import load_excel, mapping, build_matrices, sbm_super_efficiency_with_undesirable

df = load_excel("your_data.xlsx", sheet="Sheet1")
mp = mapping(df)
X, Yg, Yb, dmu_ids = build_matrices(df, mp)

res_sup = sbm_super_efficiency_with_undesirable(X, Yg, Yb, rts=0)
print(res_sup["rho"])
```

---

### 3. `Code/sbm_super_efficiency.R` —— R 版本 SBM / 超效率实现

**主要功能：**

- 从 Excel 中读取数据（第二行为表头，数据从第 3 行开始）。
- 通过 `default_mapping()` 把固定的列名映射到：
  - DMU 标识（例如 `年份`）
  - 投入、期望产出、非期望产出列。
- 提供：
  - `sbm_standard_undesirable()`：含非期望产出的标准 SBM。
  - `sbm_super_efficiency_undesirable()`：含非期望产出的超效率 SBM。
- 使用 `lpSolve` 进行线性规划求解，并输出：
  - 效率值（rho）
  - 模型状态
  - 各类松弛变量、目标值以及 λ 权重等。

**使用方式概览：**

在 R 中：

```r
source("Code/sbm_super_efficiency.R")

df <- load_clean_excel("your_data.xlsx", sheet = NULL)
mp <- default_mapping(df)
mats <- build_matrices(df, mp)

res_std <- sbm_standard_undesirable(mats$X, mats$Yg, mats$Yb, returns_to_scale = "VRS")
res_sup <- sbm_super_efficiency_undesirable(mats$X, mats$Yg, mats$Yb, returns_to_scale = "CRS")
```

脚本中通常还会包含将结果写出到 CSV/Excel 的示例代码。

---

### 4. `Code/sbm_mapping.json`

当前为一个占位文件（空文件），预留用于配置“列名 → 模型变量”映射。例如：

- 定制不同数据表的列名；
- 为不同年份/地区的数据设置不同的字段布局。

未来可以按如下结构填充：

```json
{
  "dmu_id": "代码",
  "inputs": ["建设用地面积", "耕地面积", "..."],
  "good_outputs": ["城乡居民可支配收入", "..."],
  "bad_outputs": ["单位GDP碳排放"]
}
```

然后在 Python 脚本中读取该 JSON 以替代硬编码的列名。

---

## 运行与复现建议

1. 准备好与代码中字段匹配的 Excel 数据，并放在 `Data/` 或你自己的路径下。
2. 根据需要：
   - 如果只想“一键跑完并得到 Excel 结果表”，优先使用：
     ```bash
     python Code/sbm_run.py
     ```
     并在脚本里调整 `excel_path`、`rts`、`undesirable` 等配置。
   - 如果希望在 Python/R 中做更细致的分析（如分组跑、多场景比较、绘图），建议：
     - 在 Python 中通过 `sbm_super_efficiency.py` 的函数进行调用；
     - 或在 R 中使用 `sbm_super_efficiency.R` 中的函数。
3. 所有结果文件建议写入 `Result/` 目录，以便与原始数据区分。

---

## 备注

- 代码实现遵循原始 SBM 与超效率 SBM 模型（含非期望产出）的线性化形式，并在 `sbm_run.py` 中与 R 版本进行了公式对齐。
- 如需进一步核对模型公式与原始文献，请结合 `Code/sbm_super_efficiency.py` / `Code/sbm_super_efficiency.R` 中的注释以及相关论文，对应比对各项约束与目标函数。  

