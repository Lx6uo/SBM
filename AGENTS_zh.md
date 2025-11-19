# 仓库规范（中文翻译）

## 项目结构与组织

- `Code/`：SBM 超效率模型的 Python 实现（`new.py`, `sbm_super_efficiency.py`）和 R 实现（`sbm_super_efficiency.R`）。
- `Data/`：输入的 Excel/CSV 文件和示例数据集。将大型或敏感的源数据视为外部数据，避免提交新的机密数据文件。
- `Result/`：模型生成的输出结果（CSV/XLSX）。这些是可由代码和数据重新生成的派生文件。
- `Model/` 与 `Ref/`：学术论文和方法论参考资料。
- `SPSSres/`：SPSS 分析的中间输出结果。

## 开发、构建与运行

- 使用 Python 3.10+，并尽量只依赖标准库保持脚本兼容性。
- 常用命令：
  - `python Code/new.py` – 准备数据并运行主分析工作流。
  - `python Code/sbm_super_efficiency.py` – 运行 SBM 超效率模型。
  - `Rscript Code/sbm_super_efficiency.R` – 执行 R 版参考实现。
- 如需新增依赖，请在 `requirements.txt` 或脚本顶部进行说明。

## 代码风格与命名约定

- Python：4 空格缩进，遵循 PEP 8 风格。函数/变量用 `snake_case`，类名用 `CapWords`，文件名使用小写 `snake_case`。
- R：推荐使用 2–4 空格缩进，函数命名用 `snake_case`，变量名应具有清晰语义。
- 保持函数短小、职责单一；避免硬编码绝对路径——请使用基于项目根目录的相对路径。

## 测试规范

- 目前尚无正式测试套件；如需新增测试，优先使用 `pytest`。
- 测试文件放在 `Code/tests/`（例如 `test_sbm_super_efficiency.py`），测试函数命名为 `test_*`。
- 使用 `pytest Code` 运行测试，重点覆盖核心模型逻辑和数据处理辅助函数。

## 提交与合并请求规范

- 提交信息应简洁明了，使用祈使句（例如：`Add SBM efficiency metrics export`）。
- 将相关改动放在同一个提交中；将无关的重构或数据更新拆分到单独提交。
- 在创建合并请求时，说明目标，标明受影响的脚本（`Code/*.py`, `Code/*.R`）以及数据/结果文件，并在适当情况下附上关键的前/后指标或示例输出。

## Agent 与自动化说明

- 不要覆盖 `Data/` 中的源数据；新的输出请写入 `Result/` 或新文件。
- 在没有充分说明的情况下，优先通过新增脚本而不是大幅修改参考实现。

