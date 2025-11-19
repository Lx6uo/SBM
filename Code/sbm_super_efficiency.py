"""
DEPRECATED MODULE
-----------------
This Python module implements SBM (standard) and super-efficiency SBM with
undesirable outputs in a modular way, mirroring the formulations in
Tone (2001, 2002) and the historical R implementation.

It is kept for reference and formula cross-checking only.
The canonical and maintained entry point for running the models is:
    Code/sbm_run.py

For new development and production use, prefer `sbm_run.py`.
"""

import json
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from scipy import optimize

def load_excel(path: str, sheet: str = None) -> pd.DataFrame:
    """
    读取 Excel：前两行为标签，使用第二行为表头；数据从第三行开始。
    """
    xls = pd.ExcelFile(path)
    sheet_name = sheet or xls.sheet_names[0]
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    # 使用第二行为表头，数据从第三行开始
    cols = raw.iloc[0].tolist()
    df = raw.iloc[1:].reset_index(drop=True)
    df.columns = cols
    return df


def mapping(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    根据实际列名调整构建默认字段映射。
    """
    cols = set(df.columns.tolist())
    # CFG
    inputs = [
        '建设用地面积',
        '耕地面积',
        '生态用地面积',
        '年末从业人数',
        '固定资产投资额',
        '用电量',
        '研究与试验发展（R&D）经费支出',
    ]
    good_outputs = [
        '城乡居民可支配收入',
        '粮食总产量',
        '城乡绿化水平',
    ]
    bad_outputs = [
        '单位GDP碳排放',
    ]

    missing = [c for c in inputs + good_outputs + bad_outputs if c not in cols]
    if missing:
        raise ValueError(f"以下列在数据表中未找到，请检查命名或更新映射: {missing}")

    return {
        'dmu_id': '代码',
        'inputs': inputs,
        'good_outputs': good_outputs,
        'bad_outputs': bad_outputs,
    }


def build_matrices(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    dmu_id_col = mapping['dmu_id']
    X = df[mapping['inputs']].to_numpy(dtype=float)
    Yg = df[mapping['good_outputs']].to_numpy(dtype=float)
    Yb = df[mapping['bad_outputs']].to_numpy(dtype=float)
    dmu_ids = df[dmu_id_col].astype(str).tolist()
    return X, Yg, Yb, dmu_ids


def sbm_super_efficiency_with_undesirable(
    X: np.ndarray,
    Yg: np.ndarray,
    Yb: np.ndarray,
    rts: int = 0,
    returns_to_scale: str | None = None,
    eps: float = 1e-8,
) -> Dict[str, List]:
    """
    使用 SciPy linprog 实现 Tone (2002) 含非期望产出的超效率SBM。
    - 参考集排除当前DMU（变量层面移除 λ_cur）。
    - 归一化：投入项（t - (1/m) Σ s_in'/x_o = 1）。
    - 目标：D（t + 产出松弛项/产出），最小化 D/N。

    参数：
      rts: 0=CRS，1=VRS（默认CRS）；returns_to_scale 传入则覆盖 rts。
    返回：与旧版结构一致的结果字典。
    """
    num_dmu, m = X.shape
    s1 = Yg.shape[1]
    s2 = Yb.shape[1]

    def denom_safe(arr: np.ndarray) -> np.ndarray:
        return np.where(np.abs(arr) < eps, eps, arr)

    vrs = (returns_to_scale.upper() == 'VRS') if isinstance(returns_to_scale, str) else (rts == 1)

    results = {
        'rho': [],
        'status': [],
        't': [],
        's_input': [],
        's_good': [],
        's_bad': [],
        'lambda': [],
    }

    for o in range(num_dmu):
        # 变量布局： [λ(排除o)] + [s_in(m)] + [s_g(s1)] + [s_b(s2)] + [t]
        lam_indices = []
        dmu_map = []
        for i in range(num_dmu):
            if i == o:
                continue
            lam_indices.append(len(lam_indices))
            dmu_map.append(i)
        lam_count = len(lam_indices)
        s_in_start = lam_count
        s_g_start = s_in_start + m
        s_b_start = s_g_start + s1
        t_index = s_b_start + s2
        var_count = t_index + 1

        c = np.zeros(var_count, dtype=float)
        # 目标：t + d_factor*(Σ s_g'/y_g_o + Σ s_b'/y_b_o)
        c[t_index] = 1.0
        d_factor = 1.0 / float(max(1, (s1 + s2)))
        yg_o_safe = denom_safe(Yg[o, :]) if s1 > 0 else np.array([])
        yb_o_safe = denom_safe(Yb[o, :]) if s2 > 0 else np.array([])
        for j in range(s1):
            c[s_g_start + j] = d_factor * (1.0 / float(yg_o_safe[j]))
        for k in range(s2):
            c[s_b_start + k] = d_factor * (1.0 / float(yb_o_safe[k]))

        A_eq_rows = []
        b_eq_vals = []

        # 投入平衡：Σ λ X[:,r] + s_in[r] - x_o[r] t = 0
        for r in range(m):
            row = np.zeros(var_count, dtype=float)
            for idx, i in enumerate(dmu_map):
                row[idx] = X[i, r]
            row[s_in_start + r] = 1.0
            row[t_index] = -float(X[o, r])
            A_eq_rows.append(row)
            b_eq_vals.append(0.0)

        # 期望产出平衡：Σ λ Yg[:,j] - s_g[j] - y_g_o[j] t = 0
        for j in range(s1):
            row = np.zeros(var_count, dtype=float)
            for idx, i in enumerate(dmu_map):
                row[idx] = Yg[i, j]
            row[s_g_start + j] = -1.0
            row[t_index] = -float(Yg[o, j])
            A_eq_rows.append(row)
            b_eq_vals.append(0.0)

        # 不期望产出平衡：Σ λ Yb[:,k] + s_b[k] - y_b_o[k] t = 0
        for k in range(s2):
            row = np.zeros(var_count, dtype=float)
            for idx, i in enumerate(dmu_map):
                row[idx] = Yb[i, k]
            row[s_b_start + k] = 1.0
            row[t_index] = -float(Yb[o, k])
            A_eq_rows.append(row)
            b_eq_vals.append(0.0)

        # VRS：Σ λ = t
        if vrs:
            row = np.zeros(var_count, dtype=float)
            for idx in range(lam_count):
                row[idx] = 1.0
            row[t_index] = -1.0
            A_eq_rows.append(row)
            b_eq_vals.append(0.0)

        # 归一化（投入）：t - (1/m) Σ s_in'/x_o = 1
        x_o_safe = denom_safe(X[o, :])
        row = np.zeros(var_count, dtype=float)
        row[t_index] = 1.0
        for r in range(m):
            row[s_in_start + r] = -(1.0 / float(max(1, m))) * (1.0 / float(x_o_safe[r]))
        A_eq_rows.append(row)
        b_eq_vals.append(1.0)

        bounds = [(0, None)] * var_count
        res = optimize.linprog(c=c, A_eq=np.array(A_eq_rows), b_eq=np.array(b_eq_vals), bounds=bounds, method='highs')

        def map_status(r):
            if r.success:
                return 'Optimal'
            if r.status == 2:
                return 'Infeasible'
            if r.status == 3:
                return 'Unbounded'
            return 'NotSolved'

        status = map_status(res)
        if res.success:
            x_sol = res.x
            t_val = float(x_sol[t_index])
            s_in_p = x_sol[s_in_start:s_in_start + m]
            s_g_p = x_sol[s_g_start:s_g_start + s1]
            s_b_p = x_sol[s_b_start:s_b_start + s2]
            lam_vals = x_sol[:lam_count].tolist()
            if t_val > eps:
                s_in = (s_in_p / t_val).tolist()
                s_g = (s_g_p / t_val).tolist()
                s_b = (s_b_p / t_val).tolist()
            else:
                s_in = [np.nan] * m
                s_g = [np.nan] * s1
                s_b = [np.nan] * s2
            rho_val = float(res.fun)
        else:
            t_val = np.nan
            s_in = [np.nan] * m
            s_g = [np.nan] * s1
            s_b = [np.nan] * s2
            lam_vals = [np.nan] * lam_count
            rho_val = np.nan

        results['rho'].append(rho_val)
        results['status'].append(status)
        results['t'].append(t_val)
        results['s_input'].append(s_in)
        results['s_good'].append(s_g)
        results['s_bad'].append(s_b)
        results['lambda'].append(lam_vals)

    return results


def sbm_standard_with_undesirable(
    X: np.ndarray,
    Yg: np.ndarray,
    Yb: np.ndarray,
    rts: int = 1,
    returns_to_scale: str | None = None,
    eps: float = 1e-8,
) -> Dict[str, List]:
    """
    使用 SciPy linprog 实现标准SBM（含不期望产出）。
    - 分母归一化：t + (1/(s1+s2)) [Σ s_g'/y_g_o + Σ s_b'/y_b_o] = 1。
    - 目标：ρ = t - (1/m) Σ s_in'/x_o。
    - 参考集包含自身（包括 o）。
    参数：rts 0=CRS，1=VRS（默认VRS）；returns_to_scale 传入则覆盖 rts。
    """
    num_dmu, m = X.shape
    s1 = Yg.shape[1]
    s2 = Yb.shape[1]

    def denom_safe(arr: np.ndarray) -> np.ndarray:
        return np.where(np.abs(arr) < eps, eps, arr)

    vrs = (returns_to_scale.upper() == 'VRS') if isinstance(returns_to_scale, str) else (rts == 1)

    results = {
        'rho': [],
        'status': [],
        't': [],
        's_input': [],
        's_good': [],
        's_bad': [],
        'lambda': [],
    }

    for o in range(num_dmu):
        # 变量布局： [λ(n)] + [s_in(m)] + [s_g(s1)] + [s_b(s2)] + [t]
        lam_count = num_dmu
        s_in_start = lam_count
        s_g_start = s_in_start + m
        s_b_start = s_g_start + s1
        t_index = s_b_start + s2
        var_count = t_index + 1

        c = np.zeros(var_count, dtype=float)
        # 目标：t - (1/m) Σ s_in'/x_o
        c[t_index] = 1.0
        x_o_safe = denom_safe(X[o, :])
        for r in range(m):
            c[s_in_start + r] = -(1.0 / float(max(1, m))) * (1.0 / float(x_o_safe[r]))

        A_eq_rows = []
        b_eq_vals = []

        # 投入平衡：Σ λ X[:,r] + s_in[r] - x_o[r] t = 0
        for r in range(m):
            row = np.zeros(var_count, dtype=float)
            for i in range(num_dmu):
                row[i] = X[i, r]
            row[s_in_start + r] = 1.0
            row[t_index] = -float(X[o, r])
            A_eq_rows.append(row)
            b_eq_vals.append(0.0)

        # 期望产出平衡：Σ λ Yg[:,j] - s_g[j] - y_g_o[j] t = 0
        for j in range(s1):
            row = np.zeros(var_count, dtype=float)
            for i in range(num_dmu):
                row[i] = Yg[i, j]
            row[s_g_start + j] = -1.0
            row[t_index] = -float(Yg[o, j])
            A_eq_rows.append(row)
            b_eq_vals.append(0.0)

        # 不期望产出平衡：Σ λ Yb[:,k] + s_b[k] - y_b_o[k] t = 0
        for k in range(s2):
            row = np.zeros(var_count, dtype=float)
            for i in range(num_dmu):
                row[i] = Yb[i, k]
            row[s_b_start + k] = 1.0
            row[t_index] = -float(Yb[o, k])
            A_eq_rows.append(row)
            b_eq_vals.append(0.0)

        # VRS：Σ λ = t
        if vrs:
            row = np.zeros(var_count, dtype=float)
            for i in range(num_dmu):
                row[i] = 1.0
            row[t_index] = -1.0
            A_eq_rows.append(row)
            b_eq_vals.append(0.0)

        # 分母归一化：t + d_factor*(Σ s_g'/y_g_o + Σ s_b'/y_b_o) = 1
        d_factor = 1.0 / float(max(1, (s1 + s2)))
        yg_o_safe = denom_safe(Yg[o, :]) if s1 > 0 else np.array([])
        yb_o_safe = denom_safe(Yb[o, :]) if s2 > 0 else np.array([])
        row = np.zeros(var_count, dtype=float)
        row[t_index] = 1.0
        for j in range(s1):
            row[s_g_start + j] = d_factor * (1.0 / float(yg_o_safe[j]))
        for k in range(s2):
            row[s_b_start + k] = d_factor * (1.0 / float(yb_o_safe[k]))
        A_eq_rows.append(row)
        b_eq_vals.append(1.0)

        bounds = [(0, None)] * var_count
        res = optimize.linprog(c=c, A_eq=np.array(A_eq_rows), b_eq=np.array(b_eq_vals), bounds=bounds, method='highs')

        def map_status(r):
            if r.success:
                return 'Optimal'
            if r.status == 2:
                return 'Infeasible'
            if r.status == 3:
                return 'Unbounded'
            return 'NotSolved'

        status = map_status(res)
        if res.success:
            x_sol = res.x
            t_val = float(x_sol[t_index])
            s_in_p = x_sol[s_in_start:s_in_start + m]
            s_g_p = x_sol[s_g_start:s_g_start + s1]
            s_b_p = x_sol[s_b_start:s_b_start + s2]
            lam_vals = x_sol[:lam_count].tolist()
            if t_val > eps:
                s_in = (s_in_p / t_val).tolist()
                s_g = (s_g_p / t_val).tolist()
                s_b = (s_b_p / t_val).tolist()
            else:
                s_in = [np.nan] * m
                s_g = [np.nan] * s1
                s_b = [np.nan] * s2
            rho_val = float(res.fun)
        else:
            t_val = np.nan
            s_in = [np.nan] * m
            s_g = [np.nan] * s1
            s_b = [np.nan] * s2
            lam_vals = [np.nan] * lam_count
            rho_val = np.nan

        results['rho'].append(rho_val)
        results['status'].append(status)
        results['t'].append(t_val)
        results['s_input'].append(s_in)
        results['s_good'].append(s_g)
        results['s_bad'].append(s_b)
        results['lambda'].append(lam_vals)

    return results

def compute_targets_for_dmu(
    x_o: np.ndarray, y_g_o: np.ndarray, y_b_o: np.ndarray,
    s_in: List[float], s_g: List[float], s_b: List[float]
) -> Dict[str, np.ndarray]:
    """
    使用未缩放松弛 s 计算目标值：
      x_target = x_o - s_in
      y_g_target = y_g_o + s_g
      y_b_target = y_b_o - s_b
    """
    x_t = x_o - np.array(s_in)
    yg_t = y_g_o + np.array(s_g)
    yb_t = y_b_o - np.array(s_b)
    return {
        'x_target': x_t,
        'y_g_target': yg_t,
        'y_b_target': yb_t,
    }


def main():
    # CFG
    excel_path = r"D:\Desk\MyDeskFiles\Compe&Event\Feng\Data\SupeffSbmData(2).xlsx"
    df = load_excel(excel_path)
    field_mapping = mapping(df)
    
    # # 保存字段映射到.json
    # with open('sbm_mapping.json', 'w', encoding='utf-8') as f:
    #     json.dump(field_mapping, f, ensure_ascii=False, indent=2)

    X, Yg, Yb, dmu_ids = build_matrices(df, field_mapping)
    # 在 CRS 下计算超效率（默认 rts=0）
    res_sup = sbm_super_efficiency_with_undesirable(X, Yg, Yb, rts=0)
    # 对不可行的 DMU 回退到标准 SBM（默认 rts=1 使用VRS）
    res_std = sbm_standard_with_undesirable(X, Yg, Yb, rts=1)

    # 组装输出表
    out_rows = []
    for i, dmu in enumerate(dmu_ids):
        # 仅对标准效率≈1且可行的 DMU 计算并报告超效率
        is_eff = (res_std['status'][i] == 'Optimal') and (res_std['rho'][i] is not None) and (abs(res_std['rho'][i] - 1.0) < 1e-3)
        src = res_sup if (res_sup['status'][i] == 'Optimal' and is_eff) else res_std
        targets = compute_targets_for_dmu(
            X[i, :], Yg[i, :], Yb[i, :], src['s_input'][i], src['s_good'][i], src['s_bad'][i]
        )
        row = {
            'DMU': dmu,
            'rho_sup': res_sup['rho'][i] if is_eff else np.nan,
            'status_sup': res_sup['status'][i] if is_eff else 'Skipped',
            't_sup': res_sup['t'][i] if is_eff else np.nan,
            'rho_std': res_std['rho'][i],
            'status_std': res_std['status'][i],
            't_std': res_std['t'][i],
        }
        # 将目标与松弛展平并加前缀列名
        for idx, val in enumerate(src['s_input'][i]):
            row[f'slack_in_{idx}'] = val
        for idx, val in enumerate(src['s_good'][i]):
            row[f'slack_g_{idx}'] = val
        for idx, val in enumerate(src['s_bad'][i]):
            row[f'slack_b_{idx}'] = val

        for idx, val in enumerate(targets['x_target']):
            row[f'target_in_{idx}'] = float(val)
        for idx, val in enumerate(targets['y_g_target']):
            row[f'target_g_{idx}'] = float(val)
        for idx, val in enumerate(targets['y_b_target']):
            row[f'target_b_{idx}'] = float(val)

        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(f'sbm_super_eff_results.csv', index=False, encoding='utf-8-sig')
    print(f'Saved results -> sbm_super_eff_results.csv')

if __name__ == '__main__':
    main()
