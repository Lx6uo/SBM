import numpy as np
import pandas as pd
import sys
from scipy import optimize

def load_sheet(excel_path: str):
    """
    读取 Excel 的‘格式化数据’工作表，使用‘代码’作为DMU名称，构造一次性输入矩阵。
    返回：data(np.ndarray), nx(int), ny(int), nb(int), dmus(int)
    """
    df = pd.read_excel(excel_path, sheet_name='格式化数据')
    input_cols = [
        '建设用地面积',
        '耕地面积',
        '生态用地面积',
        '年末从业人数',
        '固定资产投资额',
        '用电量',
        '研究与试验发展（R&D）经费支出',
    ]
    good_cols = [
        '城乡居民可支配收入',
        '粮食总产量',
        '城乡绿化水平',
    ]
    bad_cols = [
        '单位GDP碳排放',
    ]
    need_cols = ['代码'] + input_cols + good_cols + bad_cols
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Excel缺少必要列: {missing}")
    df_use = df[need_cols].copy()
    df_use = df_use.sort_values(by=['代码']).reset_index(drop=True)
    data = df_use.values
    nx = len(input_cols)
    ny = len(good_cols)
    nb = len(bad_cols)
    dmus = int(df_use.shape[0])
    return data, nx, ny, nb, dmus

# 1.不含非期望产出的SBM
# x为投入，y_g为产出，cur为当前计算的dmu，rts=1为规模报酬可变，rts=0为规模报酬不变
def sbmeff(x, y_g, cur, rts=0):
    m, n = x.shape  # m为投入变量数，n为决定单元数
    s1 = y_g.shape[0]
    f = np.concatenate([np.zeros(n), -1/(m*x[:, cur]),
                        np.zeros(s1), np.array([1])])
    Aeq1 = np.hstack([x,
                      np.identity(m),
                      np.zeros((m, s1)),
                      -x[:, cur, None]])
    Aeq2 = np.hstack([y_g,
                      np.zeros((s1, m)),
                      -np.identity(s1),
                      -y_g[:, cur, None]])
    Aeq4 = np.hstack([np.zeros(n),
                      np.zeros(m),
                      1/((s1)*(y_g[:, cur])),
                      np.array([1])]).reshape(1, -1)
    if (rts == 1):
        Aeq5 = np.hstack([np.ones(n),
                          np.zeros((m+s1)),
                          np.array([-1])]).reshape(1, -1)
        Aeq = np.vstack([Aeq1, Aeq2, Aeq4, Aeq5])
        beq = np.concatenate(
            [np.zeros(m+s1), np.array([1]), np.array([0])])
    else:
        Aeq = np.vstack([Aeq1, Aeq2, Aeq4])
        beq = np.concatenate([np.zeros(m+s1), np.array([1])])
    bounds = tuple([(0, None) for t in range(n+s1+m+1)])
    res = optimize.linprog(c=f, A_eq=Aeq, b_eq=beq, bounds=bounds)
    return res

# 2.不含非期望产出的超效率SBM
# x为投入，y_g为产出，cur为当前计算的dmu，rts=1为规模报酬可变，rts=0为规模报酬不变
def sup_sbmeff(x, y_g, cur, rts=0):
    m, n = x.shape  # m为投入变量数，n为决定单元数
    s1 = y_g.shape[0]
    eps = 1e-8
    # 当前DMU的投入/期望产出，用于归一化与平衡中的 x_o, y_o
    x_safe = np.where(np.abs(x[:, cur]) < eps, eps, x[:, cur])
    yg_safe = np.where(np.abs(y_g[:, cur]) < eps, eps, y_g[:, cur])
    # 参考集变量级排除：移除 λ_cur
    idx = [j for j in range(n) if j != cur]
    x_ref = x[:, idx]
    y_g_ref = y_g[:, idx]
    n_ref = n - 1
    # 目标：最小化分母 D = t + (1/s1) Σ (s_g'/y_g_o)
    f = np.concatenate([np.zeros(n_ref), np.zeros(m),
                        (1.0/s1) * (1.0/yg_safe), np.array([1])])
    # 分子归一化：t − (1/m) Σ (s_in'/x_o) = 1（不含λ）
    Aeq1 = np.hstack([np.zeros(n_ref),
                      -(1.0/m) * (1.0/x_safe),
                      np.zeros(s1),
                      np.array([1])]).reshape(1, -1)
    if (rts == 1):  # 规模报酬可变：Σ λ = t（不含被评估DMU）
        Aeq2 = np.hstack([np.ones(n_ref),
                          np.zeros((m+s1)),
                          np.array([-1])]).reshape(1, -1)
        Aeq = np.vstack([Aeq1, Aeq2])
        beq = np.concatenate([np.array([1]), np.array([0])])
    else:
        Aeq = Aeq1
        beq = np.array([1])
    # 输入平衡：Σ_{j≠cur} λ x_j + s_in − x_o t = 0
    Aeq_in = np.hstack([x_ref,
                        np.identity(m),
                        np.zeros((m, s1)),
                        -x[:, cur, None]])
    # 期望产出平衡：Σ_{j≠cur} λ y_g_j − s_g − y_g_o t = 0
    Aeq_g = np.hstack([y_g_ref,
                       np.zeros((s1, m)),
                       -np.identity(s1),
                       -y_g[:, cur, None]])
    Aeq = np.vstack([Aeq_in, Aeq_g, Aeq])
    beq = np.concatenate([np.zeros(m+s1), beq])
    bounds = tuple([(0, None) for i in range(n_ref+s1+m+1)])
    res = optimize.linprog(c=f, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
    return res

# 3.含非期望产出的SBM
# x为投入，y_g为产出，y_b为非期望产出，cur为当前计算的dmu，rts=1为规模报酬可变，rts=0为规模报酬不变
def un_sbmeff(x, y_g, y_b, cur, rts=0):
    m, n = x.shape  # m为投入变量数，n为决定单元数
    s1 = y_g.shape[0]
    s2 = y_b.shape[0]
    f = np.concatenate([np.zeros(n), -1/(m*x[:, cur]),
                        np.zeros(s1+s2), np.array([1])])
    Aeq1 = np.hstack([x,
                      np.identity(m),
                      np.zeros((m, s1+s2)),
                      -x[:, cur, None]])
    Aeq2 = np.hstack([y_g,
                      np.zeros((s1, m)),
                      -np.identity(s1),
                      np.zeros((s1, s2)),
                      -y_g[:, cur, None]])
    Aeq3 = np.hstack([y_b,
                      np.zeros((s2, m)),
                      np.zeros((s2, s1)),
                      np.identity(s2),
                      -y_b[:, cur, None]])
    d_factor = 1.0 / float(max(1, (s1 + s2)))
    Aeq4 = np.hstack([np.zeros(n),
                      np.zeros(m),
                      d_factor * (1.0/(y_g[:, cur])),
                      d_factor * (1.0/(y_b[:, cur])),
                      np.array([1])]).reshape(1, -1)
    if (rts == 1):
        Aeq5 = np.hstack([np.ones(n),
                          np.zeros((m+s1+s2)),
                          np.array([-1])]).reshape(1, -1)
        Aeq = np.vstack([Aeq1, Aeq2, Aeq3, Aeq4, Aeq5])
        beq = np.concatenate(
            [np.zeros(m+s1+s2), np.array([1]), np.array([0])])
    else:
        Aeq = np.vstack([Aeq1, Aeq2, Aeq3, Aeq4])
        beq = np.concatenate([np.zeros(m+s1+s2), np.array([1])])
    bounds = tuple([(0, None) for t in range(n+s1+s2+m+1)])
    res = optimize.linprog(c=f, A_eq=Aeq, b_eq=beq, bounds=bounds)
    return res

# 4.含非期望产出的超效率SBM
# x为投入，y_g为产出，y_b为非期望产出，cur为当前计算的dmu，rts=1为规模报酬可变，rts=0为规模报酬不变
def sup_un_sbmeff(x, y_g, y_b, cur, rts=0):
    m, n = x.shape  # m为投入变量数，n为决定单元数
    s1 = y_g.shape[0]
    s2 = y_b.shape[0]
    eps = 1e-8
    # 当前DMU数据的安全处理
    x_safe = np.where(np.abs(x[:, cur]) < eps, eps, x[:, cur])
    yg_safe = np.where(np.abs(y_g[:, cur]) < eps, eps, y_g[:, cur])
    yb_safe = np.where(np.abs(y_b[:, cur]) < eps, eps, y_b[:, cur])
    # 参考集变量级排除：移除 λ_cur
    idx = [j for j in range(n) if j != cur]
    x_ref = x[:, idx]
    y_g_ref = y_g[:, idx]
    y_b_ref = y_b[:, idx]
    n_ref = n - 1
    # 目标：最小化分母 D = t + (1/s1) Σ (s_g'/y_g_o) + (1/s2) Σ (s_b'/y_b_o)
    d_factor = 1.0 / float(max(1, (s1 + s2)))
    f = np.concatenate([np.zeros(n_ref), np.zeros(m),
                        d_factor * (1.0/yg_safe),
                        d_factor * (1.0/yb_safe),
                        np.array([1])])
    # 分子归一化：t − (1/m) Σ (s_in'/x_o) = 1（不含λ）
    Aeq1 = np.hstack([np.zeros(n_ref),
                      -(1.0/m) * (1.0/x_safe),
                      np.zeros(s1),
                      np.zeros(s2),
                      np.array([1])]).reshape(1, -1)
    if (rts == 1):  # 规模报酬可变：Σ λ = t（不含被评估DMU）
        Aeq2 = np.hstack([np.ones(n_ref),
                          np.zeros((m+s1+s2)),
                          np.array([-1])]).reshape(1, -1)
        Aeq = np.vstack([Aeq1, Aeq2])
        beq = np.concatenate([np.array([1]), np.array([0])])
    else:
        Aeq = Aeq1
        beq = np.array([1])
    # 输入平衡：Σ_{j≠cur} λ x_j + s_in − x_o t = 0
    Aeq_in = np.hstack([x_ref,
                        np.identity(m),
                        np.zeros((m, s1+s2)),
                        -x[:, cur, None]])
    # 期望产出平衡：Σ_{j≠cur} λ y_g_j − s_g − y_g_o t = 0
    Aeq_g = np.hstack([y_g_ref,
                       np.zeros((s1, m)),
                       -np.identity(s1),
                       np.zeros((s1, s2)),
                       -y_g[:, cur, None]])
    # 不期望产出平衡：Σ_{j≠cur} λ y_b_j + s_b − y_b_o t = 0
    Aeq_b = np.hstack([y_b_ref,
                       np.zeros((s2, m)),
                       np.zeros((s2, s1)),
                       np.identity(s2),
                       -y_b[:, cur, None]])
    Aeq = np.vstack([Aeq_in, Aeq_g, Aeq_b, Aeq])
    beq = np.concatenate([np.zeros(m+s1+s2), beq])
    bounds = tuple([(0, None) for i in range(n_ref+s1+s2+m+1)])
    res = optimize.linprog(c=f, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
    return res

def main():
    # CFG
    rts = 0  # 0为CRS（CCR）规模报酬；1为VRS(BCC)规模报酬可变
    sup = 1  # 0为不带超效率；1为带超效率
    undesirable = 1  # 1为存在非期望产出；0为不存在非期望产出
    tol = 1e-3  # 判定标准效率≈1的容差
    excel_path = r"d:\Desk\MyDeskFiles\Compe&Event\Feng\Data\SupeffSbmData(2).xlsx"
    data, nx, ny, nb, dmus = load_sheet(excel_path)

    theta = []  # 用于存储结果，每一个DMU的效率值
    suptheta = []  # 用于存储超效率结果，每一个DMU的效率值

    # 计算一般效率
    # 一次性计算所有行的标准效率
    yeardata = data
    dmuname = yeardata[:, 0]  # 使用“代码”列作为DMU名称
    x = yeardata[:, 1:nx+1]
    x = x.T  # 投入变量
    y_g = yeardata[:, 1+nx:1+nx+ny]
    y_g = y_g.T  # 产出变量
    if undesirable == 1:
        y_b = yeardata[:, 1+nx+ny:1+nx+ny+nb]
        y_b = y_b.T  # 非期望产出变量
    for i in range(dmus):
        if (undesirable == 1):
            res = un_sbmeff(x=x, y_g=y_g, y_b=y_b, cur=i, rts=rts)
            theta.append((dmuname[i], res.fun))
        else:
            res = sbmeff(x=x, y_g=y_g, cur=i, rts=rts)
            theta.append((dmuname[i], res.fun))
    # 不计算超效率就保存结果退出
    if (sup == 0):
        dfres = pd.DataFrame(theta, columns=('dmu', '效率'))
        dfres.to_excel("sbmeff.xlsx", sheet_name='eff', index=False)
        sys.exit(0)

    # 计算超效率结果（一次性计算所有行）
    yeardata = data
    dmuname = yeardata[:, 0]
    x = yeardata[:, 1:nx+1]
    x = x.T
    y_g = yeardata[:, 1+nx:1+nx+ny]
    y_g = y_g.T
    if undesirable == 1:
        y_b = yeardata[:, 1+nx+ny:1+nx+ny+nb]
        y_b = y_b.T
    # CCR超效率输出：仅对标准效率≈1的DMU尝试超效率；不可行或非有效标记状态
    sup_vals = []
    sup_statuses = []
    for i in range(dmus):
        is_eff = abs(theta[i][1] - 1.0) < tol
        if (undesirable == 1):
            res = sup_un_sbmeff(x=x, y_g=y_g, y_b=y_b, cur=i, rts=rts)
        else:
            res = sup_sbmeff(x=x, y_g=y_g, cur=i, rts=rts)
        if not is_eff:
            sup_vals.append(np.nan)
            sup_statuses.append('Skipped')
        else:
            if res.success and (res.fun > 1e-12):
                sup_vals.append(res.fun)
                sup_statuses.append('Optimal')
            else:
                sup_vals.append(np.nan)
                sup_statuses.append('Infeasible')

    # 组装CCR结果表：标准效率+（有效DMU的）超效率+状态
    dfres = pd.DataFrame({
        'dmu': dmuname,
        '标准效率_CCR': [t[1] for t in theta],
        '超效率_CCR': sup_vals,
        'sup_status': sup_statuses,
    })
    dfres.to_excel("effres.xlsx", sheet_name='eff', index=False)
    
if __name__ == '__main__':
    main()
