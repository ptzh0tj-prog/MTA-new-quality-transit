"""
时间序列分析模块
================
基于 MTA 每日运量数据（2020–2026），分析疫情冲击、恢复曲线、
结构断点（Chow 检验）、运输方式分化、高峰结构演化。

核心方法：statsmodels（ADF / Chow / 季节分解）+ matplotlib + seaborn
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats as scipy_stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

sns.set_theme(style="whitegrid", context="paper", font="SimHei")
plt.rcParams["axes.unicode_minus"] = False

PROCESSED_DIR = Path(__file__).parent / "processed"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

MODE_LABELS = {
    "Subway": "地铁",
    "Bus": "公交",
    "LIRR": "长岛铁路",
    "MNR": "北方铁路",
    "BT": "桥隧",
    "AAR": "通勤铁路(AAR)",
    "SIR": "史泰登岛铁路",
    "CBD Entries": "CBD进入",
    "CRZ Entries": "拥堵区进入",
}


# ================================================================
#  1. 加载 & 构建时间序列特征
# ================================================================

def load_daily_with_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取每日 Parquet，构建恢复率、月度聚合等特征。

    Returns
    -------
    daily : 每日数据（含 year, month, dayofweek 等）
    monthly : 月度聚合（含 recovery_rate）
    """
    df = pd.read_parquet(PROCESSED_DIR / "mta_daily.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"] >= 5

    # 基线：2020年3月前半月为疫情前最后窗口；回退到 2023 年均值
    baseline_pre = (
        df[(df["year"] == 2020) & (df["month"] == 3) & (df["date"].dt.day <= 15)]
        .groupby("mode")["daily_ridership"].mean()
    )
    baseline_post = (
        df[df["year"] == 2023]
        .groupby("mode")["daily_ridership"].mean()
    )
    baseline = baseline_pre.combine_first(baseline_post)

    # 月度聚合
    monthly = (
        df.groupby(["year", "month", "mode"])["daily_ridership"]
        .mean().reset_index()
    )
    monthly["baseline"] = monthly["mode"].map(baseline)
    monthly["recovery_rate"] = monthly["daily_ridership"] / monthly["baseline"]
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2) + "-01"
    )

    tqdm.write(f"每日: {len(df):,} 行,  {df['mode'].nunique()} 种模式,  "
               f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    tqdm.write(f"月度: {len(monthly):,} 行,  基线=2020年3月前半月 / 2023年均值")
    return df, monthly


# ================================================================
#  2. 平稳性检验（ADF）
# ================================================================

def test_stationarity(series: pd.Series, label: str = "") -> dict:
    """Augmented Dickey-Fuller 检验。"""
    s = series.dropna()
    if len(s) < 10:
        d = {"label": label, "adf_stat": float("nan"), "p_value": float("nan"),
             "critical_1%": float("nan"), "critical_5%": float("nan"),
             "is_stationary": False, "n_obs": len(s)}
        tqdm.write(f"  {label:20s}  数据不足 (n={len(s)}), 跳过")
        return d
    result = adfuller(s, autolag="AIC")
    d = {
        "label": label,
        "adf_stat": result[0],
        "p_value": result[1],
        "critical_1%": result[4]["1%"],
        "critical_5%": result[4]["5%"],
        "is_stationary": result[1] < 0.05,
        "n_obs": result[3],
    }
    status = "STATIONARY" if d["is_stationary"] else "NON-STATIONARY"
    tqdm.write(f"  {label:20s}  ADF={d['adf_stat']:7.3f}  p={d['p_value']:.4f}  {status}")
    return d


def run_stationarity_tests(monthly: pd.DataFrame) -> list[dict]:
    """对所有运输方式的恢复率做 ADF 检验。"""
    tqdm.write("\n[ADF 平稳性检验]")
    results = []
    for mode in tqdm(sorted(monthly["mode"].unique()), desc="  ADF", unit="个", ncols=80):
        sub = monthly[monthly["mode"] == mode].set_index("date")["recovery_rate"]
        results.append(test_stationarity(sub, mode))
    return results


# ================================================================
#  3. 疫情冲击深度 & 恢复速度
# ================================================================

def analyze_shock_recovery(monthly: pd.DataFrame) -> pd.DataFrame:
    """量化每种运输方式的冲击深度和恢复速度。"""
    tqdm.write("\n[冲击与恢复分析]")
    rows = []
    for mode in tqdm(sorted(monthly["mode"].unique()), desc="  计算", unit="个", ncols=80):
        sub = monthly[monthly["mode"] == mode].set_index("date")["recovery_rate"].dropna()
        if len(sub) < 6:
            tqdm.write(f"  {mode:12s}  数据不足 ({len(sub)} 月), 跳过")
            continue

        # 冲击深度
        shock_window = sub.loc["2020-03":"2020-06"]
        if len(shock_window) > 0:
            shock_depth = shock_window.min()
            shock_month = shock_window.idxmin()
        else:
            shock_depth = sub.min()
            shock_month = sub.idxmin()

        recent = sub.loc["2026-01":"2026-06"]
        current = recent.mean() if len(recent) > 0 else sub.iloc[-6:].mean()

        post = sub.loc[shock_month:]
        recovered = post[post >= 0.8]
        months_to_80 = (
            (recovered.index[0] - shock_month).days / 30
            if len(recovered) > 0 else None
        )

        rows.append({
            "mode": mode,
            "shock_depth": shock_depth,
            "shock_month": str(shock_month.date())[:7] if pd.notna(shock_month) else "-",
            "current_level": current,
            "months_to_80pct": months_to_80,
            "fully_recovered": current >= 1.0,
        })

    result = pd.DataFrame(rows).sort_values("shock_depth")
    for _, r in result.iterrows():
        flag = "[OK]" if r["fully_recovered"] else "[PARTIAL]"
        m80 = f"{r['months_to_80pct']:.0f}月" if pd.notna(r["months_to_80pct"]) else "未恢复"
        tqdm.write(f"  {r['mode']:12s}  冲击最低={r['shock_depth']:.1%}  "
                   f"当前={r['current_level']:.1%}  恢复至80%={m80}  {flag}")

    return result


# ================================================================
#  4. Chow 结构断点检验（新增）
# ================================================================

def chow_test(series: pd.Series, break_date: str, trend: str = "linear") -> dict:
    """对单条时间序列做一个 Chow 结构断点检验。

    Parameters
    ----------
    series : 时间序列（index 为 DatetimeIndex）
    break_date : 断点日期，如 '2022-12-01'
    trend : 'linear' 或 'quadratic'

    Returns
    -------
    dict : {break_date, f_stat, p_value, n_pre, n_post, rss_full, rss_split, is_break}
    """
    s = series.dropna()
    break_ts = pd.Timestamp(break_date)

    pre = s[s.index < break_ts]
    post = s[s.index >= break_ts]

    n_pre, n_post = len(pre), len(post)
    if n_pre < 12 or n_post < 12:
        return {"break_date": break_date, "f_stat": float("nan"),
                "p_value": float("nan"), "n_pre": n_pre, "n_post": n_post,
                "is_break": False, "error": "数据不足（<12个月）"}

    k = 3 if trend == "quadratic" else 2  # parameters: intercept + trend [+ trend²]

    def _design_matrix(idx_series):
        t = np.arange(len(idx_series))
        if trend == "quadratic":
            return np.column_stack([np.ones(len(t)), t, t**2])
        return np.column_stack([np.ones(len(t)), t])

    def _ols_rss(X, y):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        return float(resid @ resid)

    # 全样本
    y_full = s.values
    X_full = _design_matrix(s.index)
    rss_full = _ols_rss(X_full, y_full)

    # 分段
    X_pre = _design_matrix(pre.index)
    X_post = _design_matrix(post.index)
    rss_pre = _ols_rss(X_pre, pre.values)
    rss_post = _ols_rss(X_post, post.values)
    rss_split = rss_pre + rss_post

    # F 统计量
    n_total = n_pre + n_post
    f_num = (rss_full - rss_split) / k
    f_den = rss_split / (n_total - 2 * k)
    if f_den < 1e-12:
        return {"break_date": break_date, "f_stat": float("nan"),
                "p_value": float("nan"), "n_pre": n_pre, "n_post": n_post,
                "is_break": False, "error": "分母接近零"}

    f_stat = f_num / f_den
    p_value = 1 - scipy_stats.f.cdf(f_stat, k, n_total - 2 * k)
    is_break = p_value < 0.05

    return {
        "break_date": break_date,
        "f_stat": f_stat,
        "p_value": p_value,
        "n_pre": n_pre,
        "n_post": n_post,
        "rss_full": rss_full,
        "rss_split": rss_split,
        "is_break": is_break,
    }


def run_chow_tests(monthly: pd.DataFrame) -> pd.DataFrame:
    """对核心运输方式的恢复曲线做多断点 Chow 检验。

    检验节点及含义：
    - 2020-04-01：疫情冲击（数据起于 2020-03，此点仅 1 月前数据 → 标记为 N/A）
    - 2022-12-01：ChatGPT/AI 加速期（核心假设）
    - 邻近月份扫描：如果 F 统计量在整个 2022 年都高，说明恢复是非线性的
      渐进过程而非突然断点；如果仅在 2022-12 附近峰值突出，则更可能是一次性冲击

    使用二次趋势模型（允许恢复曲线自然弯曲），避免线性模型误报。
    """
    tqdm.write("\n[Chow 结构断点检验]")
    core_modes = ["Subway", "Bus", "LIRR", "MNR"]
    break_points = ["2020-04-01", "2022-12-01"]
    # 2021-06 和 2023-06 作为稳健性对照，只用线性（检验是否"到处都显著"）
    placebo_points = ["2021-06-01", "2023-06-01"]

    rows = []
    for mode in tqdm(core_modes, desc="  Chow", unit="个", ncols=80):
        sub = monthly[monthly["mode"] == mode].set_index("date")["recovery_rate"]
        for bp in break_points + placebo_points:
            trend_type = "quadratic" if bp in break_points else "linear"
            r = chow_test(sub, bp, trend=trend_type)
            r["mode"] = mode
            r["trend"] = trend_type
            rows.append(r)

    result = pd.DataFrame(rows)

    # 同时跑灵敏度扫描，判断是渐进变化还是突变压断点
    chow_scan = run_chow_sensitivity(monthly, "Subway", "2022-12-01", window=12, quiet=True)

    # 打印报告
    print(f"\n{'='*80}")
    print(f"  Chow 结构断点检验结果")
    print(f"{'='*80}")
    print(f"  {'运输方式':12s} {'断点':12s} {'趋势':>9s} {'F':>7s} {'p值':>8s}  {'结论':>12s}  n前/n后")
    print(f"  {'-'*70}")
    for _, r in result.iterrows():
        if pd.isna(r["f_stat"]):
            sig = "数据不足"
        elif r["is_break"]:
            sig = "*** 显著断点"
        elif r["p_value"] < 0.1:
            sig = "*   弱显著"
        else:
            sig = "    不显著"
        label = MODE_LABELS.get(r["mode"], r["mode"])
        if pd.isna(r["f_stat"]):
            print(f"  {label:12s} {r['break_date']:12s} {r['trend']:>9s} {'N/A':>7s} {'N/A':>8s}  {sig:>12s}")
        else:
            print(f"  {label:12s} {r['break_date']:12s} {r['trend']:>9s} {r['f_stat']:7.1f} {r['p_value']:8.4f}  {sig:>12s}  {r['n_pre']}/{r['n_post']}")

    # 灵敏度解读
    valid_scan = chow_scan[chow_scan["f_stat"].notna()]
    if len(valid_scan) > 0:
        f_max = valid_scan["f_stat"].max()
        f_at_12 = valid_scan[valid_scan["date"] == pd.Timestamp("2022-12-01")]["f_stat"]
        f_at_12 = f_at_12.values[0] if len(f_at_12) > 0 else float("nan")
        f_range = (valid_scan["f_stat"].max() - valid_scan["f_stat"].min())
        is_sharp = (f_max - valid_scan["f_stat"].min()) > valid_scan["f_stat"].std() * 1.5 \
                   if len(valid_scan) > 3 else False

        print(f"  {'-'*70}")
        print(f"  灵敏度解读 (Subway 2022-12 ±12月):")
        print(f"    F 统计量范围: {valid_scan['f_stat'].min():.1f} – {f_max:.1f}")
        print(f"    F 峰值在:     {valid_scan.loc[valid_scan['f_stat'].idxmax(), 'date'].date()}")
        print(f"    2022-12 的 F: {f_at_12:.1f}")
        if f_range < 10:
            print(f"    → F 统计量在整个窗口内波动较小，恢复曲线的非线性是渐进过程")
            print(f"      而非单一突发事件导致的结构断点。这与 AI 的渐进渗透一致。")
        else:
            print(f"    → F 统计量有明显峰值，可能存在结构性突变。")

    print(f"{'='*80}\n")

    return result


def run_chow_sensitivity(monthly: pd.DataFrame, mode: str = "Subway",
                         center: str = "2022-12-01", window: int = 12,
                         quiet: bool = False) -> pd.DataFrame:
    """对断点前后 ±window 个月的每一点做 Chow 检验（稳健性扫描）。

    如果只有中心断点显著而相邻月份不显著 → 断点是真实突变的。
    如果整个窗口全部显著 → 恢复曲线本身的非线性，而非特定事件冲击。
    """
    if not quiet:
        tqdm.write(f"\n[Chow 稳健性扫描] {mode} 围绕 {center}")
    sub = monthly[monthly["mode"] == mode].set_index("date")["recovery_rate"]

    center_ts = pd.Timestamp(center)
    dates = pd.date_range(center_ts - pd.DateOffset(months=window),
                          center_ts + pd.DateOffset(months=window), freq="MS")

    rows = []
    iterator = tqdm(dates, desc="  扫描", unit="点", ncols=80) if not quiet else dates
    for d in iterator:
        r = chow_test(sub, d.strftime("%Y-%m-%d"))
        r["mode"] = mode
        rows.append(r)

    result = pd.DataFrame(rows)
    result["date"] = dates
    return result


# ================================================================
#  5. 结构分化分析（新增）
# ================================================================

def analyze_structural_divergence(monthly: pd.DataFrame) -> pd.DataFrame:
    """计算各运输方式恢复率的月度离散度，量化结构分化程度。

    排除 CBD Entries / CRZ Entries（2025 年才开始，基线不同，会虚增离散度）。
    保留 7 种核心模式：Subway, Bus, LIRR, MNR, BT, AAR, SIR。

    Returns
    -------
    DataFrame : 每月一行，含 std, range, iqr, cv
    """
    tqdm.write("\n[结构分化分析]")

    # 排除 2025 年才开始的新模式
    core = monthly[~monthly["mode"].isin(["CBD Entries", "CRZ Entries"])]

    disp = (
        core.groupby(["year", "month", "date"])["recovery_rate"]
        .agg(
            dispersion_std=("std"),
            dispersion_range=lambda x: x.max() - x.min(),
            dispersion_iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
            mean_recovery=("mean"),
            n_modes=("count"),
        )
        .reset_index()
    )
    disp["dispersion_cv"] = disp["dispersion_std"] / disp["mean_recovery"].replace(0, np.nan)

    # 分时期统计
    eras = {
        "冲击期(2020.3-6)": disp[disp["date"].between("2020-03", "2020-06")],
        "恢复前期(2020.7-2021.12)": disp[disp["date"].between("2020-07", "2021-12")],
        "恢复后期(2022.1-2023.12)": disp[disp["date"].between("2022-01", "2023-12")],
        "新常态(2024.1-2026.6)": disp[disp["date"].between("2024-01", "2026-06")],
    }
    tqdm.write(f"  各时期平均分化度 (std):")
    for label, era_df in eras.items():
        if len(era_df) > 0:
            tqdm.write(f"    {label:20s}: mean std={era_df['dispersion_std'].mean():.4f}")

    tqdm.write(f"  当前分化 std={disp['dispersion_std'].iloc[-1]:.3f}  "
               f"(首月 std={disp['dispersion_std'].iloc[0]:.3f})")

    return disp


def analyze_mode_correlations(monthly: pd.DataFrame) -> dict:
    """分时期计算各运输方式恢复率的相关性矩阵。

    四个时期：疫情前(2020.1-2)、疫情冲击(2020.3-6)、恢复期(2021-2023)、后恢复期(2024-2026)
    """
    tqdm.write("\n[方式相关性矩阵]")

    # pivot 成 日期 × 运输方式
    pivot = monthly.pivot(index="date", columns="mode", values="recovery_rate")

    eras = {
        "P1_疫情冲击 (2020.3-6)": ("2020-03", "2020-06"),
        "P2_恢复前期 (2020.7-2021.12)": ("2020-07", "2021-12"),
        "P3_恢复后期 (2022.1-2023.12)": ("2022-01", "2023-12"),
        "P4_新常态 (2024.1-2026.6)": ("2024-01", "2026-06"),
    }

    corr_mats = {}
    for label, (start, end) in eras.items():
        window = pivot.loc[start:end]
        if len(window) > 3:
            corr_mats[label] = window.corr()
            tqdm.write(f"  {label}: {len(window)} 个月")

    return corr_mats


# ================================================================
#  6. 高峰结构演化（接入 MTA 小时数据）
# ================================================================

def load_station_features() -> pd.DataFrame:
    """加载或构建站点级特征（含 commute_ratio）。"""
    sf_path = PROCESSED_DIR / "mta_station_features.parquet"
    if sf_path.exists():
        tqdm.write(f"[加载] {sf_path.name}")
        return pd.read_parquet(sf_path)

    tqdm.write("[构建] 站点特征（需先跑 data_processor.build_station_features）...")
    from data_processor import load_hourly_agg, build_station_features
    hourly = load_hourly_agg()
    stations = build_station_features(hourly)
    return stations


def analyze_commute_ratio_evolution(hourly: pd.DataFrame) -> pd.DataFrame:
    """逐年计算每个站点的 commute_ratio，追踪通勤模式变化。

    注意：这需要从原始 hourly 数据中按年分别计算（不能用聚合后的 station_features，
    因为后者是整个时间窗口的平均）。
    """
    tqdm.write("\n[高峰结构演化]")
    df = hourly.copy()
    df["hour_int"] = df["timestamp"].dt.hour
    df["year"] = df["timestamp"].dt.year

    rows = []
    years = sorted(df["year"].unique())
    for yr in tqdm(years, desc="  逐年计算", unit="年", ncols=80):
        yr_df = df[df["year"] == yr]

        morning = (
            yr_df[yr_df["hour_int"].isin([7, 8, 9])]
            .groupby(["station_id", "station_name", "borough"])["entries"]
            .mean().reset_index().rename(columns={"entries": "peak_morning"})
        )
        evening = (
            yr_df[yr_df["hour_int"].isin([17, 18, 19])]
            .groupby(["station_id", "station_name", "borough"])["entries"]
            .mean().reset_index().rename(columns={"entries": "peak_evening"})
        )

        merged = morning.merge(evening, on=["station_id", "station_name", "borough"], how="outer")
        merged["commute_ratio"] = (
            merged["peak_morning"] / merged["peak_evening"].replace(0, np.nan)
        )
        merged["year"] = yr
        rows.append(merged)

    result = pd.concat(rows, ignore_index=True)
    # 过滤掉比值极端异常的站点（可能是数据错误或极小站）
    result = result[
        (result["commute_ratio"] > 0.1) & (result["commute_ratio"] < 10)
    ]
    tqdm.write(f"  共 {result['station_id'].nunique()} 个站点, {len(result)} 条年记录")

    # 通勤型站点占比趋势
    for yr in years:
        yr_data = result[result["year"] == yr]["commute_ratio"].dropna()
        commute_pct = (yr_data.between(0.7, 1.3)).mean()
        tqdm.write(f"  {yr}: 通勤型站点(0.7≤ratio≤1.3) 占比={commute_pct:.1%}  "
                   f"中位 commute_ratio={yr_data.median():.3f}")

    return result


# ================================================================
#  7. 可视化（原有 + 新增）
# ================================================================

def plot_recovery_curves(monthly: pd.DataFrame, df_daily: pd.DataFrame):
    """Fig 1: 各运输方式恢复曲线（双面板）。"""
    core_modes = ["Subway", "Bus", "LIRR", "MNR"]
    all_modes = sorted(monthly["mode"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    colors = sns.color_palette("tab10", len(all_modes))

    ax = axes[0]
    for mode, c in zip(all_modes, colors):
        sub = monthly[monthly["mode"] == mode]
        ax.plot(sub["date"], sub["recovery_rate"], color=c, lw=1.5, label=mode)
    ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.axvspan("2020-03-01", "2020-06-01", alpha=0.08, color="red", label="首轮疫情")
    ax.set_title("各运输方式月度恢复率 (基线=2020年3月上半月 / 2023年均值)", fontsize=13, fontweight="bold")
    ax.set_ylabel("恢复率"); ax.legend(ncol=3, fontsize=8, loc="lower right")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    ax = axes[1]
    for mode in core_modes:
        sub = monthly[monthly["mode"] == mode]
        ax.plot(sub["date"], sub["recovery_rate"], lw=2, label=mode)
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.axvspan("2020-03-01", "2020-06-01", alpha=0.08, color="red")
    for date_str, label, y in [
        ("2022-03-01", "全面解封", 0.55),
        ("2022-12-01", "ChatGPT\n发布", 0.9),
        ("2025-01-01", "AI加速期", 1.05),
    ]:
        ax.annotate(label, (pd.Timestamp(date_str), y),
                    fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    ax.set_title("核心运输方式恢复对比（虚线=疫情前水平）", fontsize=13, fontweight="bold")
    ax.set_ylabel("恢复率"); ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig1_recovery_curves.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig1_recovery_curves.png")
    plt.close()


def plot_shock_waterfall(shock_df: pd.DataFrame):
    """Fig 2: 冲击深度瀑布图。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    df = shock_df.sort_values("shock_depth")
    colors = ["#d62728" if v < 0.3 else "#ff7f0e" if v < 0.6 else "#2ca02c"
              for v in df["shock_depth"]]
    ax.bar(range(len(df)), df["shock_depth"], color=colors, edgecolor="white")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in df["mode"]], rotation=30, ha="right")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(i, r["shock_depth"] + 0.02, f"{r['shock_depth']:.0%}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("疫情冲击深度：各运输方式最低恢复率", fontsize=14, fontweight="bold")
    ax.set_ylabel("最低恢复率 (越低冲击越深)")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig2_shock_waterfall.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig2_shock_waterfall.png")
    plt.close()


def plot_seasonal_decomp(monthly: pd.DataFrame, mode: str = "Subway"):
    """Fig 3: 季节分解（以地铁为例）。"""
    sub = monthly[monthly["mode"] == mode].set_index("date")["daily_ridership"]
    decomp = seasonal_decompose(sub, model="additive", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    for ax, (name, component) in zip(axes, [
        ("Observed", decomp.observed),
        ("Trend", decomp.trend),
        ("Seasonal", decomp.seasonal),
        ("Residual", decomp.resid),
    ]):
        ax.plot(component.index, component.values, lw=0.8)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"{mode} 月度运量季节分解 (加法模型)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig3_seasonal_decomp.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig3_seasonal_decomp.png")
    plt.close()


# ---- 新增可视化 ----

def plot_chow_results(monthly: pd.DataFrame, chow_df: pd.DataFrame):
    """Fig 4: Chow 断点图——恢复曲线 + 断点标注 + p 值表。"""
    core_modes = ["Subway", "Bus", "LIRR", "MNR"]
    break_points = ["2020-04-01", "2022-12-01"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = {"2020-04-01": "#d62728", "2022-12-01": "#1f77b4"}

    for ax, mode in zip(axes.flat, core_modes):
        sub = monthly[monthly["mode"] == mode].set_index("date")
        ax.plot(sub.index, sub["recovery_rate"], color="black", lw=1.5, label="恢复率")
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)

        for bp in break_points:
            bp_ts = pd.Timestamp(bp)
            # 标注
            ax.axvline(bp_ts, color=colors[bp], ls="--", lw=1.5, alpha=0.7)
            # 该断点的 Chow 结果
            row = chow_df[(chow_df["mode"] == mode) & (chow_df["break_date"] == bp)]
            sig_text = ""
            if len(row) > 0 and not pd.isna(row.iloc[0]["f_stat"]):
                r = row.iloc[0]
                sig = "***" if r["is_break"] else ("*" if r["p_value"] < 0.1 else "ns")
                sig_text = f"F={r['f_stat']:.1f}, p={r['p_value']:.3f} {sig}"

            label = "疫情冲击" if "2020" in bp else "AI加速"
            ax.annotate(f"{label}\n{sig_text}",
                        (bp_ts, ax.get_ylim()[1] * 0.85),
                        fontsize=8, ha="center", color=colors[bp],
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(MODE_LABELS.get(mode, mode), fontsize=12, fontweight="bold")
        ax.set_ylabel("恢复率")
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    fig.suptitle("Chow 结构断点检验：疫情冲击 vs AI 加速期", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig4_chow_breakpoints.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig4_chow_breakpoints.png")
    plt.close()


def plot_chow_sensitivity(chow_scan: pd.DataFrame, mode: str = "Subway"):
    """Fig 4b: Chow 稳健性扫描——断点周围的 F 统计量曲线。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    valid = chow_scan[chow_scan["f_stat"].notna()].copy()
    ax.plot(valid["date"], valid["f_stat"], "o-", color="#1f77b4", lw=1.5, markersize=4)
    ax.axhline(
        scipy_stats.f.ppf(0.95, 2, valid["n_pre"].iloc[0] + valid["n_post"].iloc[0] - 4),
        color="red", ls="--", lw=1, label="5% 临界值"
    )
    ax.axvline(pd.Timestamp("2022-12-01"), color="orange", ls="--", lw=1.5, label="2022-12 (AI)")
    ax.set_title(f"{mode} Chow 稳健性扫描：断点前后 12 个月", fontsize=13, fontweight="bold")
    ax.set_ylabel("Chow F 统计量"); ax.set_xlabel("断点位置")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig4b_chow_sensitivity.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig4b_chow_sensitivity.png")
    plt.close()


def plot_structural_divergence(disp: pd.DataFrame):
    """Fig 5: 结构分化时序——各方式恢复率离散度 over time。"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.fill_between(disp["date"], disp["dispersion_std"] * 0, disp["dispersion_std"],
                    alpha=0.3, color="#d62728", label="标准差 std")
    ax.plot(disp["date"], disp["dispersion_std"], color="#d62728", lw=2)
    ax.plot(disp["date"], disp["dispersion_iqr"], color="#ff7f0e", lw=1.5, label="IQR")
    ax.axvline(pd.Timestamp("2022-12-01"), color="blue", ls="--", lw=1, alpha=0.7, label="AI节点")
    ax.axvspan("2020-03-01", "2020-06-01", alpha=0.08, color="red")
    ax.set_ylabel("离散度"); ax.legend(loc="upper left")
    ax.set_title("运输方式恢复率的结构分化", fontsize=13, fontweight="bold")

    ax = axes[1]
    ax.plot(disp["date"], disp["mean_recovery"], color="#2ca02c", lw=2, label="平均恢复率")
    ax.fill_between(disp["date"],
                    disp["mean_recovery"] - disp["dispersion_std"],
                    disp["mean_recovery"] + disp["dispersion_std"],
                    alpha=0.2, color="#2ca02c", label="±1 std")
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.axvline(pd.Timestamp("2022-12-01"), color="blue", ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("恢复率"); ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig5_structural_divergence.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig5_structural_divergence.png")
    plt.close()


def plot_correlation_matrices(corr_mats: dict):
    """Fig 6: 四时期方式相关性热力图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    for ax, (label, corr) in zip(axes.flat, corr_mats.items()):
        labels_cn = [MODE_LABELS.get(c, c) for c in corr.columns]
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
                    vmin=-1, vmax=1, square=True, linewidths=0.5,
                    xticklabels=labels_cn, yticklabels=labels_cn,
                    ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(label, fontsize=11, fontweight="bold")

    fig.suptitle("各运输方式恢复率相关性矩阵的时期变化", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig6_correlation_matrices.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig6_correlation_matrices.png")
    plt.close()


def plot_commute_ratio_evolution(commute_yearly: pd.DataFrame):
    """Fig 7: 通勤比分布演进——2022/2023/2024 KDE + 中位线。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：KDE 分布
    ax = axes[0]
    years = sorted(commute_yearly["year"].unique())
    colors = sns.color_palette("viridis", len(years))
    for yr, c in zip(years, colors):
        data = commute_yearly[commute_yearly["year"] == yr]["commute_ratio"].dropna()
        if len(data) > 5:
            sns.kdeplot(data, ax=ax, color=c, lw=2, label=f"{yr} (n={len(data)})", fill=True, alpha=0.15)
    ax.axvline(1.0, color="black", ls="--", lw=0.8, alpha=0.5, label="通勤平衡线")
    ax.set_xlim(0, 3)
    ax.set_title("站点 commute_ratio 分布演进", fontsize=13, fontweight="bold")
    ax.set_xlabel("早晚高峰比 (越接近1越偏向通勤)"); ax.set_ylabel("密度")
    ax.legend()

    # 右：中位线 + 四分位距
    ax = axes[1]
    medians = commute_yearly.groupby("year")["commute_ratio"].median()
    q25 = commute_yearly.groupby("year")["commute_ratio"].quantile(0.25)
    q75 = commute_yearly.groupby("year")["commute_ratio"].quantile(0.75)
    ax.fill_between(medians.index, q25.values, q75.values, alpha=0.25, color="#1f77b4")
    ax.plot(medians.index, medians.values, "o-", color="#1f77b4", lw=2, markersize=8)
    ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.5)
    for yr in medians.index:
        ax.annotate(f"{medians[yr]:.3f}", (yr, medians[yr]),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
    ax.set_title("commute_ratio 中位数趋势 (±IQR)", fontsize=13, fontweight="bold")
    ax.set_ylabel("早晚高峰比"); ax.set_xlabel("年份")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig7_commute_ratio_evolution.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig7_commute_ratio_evolution.png")
    plt.close()


def plot_commute_by_borough(commute_yearly: pd.DataFrame):
    """Fig 7b: 各 borough 通勤比年度变化。"""
    boroughs = sorted(commute_yearly["borough"].dropna().unique())
    years = sorted(commute_yearly["year"].unique())
    n_b = len(boroughs)
    cols = min(3, n_b)
    rows = (n_b + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for i, borough in enumerate(boroughs):
        ax = axes_flat[i]
        b_data = commute_yearly[commute_yearly["borough"] == borough]
        for yr in years:
            yr_data = b_data[b_data["year"] == yr]["commute_ratio"].dropna()
            if len(yr_data) > 3:
                sns.kdeplot(yr_data, ax=ax, lw=1.5, label=str(yr), fill=True, alpha=0.1)
        ax.axvline(1.0, color="black", ls="--", lw=0.8, alpha=0.4)
        ax.set_title(f"{borough} ({len(b_data['station_id'].unique())} 站)", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 3)
        if i == 0:
            ax.legend(fontsize=8)

    # 隐藏多余子图
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("各 Borough 早晚高峰比分布演进", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig7b_commute_by_borough.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig7b_commute_by_borough.png")
    plt.close()


# ================================================================
#  8. 一键运行
# ================================================================

def run_all():
    print("=" * 55)
    print("  MTA 时间序列分析 (完整版)")
    print("=" * 55)

    # ---- 加载 ----
    daily, monthly = load_daily_with_features()

    # ---- ADF 检验 ----
    adf_results = run_stationarity_tests(monthly)

    # ---- 冲击 & 恢复 ----
    shock_df = analyze_shock_recovery(monthly)

    # ---- Chow 断点检验 ----
    chow_df = run_chow_tests(monthly)
    # 灵敏度扫描已在 run_chow_tests 内部调用；单独跑一次用于绘图
    chow_scan = run_chow_sensitivity(monthly, "Subway", "2022-12-01", window=12)

    # ---- 结构分化 ----
    disp = analyze_structural_divergence(monthly)
    corr_mats = analyze_mode_correlations(monthly)

    # ---- 高峰结构演化 ----
    try:
        from data_processor import load_hourly_agg
        hourly = load_hourly_agg()
        commute_yearly = analyze_commute_ratio_evolution(hourly)
    except Exception as e:
        tqdm.write(f"[跳过] 高峰分析失败: {e}")
        commute_yearly = None

    # ---- 可视化（全量） ----
    tqdm.write("\n[绘图] 生成全部论文图表...")
    plot_recovery_curves(monthly, daily)
    plot_shock_waterfall(shock_df)
    plot_seasonal_decomp(monthly, "Subway")
    plot_chow_results(monthly, chow_df)
    plot_chow_sensitivity(chow_scan, "Subway")
    plot_structural_divergence(disp)
    plot_correlation_matrices(corr_mats)
    if commute_yearly is not None:
        plot_commute_ratio_evolution(commute_yearly)
        plot_commute_by_borough(commute_yearly)

    # ---- 核心发现 ----
    print("\n" + "=" * 55)
    print("  核心发现")
    print("=" * 55)

    subway = shock_df[shock_df["mode"] == "Subway"].iloc[0]
    print(f"  地铁冲击深度:    {subway['shock_depth']:.1%}")
    print(f"  地铁当前恢复:    {subway['current_level']:.1%}")
    print(f"  恢复至80%耗时:    {subway['months_to_80pct']:.0f} 个月")

    best = shock_df.sort_values("current_level", ascending=False).iloc[0]
    worst = shock_df.sort_values("current_level", ascending=True).iloc[0]
    print(f"  恢复最好:        {MODE_LABELS.get(best['mode'], best['mode'])} ({best['current_level']:.1%})")
    print(f"  恢复最差:        {MODE_LABELS.get(worst['mode'], worst['mode'])} ({worst['current_level']:.1%})")

    # Chow 核心发现
    ai_breaks = chow_df[(chow_df["break_date"] == "2022-12-01") & (chow_df["is_break"])]
    print(f"\n  Chow 检验 (2022-12 AI节点):")
    for _, r in ai_breaks.iterrows():
        print(f"    {MODE_LABELS.get(r['mode'], r['mode'])}: F={r['f_stat']:.1f}, p={r['p_value']:.4f} **")
    if len(ai_breaks) == 0:
        print(f"    无显著断点 —— AI 冲击可能是渐进的，而非突变")

    n_stationary = sum(1 for r in adf_results if r["is_stationary"])
    print(f"\n  平稳序列:        {n_stationary}/{len(adf_results)}")
    print(f"  分化度峰值:      {disp['dispersion_std'].max():.3f}")
    print(f"  当前分化度:      {disp['dispersion_std'].iloc[-1]:.3f}")

    print(f"\n[完成] 图表已保存至 {PLOTS_DIR}/")
    return daily, monthly, shock_df, chow_df, disp, commute_yearly


if __name__ == "__main__":
    daily, monthly, shock_df, chow_df, disp, commute_yearly = run_all()
