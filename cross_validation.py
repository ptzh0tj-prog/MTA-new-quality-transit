"""
交叉验证分析模块
================
将 LinkedIn NLP 推论与 MTA 行为证据对照，实现三层验证。

三层逻辑：
  L1 — 结构断点：LinkedIn"AI岗位2023爆发" → MTA Chow断点检验
  L2 — 通勤模式：LinkedIn"新质远程率高" → MTA commute_ratio 下降
  L3 — 运输替代：LinkedIn"高薪岗位集中曼哈顿" → LIRR/MNR 恢复更快

依赖：模块一（timeseries_analysis）的 Chow 检验 + 模块二（postings_nlp）的行业标签
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 复用模块一的 Chow 检验
from timeseries_analysis import chow_test, load_daily_with_features

sns.set_theme(style="whitegrid", context="paper", font="SimHei")
plt.rcParams["axes.unicode_minus"] = False

# ---- 路径 ----
PROCESSED_DIR = Path(__file__).parent / "processed"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

DAILY_PARQUET = PROCESSED_DIR / "mta_daily.parquet"
HOURLY_PARQUET = PROCESSED_DIR / "mta_hourly_agg.parquet"
STATION_PARQUET = PROCESSED_DIR / "mta_station_features.parquet"
POSTINGS_ENRICHED = PROCESSED_DIR / "postings_enriched.parquet"

MODE_LABELS = {
    "Subway": "地铁", "Bus": "公交", "LIRR": "长岛铁路",
    "MNR": "北方铁路", "BT": "桥隧", "AAR": "通勤铁路(AAR)",
    "SIR": "史泰登岛铁路",
}

CORE_MODES = ["Subway", "Bus", "LIRR", "MNR"]


# ====================================================================
#  数据加载
# ====================================================================

def load_all_data():
    """加载 MTA 日/时/站点 + postings NLP 结果。"""
    tqdm.write("[加载] 汇总数据...")

    # MTA 时序（模块一输出）
    daily, monthly = load_daily_with_features()

    # 站点特征
    if STATION_PARQUET.exists():
        stations = pd.read_parquet(STATION_PARQUET)
        tqdm.write(f"  站点特征: {len(stations)} 站")
    else:
        tqdm.write("[警告] 站点特征未生成，正在构建...")
        from data_processor import load_hourly_agg, build_station_features
        hourly = load_hourly_agg()
        stations = build_station_features(hourly)

    # postings NLP（模块二输出）
    if POSTINGS_ENRICHED.exists():
        postings = pd.read_parquet(POSTINGS_ENRICHED)
        tqdm.write(f"  postings NLP: {len(postings):,} 岗位, "
                   f"新质={postings['is_new_quality'].sum():,}")
    else:
        tqdm.write("[警告] postings_enriched.parquet 未生成，请先运行模块二")
        postings = None

    return daily, monthly, stations, postings


# ====================================================================
#  L1 — 结构断点验证
# ====================================================================

def validate_l1_chow(monthly: pd.DataFrame) -> dict:
    """L1: LinkedIn 推论 "AI/科技岗位 2023 起爆发" → MTA Chow 断点检验。

    核心逻辑：
    - 如果 AI 在 2022-12（ChatGPT）后加速渗透就业市场，
    - 那么就业结构变化应反映在通勤行为上，
    - MTA 恢复曲线在此节点应有结构性变化。

    对 4 种核心运输方式的恢复曲线做 Chow 检验（2022-12-01），
    并对邻近月份做灵敏度扫描确认是否为真断点。
    """
    tqdm.write("\n" + "=" * 65)
    tqdm.write("  L1 — 结构断点验证")
    tqdm.write("=" * 65)
    tqdm.write("  LinkedIn 推论: AI/科技岗位 2023 起爆发")
    tqdm.write("  MTA 检验:      恢复曲线在 2022-12 是否有 Chow 断点？")

    # 对核心模式做 Chow 检验
    rows = []
    for mode in tqdm(CORE_MODES, desc="  L1 Chow", unit="个", ncols=80):
        sub = monthly[monthly["mode"] == mode].set_index("date")["recovery_rate"]
        r = chow_test(sub, "2022-12-01", trend="quadratic")
        r["mode"] = mode
        rows.append(r)

    chow_l1 = pd.DataFrame(rows)

    # 打印判决
    print(f"\n  {'运输方式':12s} {'F':>7s} {'p值':>8s}  {'判决':>14s}")
    print(f"  {'-'*48}")
    verdicts = {}
    for _, r in chow_l1.iterrows():
        label = MODE_LABELS.get(r["mode"], r["mode"])
        if pd.isna(r["f_stat"]):
            v = "数据不足"
        elif r["is_break"]:
            v = "*** 显著断点"
        else:
            v = "不显著"
        print(f"  {label:12s} {r['f_stat']:7.1f} {r['p_value']:8.4f}  {v:>14s}")
        verdicts[r["mode"]] = v

    # 整体判断
    n_breaks = chow_l1["is_break"].sum()
    if n_breaks >= 2:
        conclusion = (
            f"[PASS] L1 验证通过: {n_breaks}/{len(CORE_MODES)} 种方式在 2022-12 存在显著断点，"
            "LinkedIn 的 AI 爆发推论被 MTA 行为数据支持。"
        )
    elif n_breaks == 1:
        conclusion = (
            f"[WEAK] L1 部分通过: 仅 {n_breaks}/{len(CORE_MODES)} 种方式显著，"
            "AI 冲击可能尚未完全传导至通勤行为。"
        )
    else:
        conclusion = (
            "[FAIL] L1 未通过: 无显著断点。AI 渗透对通勤总量的影响可能尚未显现，"
            "或其影响是渐进而非突变式的。"
        )

    print(f"\n  {conclusion}")
    print(f"{'='*65}\n")

    return {"chow_df": chow_l1, "conclusion": conclusion, "verdicts": verdicts}


# ====================================================================
#  L2 — 通勤模式验证
# ====================================================================

def compute_yearly_commute_ratio(hourly: pd.DataFrame) -> pd.DataFrame:
    """从小时数据逐年计算每个站点的 commute_ratio。"""
    df = hourly.copy()
    df["hour_int"] = df["timestamp"].dt.hour
    df["year"] = df["timestamp"].dt.year

    rows = []
    years = sorted(df["year"].unique())
    for yr in years:
        yr_df = df[df["year"] == yr]
        morning = (
            yr_df[yr_df["hour_int"].isin([7, 8, 9])]
            .groupby(["station_id", "station_name", "borough"])["entries"]
            .mean().rename("peak_morning")
        )
        evening = (
            yr_df[yr_df["hour_int"].isin([17, 18, 19])]
            .groupby(["station_id", "station_name", "borough"])["entries"]
            .mean().rename("peak_evening")
        )
        merged = pd.merge(
            morning.reset_index(), evening.reset_index(),
            on=["station_id", "station_name", "borough"], how="outer"
        )
        merged["commute_ratio"] = (
            merged["peak_morning"] / merged["peak_evening"].replace(0, np.nan)
        )
        merged["year"] = yr
        rows.append(merged)

    result = pd.concat(rows, ignore_index=True)
    result = result[(result["commute_ratio"] > 0.1) & (result["commute_ratio"] < 10)]
    return result


def cluster_stations(stations: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """KMeans 聚类站点：通勤型 / 混合型 / 居住型 / 商业型。

    基于 commute_ratio, peak_morning, entries_cv, recovery 等特征。
    """
    tqdm.write(f"\n[L2] KMeans 站点聚类 (k={n_clusters})...")

    feats = stations[["commute_ratio", "peak_morning", "peak_evening",
                       "entries_cv", "avg_daily_entries"]].copy()
    # 加入 recovery（如果有）
    if "recovery_2024_vs_2022" in stations.columns:
        feats["recovery"] = stations["recovery_2024_vs_2022"].fillna(1.0)

    feats = feats.dropna()
    feats_log = feats.copy()
    for c in ["peak_morning", "peak_evening", "avg_daily_entries"]:
        feats_log[c] = np.log1p(feats[c])

    scaler = StandardScaler()
    X = scaler.fit_transform(feats_log)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(X)

    result = stations.loc[feats.index].copy()
    result["cluster"] = labels
    result["cluster"] = result["cluster"].astype(int)

    # 解读聚类含义
    profiles = result.groupby("cluster").agg(
        n=("station_id", "count"),
        avg_commute_ratio=("commute_ratio", "median"),
        avg_morning=("peak_morning", "median"),
        avg_evening=("peak_evening", "median"),
        avg_entries=("avg_daily_entries", "median"),
        avg_cv=("entries_cv", "median"),
    )
    if "recovery_2024_vs_2022" in result.columns:
        profiles["avg_recovery"] = result.groupby("cluster")["recovery_2024_vs_2022"].median()

    # 自动命名
    names = {}
    for cid, row in profiles.iterrows():
        if row["avg_commute_ratio"] >= 1.2:
            names[cid] = f"居住型 (C{cid})"
        elif row["avg_commute_ratio"] <= 0.7:
            names[cid] = f"商业型 (C{cid})"
        elif 0.8 <= row["avg_commute_ratio"] <= 1.2:
            names[cid] = f"通勤型 (C{cid})"
        else:
            names[cid] = f"混合型 (C{cid})"

    result["cluster_name"] = result["cluster"].map(names)

    tqdm.write("  聚类画像:")
    for cid, row in profiles.iterrows():
        tqdm.write(f"    {names[cid]:20s}  n={int(row['n']):>3d}  "
                   f"commute_ratio={row['avg_commute_ratio']:.2f}  "
                   f"日均进站={row['avg_entries']:.0f}")

    return result


def validate_l2_commute(hourly: pd.DataFrame, stations: pd.DataFrame,
                        postings: pd.DataFrame = None) -> dict:
    """L2: LinkedIn 推论 "新质行业远程率高于传统" → MTA commute_ratio 下降。

    核心逻辑：
    - 如果远程办公在新质行业中更普遍，
    - 那么以通勤为主的站点（commute_ratio≈1）占比应逐年下降，
    - 站点早晚高峰比分布应向非通勤方向偏移。
    """
    tqdm.write("\n" + "=" * 65)
    tqdm.write("  L2 — 通勤模式验证")
    tqdm.write("=" * 65)
    tqdm.write("  LinkedIn 推论: 新质行业远程率高于传统行业")
    tqdm.write("  MTA 检验:      428 站点 commute_ratio 是否从 2022->2024 下降？")

    # 逐年 commute_ratio
    commute_yearly = compute_yearly_commute_ratio(hourly)

    # 逐年统计
    years = sorted(commute_yearly["year"].unique())
    trend_data = []
    for yr in years:
        yr_data = commute_yearly[commute_yearly["year"] == yr]["commute_ratio"].dropna()
        commute_pct = (yr_data.between(0.7, 1.3)).mean()
        trend_data.append({
            "year": yr,
            "median": yr_data.median(),
            "mean": yr_data.mean(),
            "std": yr_data.std(),
            "commute_pct": commute_pct,
            "n_stations": len(yr_data),
        })
    trend_df = pd.DataFrame(trend_data)

    # 判断趋势
    if len(trend_df) >= 2:
        first_commute = trend_df["commute_pct"].iloc[0]
        last_commute = trend_df["commute_pct"].iloc[-1]
        delta = last_commute - first_commute

        # 简单线性回归看斜率
        from scipy import stats
        slope, _, _, p_slope, _ = stats.linregress(
            trend_df["year"], trend_df["commute_pct"]
        )

        if p_slope < 0.05 and slope < 0:
            conclusion = (
                f"[PASS] L2 验证通过: 通勤型站点占比从 {first_commute:.1%} -> {last_commute:.1%}"
                f" (Δ={delta:+.1%})，呈显著下降趋势 (p={p_slope:.3f})。"
                "远程办公假说被 MTA 行为数据支持。"
            )
        elif slope < 0:
            conclusion = (
                f"[WEAK] L2 弱支持: 通勤型站点占比从 {first_commute:.1%} -> {last_commute:.1%}"
                f" (Δ={delta:+.1%})，方向符合预期但统计不显著 (p={p_slope:.3f})。"
            )
        else:
            conclusion = (
                f"[FAIL] L2 未通过: 通勤型站点占比未下降 (delta={delta:+.1%})。"
                "远程办公对通勤模式的影响可能被其他因素抵消。"
            )
    else:
        conclusion = "数据不足，无法判定趋势"

    # 打印
    print(f"\n  年份  中位ratio  通勤型占比  n站点")
    print(f"  {'-'*40}")
    for _, r in trend_df.iterrows():
        print(f"  {int(r['year']):4d}  {r['median']:.3f}     {r['commute_pct']:.1%}       {int(r['n_stations'])}")
    print(f"\n  {conclusion}")

    # 站点聚类
    if "commute_ratio" in stations.columns and "peak_morning" in stations.columns:
        stations_clustered = cluster_stations(stations)
    else:
        stations_clustered = None

    # borough 聚合
    borough_trend = (
        commute_yearly.groupby(["borough", "year"])["commute_ratio"]
        .median().reset_index()
    )

    print(f"{'='*65}\n")

    return {
        "trend_df": trend_df,
        "conclusion": conclusion,
        "commute_yearly": commute_yearly,
        "stations_clustered": stations_clustered,
        "borough_trend": borough_trend,
    }


# ====================================================================
#  L3 — 运输方式替代验证
# ====================================================================

def validate_l3_mode_substitution(monthly: pd.DataFrame,
                                  postings: pd.DataFrame = None) -> dict:
    """L3: LinkedIn 推论 "高薪新质岗位集中在曼哈顿" → MTA 通勤铁路恢复更快。

    核心逻辑：
    - 如果高薪新质岗位集中在曼哈顿且通勤距离拉长，
    - 那么远郊通勤铁路（LIRR/MNR）恢复率应高于市内交通（Subway/Bus），
    - 曼哈顿站点恢复率应高于外围 borough。
    """
    tqdm.write("=" * 65)
    tqdm.write("  L3 — 运输方式替代验证")
    tqdm.write("=" * 65)
    tqdm.write("  LinkedIn 推论: 高薪新质岗位集中在曼哈顿，通勤距离拉长")
    tqdm.write("  MTA 检验:      LIRR/MNR 恢复率是否高于 Subway/Bus？")

    # 各方式当前恢复率
    recent = monthly[monthly["date"] >= "2025-01-01"]
    mode_recovery = (
        recent.groupby("mode")["recovery_rate"]
        .mean().sort_values(ascending=False)
    )

    print(f"\n  各运输方式当前恢复率 (2025-2026 均值):")
    print(f"  {'方式':20s} {'恢复率':>8s}  {'评级':>10s}")
    print(f"  {'-'*42}")
    rankings = {}
    for mode, rate in mode_recovery.items():
        label = MODE_LABELS.get(mode, mode)
        stars = "[HIGH]" if rate >= 0.9 else ("[MID] " if rate >= 0.75 else "[LOW] ")
        print(f"  {label:20s} {rate:8.1%}  {stars:>10s}")
        rankings[mode] = {"rate": rate, "label": label, "tier": stars}

    # 检验 LIRR/MNR > Subway > Bus 的层级假说
    commuter_rail = mode_recovery.get("LIRR", 0) + mode_recovery.get("MNR", 0)
    commuter_rail /= 2
    subway_rate = mode_recovery.get("Subway", 0)
    bus_rate = mode_recovery.get("Bus", 0)

    # 层级判断
    tiers_ok = commuter_rail > subway_rate > bus_rate

    if tiers_ok:
        conclusion = (
            f"[PASS] L3 验证通过: LIRR/MNR 平均恢复 {commuter_rail:.1%} > "
            f"Subway {subway_rate:.1%} > Bus {bus_rate:.1%}。"
            "薪资-通勤距离假说被 MTA 行为数据支持：远郊通勤恢复强于市内。"
        )
    elif commuter_rail > subway_rate:
        conclusion = (
            f"[WEAK] L3 弱支持: LIRR/MNR ({commuter_rail:.1%}) > Subway ({subway_rate:.1%})，"
            f"但 Bus ({bus_rate:.1%}) 不符合层级。"
        )
    else:
        conclusion = (
            f"[FAIL] L3 未通过: 通勤铁路 ({commuter_rail:.1%}) 未显著高于 "
            f"Subway ({subway_rate:.1%})。远郊通勤优势不明显。"
        )

    # 曼哈顿 vs 其他 borough 的站点恢复
    if STATION_PARQUET.exists():
        stations = pd.read_parquet(STATION_PARQUET)
        if "recovery_2024_vs_2022" in stations.columns:
            manhattan_rec = stations[stations["borough"] == "Manhattan"]["recovery_2024_vs_2022"].median()
            outer_rec = stations[stations["borough"] != "Manhattan"]["recovery_2024_vs_2022"].median()
            tqdm.write(f"\n  曼哈顿站点恢复率中位: {manhattan_rec:.3f}")
            tqdm.write(f"  外围站点恢复率中位: {outer_rec:.3f}")
            if manhattan_rec > outer_rec:
                tqdm.write("  -> 曼哈顿恢复强于外围，与高薪岗位集中假说一致")
            else:
                tqdm.write("  -> 曼哈顿恢复弱于外围，需重新审视假说")

    print(f"\n  {conclusion}")
    print(f"{'='*65}\n")

    return {
        "mode_recovery": mode_recovery,
        "conclusion": conclusion,
        "tiers_ok": tiers_ok,
        "rankings": rankings,
    }


# ====================================================================
#  综合判决
# ====================================================================

def synthesize_verdict(l1: dict, l2: dict, l3: dict) -> str:
    """汇总三层验证结果，给出综合判决。"""
    passed = 0
    for result in [l1, l2, l3]:
        if result["conclusion"].startswith("[PASS]"):
            passed += 1

    if passed == 3:
        overall = "强支持"
    elif passed == 2:
        overall = "中等支持"
    elif passed == 1:
        overall = "弱支持"
    else:
        overall = "不支持"

    summary = (
        f"三层交叉验证综合判决: {overall} (通过 {passed}/3 层)\n"
        f"  L1 结构断点: {l1['conclusion'][:80]}...\n"
        f"  L2 通勤模式: {l2['conclusion'][:80]}...\n"
        f"  L3 运输替代: {l3['conclusion'][:80]}..."
    )
    print(f"\n{'='*65}")
    print(f"  {summary}")
    print(f"{'='*65}")
    return summary


# ====================================================================
#  可视化
# ====================================================================

def plot_three_layer_validation(l1: dict, l2: dict, l3: dict, monthly: pd.DataFrame):
    """Fig 11: 三层验证综合面板图——三行，每层一个核心结果。"""
    tqdm.write("\n[绘图] Fig 11: 三层验证综合面板图...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # ---- 面板 1: L1 Chow 断点 ----
    ax = axes[0]
    core_modes = ["Subway", "Bus", "LIRR", "MNR"]
    colors = sns.color_palette("tab10", len(core_modes))
    for mode, c in zip(core_modes, colors):
        sub = monthly[monthly["mode"] == mode].set_index("date")
        ax.plot(sub.index, sub["recovery_rate"], color=c, lw=1.8, label=MODE_LABELS.get(mode, mode))
    ax.axvline(pd.Timestamp("2022-12-01"), color="red", ls="--", lw=1.5, alpha=0.7)
    ax.axhline(1.0, color="black", ls=":", lw=0.8, alpha=0.4)
    ax.annotate("ChatGPT 发布\n2022-12", (pd.Timestamp("2022-12-01"), 0.55),
                fontsize=9, ha="center", color="red",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax.set_title("L1 — 结构断点验证：恢复曲线在 AI 节点的 Chow 检验", fontsize=13, fontweight="bold")
    ax.set_ylabel("恢复率"); ax.legend(ncol=4, fontsize=8, loc="lower right")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    # 标注 Chow 结果
    chow_df = l1.get("chow_df")
    if chow_df is not None:
        text_lines = ["Chow(2022-12) 结果:"]
        for _, r in chow_df.iterrows():
            sig = "**" if r.get("is_break") else "ns"
            text_lines.append(
                f"  {MODE_LABELS.get(r['mode'], r['mode'])}: F={r['f_stat']:.1f} p={r['p_value']:.3f} {sig}"
            )
        ax.text(0.98, 0.25, "\n".join(text_lines), transform=ax.transAxes,
                fontsize=8, fontfamily="sans-serif", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.85))

    # ---- 面板 2: L2 commute_ratio 趋势 ----
    ax = axes[1]
    commute_yearly = l2.get("commute_yearly")
    if commute_yearly is not None:
        years = sorted(commute_yearly["year"].unique())
        # 每年 KDE
        yr_colors = sns.color_palette("viridis", len(years))
        for yr, c in zip(years, yr_colors):
            data = commute_yearly[commute_yearly["year"] == yr]["commute_ratio"].dropna()
            if len(data) > 5:
                sns.kdeplot(data, ax=ax, color=c, lw=2, label=f"{yr}", fill=True, alpha=0.12)
        ax.axvline(1.0, color="black", ls="--", lw=0.8, alpha=0.4)
        ax.set_xlim(0, 3)
        ax.legend(title="年份", fontsize=9)

    ax.set_title("L2 — 通勤模式验证：站点 commute_ratio 分布演进", fontsize=13, fontweight="bold")
    ax.set_xlabel("早晚高峰比 (越接近 1 越偏向通勤)")
    ax.set_ylabel("密度")

    # 通勤占比趋势 inset
    trend_df = l2.get("trend_df")
    if trend_df is not None and len(trend_df) >= 2:
        inset_text = (f"通勤型站点(0.7≤ratio≤1.3)占比:\n"
                     f"  {int(trend_df['year'].iloc[0])}: {trend_df['commute_pct'].iloc[0]:.1%}\n"
                     f"  {int(trend_df['year'].iloc[-1])}: {trend_df['commute_pct'].iloc[-1]:.1%}")
        ax.text(0.02, 0.95, inset_text, transform=ax.transAxes,
                fontsize=9, fontfamily="sans-serif", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8))

    # ---- 面板 3: L3 方式恢复率排序 ----
    ax = axes[2]
    mode_recovery = l3.get("mode_recovery")
    if mode_recovery is not None:
        core_rec = mode_recovery[mode_recovery.index.isin(CORE_MODES)]
        bars = ax.bar(
            range(len(core_rec)),
            [core_rec[m] for m in core_rec.index],
            color=["#2ca02c" if m in ["LIRR", "MNR"] else "#1f77b4" if m == "Subway" else "#ff7f0e"
                   for m in core_rec.index],
            edgecolor="white",
        )
        ax.set_xticks(range(len(core_rec)))
        ax.set_xticklabels([MODE_LABELS.get(m, m) for m in core_rec.index], fontsize=11)
        ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.4)
        for i, m in enumerate(core_rec.index):
            rate = core_rec[m]
            ax.text(i, rate + 0.01, f"{rate:.1%}", ha="center", fontsize=10, fontweight="bold")

    ax.set_title("L3 — 运输方式替代：核心方式当前恢复率排序", fontsize=13, fontweight="bold")
    ax.set_ylabel("恢复率"); ax.set_ylim(0, 1.2)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    # 总判决
    fig.suptitle("三层交叉验证综合面板", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig11_three_layer_validation.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig11_three_layer_validation.png")
    plt.close()


def plot_station_cluster_map(stations_clustered: pd.DataFrame):
    """Fig 12: 站点聚类地图——Manhattan/Brooklyn 站点按类别着色。"""
    tqdm.write("\n[绘图] Fig 12: 站点聚类地图...")

    if stations_clustered is None:
        tqdm.write("[跳过] 无聚类数据")
        return

    # 聚焦 Manhattan 和 Brooklyn（站点最多）
    focus = stations_clustered[
        stations_clustered["borough"].isin(["Manhattan", "Brooklyn"])
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    cluster_names = sorted(focus["cluster_name"].unique())
    palette = sns.color_palette("Set2", len(cluster_names))
    color_map = dict(zip(cluster_names, palette))

    for ax, borough in zip(axes, ["Manhattan", "Brooklyn"]):
        b_data = focus[focus["borough"] == borough].sort_values("station_id").reset_index(drop=True)

        x_positions = np.arange(len(b_data))
        for cname in cluster_names:
            mask = b_data["cluster_name"].values == cname
            if mask.sum() > 0:
                ax.scatter(
                    x_positions[mask],
                    b_data.loc[mask, "commute_ratio"],
                    c=[color_map[cname]], label=cname, s=40, alpha=0.7, edgecolors="white", lw=0.3,
                )

        ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.4)
        ax.set_title(f"{borough} ({len(b_data)} 站)", fontsize=12, fontweight="bold")
        ax.set_xlabel("站点序号 (按 ID 排列)")
        ax.set_ylabel("commute_ratio")
        if borough == "Manhattan":
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("站点聚类：Manhattan vs Brooklyn 通勤模式分类", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig12_station_cluster_map.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig12_station_cluster_map.png")
    plt.close()


def plot_mode_recovery_ranking(l3: dict):
    """Fig 13: 方式恢复率排序——条形图 + 远程率标注。"""
    tqdm.write("\n[绘图] Fig 13: 方式恢复率排序...")

    mode_recovery = l3.get("mode_recovery")
    if mode_recovery is None:
        return

    # 过滤核心 + 其他方式
    all_modes = mode_recovery.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    for m in all_modes.index:
        if m in ["LIRR", "MNR"]:
            colors.append("#2ca02c")  # 通勤铁路绿
        elif m == "Subway":
            colors.append("#1f77b4")  # 地铁蓝
        elif m == "Bus":
            colors.append("#ff7f0e")  # 公交橙
        else:
            colors.append("#7f7f7f")

    bars = ax.barh(
        range(len(all_modes)),
        [all_modes[m] for m in all_modes.index],
        color=colors, edgecolor="white", height=0.6,
    )

    ax.set_yticks(range(len(all_modes)))
    ax.set_yticklabels([MODE_LABELS.get(m, m) for m in all_modes.index], fontsize=10)
    ax.axvline(1.0, color="black", ls="--", lw=1, alpha=0.5, label="疫情前基线")
    ax.set_xlabel("恢复率")
    ax.set_title("各运输方式恢复率排序 (2025-2026 均值)", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    for i, m in enumerate(all_modes.index):
        rate = all_modes[m]
        ax.text(rate + 0.01, i, f"{rate:.1%}", va="center", fontsize=9, fontweight="bold")

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="通勤铁路 (LIRR/MNR)"),
        Patch(facecolor="#1f77b4", label="地铁 (Subway)"),
        Patch(facecolor="#ff7f0e", label="公交 (Bus)"),
        Patch(facecolor="#7f7f7f", label="其他"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig13_mode_recovery_ranking.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig13_mode_recovery_ranking.png")
    plt.close()


def plot_commute_vs_recovery(stations: pd.DataFrame):
    """Fig 14: commute_ratio × recovery 散点——每站点一个点，按 borough 着色。"""
    tqdm.write("\n[绘图] Fig 14: commute_ratio × recovery 散点...")

    if "recovery_2024_vs_2022" not in stations.columns:
        tqdm.write("[跳过] 缺少 recovery 列")
        return

    plot_data = stations.dropna(subset=["commute_ratio", "recovery_2024_vs_2022"])
    # 过滤极端值
    plot_data = plot_data[
        (plot_data["commute_ratio"].between(0.1, 3)) &
        (plot_data["recovery_2024_vs_2022"].between(0.5, 2))
    ]

    boroughs = sorted(plot_data["borough"].dropna().unique())
    colors = sns.color_palette("Set2", len(boroughs))

    fig, ax = plt.subplots(figsize=(10, 7))

    for b, c in zip(boroughs, colors):
        subset = plot_data[plot_data["borough"] == b]
        ax.scatter(
            subset["commute_ratio"], subset["recovery_2024_vs_2022"],
            c=[c], label=f"{b} (n={len(subset)})", s=30, alpha=0.6, edgecolors="white"
        )

    ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.3)
    ax.axvline(1.0, color="black", ls="--", lw=0.8, alpha=0.3)
    ax.set_xlabel("commute_ratio (早晚高峰比)")
    ax.set_ylabel("恢复率 (2024 vs 2022)")
    ax.set_title("站点通勤模式 vs 恢复率", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best")

    # 添加趋势线
    x = plot_data["commute_ratio"].values
    y = plot_data["recovery_2024_vs_2022"].values
    if len(x) > 10:
        from numpy.polynomial.polynomial import polyfit
        coef = polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = coef[0] + coef[1] * x_line
        ax.plot(x_line, y_line, color="red", lw=1.5, alpha=0.6, label="线性趋势")
        # Pearson 相关
        r, p = scipy_stats.pearsonr(x, y)
        ax.text(0.02, 0.95, f"Pearson r={r:.3f}  p={p:.3f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig14_commute_vs_recovery.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig14_commute_vs_recovery.png")
    plt.close()


# ====================================================================
#  一键运行
# ====================================================================

def run_all():
    print("=" * 60)
    print("  交叉验证分析 (模块三)")
    print("=" * 60)

    # ---- 加载数据 ----
    tqdm.write("\n>>> 加载数据")
    daily, monthly, stations, postings = load_all_data()

    # ---- 补充：从 hourly 数据加载用于 commute_ratio 逐年计算 ----
    hourly = pd.read_parquet(HOURLY_PARQUET)
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])

    # ---- L1: 结构断点验证 ----
    tqdm.write("\n>>> L1: 结构断点验证")
    l1 = validate_l1_chow(monthly)

    # ---- L2: 通勤模式验证 ----
    tqdm.write("\n>>> L2: 通勤模式验证")
    l2 = validate_l2_commute(hourly, stations, postings)

    # ---- L3: 运输方式替代验证 ----
    tqdm.write("\n>>> L3: 运输方式替代验证")
    l3 = validate_l3_mode_substitution(monthly, postings)

    # ---- 综合判决 ----
    verdict = synthesize_verdict(l1, l2, l3)

    # ---- 可视化 ----
    tqdm.write("\n>>> 产出图表")
    plot_three_layer_validation(l1, l2, l3, monthly)
    if l2.get("stations_clustered") is not None:
        plot_station_cluster_map(l2["stations_clustered"])
    plot_mode_recovery_ranking(l3)
    plot_commute_vs_recovery(stations)

    # ---- 输出验证矩阵 ----
    print(f"\n{'='*60}")
    print(f"  交叉验证矩阵")
    print(f"{'='*60}")
    print(f"  {'层':5s} {'推论来源':15s} {'检验方法':15s} {'结果':15s}")
    print(f"  {'-'*55}")
    print(f"  {'L1':5s} {'LinkedIn NLP':15s} {'Chow 断点':15s} "
          f"{'通过' if l1['conclusion'].startswith('[PASS]') else '待定':>15s}")
    print(f"  {'L2':5s} {'LinkedIn NLP':15s} {'commute_ratio':15s} "
          f"{'通过' if l2['conclusion'].startswith('[PASS]') else '待定':>15s}")
    print(f"  {'L3':5s} {'LinkedIn NLP':15s} {'方式恢复排序':15s} "
          f"{'通过' if l3['conclusion'].startswith('[PASS]') else '待定':>15s}")
    print(f"{'='*60}")

    return daily, monthly, stations, l1, l2, l3


if __name__ == "__main__":
    daily, monthly, stations, l1, l2, l3 = run_all()
