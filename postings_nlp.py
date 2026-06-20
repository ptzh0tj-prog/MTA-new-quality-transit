"""
postings.csv NLP 管道
=====================
从 postings.csv 的 description + title 中提取行业结构画像，
输出"新质 vs 传统"的对比特征。

核心技术：DuckDB 加载 → TF-IDF + NMF 主题建模（scikit-learn）
复用 nlp_extractor.py 的 NEW_QUALITY_SEEDS 种子词库（7 大类）。
"""

import re
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from tqdm import tqdm

# 复用 nlp_extractor 的种子词库
from nlp_extractor import NEW_QUALITY_SEEDS, REMOTE_KEYWORDS

sns.set_theme(style="whitegrid", context="paper", font="SimHei")
plt.rcParams["axes.unicode_minus"] = False

# ---- 路径 ----
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

POSTINGS_CSV = Path(__file__).parent / "csv" / "postings.csv" / "postings.csv"
POSTINGS_PARQUET = PROCESSED_DIR / "postings_processed.parquet"
POSTINGS_ENRICHED = PROCESSED_DIR / "postings_enriched.parquet"

# ---- 种子词库扩展（针对 job descriptions） ----
# 在原有 7 大类基础上，加入从 description 中更常出现的上下文词
DESCRIPTION_BOOST = {
    "ai_ml": {"large language model", "llm", "generative ai", "transformer",
              "prompt engineering", "rag", "langchain", "hugging face",
              "stable diffusion", "chatbot", "ai engineer", "machine learning engineer"},
    "cloud_devops": {"ci/cd pipeline", "infrastructure as code", "serverless",
                     "cloud native", "containerization", "orchestration", "helm",
                     "istio", "prometheus", "grafana"},
    "data_analytics": {"data pipeline", "data lake", "snowflake", "databricks",
                        "dbt", "airflow", "spark", "feature engineering",
                        "a/b testing", "experimentation"},
    "renewable_energy": {"carbon neutral", "net zero", "esg reporting", "green building",
                          "smart grid", "solar panel", "wind turbine",
                          "circular economy", "climate tech", "cleantech",
                          "renewable energy system", "energy storage system",
                          "carbon capture", "decarbonization strategy"},
    "biotech_health": {"clinical trial", "fda regulation", "therapeutic area",
                        "medical device", "healthcare ai", "telemedicine",
                        "ehr system", "hipaa compliance", "regulatory affairs"},
    "semiconductor": {"silicon design", "wafer fabrication", "semiconductor manufacturing",
                       "cmos technology", "pcb layout", "rf design", "signal integrity",
                       "power electronics", "electromagnetics", "vlsi design"},
    "fintech": {"regtech", "insurtech", "open banking", "embedded finance",
                "kyc", "aml", "fraud detection", "payment gateway",
                "trading system", "market data"},
}

# 合并种子词库，过滤长度≤2的短词（它们在文本中误匹配率极高）
SHORT_SEED_BLACKLIST = {"ev", "ai", "ml", "gcp", "kyc", "aml", "r", "c", "go", "aws"}
MIN_SEED_LEN = 3

ENRICHED_SEEDS = {}
for category in NEW_QUALITY_SEEDS:
    merged = set()
    for w in NEW_QUALITY_SEEDS[category]:
        if len(w) >= MIN_SEED_LEN and w.lower() not in SHORT_SEED_BLACKLIST:
            merged.add(w)
    if category in DESCRIPTION_BOOST:
        for w in DESCRIPTION_BOOST[category]:
            if len(w) >= MIN_SEED_LEN:
                merged.add(w)
    ENRICHED_SEEDS[category] = merged


# ====================================================================
#  2.1 DuckDB 加载 + 预处理
# ====================================================================

def load_postings_raw() -> pd.DataFrame:
    """DuckDB 读取 postings.csv（516MB），清洗并缓存 Parquet。

    如 Parquet 已存在则直接读取。
    """
    if POSTINGS_PARQUET.exists():
        tqdm.write(f"[跳过] {POSTINGS_PARQUET.name} 已存在，直接读取")
        return pd.read_parquet(POSTINGS_PARQUET)

    tqdm.write(f"[DuckDB] 读取 {POSTINGS_CSV.name}（~516MB）...")

    con = duckdb.connect()
    # 使用 DuckDB 的 CSV 读取能力，自动处理引号和换行
    df = con.execute(f"""
        SELECT
            job_id,
            company_name,
            title,
            description,
            formatted_work_type,
            remote_allowed,
            normalized_salary,
            location,
            work_type,
            currency,
            formatted_experience_level,
            skills_desc,
            listed_time,
            posting_domain,
            company_id,
        FROM read_csv_auto('{POSTINGS_CSV.as_posix()}')
    """).df()
    con.close()

    tqdm.write(f"      原始记录: {len(df):,}")

    # ---- 清洗 ----
    tqdm.write("[清洗] 去重、去空、构建 NLP 文本...")

    # 去重：同公司 + 同 title
    before = len(df)
    df = df.drop_duplicates(subset=["company_name", "title"], keep="first").reset_index(drop=True)
    tqdm.write(f"      去重后: {len(df):,}（丢弃 {before - len(df):,}）")

    # 合并 title + description 为 nlp_text
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["nlp_text"] = (df["title"] + " " + df["description"]).str.strip()

    # 文本清洗：小写 + 去特殊字符 + 压缩空白
    tqdm.write("      文本清洗（向量化）...")
    df["nlp_text_clean"] = (
        df["nlp_text"]
        .str.lower()
        .str.replace(r"[^a-z0-9+#\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 过滤无意义短文本
    before = len(df)
    df = df[df["nlp_text_clean"].str.len() > 20].reset_index(drop=True)
    tqdm.write(f"      过滤短文本后: {len(df):,}（丢弃 {before - len(df):,}）")

    # 文本长度统计
    df["text_len"] = df["nlp_text_clean"].str.len()
    tqdm.write(f"      文本长度: 中位={df['text_len'].median():.0f}  均值={df['text_len'].mean():.0f}")

    # 标准化结构化字段
    df["remote_allowed"] = df["remote_allowed"].fillna(0).astype(int)
    df["normalized_salary"] = pd.to_numeric(df["normalized_salary"], errors="coerce")
    # work_type 标准化
    df["work_type_clean"] = df["work_type"].fillna("UNKNOWN").str.upper().str.strip()
    df["is_fulltime"] = df["work_type_clean"].isin(["FULL_TIME", "FULL-TIME", "FULLTIME"])

    # 从 location 中提取城市/州信息
    df["location"] = df["location"].fillna("Unknown")

    tqdm.write(f"[缓存] {POSTINGS_PARQUET.name}")
    df.to_parquet(POSTINGS_PARQUET, index=False)
    return df


def preprocess_structured_fields(df: pd.DataFrame) -> dict:
    """提取并统计结构化字段，输出摘要。"""
    stats = {}
    stats["total"] = len(df)

    # 远程率
    stats["remote_count"] = df["remote_allowed"].sum()
    stats["remote_pct"] = stats["remote_count"] / stats["total"]

    # 薪资
    has_salary = df["normalized_salary"].notna() & (df["normalized_salary"] > 0)
    stats["has_salary_pct"] = has_salary.mean()
    stats["salary_median"] = df.loc[has_salary, "normalized_salary"].median()
    stats["salary_mean"] = df.loc[has_salary, "normalized_salary"].mean()

    # 工作类型分布
    stats["work_type_dist"] = df["work_type_clean"].value_counts().to_dict()
    stats["fulltime_pct"] = df["is_fulltime"].mean()

    # 经验水平
    if "formatted_experience_level" in df.columns:
        stats["exp_level_dist"] = df["formatted_experience_level"].value_counts().to_dict()

    tqdm.write(f"\n  结构化字段摘要:")
    tqdm.write(f"    总岗位:        {stats['total']:,}")
    tqdm.write(f"    远程率:        {stats['remote_pct']:.1%} ({stats['remote_count']:,})")
    tqdm.write(f"    薪资中位数:    ${stats['salary_median']:,.0f}")
    tqdm.write(f"    全职占比:      {stats['fulltime_pct']:.1%}")
    tqdm.write(f"    有薪资数据:    {stats['has_salary_pct']:.1%}")

    return stats


# ====================================================================
#  2.2 TF-IDF + NMF 主题建模
# ====================================================================

def compute_seed_scores(df: pd.DataFrame) -> pd.DataFrame:
    """策略 B：种子词库半监督打分。

    对每个岗位计算 7 大类的新质技能匹配得分，
    输出每类的命中数和总密度。
    """
    tqdm.write("\n[策略 B] 种子词库半监督打分...")

    result = pd.DataFrame(index=df.index)
    all_seeds = set()
    for seeds in ENRICHED_SEEDS.values():
        all_seeds.update(seeds)

    for category, seeds in tqdm(ENRICHED_SEEDS.items(), desc="  类别", unit="个", ncols=80):
        pattern = "|".join(sorted(seeds, key=len, reverse=True))  # 长词优先
        # 使用 str.count 向量化
        result[f"hits_{category}"] = (
            df["nlp_text_clean"].str.count(pattern, flags=re.IGNORECASE)
        )

    # 总命中数
    hit_cols = [c for c in result.columns if c.startswith("hits_")]
    result["hits_total"] = result[hit_cols].sum(axis=1)

    # 新质密度：命中数 / 文本长度（归一化）
    result["seed_density"] = (
        result["hits_total"] / (df["text_len"] / 1000)  # 每千词命中数
    ).clip(upper=10.0)

    # is_new_quality_seed: 至少在 2 个以上类别有命中
    result["is_new_quality_seed"] = (
        (result[hit_cols] > 0).sum(axis=1) >= 2
    )

    n_nq = result["is_new_quality_seed"].sum()
    tqdm.write(f"  种子词库判定新质: {n_nq:,} ({n_nq/len(df)*100:.1f}%)")
    tqdm.write(f"  命中密度中位:     {result['seed_density'].median():.2f}/千词")

    # 各类命中占比
    for c in hit_cols:
        cat = c.replace("hits_", "")
        n = (result[c] > 0).sum()
        tqdm.write(f"    {cat:20s}: {n:>8,} ({n/len(df)*100:5.1f}%)")

    return result


def run_topic_model(df: pd.DataFrame, n_topics: int = 20,
                    max_features: int = 5000, sample_size: int = 80_000):
    """TF-IDF + NMF 主题建模（策略 A：无监督）。

    由于 postings 约 120K，全量 fit + transform 可行。
    采样 fit 以加速，全量 transform。
    """
    tqdm.write(f"\n[策略 A] TF-IDF + NMF 主题建模 (n_topics={n_topics})...")

    tqdm.write(f"  TF-IDF 向量化 (vocab≤{max_features}, ngram=(1,2))...")
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5,
        max_df=0.7,
        sublinear_tf=True,
    )

    if sample_size and len(df) > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), sample_size, replace=False)
        tqdm.write(f"      采样 {sample_size:,} 条 fit TF-IDF...")
        X_sample = vec.fit_transform(df["nlp_text_clean"].iloc[idx])

        tqdm.write(f"      全量 transform ({len(df):,} 条)...")
        chunk_size = 50_000
        from scipy.sparse import vstack
        chunks = []
        for i in tqdm(range(0, len(df), chunk_size),
                      desc="  tfidf", unit="块", ncols=80):
            chunks.append(vec.transform(df["nlp_text_clean"].iloc[i:i + chunk_size]))
        X = vstack(chunks)
    else:
        X = vec.fit_transform(df["nlp_text_clean"])

    tqdm.write(f"  NMF 分解 ({n_topics} 个主题)...")
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=400)

    if sample_size and len(df) > sample_size:
        nmf.fit(X_sample)
        chunk_size = 50_000
        from scipy.sparse import vstack
        chunks = []
        for i in tqdm(range(0, len(df), chunk_size),
                      desc="  nmf  ", unit="块", ncols=80):
            chunks.append(nmf.transform(X[i:i + chunk_size]))
        W = np.vstack(chunks)
    else:
        W = nmf.fit_transform(X)

    tqdm.write(f"      完成 W{W.shape}  重构误差={nmf.reconstruction_err_:.2f}")
    return W, nmf.components_, vec, nmf


def label_topics(H: np.ndarray, vec: TfidfVectorizer, top_n: int = 15) -> dict:
    """输出每个主题的 Top 关键词，自动标注新质/传统。"""
    names = vec.get_feature_names_out()
    labels = {}

    print(f"\n{'='*75}")
    print(f"  NMF 主题 Top-8 关键词（postings.csv）")
    print(f"{'='*75}")

    for tid in range(H.shape[0]):
        top = H[tid].argsort()[::-1][:top_n]
        words = [names[i] for i in top]
        scores = H[tid][top]

        word_set = set(words)
        overlap = {}
        for g, seeds in ENRICHED_SEEDS.items():
            hit = word_set & seeds
            if hit:
                overlap[g] = hit

        tag = "新质→" + "+".join(overlap.keys()) if overlap else "传统行业"
        labels[tid] = {
            "top_words": words,
            "top_scores": scores.round(4).tolist(),
            "suggested_label": tag,
            "overlap_categories": list(overlap.keys()),
        }

        line = " | ".join(f"{w}({s:.3f})" for w, s in zip(words[:8], scores[:8]))
        print(f"  T{tid:2d} [{tag:35s}] {line}")

    print(f"{'='*75}\n")
    return labels


def classify_postings(W: np.ndarray, labels: dict,
                      seed_scores: pd.DataFrame) -> pd.DataFrame:
    """综合策略 A（NMF 主题）和策略 B（种子词库），分配标签。

    最终 is_new_quality 判定：NMF 新质 OR 种子词库新质（任一满足即可）。
    """
    tqdm.write("\n[分类] 综合双策略判定...")

    primary = W.argmax(axis=1)
    conf = W.max(axis=1)

    # 策略 A：NMF 主题是否为"新质"
    is_nq_nmf = np.array([
        "新质" in labels.get(t, {}).get("suggested_label", "")
        for t in tqdm(primary, desc="  NMF判定", unit="条", ncols=80)
    ])

    n_nmf = is_nq_nmf.sum()
    tqdm.write(f"  策略 A (NMF) 新质: {n_nmf:,} ({n_nmf/len(primary)*100:.1f}%)")

    # 策略 B：种子词库判定
    n_seed = seed_scores["is_new_quality_seed"].sum()
    tqdm.write(f"  策略 B (种子) 新质: {n_seed:,} ({n_seed/len(primary)*100:.1f}%)")

    # 综合判定
    is_nq = is_nq_nmf | seed_scores["is_new_quality_seed"].values
    tqdm.write(f"  综合判定新质:     {is_nq.sum():,} ({is_nq.sum()/len(primary)*100:.1f}%)")
    tqdm.write(f"  仅NMF: {is_nq_nmf.sum() - (is_nq_nmf & seed_scores['is_new_quality_seed'].values).sum():,}")
    tqdm.write(f"  仅种子: {n_seed - (is_nq_nmf & seed_scores['is_new_quality_seed'].values).sum():,}")
    tqdm.write(f"  双命中: {(is_nq_nmf & seed_scores['is_new_quality_seed'].values).sum():,}")

    result = pd.DataFrame({
        "primary_topic": primary,
        "topic_confidence": conf,
        "is_new_quality_nmf": is_nq_nmf,
        "is_new_quality_seed": seed_scores["is_new_quality_seed"],
        "is_new_quality": is_nq,
        "seed_density": seed_scores["seed_density"].values,
    })

    # 附上各类命中数
    for c in seed_scores.columns:
        if c.startswith("hits_"):
            result[c] = seed_scores[c].values

    return result


# ====================================================================
#  2.3 新质 vs 传统行业画像
# ====================================================================

def compare_new_vs_traditional(df: pd.DataFrame) -> pd.DataFrame:
    """对比新质行业 vs 传统行业在远程率、薪资、全职占比等维度的差异。

    Returns
    -------
    DataFrame : 每维度一行，含新质值、传统值、差异、检验结果
    """
    tqdm.write("\n[画像] 新质 vs 传统行业对比...")

    nq = df[df["is_new_quality"]]
    trad = df[~df["is_new_quality"]]

    rows = []

    # ---- 远程率 ----
    remote_nq = nq["remote_allowed"].mean()
    remote_trad = trad["remote_allowed"].mean()
    # χ² 检验
    cont_remote = pd.crosstab(
        df["is_new_quality"], df["remote_allowed"]
    )
    if cont_remote.shape == (2, 2):
        chi2, p_remote, _, _ = scipy_stats.chi2_contingency(cont_remote)
    else:
        chi2, p_remote = float("nan"), float("nan")

    rows.append({
        "维度": "远程率", "新质行业": f"{remote_nq:.1%}",
        "传统行业": f"{remote_trad:.1%}",
        "差异": f"{remote_nq - remote_trad:+.1%}",
        "检验": "χ²", "统计量": f"{chi2:.1f}", "p值": f"{p_remote:.4f}",
        "显著": "***" if p_remote < 0.001 else ("**" if p_remote < 0.01 else ("*" if p_remote < 0.05 else "ns")),
    })

    # ---- 平均薪资 ----
    has_sal = df["normalized_salary"].notna() & (df["normalized_salary"] > 0)
    sal_nq = nq.loc[has_sal[nq.index], "normalized_salary"].dropna()
    sal_trad = trad.loc[has_sal[trad.index], "normalized_salary"].dropna()
    if len(sal_nq) > 1 and len(sal_trad) > 1:
        t_stat, p_sal = scipy_stats.ttest_ind(sal_nq, sal_trad, equal_var=False)
    else:
        t_stat, p_sal = float("nan"), float("nan")

    rows.append({
        "维度": "平均薪资",
        "新质行业": f"${sal_nq.mean():,.0f}",
        "传统行业": f"${sal_trad.mean():,.0f}",
        "差异": f"${sal_nq.mean() - sal_trad.mean():+,.0f}",
        "检验": "t-test", "统计量": f"{t_stat:.2f}", "p值": f"{p_sal:.4f}",
        "显著": "***" if p_sal < 0.001 else ("**" if p_sal < 0.01 else ("*" if p_sal < 0.05 else "ns")),
    })

    # ---- 全职占比 ----
    ft_nq = nq["is_fulltime"].mean()
    ft_trad = trad["is_fulltime"].mean()
    cont_ft = pd.crosstab(df["is_new_quality"], df["is_fulltime"])
    if cont_ft.shape == (2, 2):
        chi2_ft, p_ft, _, _ = scipy_stats.chi2_contingency(cont_ft)
    else:
        chi2_ft, p_ft = float("nan"), float("nan")

    rows.append({
        "维度": "全职占比",
        "新质行业": f"{ft_nq:.1%}",
        "传统行业": f"{ft_trad:.1%}",
        "差异": f"{ft_nq - ft_trad:+.1%}",
        "检验": "χ²", "统计量": f"{chi2_ft:.1f}", "p值": f"{p_ft:.4f}",
        "显著": "***" if p_ft < 0.001 else ("**" if p_ft < 0.01 else ("*" if p_ft < 0.05 else "ns")),
    })

    # ---- 技能集中度（种子命中种类数） ----
    hit_cols = [c for c in df.columns if c.startswith("hits_")]
    if hit_cols:
        nq_diversity = (nq[hit_cols] > 0).sum(axis=1).mean()
        trad_diversity = (trad[hit_cols] > 0).sum(axis=1).mean()
        t_stat_div, p_div = scipy_stats.ttest_ind(
            (nq[hit_cols] > 0).sum(axis=1),
            (trad[hit_cols] > 0).sum(axis=1),
            equal_var=False,
        )
        rows.append({
            "维度": "技能集中度(类数)",
            "新质行业": f"{nq_diversity:.1f}",
            "传统行业": f"{trad_diversity:.1f}",
            "差异": f"{nq_diversity - trad_diversity:+.1f}",
            "检验": "t-test", "统计量": f"{t_stat_div:.2f}", "p值": f"{p_div:.4f}",
            "显著": "***" if p_div < 0.001 else ("**" if p_div < 0.01 else ("*" if p_div < 0.05 else "ns")),
        })

    # ---- 描述文本长度 ----
    nq_len = nq["text_len"].mean()
    trad_len = trad["text_len"].mean()
    t_stat_len, p_len = scipy_stats.ttest_ind(nq["text_len"], trad["text_len"], equal_var=False)
    rows.append({
        "维度": "描述文本长度(字符)",
        "新质行业": f"{nq_len:.0f}",
        "传统行业": f"{trad_len:.0f}",
        "差异": f"{nq_len - trad_len:+.0f}",
        "检验": "t-test", "统计量": f"{t_stat_len:.2f}", "p值": f"{p_len:.4f}",
        "显著": "***" if p_len < 0.001 else ("**" if p_len < 0.01 else ("*" if p_len < 0.05 else "ns")),
    })

    result = pd.DataFrame(rows)

    # 打印
    print(f"\n{'='*85}")
    print(f"  新质行业 vs 传统行业画像对比")
    print(f"  总岗位: {len(df):,}  新质: {len(nq):,} ({len(nq)/len(df)*100:.1f}%)  传统: {len(trad):,} ({len(trad)/len(df)*100:.1f}%)")
    print(f"{'='*85}")
    print(f"  {'维度':20s} {'新质':>10s} {'传统':>10s} {'差异':>8s} {'检验':>8s} {'显著':>6s}")
    print(f"  {'-'*70}")
    for _, r in result.iterrows():
        print(f"  {r['维度']:20s} {r['新质行业']:>10s} {r['传统行业']:>10s} {r['差异']:>8s} {r['统计量']:>8s} {r['显著']:>6s}")
    print(f"{'='*85}\n")

    return result


# ====================================================================
#  2.4 可视化
# ====================================================================

def plot_topic_heatmap(H: np.ndarray, vec: TfidfVectorizer, labels: dict):
    """Fig 8: 主题关键词热力图——20 主题 × top 10 词，按词权着色。"""
    tqdm.write("\n[绘图] Fig 8: 主题关键词热力图...")

    n_topics = H.shape[0]
    top_n_words = 10
    names = vec.get_feature_names_out()

    # 构建矩阵和词标注
    matrix = np.zeros((n_topics, top_n_words))
    word_matrix = np.empty((n_topics, top_n_words), dtype=object)
    y_labels = []
    for tid in range(n_topics):
        top_idx = H[tid].argsort()[::-1][:top_n_words]
        matrix[tid] = H[tid][top_idx]
        word_matrix[tid] = [names[i] for i in top_idx]
        tag = labels.get(tid, {}).get("suggested_label", f"T{tid}")
        y_labels.append(f"T{tid:2d} [{tag[:30]}]")

    # 构建标注文本矩阵（词 + 权重）
    annot_text = np.empty((n_topics, top_n_words), dtype=object)
    for tid in range(n_topics):
        for i in range(top_n_words):
            annot_text[tid, i] = f"{word_matrix[tid, i]}\n{matrix[tid, i]:.3f}"

    fig, ax = plt.subplots(figsize=(18, max(8, n_topics * 0.45)))
    sns.heatmap(
        matrix, cmap="YlOrRd", annot=annot_text,
        fmt="", linewidths=0.5, ax=ax,
        xticklabels=[f"#{i+1}" for i in range(top_n_words)],
        yticklabels=y_labels,
        cbar_kws={"label": "NMF 权重", "shrink": 0.6},
        annot_kws={"fontsize": 6.5, "va": "center", "ha": "center"},
    )
    ax.set_title("postings.csv NMF 主题 Top-10 关键词权重热力图", fontsize=14, fontweight="bold")
    ax.set_ylabel("主题")
    ax.set_xlabel("关键词排名")
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig8_topic_keywords_heatmap.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig8_topic_keywords_heatmap.png")
    plt.close()


def plot_salary_distribution(df: pd.DataFrame):
    """Fig 9: 新质 vs 传统薪资分布——双 KDE + 中位线。"""
    tqdm.write("\n[绘图] Fig 9: 新质 vs 传统薪资分布...")

    has_sal = df["normalized_salary"].notna() & (df["normalized_salary"] > 0)
    nq_sal = df.loc[has_sal & df["is_new_quality"], "normalized_salary"]
    trad_sal = df.loc[has_sal & ~df["is_new_quality"], "normalized_salary"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：完整分布 KDE
    ax = axes[0]
    # 截断极端值
    upper = df.loc[has_sal, "normalized_salary"].quantile(0.99)
    lower = df.loc[has_sal, "normalized_salary"].quantile(0.01)

    sns.kdeplot(nq_sal.clip(lower, upper), ax=ax, color="#d62728", lw=2.5,
                fill=True, alpha=0.2, label=f"新质行业 (n={len(nq_sal):,})")
    sns.kdeplot(trad_sal.clip(lower, upper), ax=ax, color="#1f77b4", lw=2.5,
                fill=True, alpha=0.2, label=f"传统行业 (n={len(trad_sal):,})")

    ax.axvline(nq_sal.median(), color="#d62728", ls="--", lw=1.2)
    ax.axvline(trad_sal.median(), color="#1f77b4", ls="--", lw=1.2)
    ax.annotate(f"中位 ${nq_sal.median():,.0f}",
                (nq_sal.median(), ax.get_ylim()[1] * 0.8),
                color="#d62728", fontsize=9, ha="left")
    ax.annotate(f"中位 ${trad_sal.median():,.0f}",
                (trad_sal.median(), ax.get_ylim()[1] * 0.65),
                color="#1f77b4", fontsize=9, ha="right")

    ax.set_title("薪资分布：新质 vs 传统", fontsize=13, fontweight="bold")
    ax.set_xlabel("标准化年薪 (USD)")
    ax.set_ylabel("密度")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    # 右：分主题薪资箱线
    ax = axes[1]
    df_sal = df[has_sal].copy()
    # 取前 10 个最大的主题
    top_topics = df_sal["primary_topic"].value_counts().head(10).index
    plot_data = df_sal[df_sal["primary_topic"].isin(top_topics)]
    # 主题标签
    topic_order = plot_data.groupby("primary_topic")["normalized_salary"].median().sort_values().index

    bp = sns.boxplot(
        data=plot_data, x="primary_topic", y="normalized_salary",
        order=topic_order, ax=ax, hue="primary_topic",
        palette="coolwarm", showfliers=False,
        linewidth=0.8, legend=False,
    )
    ax.set_title("各主题薪资分布 (Top 10)", fontsize=13, fontweight="bold")
    ax.set_xlabel("主题编号")
    ax.set_ylabel("标准化年薪 (USD)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig9_salary_distribution.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig9_salary_distribution.png")
    plt.close()


def plot_remote_industry_bubble(df: pd.DataFrame):
    """Fig 10: 远程率 × 行业 × 薪资 散点气泡图。

    每个主题一个气泡：x=远程率, y=平均薪资, size=岗位数, color=新质/传统。
    """
    tqdm.write("\n[绘图] Fig 10: 远程率 × 行业 × 薪资 气泡图...")

    has_sal = df["normalized_salary"].notna() & (df["normalized_salary"] > 0)

    topic_stats = df.groupby("primary_topic").agg(
        n_jobs=("job_id", "count"),
        remote_pct=("remote_allowed", "mean"),
        avg_salary=("normalized_salary", lambda x: x[has_sal[x.index]].mean()),
        is_new_quality=("is_new_quality", "mean"),  # 新质占比
        fulltime_pct=("is_fulltime", "mean"),
    ).reset_index()

    topic_stats = topic_stats[topic_stats["n_jobs"] >= 50]  # 过滤小主题

    # 标注主题
    topic_labels = {}
    for tid in topic_stats["primary_topic"]:
        if topic_stats.loc[topic_stats["primary_topic"] == tid, "is_new_quality"].values[0] > 0.5:
            topic_labels[tid] = f"T{tid}[NQ]"
        else:
            topic_labels[tid] = f"T{tid}"

    fig, ax = plt.subplots(figsize=(12, 7))

    # 颜色：新质占比
    scatter = ax.scatter(
        topic_stats["remote_pct"] * 100,
        topic_stats["avg_salary"] / 1000,
        s=topic_stats["n_jobs"] / 10,  # 气泡大小
        c=topic_stats["is_new_quality"] * 100,
        cmap="RdYlBu",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        vmin=0, vmax=100,
    )

    # 标注
    for _, row in topic_stats.iterrows():
        tid = int(row["primary_topic"])
        ax.annotate(
            topic_labels.get(tid, f"T{tid}"),
            (row["remote_pct"] * 100, row["avg_salary"] / 1000),
            fontsize=8, ha="center", va="bottom",
            textcoords="offset points", xytext=(0, 6),
        )

    ax.set_xlabel("远程率 (%)")
    ax.set_ylabel("平均年薪 (千 USD)")
    ax.set_title("远程率 × 薪资 × 岗位规模：postings.csv 主题气泡图", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(scatter, ax=ax, label="新质占比 (%)")
    cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter())

    # 添加平均线
    ax.axvline(topic_stats["remote_pct"].mean() * 100, color="gray", ls="--", alpha=0.4)
    ax.axhline(topic_stats["avg_salary"].mean() / 1000, color="gray", ls="--", alpha=0.4)

    # 图例：气泡大小
    for size, label in [(500, "5K"), (2000, "20K"), (5000, "50K")]:
        ax.scatter([], [], s=size, alpha=0.5, color="gray", edgecolors="black",
                   linewidth=0.5, label=f"{label} 岗位")
    ax.legend(title="气泡大小", loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig10_remote_industry_bubble.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig10_remote_industry_bubble.png")
    plt.close()

    return topic_stats


def plot_topic_distribution(df: pd.DataFrame, labels: dict):
    """补充：主题分布柱状图 + 新质占比 + 远程率。"""
    tqdm.write("\n[绘图] 补充: 主题分布...")

    topic_dist = df.groupby("primary_topic").agg(
        n=("job_id", "count"),
        remote=("remote_allowed", "mean"),
        nq_pct=("is_new_quality", "mean"),
        avg_salary=("normalized_salary", "mean"),
    ).sort_values("n", ascending=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 上：柱状图 + 新质占比标注
    ax = axes[0]
    colors = ["#d62728" if topic_dist.loc[t, "nq_pct"] > 0.5 else "#1f77b4"
              for t in topic_dist.index]
    bars = ax.bar(range(len(topic_dist)), topic_dist["n"], color=colors, edgecolor="white")

    for i, (tid, row) in enumerate(topic_dist.iterrows()):
        label_tag = labels.get(tid, {}).get("suggested_label", "")
        ax.text(i, row["n"] + topic_dist["n"].max() * 0.02,
                f"T{tid}", ha="center", fontsize=8, fontweight="bold")
        ax.text(i, row["n"] + topic_dist["n"].max() * 0.07,
                f"{label_tag[:25]}", ha="center", fontsize=6, color="gray")

    ax.set_title("各主题岗位数分布（红=新质，蓝=传统）", fontsize=13, fontweight="bold")
    ax.set_ylabel("岗位数")
    ax.set_xticks([])

    # 下：远程率 + 新质占比折线
    ax = axes[1]
    ax.bar(range(len(topic_dist)), topic_dist["remote"] * 100,
           color="#ff7f0e", alpha=0.6, label="远程率 (%)")
    ax.plot(range(len(topic_dist)), topic_dist["nq_pct"] * 100,
            "o-", color="#d62728", lw=2, markersize=6, label="新质占比 (%)")
    ax.axhline(topic_dist["remote"].mean() * 100, color="#ff7f0e", ls="--", alpha=0.5)
    ax.set_xticks(range(len(topic_dist)))
    ax.set_xticklabels([f"T{t}" for t in topic_dist.index], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("百分比 (%)")
    ax.set_title("各主题远程率 & 新质占比", fontsize=13, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig8b_topic_distribution.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig8b_topic_distribution.png")
    plt.close()


# ====================================================================
#  5. 一键运行
# ====================================================================

def run_all():
    print("=" * 60)
    print("  postings.csv NLP 管道 (模块二)")
    print("=" * 60)

    # ---- 2.1 加载 + 预处理 ----
    tqdm.write("\n>>> 2.1 DuckDB 加载 + 预处理")
    df = load_postings_raw()
    struct_stats = preprocess_structured_fields(df)

    # ---- 2.2 策略 B：种子词库打分 ----
    tqdm.write("\n>>> 2.2 策略 B: 种子词库半监督打分")
    seed_scores = compute_seed_scores(df)

    # ---- 2.2 策略 A：NMF 主题建模 ----
    tqdm.write("\n>>> 2.2 策略 A: TF-IDF + NMF 主题建模")
    W, H, vec, nmf = run_topic_model(df, n_topics=20, max_features=5000, sample_size=80_000)
    topic_labels = label_topics(H, vec)

    # ---- 综合分类 ----
    topic_df = classify_postings(W, topic_labels, seed_scores)
    df = pd.concat([df.reset_index(drop=True), topic_df.reset_index(drop=True)], axis=1)

    # ---- 2.3 画像对比 ----
    tqdm.write("\n>>> 2.3 新质 vs 传统行业画像")
    comparison = compare_new_vs_traditional(df)

    # ---- 2.4 可视化 ----
    tqdm.write("\n>>> 2.4 产出图表")
    plot_topic_heatmap(H, vec, topic_labels)
    plot_topic_distribution(df, topic_labels)
    plot_salary_distribution(df)
    topic_bubble = plot_remote_industry_bubble(df)

    # ---- 保存 ----
    tqdm.write("\n[保存] 输出 enriched Parquet...")
    # 选择关键列保存（去掉过长的原始文本以减小体积）
    save_cols = [
        "job_id", "company_name", "title", "location",
        "remote_allowed", "normalized_salary", "work_type_clean",
        "is_fulltime", "formatted_experience_level",
        "text_len", "primary_topic", "topic_confidence",
        "is_new_quality_nmf", "is_new_quality_seed", "is_new_quality",
        "seed_density",
    ]
    # 附上命中列
    hit_cols = [c for c in df.columns if c.startswith("hits_")]
    save_cols += hit_cols
    save_cols = [c for c in save_cols if c in df.columns]

    df_out = df[save_cols]
    df_out.to_parquet(POSTINGS_ENRICHED, index=False)
    tqdm.write(f"[完成] {POSTINGS_ENRICHED}")

    # ---- 最终摘要 ----
    n_nq = df["is_new_quality"].sum()
    print(f"\n{'='*60}")
    print(f"  模块二完成")
    print(f"{'='*60}")
    print(f"  总岗位:      {len(df):,}")
    print(f"  新质行业:    {n_nq:,} ({n_nq/len(df)*100:.1f}%)")
    print(f"  主题数:      {H.shape[0]}")
    print(f"  新质远程率:  {df[df['is_new_quality']]['remote_allowed'].mean():.1%}")
    print(f"  传统远程率:  {df[~df['is_new_quality']]['remote_allowed'].mean():.1%}")
    nq_sal = df.loc[df["is_new_quality"] & df["normalized_salary"].notna(), "normalized_salary"]
    trad_sal = df.loc[~df["is_new_quality"] & df["normalized_salary"].notna(), "normalized_salary"]
    if len(nq_sal) > 0 and len(trad_sal) > 0:
        print(f"  新质薪资中位: ${nq_sal.median():,.0f}")
        print(f"  传统薪资中位: ${trad_sal.median():,.0f}")
        print(f"  薪资溢价:     ${nq_sal.median() - trad_sal.median():+,.0f}")
    print(f"  图表已保存至: {PLOTS_DIR}/")
    print(f"{'='*60}")

    return df, topic_labels, comparison, struct_stats


if __name__ == "__main__":
    df, topic_labels, comparison, struct_stats = run_all()
