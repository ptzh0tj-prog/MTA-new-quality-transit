"""
LinkedIn NLP 提取模块
=====================
从 job_skills 文本中自动发现行业结构、识别"新质生产力"岗位。

核心技术：TF-IDF + NMF 主题建模（scikit-learn）
输出：岗位级标签 + 技能聚类 + 行业画像
"""

import re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# ---- 路径 ----
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)
JOBS_PARQUET = PROCESSED_DIR / "linkedin_processed.parquet"
SKILL_COOC_PARQUET = PROCESSED_DIR / "skill_cooccurrence.parquet"

# ---- 种子词库 ----
NEW_QUALITY_SEEDS = {
    "ai_ml": {"artificial intelligence", "machine learning", "deep learning",
              "neural network", "nlp", "computer vision", "data science",
              "data engineering", "mlops", "predictive modeling"},
    "cloud_devops": {"aws", "azure", "gcp", "cloud", "docker", "kubernetes",
                     "devops", "ci/cd", "terraform", "microservices"},
    "data_analytics": {"data analysis", "data mining", "big data", "sql",
                       "etl", "data warehouse", "business intelligence",
                       "tableau", "power bi", "looker"},
    "renewable_energy": {"solar", "wind energy", "renewable energy",
                         "energy storage", "electric vehicle", "ev",
                         "battery", "sustainability", "carbon", "clean energy"},
    "biotech_health": {"biotechnology", "bioinformatics", "genomics",
                       "clinical research", "drug discovery", "molecular biology",
                       "crispr", "precision medicine", "immunology", "pharmaceutical"},
    "semiconductor": {"semiconductor", "chip design", "vlsi", "asic",
                      "fpga", "embedded systems", "verilog", "rtl",
                      "pcb design", "systemverilog"},
    "fintech": {"blockchain", "cryptocurrency", "defi", "fintech",
                "quantitative finance", "algorithmic trading", "risk modeling",
                "payment systems", "digital banking"},
}

REMOTE_KEYWORDS = {
    "remote", "work from home", "wfh", "hybrid", "hybrid work",
    "flexible schedule", "flexible hours", "telecommute", "virtual",
    "distributed team", "remote-first", "remote first",
}


def load_and_preprocess() -> pd.DataFrame:
    """读取 CSV → 清洗 → 缓存 Parquet。"""
    if JOBS_PARQUET.exists():
        tqdm.write(f"[跳过] {JOBS_PARQUET.name} 已存在，直接读取")
        return pd.read_parquet(JOBS_PARQUET)

    csv_path = Path(__file__).parent / "csv" / "job_skills.csv"
    tqdm.write("[1/4] 读取 CSV（642MB）...")
    df = pd.read_csv(csv_path)
    tqdm.write(f"      原始记录: {len(df):,}")

    tqdm.write("[2/4] 文本清洗（向量化，几秒完成）...")
    df["text_clean"] = (
        df["job_skills"].fillna("")
        .str.lower()
        .str.replace(r"[^a-z0-9+#\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["job_title_raw"] = df["job_link"].str.extract(r"jobs/view/(.+?)-at-", expand=False)
    df["job_title_clean"] = df["job_title_raw"].fillna("").str.replace("-", " ").str.lower().str.strip()

    tqdm.write("[3/4] 拆分技能列表...")
    df["skills_list"] = df["text_clean"].str.split(r"\s*,\s*")
    # 用普通循环 + tqdm 统计技能数
    skill_counts = []
    for skills in tqdm(df["skills_list"], desc="  技能计数", unit="条", ncols=80):
        skill_counts.append(len([s for s in skills if s.strip()]))
    df["skill_count"] = skill_counts

    before = len(df)
    df = df[df["text_clean"].str.len() > 5].reset_index(drop=True)
    tqdm.write(f"      清洗后: {len(df):,}（丢弃 {before - len(df):,} 空记录）")

    tqdm.write("[4/4] 缓存 Parquet...")
    df.to_parquet(JOBS_PARQUET, index=False)
    return df


def extract_remote_flag(df: pd.DataFrame) -> pd.Series:
    """检测远程/混合办公岗位。"""
    tqdm.write("[NLP] 检测远程办公...")
    pattern = "|".join(REMOTE_KEYWORDS)
    result = df["text_clean"].str.contains(pattern, regex=True).fillna(False)
    tqdm.write(f"      远程岗位: {result.sum():,} ({result.sum() / len(df) * 100:.1f}%)")
    return result


def compute_tech_density(df: pd.DataFrame) -> np.ndarray:
    """向量化计算新质技能密度（0-1），秒级完成。

    Note: 不做逐行去重（性能优先），用 str.count 向量化。
          精确分类依赖后续 TF-IDF + NMF 主题模型。
    """
    tqdm.write("[NLP] 计算新质技能密度（向量化）...")
    all_seeds = set()
    for g in NEW_QUALITY_SEEDS.values():
        all_seeds.update(g)
    pattern = "|".join(all_seeds)

    hit_counts = df["text_clean"].str.count(pattern, flags=re.IGNORECASE)
    densities = (hit_counts / df["skill_count"].replace(0, 1)).clip(upper=1.0).values.astype(np.float32)

    tqdm.write(f"      均值={densities.mean():.4f}  中位数={np.median(densities):.4f}")
    return densities


# ====================================================================
#  TF-IDF + NMF
# ====================================================================

def run_topic_model(df: pd.DataFrame, n_topics: int = 20,
                    max_features: int = 5000, sample_size: int = 200_000):
    """TF-IDF + NMF 主题建模。大样本时采样 fit，全量 transform。"""
    tqdm.write(f"[NLP] TF-IDF 向量化 (vocab≤{max_features})...")
    vec = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2),
        stop_words="english", min_df=5, max_df=0.7, sublinear_tf=True,
    )

    if sample_size and len(df) > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), sample_size, replace=False)
        tqdm.write(f"      采样 {sample_size:,} 条 fit...")
        # sklearn 不支持 tqdm，直接跑（采样量小，很快）
        X_sample = vec.fit_transform(df["text_clean"].iloc[idx])
        tqdm.write(f"      全量 transform ({len(df):,} 条)...")
        # 分块 transform + 拼接，每块报告进度
        chunk_size = 100_000
        n_chunks = (len(df) + chunk_size - 1) // chunk_size
        chunks = []
        for i in tqdm(range(0, len(df), chunk_size), total=n_chunks,
                      desc="  tfidf", unit="块", ncols=80):
            chunks.append(vec.transform(df["text_clean"].iloc[i:i + chunk_size]))
        from scipy.sparse import vstack
        X = vstack(chunks)
    else:
        X = vec.fit_transform(df["text_clean"])

    tqdm.write(f"[NLP] NMF 分解 ({n_topics} 个主题)...")
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=400)
    if sample_size and len(df) > sample_size:
        nmf.fit(X_sample)
        # 分块 transform
        chunks = []
        for i in tqdm(range(0, len(df), chunk_size), total=n_chunks,
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

    print(f"\n{'='*65}")
    print("  NMF 主题 Top-8 关键词")
    print(f"{'='*65}")

    for tid in range(H.shape[0]):
        top = H[tid].argsort()[::-1][:top_n]
        words = [names[i] for i in top]
        scores = H[tid][top]

        word_set = set(words)
        overlap = {}
        for g, seeds in NEW_QUALITY_SEEDS.items():
            hit = word_set & seeds
            if hit:
                overlap[g] = hit

        tag = "新质→" + "+".join(overlap) if overlap else "传统行业"
        labels[tid] = {"top_words": words, "top_scores": scores.round(4).tolist(),
                        "suggested_label": tag}

        line = " | ".join(f"{w}({s:.3f})" for w, s in zip(words[:8], scores[:8]))
        print(f"  T{tid:2d} [{tag:30s}] {line}")

    print(f"{'='*65}\n")
    return labels


def classify_jobs_by_topic(W: np.ndarray, labels: dict) -> pd.DataFrame:
    """分配主 topic + 新质/传统标签。"""
    tqdm.write("[NLP] 岗位主题分类...")
    primary = W.argmax(axis=1)
    conf = W.max(axis=1)

    is_nq = np.empty(len(primary), dtype=bool)
    for i, t in enumerate(
        tqdm(primary, desc="  分类", unit="条", ncols=80)
    ):
        is_nq[i] = "新质" in labels.get(t, {}).get("suggested_label", "")

    tqdm.write(f"      新质岗位: {is_nq.sum():,} ({is_nq.sum() / len(is_nq) * 100:.1f}%)")
    return pd.DataFrame({"primary_topic": primary, "topic_confidence": conf,
                         "is_new_quality_nlp": is_nq})


# ====================================================================
#  技能共现
# ====================================================================

def build_skill_cooccurrence(df: pd.DataFrame, top_n: int = 100,
                             sample_size: int = 100_000) -> pd.DataFrame:
    """Top-N 技能共现矩阵。"""
    tqdm.write(f"[NLP] 技能共现 (top {top_n})...")
    all_skills = df["skills_list"].explode().str.strip()
    top_set = set(all_skills.value_counts().head(top_n).index)

    if sample_size and len(df) > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), sample_size, replace=False)
        sample = df["skills_list"].iloc[idx]
    else:
        sample = df["skills_list"]

    cooc = Counter()
    for skills in tqdm(sample, desc="  共现", unit="条", ncols=80):
        f = [s.strip() for s in skills if s.strip() in top_set]
        for i, a in enumerate(f):
            for b in f[i + 1:]:
                cooc[(a, b) if a < b else (b, a)] += 1

    rows = [{"skill_a": k[0], "skill_b": k[1], "weight": v}
            for k, v in cooc.most_common(5000)]
    result = pd.DataFrame(rows)
    tqdm.write(f"      共现边: {len(result):,}")
    result.to_parquet(SKILL_COOC_PARQUET, index=False)
    return result


# ====================================================================
#  一键运行
# ====================================================================

def run_all():
    print("=" * 55)
    print("  LinkedIn NLP 提取  (1.3M 岗位)")
    print("=" * 55)

    steps = tqdm(range(6), desc="总体", unit="步", ncols=80)

    steps.set_postfix_str("加载+清洗")
    df = load_and_preprocess()
    steps.update(1)

    steps.set_postfix_str("远程检测")
    df["is_remote"] = extract_remote_flag(df)
    steps.update(1)

    steps.set_postfix_str("技能密度")
    df["tech_density"] = compute_tech_density(df)
    steps.update(1)

    steps.set_postfix_str("主题建模")
    W, H, vec, nmf = run_topic_model(df)
    topic_labels = label_topics(H, vec)
    steps.update(1)

    steps.set_postfix_str("岗位分类")
    topic_df = classify_jobs_by_topic(W, topic_labels)
    df = pd.concat([df.reset_index(drop=True), topic_df.reset_index(drop=True)], axis=1)
    steps.update(1)

    steps.set_postfix_str("技能共现")
    cooc = build_skill_cooccurrence(df)
    steps.update(1)
    steps.close()

    # ---- 摘要 ----
    n_new = df["is_new_quality_nlp"].sum()
    n_remote = df["is_remote"].sum()
    print(f"\n{'='*55}")
    print(f"  总: {len(df):,} | 新质: {n_new:,} ({n_new/len(df)*100:.1f}%)")
    print(f"  远程: {n_remote:,} ({n_remote/len(df)*100:.1f}%)")
    print(f"  tech密度均值: {df['tech_density'].mean():.4f}")

    # 主题分布
    print(f"\n  主题分布:")
    dist = df.groupby("primary_topic").agg(
        n=("job_link", "count"), remote=("is_remote", "mean")
    ).sort_values("n", ascending=False)
    for tid, row in dist.head(10).iterrows():
        label = topic_labels.get(tid, {}).get("suggested_label", "?")
        bar = "█" * int(row["n"] / dist["n"].max() * 25)
        print(f"  T{tid:2d} [{label:30s}] {int(row['n']):>8,} {bar}")

    out = PROCESSED_DIR / "linkedin_enriched.parquet"
    # 去掉大列，只保留后续分析需要的
    drop_cols = ["skills_list", "text_clean", "skill_count"]
    df_out = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df_out.to_parquet(out, index=False)
    print(f"\n[完成] {out}")

    return df, topic_labels, cooc


if __name__ == "__main__":
    run_all()
