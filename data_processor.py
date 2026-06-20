"""
MTA 数据处理模块
===============
核心思路：大文件入口走 DuckDB（聚合 → 导出 Parquet），后续走 Pandas。
"""

from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---- 路径配置 ----
DATA_DIR = Path(__file__).parent / "csv"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

DAILY_CSV = DATA_DIR / "MTA_Daily_Ridership_and_Traffic__Beginning_2020.csv"
HOURLY_CSV = DATA_DIR / "MTA_Subway_Hourly_Ridership.csv"
JOBS_CSV = DATA_DIR / "job_skills.csv"

DAILY_PARQUET = PROCESSED_DIR / "mta_daily.parquet"
HOURLY_PARQUET = PROCESSED_DIR / "mta_hourly_agg.parquet"
STATION_PARQUET = PROCESSED_DIR / "mta_station_features.parquet"


# ============================================================
#  1. MTA 每日运量处理（DuckDB 聚合 → Pandas 特征）
# ============================================================

def load_daily_raw() -> pd.DataFrame:
    """DuckDB 读取原始每日 CSV，按日期 + 运输方式聚合，导出 Parquet。

    如 Parquet 已存在则直接读取（避免重复跑 DuckDB）。
    """
    if DAILY_PARQUET.exists():
        print(f"[跳过] {DAILY_PARQUET.name} 已存在，直接读取")
        return pd.read_parquet(DAILY_PARQUET)

    print(f"[DuckDB] 聚合 {DAILY_CSV.name} ...")
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            CAST("Date" AS DATE) AS date,
            "Mode" AS mode,
            SUM("Count") AS daily_ridership
        FROM read_csv('{DAILY_CSV.as_posix()}')
        GROUP BY date, mode
        ORDER BY date, mode
    """).df()
    con.close()

    df.to_parquet(DAILY_PARQUET, index=False)
    print(f"[完成] 导出 {DAILY_PARQUET.name}，{len(df):,} 行")
    return df


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """在聚合后的每日数据上构建时间特征和恢复率。

    Parameters
    ----------
    df : DataFrame，含 date, mode, daily_ridership

    Returns
    -------
    DataFrame，新增 year, month, dayofweek, recovery_rate 等
    """
    print("[Pandas] 构建每日特征 ...")
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"] >= 5

    # 以 2020 年 1-2 月（疫情前）为基线
    baseline = (
        df[(df["year"] == 2020) & (df["month"].isin([1, 2]))]
        .groupby("mode")["daily_ridership"]
        .mean()
    )

    # 按模式计算月度恢复率
    monthly = (
        df.groupby(["year", "month", "mode"])["daily_ridership"]
        .mean()
        .reset_index()
    )
    monthly["baseline"] = monthly["mode"].map(baseline)
    monthly["recovery_rate"] = monthly["daily_ridership"] / monthly["baseline"]

    # 按模式汇总年度恢复率
    yearly_summary = (
        monthly.groupby(["year", "mode"])
        .agg(avg_recovery=("recovery_rate", "mean"),
             min_recovery=("recovery_rate", "min"),
             max_recovery=("recovery_rate", "max"))
        .reset_index()
    )

    print(f"[完成] 月度数据 {len(monthly):,} 行，年度汇总 {len(yearly_summary):,} 行")
    return monthly


def get_mode_recovery_curve(df: pd.DataFrame, target_mode: str = "subway") -> pd.DataFrame:
    """提取单个运输方式的月度恢复曲线，方便直接画图。

    Parameters
    ----------
    df : build_daily_features 的返回结果（月度数据）
    target_mode : 运输方式名称，如 'subway', 'bus', 'LIRR' 等
    """
    sub = df[df["mode"].str.lower() == target_mode.lower()].copy()
    sub["year_month"] = sub["year"].astype(str) + "-" + sub["month"].astype(str).str.zfill(2)
    return sub.sort_values(["year", "month"])


# ============================================================
#  2. MTA 地铁每小时刷卡处理（~8GB，DuckDB 聚合）
# ============================================================

def load_hourly_agg() -> pd.DataFrame:
    """DuckDB 按小时 + 站点聚合 8GB 刷卡 CSV，导出 Parquet。

    聚合后约 100-300 万行（视站点数 × 小时数），远小于原始文件。
    """
    if HOURLY_PARQUET.exists():
        print(f"[跳过] {HOURLY_PARQUET.name} 已存在，直接读取")
        return pd.read_parquet(HOURLY_PARQUET)

    print(f"[DuckDB] 聚合 {HOURLY_CSV.name}（~8GB，请耐心等待）...")
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            strptime("transit_timestamp", '%m/%d/%Y %I:%M:%S %p') AS timestamp,
            "station_complex_id" AS station_id,
            "station_complex" AS station_name,
            "borough" AS borough,
            SUM("ridership"::INTEGER) AS entries,
            SUM("transfers"::INTEGER) AS transfers
        FROM read_csv('{HOURLY_CSV.as_posix()}',
                       header=true,
                       ignore_errors=true,
                       all_varchar=true)
        GROUP BY timestamp, station_id, station_name, borough
        ORDER BY timestamp, station_id
    """).df()
    con.close()

    df.to_parquet(HOURLY_PARQUET, index=False)
    print(f"[完成] 导出 {HOURLY_PARQUET.name}，{len(df):,} 行")
    return df


def build_station_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """从小时级数据中提取站点级时空特征。

    输出每个站点的：
    - commute_ratio：早晚高峰比，衡量该站是否以通勤为主
    - peak_morning：早高峰(7-9点)日均进站量
    - peak_evening：晚高峰(17-19点)日均出站量
    - recovery_2024_vs_2022：2024 年相对 2022 年的恢复率
    - entries_cv：进站量变异系数，衡量客流波动性
    """
    if STATION_PARQUET.exists():
        print(f"[跳过] {STATION_PARQUET.name} 已存在，直接读取")
        return pd.read_parquet(STATION_PARQUET)

    print("[Pandas] 构建站点特征 ...")
    df = hourly.copy()
    df["hour_int"] = df["timestamp"].dt.hour
    df["year"] = df["timestamp"].dt.year

    # 早晚高峰比
    mask_morning = df["hour_int"].isin([7, 8, 9])
    mask_evening = df["hour_int"].isin([17, 18, 19])

    morning_entries = (
        df[mask_morning].groupby(["station_id", "station_name", "borough"])["entries"]
        .mean().reset_index().rename(columns={"entries": "peak_morning"})
    )
    evening_exits = (
        df[mask_evening].groupby(["station_id", "station_name", "borough"])["entries"]
        .mean().reset_index().rename(columns={"entries": "peak_evening"})  # entries 作为出站代理
    )

    stations = morning_entries.merge(
        evening_exits, on=["station_id", "station_name", "borough"], how="outer"
    )
    stations["commute_ratio"] = stations["peak_morning"] / stations["peak_evening"].replace(0, np.nan)

    # 2024 vs 2022 恢复率
    yearly = (
        df.groupby(["station_id", "year"])["entries"].mean().reset_index()
    )
    pivot = yearly.pivot(index="station_id", columns="year", values="entries")
    if 2022 in pivot.columns and 2024 in pivot.columns:
        stations = stations.merge(
            (pivot[2024] / pivot[2022].replace(0, np.nan))
            .rename("recovery_2024_vs_2022")
            .reset_index(),
            on="station_id", how="left"
        )

    # 进站量变异系数（日均波动 / 日均均值）
    cv = (
        df.groupby(["station_id"])["entries"]
        .agg(["mean", "std"]).reset_index()
    )
    cv["entries_cv"] = cv["std"] / cv["mean"].replace(0, np.nan)
    stations = stations.merge(
        cv[["station_id", "entries_cv"]], on="station_id", how="left"
    )

    # 总日均进站
    total_entries = (
        df.groupby(["station_id", "station_name", "borough"])["entries"]
        .mean().reset_index().rename(columns={"entries": "avg_daily_entries"})
    )
    stations = stations.merge(total_entries, on=["station_id", "station_name", "borough"], how="left")

    print(f"[完成] {len(stations):,} 个站点特征已构建")
    stations.to_parquet(STATION_PARQUET, index=False)
    return stations


# ============================================================
#  3. LinkedIn 数据（文本挖掘入口）
# ============================================================

def load_jobs() -> pd.DataFrame:
    """读取 LinkedIn 岗位数据。"""
    print(f"[Pandas] 读取 {JOBS_CSV.name} ...")
    df = pd.read_csv(JOBS_CSV)
    print(f"[完成] {len(df):,} 条岗位记录")
    return df


def preprocess_jobs_text(df: pd.DataFrame) -> pd.DataFrame:
    """清洗 LinkedIn 技能文本，为 NLP 做准备。

    将 job_skills 列处理为：
    - text_clean: 小写 + 去特殊字符
    - skills_list: 拆分为技能列表
    """
    print("[Pandas] 文本预处理 ...")
    df = df.copy()
    df["text_clean"] = (
        df["job_skills"]
        .str.lower()
        .str.replace(r"[^a-z0-9+#,\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["skills_list"] = df["text_clean"].str.split(r"\s*,\s*")
    return df


# ============================================================
#  4. 一键运行入口
# ============================================================

PROCESSING_STEPS = [
    ("每日 MTA 运量", load_daily_raw),
    ("地铁刷卡聚合", load_hourly_agg),
]


def run_all():
    """从原始 CSV 到中间特征，一键执行全部处理。

    各步结果缓存为 Parquet，重复运行自动跳过已完成步骤。
    """
    print("=" * 50)
    print("MTA 数据处理流水线")
    print("=" * 50)

    for name, func in tqdm(PROCESSING_STEPS, desc="总体进度"):
        print(f"\n--- {name} ---")
        func()

    print("\n" + "=" * 50)
    print("全部处理完成。中间结果已缓存至 processed/ 目录。")
    print("=" * 50)


if __name__ == "__main__":
    run_all()
