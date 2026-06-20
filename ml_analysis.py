"""
机器学习验证模块（模块四）
========================
用 XGBoost + SHAP 量化各特征对站点恢复率的贡献，
并与 RandomForest / LinearRegression 基准对比。
可选：SARIMAX vs XGBoost 短时客流预测对比。

子模块：
  4.1 特征矩阵构造 —— 站点级 X/y
  4.2 XGBoost + SHAP —— 训练 / SHAP 可解释性 / 基准对比
  4.3 SARIMAX vs XGBoost —— 短时客流预测方法对比（可选）
  4.4 图表 —— fig15–fig18

依赖：模块一（timeseries_analysis）的站点特征
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sns.set_theme(style="whitegrid", context="paper", font="SimHei")
plt.rcParams["axes.unicode_minus"] = False

# ---- 路径 ----
PROCESSED_DIR = Path(__file__).parent / "processed"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

STATION_PARQUET = PROCESSED_DIR / "mta_station_features.parquet"
HOURLY_PARQUET = PROCESSED_DIR / "mta_hourly_agg.parquet"

# ---- 常数 ----
RANDOM_STATE = 42
N_SPLITS = 5

BOROUGH_NAMES = {
    "Manhattan": "曼哈顿", "Brooklyn": "布鲁克林", "Queens": "皇后区",
    "Bronx": "布朗克斯", "Staten Island": "史泰登岛",
}


# ====================================================================
#  4.1 特征矩阵构造
# ====================================================================

def compute_yoy_change(hourly: pd.DataFrame) -> pd.DataFrame:
    """从小时数据逐年计算每个站点的年度变化率。

    Returns
    -------
    DataFrame : station_id, yoy_2023_vs_2022, yoy_2024_vs_2023
    """
    df = hourly.copy()
    df["year"] = df["timestamp"].dt.year

    yearly = (
        df.groupby(["station_id", "year"])["entries"]
        .mean().reset_index()
    )
    pivot = yearly.pivot(index="station_id", columns="year", values="entries")

    result = pd.DataFrame(index=pivot.index)
    result.index.name = "station_id"

    if 2022 in pivot.columns and 2023 in pivot.columns:
        result["yoy_2023_vs_2022"] = (
            (pivot[2023] - pivot[2022]) / pivot[2022].replace(0, np.nan)
        )
    if 2023 in pivot.columns and 2024 in pivot.columns:
        result["yoy_2024_vs_2023"] = (
            (pivot[2024] - pivot[2023]) / pivot[2023].replace(0, np.nan)
        )

    return result.reset_index()


def build_feature_matrix(stations: pd.DataFrame,
                         hourly: pd.DataFrame | None = None
                         ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """构造站点级特征矩阵 X 和标签 y。

    Parameters
    ----------
    stations : 站点特征（含 commute_ratio, entries_cv 等）
    hourly : 小时聚合数据（可选，用于计算 YoY 变化）

    Returns
    -------
    X : 特征矩阵（含 one-hot borough）
    y : 恢复率标签
    feature_info : 特征信息表（名称 / 类型 / 缺失率）
    """
    tqdm.write("\n[4.1] 构造站点级特征矩阵...")

    df = stations.copy()

    # ---- 基础数值特征 ----
    numeric_features = [
        "commute_ratio",
        "entries_cv",
        "peak_morning",
        "peak_evening",
        "avg_daily_entries",
    ]
    # 确保全部存在
    for c in numeric_features:
        if c not in df.columns:
            tqdm.write(f"  [警告] 缺少列 {c}，用 NaN 填充")
            df[c] = np.nan

    # ---- year_over_year_change ----
    if hourly is not None:
        yoy = compute_yoy_change(hourly)
        df = df.merge(yoy, on="station_id", how="left")
        if "yoy_2023_vs_2022" in df.columns:
            numeric_features.append("yoy_2023_vs_2022")
        if "yoy_2024_vs_2023" in df.columns:
            numeric_features.append("yoy_2024_vs_2023")
    else:
        tqdm.write("  [跳过] YoY 变化（无小时数据）")

    # ---- borough one-hot ----
    if "borough" in df.columns:
        df["borough"] = df["borough"].fillna("Unknown")
        borough_dummies = pd.get_dummies(df["borough"], prefix="borough", drop_first=True)
        borough_dummies = borough_dummies.astype(float)  # SHAP 需要 float
        df = pd.concat([df, borough_dummies], axis=1)
        borough_cols = list(borough_dummies.columns)
    else:
        borough_cols = []

    # ---- 标签 ----
    y = df["recovery_2024_vs_2022"].copy()

    # ---- 组装 X ----
    feature_cols = numeric_features + borough_cols
    # 只保留存在的列
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()

    # ---- 缺失处理 ----
    missing_before = X.isna().sum().sum()
    X = X.fillna(X.median())
    missing_after = X.isna().sum().sum()
    if missing_before > 0:
        tqdm.write(f"  缺失值: {missing_before} → 中位数填充后 {missing_after}")

    # 对齐 X 和 y（去掉 y 缺失的行）
    valid = y.notna() & X.notna().all(axis=1)
    X = X.loc[valid]
    y = y.loc[valid]

    # ---- 特征信息表 ----
    feature_info_rows = []
    for col in X.columns:
        ftype = "numeric"
        if col.startswith("borough_"):
            ftype = "one-hot (borough)"
        elif col.startswith("yoy_"):
            ftype = "numeric (YoY)"
        feature_info_rows.append({
            "feature": col,
            "type": ftype,
            "missing": int(df[col].isna().sum()) if col in df.columns else 0,
            "mean": X[col].mean(),
            "std": X[col].std(),
        })
    feature_info = pd.DataFrame(feature_info_rows)

    tqdm.write(f"  X: {X.shape[0]} 站点 × {X.shape[1]} 特征")
    tqdm.write(f"  y: recovery_2024_vs_2022, 均值={y.mean():.3f}, std={y.std():.3f}")
    tqdm.write(f"  特征列表: {feature_cols}")

    return X, y, feature_info


# ====================================================================
#  4.2 XGBoost + SHAP + 基准对比
# ====================================================================

def train_xgboost(X: pd.DataFrame, y: pd.Series) -> dict:
    """训练 XGBoost 回归模型，返回模型和交叉验证指标。

    Returns
    -------
    dict : model, cv_rmse, cv_mae, cv_r2, y_test, y_pred, test_rmse, ...
    """
    import xgboost as xgb

    tqdm.write("\n[4.2] 训练 XGBoost...")

    # 训练/测试拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # XGBoost 模型
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    # 交叉验证
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_rmse = np.sqrt(-cross_val_score(
        model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error"
    ))
    cv_mae = -cross_val_score(
        model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error"
    )
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

    tqdm.write(f"  XGBoost CV (k={N_SPLITS}):")
    tqdm.write(f"    RMSE = {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    tqdm.write(f"    MAE  = {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
    tqdm.write(f"    R^2   = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # 全训练集 fit + 测试集评估
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    tqdm.write(f"  XGBoost 测试集:")
    tqdm.write(f"    RMSE = {test_rmse:.4f}")
    tqdm.write(f"    MAE  = {test_mae:.4f}")
    tqdm.write(f"    R^2   = {test_r2:.4f}")

    return {
        "model": model,
        "model_type": "XGBoost",
        "cv_rmse": cv_rmse, "cv_mae": cv_mae, "cv_r2": cv_r2,
        "test_rmse": test_rmse, "test_mae": test_mae, "test_r2": test_r2,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred,
    }


def train_baselines(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """训练 RandomForest 和 LinearRegression/Ridge 作为基准模型。

    Returns
    -------
    list[dict] : 每个模型的结果字典
    """
    tqdm.write("\n[4.2] 训练基准模型...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    # ---- RandomForest ----
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_cv_rmse = np.sqrt(-cross_val_score(
        rf, X_train, y_train, cv=kf, scoring="neg_mean_squared_error"
    ))
    rf_cv_r2 = cross_val_score(rf, X_train, y_train, cv=kf, scoring="r2")
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    rf_result = {
        "model": rf,
        "model_type": "RandomForest",
        "cv_rmse": rf_cv_rmse,
        "cv_mae": -cross_val_score(rf, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error"),
        "cv_r2": rf_cv_r2,
        "test_rmse": np.sqrt(mean_squared_error(y_test, rf_pred)),
        "test_mae": mean_absolute_error(y_test, rf_pred),
        "test_r2": r2_score(y_test, rf_pred),
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": rf_pred,
    }
    results.append(rf_result)
    tqdm.write(f"  RandomForest CV R^2={rf_cv_r2.mean():.4f} ± {rf_cv_r2.std():.4f}, "
               f"Test R^2={rf_result['test_r2']:.4f}")

    # ---- Ridge 回归（带标准化） ----
    pipe = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    # 手动标准化用于线性模型
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge_cv_rmse = np.sqrt(-cross_val_score(
        pipe, X_train_scaled, y_train, cv=kf, scoring="neg_mean_squared_error"
    ))
    ridge_cv_r2 = cross_val_score(pipe, X_train_scaled, y_train, cv=kf, scoring="r2")
    pipe.fit(X_train_scaled, y_train)
    ridge_pred = pipe.predict(X_test_scaled)

    ridge_result = {
        "model": pipe,
        "model_type": "Ridge",
        "cv_rmse": ridge_cv_rmse,
        "cv_mae": -cross_val_score(pipe, X_train_scaled, y_train, cv=kf, scoring="neg_mean_absolute_error"),
        "cv_r2": ridge_cv_r2,
        "test_rmse": np.sqrt(mean_squared_error(y_test, ridge_pred)),
        "test_mae": mean_absolute_error(y_test, ridge_pred),
        "test_r2": r2_score(y_test, ridge_pred),
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": ridge_pred,
    }
    results.append(ridge_result)
    tqdm.write(f"  Ridge        CV R^2={ridge_cv_r2.mean():.4f} ± {ridge_cv_r2.std():.4f}, "
               f"Test R^2={ridge_result['test_r2']:.4f}")

    return results


def compute_shap(xgb_result: dict, X: pd.DataFrame,
                 max_display: int = 20) -> dict:
    """计算 SHAP 值并返回解释对象。

    Parameters
    ----------
    xgb_result : XGBoost 训练结果字典
    X : 完整特征矩阵（用于背景数据采样）
    max_display : SHAP 图中最多显示的特征数

    Returns
    -------
    dict : shap_values, explainer, shap_df（特征重要性排名表）
    """
    import shap

    tqdm.write("\n[4.2] 计算 SHAP 值...")

    model = xgb_result["model"]
    X_train = xgb_result["X_train"]

    # 用训练集子集做背景（加速），确保 float64
    bg_size = min(100, len(X_train))
    background = X_train.sample(n=bg_size, random_state=RANDOM_STATE).astype(float)

    # TreeExplainer 对 XGBoost 最有效
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
    )

    # 测试集也转 float64（SHAP 要求）
    X_test = xgb_result["X_test"].astype(float)
    shap_values = explainer(X_test, check_additivity=False)

    # 特征重要性排名
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": X_test.columns,
        "shap_importance": mean_abs_shap,
    }).sort_values("shap_importance", ascending=False)

    tqdm.write("  SHAP 特征重要性 Top 5:")
    for _, row in shap_df.head(5).iterrows():
        tqdm.write(f"    {row['feature']:25s} |SHAP|={row['shap_importance']:.4f}")

    return {
        "shap_values": shap_values,
        "explainer": explainer,
        "shap_df": shap_df,
    }


def compare_models(xgb_result: dict, baseline_results: list[dict]) -> pd.DataFrame:
    """汇总模型对比表。"""
    rows = []
    for r in [xgb_result] + baseline_results:
        rows.append({
            "模型": r["model_type"],
            "CV RMSE (mean)": r["cv_rmse"].mean(),
            "CV RMSE (std)": r["cv_rmse"].std(),
            "CV R^2 (mean)": r["cv_r2"].mean(),
            "Test RMSE": r["test_rmse"],
            "Test MAE": r["test_mae"],
            "Test R^2": r["test_r2"],
        })

    comparison = pd.DataFrame(rows).set_index("模型")
    tqdm.write("\n[4.2] 模型对比:")
    tqdm.write(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    return comparison


# ====================================================================
#  4.3 SARIMAX vs XGBoost 短时客流预测（可选）
# ====================================================================

def _build_lag_features(series: np.ndarray, lags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """从一维序列构造滞后特征矩阵。

    Parameters
    ----------
    series : 一维时间序列
    lags : 滞后阶数列表，如 [1, 2, 3, 7, 14]

    Returns
    -------
    X : (n - max(lags), len(lags)) 特征矩阵
    y : (n - max(lags),) 标签
    """
    max_lag = max(lags)
    n = len(series)
    X_rows = []
    for i in range(max_lag, n):
        X_rows.append([series[i - lag] for lag in lags])
    X = np.array(X_rows)
    y = series[max_lag:]
    return X, y


def predict_with_xgb_ts(train: np.ndarray, test: np.ndarray,
                        lags: list[int] | None = None) -> np.ndarray:
    """用 XGBoost + 滞后特征做时间序列预测。

    对测试集的每个时点做一步超前预测（walk-forward）。
    """
    import xgboost as xgb

    if lags is None:
        lags = [1, 2, 3, 7, 14]

    # 用训练数据构造滞后特征
    X_train, y_train = _build_lag_features(train, lags)

    if len(X_train) < 20:
        # 数据太少，回退到简单均值预测
        return np.full(len(test), np.mean(train))

    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=RANDOM_STATE, verbosity=0,
    )
    model.fit(X_train, y_train)

    # walk-forward 预测测试集
    history = list(train)
    preds = []
    for _ in range(len(test)):
        if len(history) < max(lags):
            preds.append(np.mean(history[-10:]))
            history.append(preds[-1])
            continue
        feats = np.array([[history[-lag] for lag in lags]])
        pred = model.predict(feats)[0]
        preds.append(pred)
        history.append(pred)

    return np.array(preds)


def predict_with_sarimax(train: np.ndarray, test: np.ndarray,
                         order: tuple = (1, 1, 1)) -> np.ndarray:
    """用 SARIMAX 做时间序列预测。

    对测试集做动态预测（逐步向前，不 refit）。
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        tqdm.write("  [跳过] SARIMAX 需要 statsmodels")
        return np.full(len(test), np.nan)

    try:
        model = SARIMAX(train, order=order, trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=200, method="lbfgs")
        forecast = fitted.get_forecast(steps=len(test))
        return forecast.predicted_mean
    except Exception as e:
        tqdm.write(f"  [警告] SARIMAX 拟合失败: {e}")
        return np.full(len(test), np.mean(train))


def run_ts_comparison(hourly: pd.DataFrame,
                      n_stations: int = 3,
                      test_days: int = 60) -> dict | None:
    """对代表性站点做 SARIMAX vs XGBoost 短时预测对比。

    Parameters
    ----------
    hourly : 小时聚合数据
    n_stations : 选取的站点数
    test_days : 测试集天数

    Returns
    -------
    dict | None : 对比结果（失败返回 None）
    """
    import xgboost as xgb

    tqdm.write(f"\n[4.3] SARIMAX vs XGBoost 短时客流预测...")

    # ---- 选取代表性站点 ----
    # 按日均进站量分大/中/小三类，各取一个
    df = hourly.copy()
    station_sizes = (
        df.groupby("station_id")["entries"].mean()
        .sort_values(ascending=False)
    )

    n_total = len(station_sizes)
    if n_total < n_stations:
        n_stations = n_total

    # 选大、中、小各一
    indices = [0, n_total // 2, n_total - 1]
    selected_ids = station_sizes.index[indices].tolist()
    # 去重
    selected_ids = list(dict.fromkeys(selected_ids))[:n_stations]

    # 获取站名
    station_names = (
        df[["station_id", "station_name"]]
        .drop_duplicates()
        .set_index("station_id")
    )

    tqdm.write(f"  选取站点: {selected_ids}")

    results = []
    lags = [1, 2, 3, 7, 14]

    for sid in tqdm(selected_ids, desc="  站点", unit="个", ncols=80):
        # 聚合到日度
        sub = df[df["station_id"] == sid].copy()
        sub["date"] = sub["timestamp"].dt.date
        daily = sub.groupby("date")["entries"].sum().sort_index()

        if len(daily) < test_days + 30:
            tqdm.write(f"    {sid}: 数据不足 ({len(daily)} 天)")
            continue

        train = daily.iloc[:-test_days].values.astype(float)
        test = daily.iloc[-test_days:].values.astype(float)
        train_dates = daily.index[:-test_days]
        test_dates = daily.index[-test_days:]

        # ---- XGBoost ----
        xgb_pred = predict_with_xgb_ts(train, test, lags)
        xgb_rmse = np.sqrt(mean_squared_error(test, xgb_pred))
        xgb_mae = mean_absolute_error(test, xgb_pred)

        # ---- SARIMAX ----
        sarimax_pred = predict_with_sarimax(train, test)
        if np.all(np.isnan(sarimax_pred)):
            sarimax_rmse = float("nan")
            sarimax_mae = float("nan")
        else:
            sarimax_rmse = np.sqrt(mean_squared_error(test, sarimax_pred))
            sarimax_mae = mean_absolute_error(test, sarimax_pred)

        name = station_names.loc[sid, "station_name"] if sid in station_names.index else str(sid)
        name_short = name[:30] if isinstance(name, str) else str(name)[:30]

        tqdm.write(f"    {name_short:30s}  "
                   f"XGBoost RMSE={xgb_rmse:.0f}  "
                   f"SARIMAX RMSE={sarimax_rmse:.0f}" if not np.isnan(sarimax_rmse)
                   else f"    {name_short:30s}  XGBoost RMSE={xgb_rmse:.0f}  SARIMAX 失败")

        results.append({
            "station_id": sid,
            "station_name": name,
            "train_size": len(train),
            "test_size": len(test),
            "xgb_rmse": xgb_rmse,
            "xgb_mae": xgb_mae,
            "sarimax_rmse": sarimax_rmse,
            "sarimax_mae": sarimax_mae,
            "xgb_pred": xgb_pred,
            "sarimax_pred": sarimax_pred,
            "train": train,
            "test": test,
            "train_dates": train_dates,
            "test_dates": test_dates,
        })

    if not results:
        tqdm.write("  [失败] 无足够数据用于对比")
        return None

    # 汇总
    summary_df = pd.DataFrame([
        {
            "站点": r["station_name"][:25],
            "XGBoost RMSE": r["xgb_rmse"],
            "XGBoost MAE": r["xgb_mae"],
            "SARIMAX RMSE": r["sarimax_rmse"],
            "SARIMAX MAE": r["sarimax_mae"],
            "RMSE比 (XGB/SARIMAX)": r["xgb_rmse"] / r["sarimax_rmse"] if r["sarimax_rmse"] > 0 else float("nan"),
        }
        for r in results
    ])

    tqdm.write("\n  SARIMAX vs XGBoost 对比汇总:")
    tqdm.write(summary_df.to_string(float_format=lambda x: f"{x:.1f}" if abs(x) < 100 else f"{x:.0f}"))

    return {
        "results": results,
        "summary_df": summary_df,
    }


# ====================================================================
#  4.4 可视化
# ====================================================================

def plot_shap_beeswarm(shap_result: dict, feature_info: pd.DataFrame):
    """Fig 15: SHAP beeswarm —— 特征重要性散点。"""
    import shap

    tqdm.write("\n[绘图] Fig 15: SHAP beeswarm...")

    shap_values = shap_result["shap_values"]

    # 特征名中文化映射
    name_map = {
        "commute_ratio": "通勤比", "entries_cv": "客流CV",
        "peak_morning": "早高峰强度", "peak_evening": "晚高峰强度",
        "avg_daily_entries": "日均进站量",
        "yoy_2023_vs_2022": "YoY变化(23vs22)",
        "yoy_2024_vs_2023": "YoY变化(24vs23)",
    }
    for b in BOROUGH_NAMES:
        name_map[f"borough_{b}"] = f"区-{BOROUGH_NAMES[b]}"

    display_features = shap_values.feature_names
    display_names = [name_map.get(f, f) for f in display_features]

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False,
                        color_bar=True, plot_size=None)
    # 替换 y 轴标签为中文
    ax = plt.gca()
    yticks = ax.get_yticks()
    if len(yticks) <= len(display_names):
        ax.set_yticklabels([display_names[min(int(t), len(display_names) - 1)]
                           if int(t) < len(display_names) else ""
                           for t in yticks])

    ax.set_title("SHAP 特征重要性 (beeswarm)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig15_shap_beeswarm.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig15_shap_beeswarm.png")
    plt.close()


def plot_shap_bar(shap_result: dict):
    """Fig 16: SHAP bar —— 特征重要性均值排名。"""
    import shap

    tqdm.write("\n[绘图] Fig 16: SHAP bar...")

    shap_values = shap_result["shap_values"]

    name_map = {
        "commute_ratio": "通勤比", "entries_cv": "客流CV",
        "peak_morning": "早高峰强度", "peak_evening": "晚高峰强度",
        "avg_daily_entries": "日均进站量",
        "yoy_2023_vs_2022": "YoY变化(23vs22)",
        "yoy_2024_vs_2023": "YoY变化(24vs23)",
    }
    for b in BOROUGH_NAMES:
        name_map[f"borough_{b}"] = f"区-{BOROUGH_NAMES[b]}"

    display_features = shap_values.feature_names
    display_names = [name_map.get(f, f) for f in display_features]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=20, show=False)
    ax = plt.gca()
    yticks = ax.get_yticks()
    if len(yticks) <= len(display_names):
        ax.set_yticklabels([display_names[min(int(abs(t)), len(display_names) - 1)]
                           if int(abs(t)) < len(display_names) else ""
                           for t in yticks])

    ax.set_title("SHAP 特征重要性 (mean |SHAP|)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig16_shap_bar.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig16_shap_bar.png")
    plt.close()


def plot_prediction_vs_actual(xgb_result: dict, baseline_results: list[dict]):
    """Fig 17: 预测 vs 实际散点图——三模型并排。"""
    tqdm.write("\n[绘图] Fig 17: 预测 vs 实际...")

    all_models = [xgb_result] + baseline_results
    n_models = len(all_models)

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, r in zip(axes, all_models):
        y_test = r["y_test"]
        y_pred = r["y_pred"]
        r2 = r["test_r2"]
        rmse = r["test_rmse"]

        ax.scatter(y_test, y_pred, alpha=0.6, s=40, edgecolors="white", lw=0.3,
                   color="#1f77b4")
        # 对角线
        lims = [min(y_test.min(), y_pred.min()) - 0.05,
                max(y_test.max(), y_pred.max()) + 0.05]
        ax.plot(lims, lims, "r--", lw=1, alpha=0.7, label="完美预测")
        ax.set_xlim(lims); ax.set_ylim(lims)

        ax.set_xlabel("实际恢复率"); ax.set_ylabel("预测恢复率")
        ax.set_title(f"{r['model_type']}\nR^2={r2:.3f}  RMSE={rmse:.3f}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("站点恢复率：预测 vs 实际", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig17_prediction_vs_actual.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig17_prediction_vs_actual.png")
    plt.close()


def plot_ts_comparison(ts_result: dict):
    """Fig 18: SARIMAX vs XGBoost 短时预测曲线对比。"""
    tqdm.write("\n[绘图] Fig 18: SARIMAX vs XGBoost...")

    if ts_result is None or not ts_result.get("results"):
        tqdm.write("[跳过] 无对比结果")
        return

    results = ts_result["results"]
    n_stations = len(results)

    fig, axes = plt.subplots(n_stations, 2, figsize=(16, 4 * n_stations))
    if n_stations == 1:
        axes = axes.reshape(1, -1)

    for i, r in enumerate(results):
        # ---- 左列：完整序列 ----
        ax = axes[i, 0]
        train = r["train"]
        test = r["test"]
        xgb_pred = r["xgb_pred"]
        sarimax_pred = r["sarimax_pred"]
        train_dates = r["train_dates"]
        test_dates = r["test_dates"]

        # 只显示最后 180 天（6个月）以保持可读性
        show_days = min(180, len(train))
        ax.plot(range(len(train) - show_days, len(train)),
                train[-show_days:], color="gray", lw=0.8, alpha=0.6, label="训练数据")
        ax.plot(range(len(train), len(train) + len(test)),
                test, color="black", lw=1.5, label="实际值")
        ax.plot(range(len(train), len(train) + len(test)),
                xgb_pred, color="#1f77b4", lw=1.5, ls="--", label="XGBoost")
        if not np.all(np.isnan(sarimax_pred)):
            ax.plot(range(len(train), len(train) + len(test)),
                    sarimax_pred, color="#ff7f0e", lw=1.5, ls="--", label="SARIMAX")

        ax.axvline(len(train), color="red", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(f"{r['station_name'][:25]} (完整)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8); ax.set_ylabel("日进站量")

        # ---- 右列：放大测试区间 ----
        ax = axes[i, 1]
        x_axis = range(len(test))
        ax.plot(x_axis, test, "ko-", lw=1.5, markersize=3, label="实际值")
        ax.plot(x_axis, xgb_pred, "s--", color="#1f77b4", lw=1.5, markersize=3, label="XGBoost")
        if not np.all(np.isnan(sarimax_pred)):
            ax.plot(x_axis, sarimax_pred, "^--", color="#ff7f0e", lw=1.5, markersize=3, label="SARIMAX")

        # RMSE 标注
        text = f"XGBoost RMSE={r['xgb_rmse']:.0f}"
        if not np.isnan(r.get("sarimax_rmse", float("nan"))):
            text += f"\nSARIMAX RMSE={r['sarimax_rmse']:.0f}"
        ax.text(0.02, 0.95, text, transform=ax.transAxes,
                fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.85))

        ax.set_title(f"{r['station_name'][:25]} (测试区间放大)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8); ax.set_xlabel("测试天数"); ax.set_ylabel("日进站量")

    fig.suptitle("SARIMAX vs XGBoost 短时客流预测对比",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig18_sarimax_vs_xgboost.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig18_sarimax_vs_xgboost.png")
    plt.close()


def plot_feature_correlations(X: pd.DataFrame, y: pd.Series):
    """补充：特征相关性热力图（帮助解读 SHAP）。"""
    tqdm.write("\n[绘图] 补充: 特征相关性热力图...")

    df_corr = X.copy()
    df_corr["恢复率"] = y.values

    name_map = {
        "commute_ratio": "通勤比", "entries_cv": "客流CV",
        "peak_morning": "早高峰强度", "peak_evening": "晚高峰强度",
        "avg_daily_entries": "日均进站量",
        "yoy_2023_vs_2022": "YoY变化(23vs22)",
        "yoy_2024_vs_2023": "YoY变化(24vs23)",
    }
    for b in BOROUGH_NAMES:
        name_map[f"borough_{b}"] = f"区-{BOROUGH_NAMES[b]}"
    df_corr = df_corr.rename(columns=name_map)

    corr = df_corr.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("特征相关性矩阵（含目标变量：恢复率）", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig_ml_feature_corr.png", dpi=200, bbox_inches="tight")
    tqdm.write("[保存] fig_ml_feature_corr.png")
    plt.close()


# ====================================================================
#  汇总报告
# ====================================================================

def print_report(xgb_result: dict, baseline_results: list[dict],
                 comparison: pd.DataFrame, shap_result: dict,
                 feature_info: pd.DataFrame):
    """打印完整的 ML 验证报告。"""
    print("\n" + "=" * 65)
    print("  模块四 — ML 验证报告")
    print("=" * 65)

    # 模型对比
    print(f"\n  {'模型':15s} {'CV R^2':>8s} {'Test R^2':>8s} {'Test RMSE':>10s} {'Test MAE':>10s}")
    print(f"  {'-'*55}")
    for r in [xgb_result] + baseline_results:
        print(f"  {r['model_type']:15s} {r['cv_r2'].mean():8.4f} {r['test_r2']:8.4f} "
              f"{r['test_rmse']:10.4f} {r['test_mae']:10.4f}")

    # 关键结论
    best_model = max([xgb_result] + baseline_results, key=lambda r: r["test_r2"])
    print(f"\n  最佳模型: {best_model['model_type']} (Test R^2={best_model['test_r2']:.4f})")

    # SHAP Top 特征
    print(f"\n  SHAP 最强预测因子 (Top 5):")
    for i, (_, row) in enumerate(shap_result["shap_df"].head(5).iterrows()):
        feat = row["feature"]
        imp = row["shap_importance"]
        # 查与 y 的相关性
        if feat in best_model["X_test"].columns:
            corr_val = np.corrcoef(best_model["X_test"][feat], best_model["y_test"])[0, 1]
            corr_sign = "+" if corr_val > 0 else "-"
        else:
            corr_sign = "?"
        print(f"    {i+1}. {feat:28s} |SHAP|={imp:.4f}  (与恢复率相关: {corr_sign})")

    # 结论
    feature_count = len(shap_result["shap_df"])
    top_feat = shap_result["shap_df"].iloc[0]["feature"]
    print(f"\n  ML 验证结论:")
    print(f"    模型能解释站点恢复率 {best_model['test_r2']:.1%} 的方差")
    print(f"    最强预测因子为 {top_feat}，在 {feature_count} 个特征中 SHAP 值最高")
    print(f"    交叉验证 (k=5) 确认模型稳定性: CV R^2 = {best_model['cv_r2'].mean():.4f} ± {best_model['cv_r2'].std():.4f}")
    print(f"{'='*65}\n")


# ====================================================================
#  一键运行入口
# ====================================================================

def run_all(skip_ts: bool = False):
    """运行模块四全部流程。

    Parameters
    ----------
    skip_ts : 跳过 4.3 时序预测对比（耗时较长，默认 False）
    """
    print("=" * 60)
    print("  机器学习验证 (模块四)")
    print("=" * 60)

    # ---- 加载数据 ----
    tqdm.write("\n>>> 加载数据")

    if not STATION_PARQUET.exists():
        tqdm.write("[错误] 站点特征未生成，请先运行 data_processor.py")
        return None

    stations = pd.read_parquet(STATION_PARQUET)
    tqdm.write(f"  站点特征: {len(stations)} 站")

    hourly = None
    if HOURLY_PARQUET.exists():
        hourly = pd.read_parquet(HOURLY_PARQUET)
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
        tqdm.write(f"  小时数据: {len(hourly):,} 行")
    else:
        tqdm.write("[警告] 小时聚合数据未生成，YoY 特征不可用")

    # ---- 4.1 特征矩阵构造 ----
    X, y, feature_info = build_feature_matrix(stations, hourly)

    # ---- 4.2 XGBoost 训练 ----
    xgb_result = train_xgboost(X, y)

    # ---- 4.2 基准模型 ----
    baseline_results = train_baselines(X, y)

    # ---- 4.2 模型对比 ----
    comparison = compare_models(xgb_result, baseline_results)

    # ---- 4.2 SHAP ----
    shap_result = compute_shap(xgb_result, X)

    # ---- 4.3 SARIMAX vs XGBoost（可选） ----
    ts_result = None
    if not skip_ts and hourly is not None:
        ts_result = run_ts_comparison(hourly, n_stations=3, test_days=60)
    elif skip_ts:
        tqdm.write("\n[4.3] 跳过时序对比（skip_ts=True）")

    # ---- 4.4 可视化 ----
    tqdm.write("\n>>> 产出图表")
    plot_shap_beeswarm(shap_result, feature_info)
    plot_shap_bar(shap_result)
    plot_prediction_vs_actual(xgb_result, baseline_results)
    if ts_result is not None:
        plot_ts_comparison(ts_result)
    plot_feature_correlations(X, y)

    # ---- 报告 ----
    print_report(xgb_result, baseline_results, comparison, shap_result, feature_info)

    return {
        "X": X, "y": y, "feature_info": feature_info,
        "xgb_result": xgb_result,
        "baseline_results": baseline_results,
        "comparison": comparison,
        "shap_result": shap_result,
        "ts_result": ts_result,
    }


if __name__ == "__main__":
    run_all(skip_ts=False)
