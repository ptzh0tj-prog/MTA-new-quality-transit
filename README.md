# MTA — 新质生产力冲击下的城市交通复苏与结构演化

> 基于纽约多源时空数据，检验新质生产力（AI/大数据/新能源等）通过改变就业结构
> 重塑城市交通格局的假设。三份数据各自独立推断，互相印证。

## 数据资产

| 数据 | 时间跨度 | 粒度 | 规模 | 关键字段 |
|------|----------|------|------|----------|
| MTA 每日运量 | 2020-03 ~ 2026-06 | 日 × 9 种运输方式 | 17,115 行 | date, mode, daily_ridership |
| MTA 地铁刷卡 | 2022-05 ~ 2024-12 | 小时 × 428 站点 | 740 万行（聚合后）| timestamp, station, borough, entries |
| LinkedIn 岗位技能 | 截面（~2024） | 岗位级 | 129.6 万条 | job_link, job_skills |
| LinkedIn 岗位详情 | 2023-12 ~ 2024-04 | 岗位级 | 12.4 万条（去重后 9.7 万）| title, description, location, salary, remote, work_type |

## 项目结构

```
MTA/
├── csv/                        # 原始数据
│   ├── MTA_Daily_Ridership_and_Traffic__Beginning_2020.csv
│   ├── MTA_Subway_Hourly_Ridership.csv   (~8GB)
│   ├── job_skills.csv                    (642MB, 129.6万条)
│   └── postings.csv/postings.csv         (493MB, 12.4万条)
├── processed/                  # 中间数据（Parquet 缓存）
│   ├── mta_daily.parquet
│   ├── mta_hourly_agg.parquet
│   ├── mta_station_features.parquet
│   ├── linkedin_enriched_light.parquet
│   ├── postings_processed.parquet
│   └── postings_enriched.parquet
├── plots/                      # 全部论文图表（19 张）
├── data_processor.py           # 数据管道：DuckDB ETL + 特征工程
├── nlp_extractor.py            # job_skills NLP：TF-IDF + NMF 主题建模
├── timeseries_analysis.py      # 模块一：MTA 时序深化（Chow/ADF/高峰演化）
├── postings_nlp.py             # 模块二：postings.csv NLP 管道
├── cross_validation.py         # 模块三：三层交叉验证
├── ml_analysis.py              # 模块四：ML 验证（XGBoost + SHAP）
├── main.py                     # 主流程入口
└── 工作安排.md                  # 分步推进计划
```

## 模块总览

### 模块一：MTA 时序深化 `timeseries_analysis.py`

**目标**：从 MTA 日/时数据中提取全部时间维度的证据。

| 函数 | 方法 | 产出 |
|------|------|------|
| `load_daily_with_features()` | 基线=2020.3 前半月；恢复率=日均/基线 | 每日 + 月度时序数据 |
| `test_stationarity()` / `run_stationarity_tests()` | **ADF 单位根检验**（statsmodels） | 9 种方式的平稳性判断 |
| `analyze_shock_recovery()` | 冲击深度（最低恢复率）、恢复至 80% 月数 | 冲击-恢复汇总表 |
| `chow_test()` | **Chow 结构断点检验**：拟合线性/二次趋势，比较全样本 RSS vs 分段 RSS，构造 F 统计量 | F 值 + p 值 + 断点判断 |
| `run_chow_tests()` | 对 Subway/Bus/LIRR/MNR 检验 2022-12（AI 节点）；二次趋势避免线性误报；内部调用灵敏度扫描 | Chow 结果矩阵 + 渐进 vs 突变解读 |
| `run_chow_sensitivity()` | 断点 ±12 个月逐月扫描，观察 F 统计量波动 | 稳健性曲线 |
| `analyze_structural_divergence()` | 月度恢复率跨方式 std/IQR/CV；排除 CBD/CRZ（2025 年才出现） | 四时期分化均值 |
| `analyze_mode_correlations()` | 四时期（冲击/恢复前期/恢复后期/新常态）相关性矩阵 | 方式间替代/互补关系 |
| `analyze_commute_ratio_evolution()` | 逐年计算 428 站点 commute_ratio = 早高峰 entries / 晚高峰 entries；通勤型站点(0.7≤ratio≤1.3)占比 | 3 年 × 5 borough 趋势 |

**核心发现：**
- Chow 二次趋势检验：**Bus 在 AI 节点断点最强 (F=9.4, p<.001)，LIRR 不显著 (F=1.6, p=.195)**
- 运输方式分化在新常态时期达到最高 (std=0.335)，是冲击期的 1.9 倍
- 通勤型站点占比 16.9%→15.0%，commute_ratio 中位数持续远离 1.0 → **高峰极化而非扁平化**
- Bus 仅恢复至 65%、SIR 54%——两者可能面临永久性结构替代

### 模块二：postings.csv NLP 管道 `postings_nlp.py`

**目标**：从 12.4 万条岗位描述中提取行业结构画像，输出"新质 vs 传统"对比。

| 函数 | 方法 | 产出 |
|------|------|------|
| `load_postings_raw()` | **DuckDB** 读取 CSV → 时间戳转换 → 文本合并 → 缓存 Parquet | postings_processed.parquet |
| `preprocess_structured_fields()` | 提取 work_type / remote / salary / location；NYC 筛选 + borough 分类 | 结构化统计 |
| `compute_seed_scores()` | 向量化：用 `str.count` 对 7 大类种子词库（AI/ML、Cloud/DevOps、数据分析、新能源、生物科技、半导体、金融科技）打分，密度 = hits / total_skills | 7 维新质得分向量 |
| `run_topic_model()` | **TF-IDF**（max_features=5000, ngram 1-2）+ **NMF** 分解（n_components=20）；80K 采样 fit，全量分块 transform | 20 主题 × 关键词矩阵 |
| `label_topics()` | 每个主题 top-15 词与种子词库交叉比对，自动标注"新质→AI_ML+Data"或"传统行业" | 主题-行业映射表 |
| `classify_postings()` | 综合 NMF topic + seed_scores 双重判定：NMF 分配到新质主题 OR 种子词库 ≥2 类命中 → is_new_quality | 岗位级标签 |
| `compare_new_vs_traditional()` | 分组对比：远程率、平均薪资、全职占比、技能密度；t 检验 + χ² | 新质 vs 传统画像表 |

**两种分类策略的互补：**
- **策略 A（NMF 无监督）**：从数据中自动发现行业结构，避免主观关键词偏差
- **策略 B（种子词库半监督）**：利用 nlp_extractor 在 129.6 万条 job_skills 上验证过的词库，做有根有据的打分
- 最终判定 = NMF 主题匹配 **OR** 种子密度高于阈值 → 双重保险

**核心发现：**
- 全样本新质岗位占比 72.4%（策略 A NMF 9.3% + 策略 B 种子词库 70.8%，双命中 7,513）
- 新质行业远程率 13.8% > 传统行业 10.5%（χ² p<.001）；新质薪资中位 $90K vs 传统 $75K（溢价 +$15K）
- 新质行业技能集中度 3.7 类 vs 传统 1.7 类；描述文本长 4,227 vs 2,451 字符
- NMF 识别出 1 个清晰的新质主题（T4: cloud+data+software），其余为传统行业细分

### 模块三：三层交叉验证 `cross_validation.py`

**目标**：将 LinkedIn NLP 的推论与 MTA 行为证据对照，实现三个独立维度的互相印证。

#### L1 — 结构断点验证

```
LinkedIn 推论：AI/科技岗位 2023 起爆发 → MTA 恢复曲线应有结构断点
方法：Chow 检验（二次趋势），对 4 种核心方式逐一检验
判定：≥2/4 方式 p<.05 → 部分支持；≥3/4 → 强支持
```

#### L2 — 通勤模式验证

```
LinkedIn 推论：新质行业远程率显著高于传统行业 → 地铁通勤高峰应弱化
方法：逐年计算 commute_ratio，KMeans 聚类站点（4 类）；
     检验通勤型站点占比是否下降、commute_ratio 中位数是否远离 1.0
判定：占比下降 + 中位数偏移 → 支持远程办公假说
```

#### L3 — 运输方式替代验证

```
LinkedIn 推论：高薪新质岗位集中在曼哈顿 → 远郊通勤铁路恢复应优于短途公交
方法：各方式当前恢复率排序；与 LinkedIn 薪资/远程率做 Spearman 相关
判定：LIRR/MNR > Subway > Bus → 支持薪资-通勤距离假说
```

| 函数 | 方法 | 产出 |
|------|------|------|
| `validate_l1_chow()` | 复用 timeseries_analysis 的 chow_test()；对 Subway/Bus/LIRR/MNR 检验 2022-12 | L1 判决 + Chow 统计表 |
| `validate_l2_commute()` | yearly commute_ratio → KMeans(n=4) 站点聚类；通勤型占比趋势检验 | L2 判决 + 站点聚类标签 |
| `validate_l3_mode_substitution()` | 恢复率排序 + Spearman 秩相关（恢复率 ~ 远程率/薪资） | L3 判决 + 相关系数 |
| `synthesize_verdict()` | 综合三层证据强度，输出总体判决文本 | 论文级结论摘要 |
| `cluster_stations()` | StandardScaler + **KMeans**（4 类）：通勤型/混合型/居住型/商业型 | 站点-类别映射 |

**核心发现：**
- L1（断点）：Subway (F=3.9, p=.012) / Bus (F=9.4, p<.001) / MNR (F=5.0, p=.003) 显著；LIRR (F=1.6, p=.195) 不显著 → **强支持**（3/4 通过）
- L2（通勤）：通勤型站点占比 16.9%→15.0% (Δ=-1.9%, p=0.26) → **弱支持**（方向符合假说但幅度温和）
- L3（替代）：LIRR 109.2% / MNR 135.8% / Subway 89.9% / Bus 68.3% — 完美层级 → **强支持**
- 综合判决：**中等支持 (2/3 层通过)**

### 模块四：机器学习验证 `ml_analysis.py`

**目标**：用 XGBoost + SHAP 量化各特征对站点恢复率的贡献，并与基准模型对比。

#### 4.1 特征矩阵构造

```
X（站点级特征，428 站 × 11 维）：
  - commute_ratio（通勤比）
  - entries_cv（客流波动性）
  - peak_morning / peak_evening（高峰强度）
  - avg_daily_entries（规模）
  - yoy_2023_vs_2022 / yoy_2024_vs_2023（年度变化率）
  - borough（one-hot × 4）

y：recovery_2024_vs_2022（站点恢复率）
```

#### 4.2 XGBoost + SHAP + 基准对比

| 函数 | 方法 | 产出 |
|------|------|------|
| `train_xgboost()` | **XGBoost** 回归（n=200, depth=5）+ 5 折 CV | 模型 + CV/Test 指标 |
| `train_baselines()` | **RandomForest** + **Ridge** 回归对比 | 三模型对比表 |
| `compute_shap()` | **TreeExplainer**（interventional）+ 特征重要性排名 | SHAP values + 重要性 DataFrame |
| `compare_models()` | RMSE / MAE / R^2 汇总 | 模型选择依据 |

#### 4.3 SARIMAX vs XGBoost 短时客流预测（可选）

| 函数 | 方法 | 产出 |
|------|------|------|
| `predict_with_sarimax()` | **SARIMAX**(1,1,1) 动态预测 | 传统时序基线 |
| `predict_with_xgb_ts()` | **XGBoost** + 滞后特征（lag 1/2/3/7/14）walk-forward | ML 方法预测 |
| `run_ts_comparison()` | 3 个代表性站点（大/中/小）对比 RMSE/MAE | 方法论选择自觉性 |

#### 4.4 产出图表

| 编号 | 文件 | 内容 |
|------|-----|------|
| fig15 | `fig15_shap_beeswarm.png` | SHAP 特征重要性散点（beeswarm） |
| fig16 | `fig16_shap_bar.png` | SHAP 特征重要性均值排名 |
| fig17 | `fig17_prediction_vs_actual.png` | 三模型预测 vs 实际散点对比 |
| fig18 | `fig18_sarimax_vs_xgboost.png` | SARIMAX vs XGBoost 预测曲线对比 |

**核心发现：**
- Ridge 回归（R^2=0.995）> RandomForest（R^2=0.955）> XGBoost（R^2=0.950）；样本量小（428）、特征线性关系强
- **SHAP 最强预测因子：yoy_2024_vs_2023（年度客流变化率）**，其次是 yoy_2023_vs_2022 和 commute_ratio
- XGBoost 在 3 个代表性站点上均优于 SARIMAX（RMSE 比 0.4–0.7），验证了 ML 方法对小样本短时预测的适用性

### 支撑模块

#### 数据管道 `data_processor.py`

| 函数 | 方法 |
|------|------|
| `load_daily_raw()` | **DuckDB** SQL 聚合 2020-2026 日数据 → Parquet |
| `build_daily_features()` | 基线(2020.1-2) → 月度恢复率 → 年度汇总 |
| `load_hourly_agg()` | **DuckDB** 聚合 8GB 刷卡数据（按小时+站点）→ Parquet |
| `build_station_features()` | 早晚高峰比 / 2024 vs 2022 恢复率 / 变异系数 CV / 日均进站量 |

#### LinkedIn NLP 提取 `nlp_extractor.py`

| 函数 | 方法 |
|------|------|
| `load_and_preprocess()` | 读取 642MB CSV → 文本清洗 → 技能分词 |
| `extract_remote_flag()` | 正则匹配 12 个远程办公关键词 |
| `compute_tech_density()` | 向量化 `str.count`：新质技能命中数 / 总技能数 |
| `run_topic_model()` | **TF-IDF**（max_features=5000）+ **NMF**（n=20）；200K 采样 fit，10 万/块 transform |
| `label_topics()` | Top-15 词 × 7 类种子词库交叉 → 自动标注 |
| `classify_jobs_by_topic()` | 主 topic 分配 + 新质/传统二分类 |
| `build_skill_cooccurrence()` | 100K 采样，Top-100 技能共现矩阵 |

## 技术栈

| 库 | 场景 |
|------|------|
| **duckdb** | 大文件 SQL 聚合（8GB CSV → 200MB 内存峰值） |
| **pandas + numpy** | 全流程数据处理与特征工程 |
| **scikit-learn** | TF-IDF 向量化、NMF 主题分解、KMeans 聚类、StandardScaler |
| **statsmodels** | ADF 平稳性检验、季节分解（seasonal_decompose）、SARIMAX |
| **xgboost + shap** | XGBoost 回归 + SHAP 可解释性分析 |
| **scipy** | Chow 检验 F 分布临界值、Spearman/Pearson 相关、t 检验 |
| **matplotlib + seaborn** | 全部 19 张论文图表 |
| **tqdm** | 大规模数据处理的进度反馈 |

## 图表清单

| 编号 | 文件 | 内容 | 来源模块 |
|:--:|------|------|:--:|
| fig1 | `fig1_recovery_curves.png` | 各运输方式恢复曲线（双面板） | 一 |
| fig2 | `fig2_shock_waterfall.png` | 疫情冲击深度瀑布图 | 一 |
| fig3 | `fig3_seasonal_decomp.png` | 地铁月度运量季节分解 | 一 |
| fig4 | `fig4_chow_breakpoints.png` | Chow 断点图：四方式 + 两断点 | 一 |
| fig4b | `fig4b_chow_sensitivity.png` | Chow 稳健性扫描（Subway ±12月） | 一 |
| fig5 | `fig5_structural_divergence.png` | 结构分化时序：离散度 + 均值 | 一 |
| fig6 | `fig6_correlation_matrices.png` | 四时期方式相关性热力图 | 一 |
| fig7 | `fig7_commute_ratio_evolution.png` | 通勤比分布演进（KDE + 中位趋势） | 一 |
| fig7b | `fig7b_commute_by_borough.png` | 各 Borough 通勤比年度变化 | 一 |
| fig8 | `fig8_topic_keywords_heatmap.png` | NMF 20 主题 × Top-10 关键词 | 二 |
| fig8b | `fig8b_topic_distribution.png` | 主题分布 + 新质/传统着色 | 二 |
| fig9 | `fig9_salary_distribution.png` | 新质 vs 传统薪资双 KDE | 二 |
| fig10 | `fig10_remote_industry_bubble.png` | 远程率 × 行业 × 薪资气泡图 | 二 |
| fig11 | `fig11_three_layer_validation.png` | 三层验证综合面板图 | 三 |
| fig12 | `fig12_station_cluster_map.png` | 站点 KMeans 聚类空间分布 | 三 |
| fig13 | `fig13_mode_recovery_ranking.png` | 方式恢复率排序 + LinkedIn 特征标注 | 三 |
| fig14 | `fig14_commute_vs_recovery.png` | commute_ratio × recovery 散点 | 三 |
| fig15 | `fig15_shap_beeswarm.png` | SHAP beeswarm 特征重要性散点 | 四 |
| fig16 | `fig16_shap_bar.png` | SHAP 特征重要性均值排名 | 四 |
| fig17 | `fig17_prediction_vs_actual.png` | 三模型预测 vs 实际对比 | 四 |
| fig18 | `fig18_sarimax_vs_xgboost.png` | SARIMAX vs XGBoost 预测曲线 | 四 |

## 运行

```bash
# 安装依赖
uv sync

# 单独运行各模块
uv run python timeseries_analysis.py    # 模块一：MTA 时序
uv run python postings_nlp.py           # 模块二：LinkedIn NLP（首次运行 ~10 分钟）
uv run python cross_validation.py       # 模块三：交叉验证（依赖模块一、二）
uv run python ml_analysis.py            # 模块四：ML 验证（XGBoost + SHAP）

# 或通过主流程入口
uv run python main.py                   # 全流程
uv run python main.py --step ml         # 只跑 ML 验证

# 数据预处理（首次运行，或重新生成中间表）
uv run python data_processor.py
uv run python nlp_extractor.py
```

中间结果缓存在 `processed/` 目录（Parquet 格式），重复运行自动跳过已完成步骤。

## 方法论关键词

`Chow 结构断点检验` `ADF 平稳性检验` `TF-IDF 向量化` `NMF 非负矩阵分解`
`KMeans 聚类` `季节分解` `Spearman 秩相关` `向量化文本挖掘`
`XGBoost 回归` `SHAP 可解释性` `SARIMAX 时序预测` `模型对比验证`
`DuckDB OLAP` `假设驱动交叉验证` `多源时空数据融合`
