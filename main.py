"""
MTA 主流程入口
=============
用法：
  uv run python main.py                # 全流程
  uv run python main.py --step mta     # 只跑 MTA 时序
  uv run python main.py --step nlp     # 只跑 LinkedIn NLP
  uv run python main.py --step cv      # 只跑交叉验证
  uv run python main.py --step ml      # 只跑 ML 验证
  uv run python main.py --step data    # 只跑数据预处理
  uv run python main.py --figs         # 仅重新出图（跳过 SARIMAX）
"""

import argparse
import sys
import time
from pathlib import Path


# ====================================================================
#  各模块运行函数
# ====================================================================

def run_data():
    """运行数据预处理：data_processor + nlp_extractor。"""
    print("\n>>> 数据预处理 (DuckDB ETL)")
    import data_processor
    data_processor.run_all()

    print("\n>>> LinkedIn NLP 提取 (job_skills TF-IDF + NMF)")
    import nlp_extractor
    nlp_extractor.run_all()


def run_mta():
    """模块一: MTA 时序深化。"""
    print("\n>>> 模块一: MTA 时序深化")
    import timeseries_analysis
    timeseries_analysis.run_all()


def run_nlp():
    """模块二: postings.csv NLP 管道。"""
    print("\n>>> 模块二: postings NLP 管道")
    import postings_nlp
    postings_nlp.run_all()


def run_cv():
    """模块三: 三层交叉验证。"""
    print("\n>>> 模块三: 三层交叉验证")
    import cross_validation
    cross_validation.run_all()


def run_ml(skip_ts: bool = False):
    """模块四: ML 验证（XGBoost + SHAP）。"""
    print("\n>>> 模块四: ML 验证")
    import ml_analysis
    ml_analysis.run_all(skip_ts=skip_ts)


# ====================================================================
#  编排逻辑
# ====================================================================

STEPS = {
    "data": ("数据预处理 (data_processor + nlp_extractor)", run_data),
    "mta":  ("模块一: MTA 时序深化", run_mta),
    "nlp":  ("模块二: postings NLP 管道", run_nlp),
    "cv":   ("模块三: 三层交叉验证", run_cv),
    "ml":   ("模块四: ML 验证 (XGBoost + SHAP)", run_ml),
}


def run_all(skip_ts: bool = False):
    """全流程：按依赖顺序依次运行所有模块。

    Parameters
    ----------
    skip_ts : 传递给模块四，跳过 SARIMAX 时序对比
    """
    ordered = ["data", "mta", "nlp", "cv", "ml"]

    print("=" * 60)
    print("  MTA 全流程分析")
    print("=" * 60)
    print(f"  模块: {' → '.join(ordered)}")
    print(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start_all = time.time()
    results = {}

    for i, key in enumerate(ordered, 1):
        name, func = STEPS[key]
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(ordered)}] {name}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            if key == "ml":
                result = func(skip_ts=skip_ts)
            else:
                result = func()
            elapsed = time.time() - t0
            print(f"\n[完成] {name} ({elapsed:.0f}s)")
            results[key] = result
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n[错误] {name} 失败 ({elapsed:.0f}s): {e}")
            import traceback
            traceback.print_exc()
            results[key] = None

    total_elapsed = time.time() - start_all
    n_ok = sum(1 for v in results.values() if v is not None)
    print(f"\n{'='*60}")
    print(f"  全流程完成: {n_ok}/{len(ordered)} 模块成功")
    print(f"  总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*60}")

    return results


def run_figs_only():
    """仅重新出图模式：运行所有模块以重新生成全部论文图表。

    各模块的 run_all() 会检测已有的 Parquet 缓存，跳过耗时的数据聚合和
    模型训练步骤，仅重新执行可视化部分。模块四跳过 SARIMAX 以加速。
    """
    print("=" * 60)
    print("  仅重新出图模式 (--figs)")
    print("  读取缓存数据 → 重新生成全部图表")
    print("=" * 60)
    run_all(skip_ts=True)


# ====================================================================
#  CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MTA — 新质生产力冲击下的城市交通复苏与结构演化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  uv run python main.py                # 全流程\n"
            "  uv run python main.py --step mta     # 只跑 MTA 时序\n"
            "  uv run python main.py --step nlp     # 只跑 LinkedIn NLP\n"
            "  uv run python main.py --step cv      # 只跑交叉验证\n"
            "  uv run python main.py --step ml      # 只跑 ML 验证\n"
            "  uv run python main.py --step data    # 只跑数据预处理\n"
            "  uv run python main.py --figs         # 仅重新出图\n"
            "  uv run python main.py --step ml --skip-ts  # ML 跳过 SARIMAX"
        ),
    )
    parser.add_argument(
        "--step", "-s",
        choices=list(STEPS.keys()),
        default=None,
        help="只运行指定模块（默认运行全流程）",
    )
    parser.add_argument(
        "--figs",
        action="store_true",
        help="仅重新出图：读取缓存数据，重新生成全部论文图表",
    )
    parser.add_argument(
        "--skip-ts",
        action="store_true",
        help="跳过 4.3 SARIMAX 时序对比（加速 ML 模块，出图模式默认启用）",
    )

    args = parser.parse_args()

    # --figs 模式：运行全流程但跳过 SARIMAX
    if args.figs:
        run_figs_only()
        return

    # --step 模式：只跑单个模块
    if args.step:
        name, func = STEPS[args.step]
        print(f"[单步] {name}")
        if args.step == "ml":
            run_ml(skip_ts=args.skip_ts)
        else:
            func()
        return

    # 默认：全流程
    run_all(skip_ts=args.skip_ts)


if __name__ == "__main__":
    main()
