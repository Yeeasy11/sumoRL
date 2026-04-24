import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_matching(src: Path, dst: Path, patterns: list[str]) -> int:
    if not src.exists():
        return 0
    ensure_dir(dst)
    copied = 0
    for pattern in patterns:
        for fp in src.rglob(pattern):
            if fp.is_file():
                target = dst / fp.name
                if target.exists() and fp.resolve() == target.resolve():
                    continue
                shutil.copy2(fp, target)
                copied += 1
    return copied


def copy_tree_files(src: Path, dst: Path, patterns: list[str]) -> int:
    if not src.exists():
        return 0
    copied = 0
    for pattern in patterns:
        for fp in src.rglob(pattern):
            if fp.is_file():
                rel = fp.relative_to(src)
                target = dst / rel
                if target.exists() and fp.resolve() == target.resolve():
                    continue
                ensure_dir(target.parent)
                shutil.copy2(fp, target)
                copied += 1
    return copied


def main() -> None:
    models_dir = ROOT / "models"
    outputs_dir = ROOT / "outputs"
    logs_dir = ROOT / "logs"

    models_repro = models_dir / "repro"
    figures_repro = outputs_dir / "figures_repro"
    eval_repro = outputs_dir / "eval_repro"
    reports_repro = outputs_dir / "reports_repro"
    logs_repro = logs_dir / "repro"

    ensure_dir(models_repro)
    ensure_dir(figures_repro)
    ensure_dir(eval_repro)
    ensure_dir(reports_repro)
    ensure_dir(logs_repro)

    c1 = copy_matching(models_dir, models_repro, ["ppo_*.zip", "dqn_*.zip"])
    c2 = copy_matching(models_dir, models_repro / "ablation_single", ["ppo_ablation_*.zip"])

    c3 = copy_tree_files(
        outputs_dir / "2way-single-intersection" / "ablation",
        eval_repro / "2way-single-intersection" / "ablation_legacy",
        ["*.csv", "*.md", "*.png", "*.xml"],
    )

    c4 = copy_tree_files(
        outputs_dir / "2way-single-intersection" / "ablation",
        figures_repro / "ablation_2way_legacy",
        ["*.png"],
    )

    c5 = copy_tree_files(outputs_dir / "thesis_cn", reports_repro / "thesis_cn_legacy", ["*.csv", "*.md", "*.png"])
    c6 = copy_tree_files(logs_dir / "thesis_cn_2way", logs_repro / "thesis_cn_2way_legacy", ["*.csv", "*.zip", "*.json"])

    print(f"Copied model files to models/repro: {c1}")
    print(f"Copied ablation model files to models/repro/ablation_single: {c2}")
    print(f"Copied 2way ablation artifacts to outputs/eval_repro: {c3}")
    print(f"Copied 2way ablation figures to outputs/figures_repro: {c4}")
    print(f"Copied thesis_cn legacy outputs to outputs/reports_repro: {c5}")
    print(f"Copied thesis_cn_2way legacy logs to logs/repro: {c6}")


if __name__ == "__main__":
    main()
