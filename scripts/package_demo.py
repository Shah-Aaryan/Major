"""
Demo Packaging Script.

Bundles sample dataset, demo configuration, benchmark outputs, and execution scripts
into a clean, self-contained ZIP archive for client presentation or deployment.
"""

import os
import shutil
import zipfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("package_demo")


def package_demo(output_zip: str = "BeyondAlgo_Demo_Package.zip"):
    root_dir = Path(__file__).resolve().parent.parent
    dist_dir = root_dir / "demo_package"
    
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True)

    files_to_copy = [
        "cli.py",
        "main.py",
        "README.md",
        "RUN_AND_AUTOMATION_GUIDE.md",
        "RESEARCH_VALIDATION_REPORT.md",
        "requirements.txt"
    ]
    
    dirs_to_copy = [
        "analysis",
        "audit",
        "backtesting",
        "config",
        "data",
        "evaluation",
        "features",
        "market_data",
        "optimization",
        "pipeline",
        "realtime",
        "strategies",
        "visualization"
    ]

    logger.info("Copying core files...")
    for f in files_to_copy:
        src = root_dir / f
        if src.exists():
            shutil.copy2(src, dist_dir / f)

    logger.info("Copying modules...")
    for d in dirs_to_copy:
        src = root_dir / d
        if src.exists():
            shutil.copytree(src, dist_dir / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))

    zip_path = root_dir / output_zip
    logger.info(f"Compressing package into {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dist_dir):
            for file in files:
                filepath = Path(root) / file
                arcname = filepath.relative_to(dist_dir)
                zipf.write(filepath, arcname)

    shutil.rmtree(dist_dir)
    logger.info(f"Successfully packaged demo into {zip_path}")
    print(f"\n[SUCCESS] Demo package created: {zip_path}\n")


if __name__ == "__main__":
    package_demo()
