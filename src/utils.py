"""
src/utils.py — Logging Setup, Configuration Loader & Error Handling
====================================================================
Provides centralized logging, project configuration management,
and custom exception classes for the Downstream Activation Engine.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Project paths (resolved relative to this file's location)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
LOGS_DIR: Path = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # --- Paths ---
    "data_raw_dir": str(DATA_RAW_DIR),
    "data_processed_dir": str(DATA_PROCESSED_DIR),
    "docs_dir": str(DOCS_DIR),
    "logs_dir": str(LOGS_DIR),

    # --- Synthetic data sizing ---
    "num_campaigns": 10,
    "impression_rows": 50_000,
    "signup_target": 5_000,
    "event_multiplier": 3,          # events per signup on average
    "random_seed": 42,

    # --- Business-rule thresholds ---
    "activation_window_days": 7,
    "activation_rate_target_pct": 25.0,
    "cpau_target_usd": 45.00,
    "vanity_ratio_flag_threshold": 3.0,

    # --- Data-quality injection (for synthetic data) ---
    "duplicate_rate": 0.02,         # ≈ 2 %
    "null_injection_rate": 0.01,    # ≈ 1 %

    # --- File names ---
    "google_ads_file": "google_ads_impressions.csv",
    "meta_ads_file": "meta_ads_clicks.csv",
    "signups_file": "platform_signups.json",
    "events_file": "user_event_logs.csv",
    "master_output_file": "master_activated_campaigns.parquet",
    "data_dictionary_file": "data_dictionary.md",
}


# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------
class PipelineError(Exception):
    """Base exception for all pipeline-related errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.context = context or {}
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            details = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} | Context: {details}"
        return base


class DataIngestionError(PipelineError):
    """Raised when a data file cannot be read or parsed."""


class SchemaValidationError(PipelineError):
    """Raised when a dataset fails schema / type validation."""


class DataQualityError(PipelineError):
    """Raised when data-quality checks detect unacceptable issues."""


class JoinIntegrityError(PipelineError):
    """Raised when a multi-source join loses or duplicates records unexpectedly."""


class FeatureEngineeringError(PipelineError):
    """Raised during derived-column computation failures."""


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
_LOGGERS_INITIALIZED: Dict[str, bool] = {}


def setup_logger(
    name: str = "downstream_activation",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Create (or retrieve) a named logger with console + optional file handlers.

    Parameters
    ----------
    name : str
        Logger name (use module __name__ for per-module loggers).
    level : int
        Logging level (default: INFO).
    log_to_file : bool
        Whether to also write to a log file.
    log_dir : Path | None
        Directory for log files. Defaults to ``<project>/logs/``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if name in _LOGGERS_INITIALIZED:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Structured format: timestamp | level | module | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stdout) — force UTF-8 on Windows to avoid cp1252 errors
    import io
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        target_dir = log_dir or LOGS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp_tag = datetime.now().strftime("%Y%m%d")
        log_file = target_dir / f"{name}_{timestamp_tag}.log"
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS_INITIALIZED[name] = True
    return logger


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------
def load_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return the project configuration dictionary, merged with any overrides.

    Parameters
    ----------
    overrides : dict | None
        Key-value pairs that override defaults.

    Returns
    -------
    dict
        Final merged configuration.
    """
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def ensure_directories(config: Optional[Dict[str, Any]] = None) -> None:
    """Create all required project directories if they don't exist."""
    cfg = config or DEFAULT_CONFIG
    for key in ("data_raw_dir", "data_processed_dir", "docs_dir", "logs_dir"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger = setup_logger()
    cfg = load_config()
    ensure_directories(cfg)
    logger.info("Utils module self-test passed.")
    logger.info("Project root : %s", PROJECT_ROOT)
    logger.info("Config keys  : %s", list(cfg.keys()))
