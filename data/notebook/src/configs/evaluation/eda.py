"""
eda.py
======
Exploratory Data Analysis (EDA) module for the IoT Predictive Maintenance project.

This module provides an EDAAnalyser class that accepts a pandas DataFrame
(raw or preprocessed) and produces a comprehensive, publication-quality
analysis suite covering:

    1. Dataset overview       — shape, dtypes, memory usage
    2. Summary statistics     — descriptive stats for numerical & categorical cols
    3. Missing value analysis — tabular report + heatmap visualisation
    4. Feature distributions  — per-column histograms with KDE overlays
    5. Box plots              — per-column spread & outlier visualisation
    6. Correlation heatmap    — Pearson correlation matrix with annotation
    7. Target distribution    — class balance chart (optional)

All plots are saved to the configured output directory (default: outputs/plots/).
Each figure is also returned so callers can display them interactively in
Jupyter notebooks.

Part of the Infotact Solutions Data Science & Machine Learning Internship project.

Note:
    This module is read-only — it never modifies the input DataFrame.
    Model training and feature engineering are intentionally out-of-scope.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Suppress non-critical matplotlib / seaborn warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Centralised configuration defaults
# ---------------------------------------------------------------------------
try:
    from src.configs.config import get_config as _get_cfg
    _cfg = _get_cfg()
    _DEFAULT_PLOTS_DIR    = _cfg.paths.plots_dir
    _DEFAULT_RANDOM_STATE = _cfg.project.random_seed
    _DEFAULT_MAX_COLS     = _cfg.evaluation.eda_max_cols_per_figure
    _DEFAULT_HIST_BINS    = _cfg.evaluation.eda_hist_bins
    _DEFAULT_SAMPLE_ROWS  = _cfg.evaluation.eda_missing_heatmap_sample
except Exception:   # fallback when running module in isolation
    _DEFAULT_PLOTS_DIR    = "outputs/plots"
    _DEFAULT_RANDOM_STATE = 42
    _DEFAULT_MAX_COLS     = 20
    _DEFAULT_HIST_BINS    = 30
    _DEFAULT_SAMPLE_ROWS  = 300

# ---------------------------------------------------------------------------
# Global aesthetic defaults
# ---------------------------------------------------------------------------
_PALETTE      = "muted"          # seaborn colour palette
_FIG_DPI      = 150              # saved figure resolution
_GRID_ALPHA   = 0.35             # grid line opacity
_SPINE_COLOR  = "#CCCCCC"        # axis spine colour
_TITLE_SIZE   = 13               # plot title font size
_LABEL_SIZE   = 10               # axis label font size
_TICK_SIZE    = 8                # tick label font size

# Apply a clean, consistent style globally
sns.set_theme(style="whitegrid", palette=_PALETTE, font_scale=0.95)
plt.rcParams.update({
    "figure.dpi":        _FIG_DPI,
    "axes.titlesize":    _TITLE_SIZE,
    "axes.labelsize":    _LABEL_SIZE,
    "xtick.labelsize":   _TICK_SIZE,
    "ytick.labelsize":   _TICK_SIZE,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#FAFAFA",
})


# ---------------------------------------------------------------------------
# EDAAnalyser class
# ---------------------------------------------------------------------------


class EDAAnalyser:
    """
    Performs a structured Exploratory Data Analysis on an IoT sensor DataFrame.

    The class is stateless with respect to the DataFrame — the input is never
    mutated. Plots are saved automatically under :attr:`plots_dir` and returned
    to the caller as ``matplotlib.figure.Figure`` objects for interactive use.

    Attributes:
        df          (pd.DataFrame):  The dataset under analysis (read-only).
        plots_dir   (Path):          Directory where plots are saved.
        target_col  (str | None):    Optional target/label column name.
        num_cols    (List[str]):      Detected numerical column names.
        cat_cols    (List[str]):      Detected categorical column names.
        _saved_paths (List[Path]):   Paths of every plot saved this session.

    Example::

        analyser = EDAAnalyser(
            df=clean_df,
            plots_dir="outputs/plots/",
            target_col="failure",
        )
        analyser.run_full_eda()
    """

    def __init__(
        self,
        df:         pd.DataFrame,
        plots_dir:  Union[str, Path] = _DEFAULT_PLOTS_DIR,
        target_col: Optional[str]    = None,
    ) -> None:
        """
        Initialise the EDAAnalyser.

        Args:
            df (pd.DataFrame):
                The dataset to analyse. A defensive copy is stored internally
                so the caller's object is never modified.
            plots_dir (str | Path):
                Directory where all generated plots are saved.
                Created automatically if it does not exist.
                Defaults to ``"outputs/plots"``.
            target_col (str, optional):
                Name of the target / label column. When supplied, an extra
                target-distribution plot is generated during :meth:`run_full_eda`.

        Raises:
            TypeError:  If *df* is not a ``pd.DataFrame``.
            ValueError: If *target_col* is provided but absent from *df*.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pd.DataFrame, got {type(df).__name__}."
            )
        if target_col is not None and target_col not in df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in DataFrame columns."
            )

        # Store a defensive copy — EDA never mutates the original
        self.df:         pd.DataFrame = df.copy(deep=True)
        self.plots_dir:  Path         = Path(plots_dir).resolve()
        self.target_col: Optional[str] = target_col

        # Ensure the plots directory exists
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Detect column types once at construction time
        self.num_cols: List[str] = (
            df.select_dtypes(include=[np.number]).columns.tolist()
        )
        self.cat_cols: List[str] = (
            df.select_dtypes(include=["object", "category"]).columns.tolist()
        )

        # Track saved figure paths for the session
        self._saved_paths: List[Path] = []

        logger.info(
            "EDAAnalyser ready — %d rows, %d cols "
            "(%d numerical, %d categorical). Plots → %s",
            df.shape[0], df.shape[1],
            len(self.num_cols), len(self.cat_cols),
            self.plots_dir,
        )

    # ------------------------------------------------------------------
    # Convenience orchestrator
    # ------------------------------------------------------------------

    def run_full_eda(self) -> None:
        """
        Execute the complete EDA pipeline in a single call.

        Runs all analysis steps in logical order:
            1. :meth:`display_overview`
            2. :meth:`display_summary_statistics`
            3. :meth:`plot_missing_values`
            4. :meth:`plot_distributions`
            5. :meth:`plot_boxplots`
            6. :meth:`plot_correlation_heatmap`
            7. :meth:`plot_target_distribution`  (only when *target_col* is set)
            8. :meth:`display_saved_paths`

        All plots are saved to :attr:`plots_dir` automatically.
        """
        logger.info("Starting full EDA pipeline …")

        self.display_overview()
        self.display_summary_statistics()
        self.plot_missing_values()
        self.plot_distributions()
        self.plot_boxplots()
        self.plot_correlation_heatmap()

        if self.target_col:
            self.plot_target_distribution()

        self.display_saved_paths()
        logger.info("Full EDA pipeline complete. %d plot(s) saved.", len(self._saved_paths))

    # ------------------------------------------------------------------
    # 1. Dataset overview
    # ------------------------------------------------------------------

    def display_overview(self) -> None:
        """
        Print a concise structural overview of the dataset.

        Displays:
            - Shape (rows × columns)
            - Column names, dtypes, non-null counts, and missing percentages
            - Estimated memory usage
            - Duplicate row count
        """
        df  = self.df
        sep = "─" * 72

        print(f"\n{sep}")
        print("  DATASET OVERVIEW")
        print(sep)
        print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        print(f"  Memory usage   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        print(f"  Duplicate rows : {int(df.duplicated().sum()):,}")
        print(f"  Numerical cols : {len(self.num_cols)}")
        print(f"  Categorical cols: {len(self.cat_cols)}")
        print(sep)

        # Per-column summary table
        print(f"  {'#':<4} {'Column':<30} {'Dtype':<15} {'Non-Null':>9} {'Missing %':>10}")
        print(f"  {'─'*4} {'─'*30} {'─'*15} {'─'*9} {'─'*10}")
        for i, col in enumerate(df.columns, 1):
            non_null   = int(df[col].notna().sum())
            miss_pct   = (df[col].isna().sum() / len(df)) * 100
            dtype_str  = str(df[col].dtype)
            print(
                f"  {i:<4} {col:<30} {dtype_str:<15} "
                f"{non_null:>9,} {miss_pct:>9.2f}%"
            )

        print(f"{sep}\n")
        logger.info("Overview displayed — %d columns profiled.", df.shape[1])

    # ------------------------------------------------------------------
    # 2. Summary statistics
    # ------------------------------------------------------------------

    def display_summary_statistics(self) -> None:
        """
        Print descriptive statistics for both numerical and categorical columns.

        Numerical:  count, mean, std, min, 25th/50th/75th percentile, max.
        Categorical: count, unique, top value, frequency of top value.
        """
        sep = "─" * 72

        # ── Numerical ────────────────────────────────────────────────────
        if self.num_cols:
            print(f"\n{sep}")
            print("  NUMERICAL SUMMARY STATISTICS")
            print(sep)
            stats = self.df[self.num_cols].describe().T
            stats.insert(0, "column", stats.index)
            print(stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print(f"{sep}\n")

        # ── Categorical ───────────────────────────────────────────────────
        if self.cat_cols:
            print(f"\n{sep}")
            print("  CATEGORICAL SUMMARY STATISTICS")
            print(sep)
            cat_stats = self.df[self.cat_cols].describe(include="all").T
            cat_stats.insert(0, "column", cat_stats.index)
            print(cat_stats.to_string(index=False))
            print(f"{sep}\n")

        logger.info("Summary statistics displayed.")

    # ------------------------------------------------------------------
    # 3. Missing value analysis
    # ------------------------------------------------------------------

    def plot_missing_values(self) -> Optional[plt.Figure]:
        """
        Visualise missing values with a two-panel figure.

        Panel 1 — Bar chart: percentage of missing values per column
                              (only columns with ≥ 1 missing value are shown).
        Panel 2 — Heatmap:   binary (present / missing) matrix across all rows,
                              useful for spotting systematic missingness patterns.

        Returns:
            matplotlib.figure.Figure | None:
                The figure object, or ``None`` if no missing values exist.

        Side-effect:
            Saves the figure to ``{plots_dir}/missing_values.png``.
        """
        df = self.df
        missing_counts = df.isna().sum()
        missing_cols   = missing_counts[missing_counts > 0]

        total_missing = int(missing_counts.sum())
        print(f"\n  Missing values — total: {total_missing:,} "
              f"across {len(missing_cols)} column(s).")

        if total_missing == 0:
            print("  ✓ No missing values detected.\n")
            logger.info("No missing values — plot skipped.")
            return None

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, max(4, len(missing_cols) * 0.5 + 2)),
            gridspec_kw={"width_ratios": [1, 2]},
        )
        fig.suptitle("Missing Value Analysis", fontsize=_TITLE_SIZE + 1, fontweight="bold", y=1.01)

        # ── Panel 1: Bar chart ────────────────────────────────────────────
        miss_pct = (missing_cols / len(df) * 100).sort_values(ascending=True)
        colours  = [
            "#E63946" if v > 30 else "#F4A261" if v > 10 else "#A8DADC"
            for v in miss_pct.values
        ]
        axes[0].barh(miss_pct.index, miss_pct.values, color=colours, edgecolor="white")
        axes[0].set_xlabel("Missing (%)", fontsize=_LABEL_SIZE)
        axes[0].set_title("Missing % per Column", fontsize=_TITLE_SIZE)
        axes[0].axvline(x=30, color="#E63946", linestyle="--", linewidth=0.8,
                        alpha=0.6, label=">30% threshold")
        axes[0].legend(fontsize=7)
        axes[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

        # Annotate bars
        for bar, val in zip(axes[0].patches, miss_pct.values):
            axes[0].text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=7,
            )

        # ── Panel 2: Heatmap (sample up to 300 rows for clarity) ─────────
        sample = df[missing_cols.index].isna().astype(int)
        if len(sample) > _DEFAULT_SAMPLE_ROWS:
            sample = sample.sample(_DEFAULT_SAMPLE_ROWS, random_state=_DEFAULT_RANDOM_STATE)

        sns.heatmap(
            sample.T,
            ax=axes[1],
            cmap=["#FAFAFA", "#E63946"],
            cbar=False,
            linewidths=0,
            yticklabels=True,
        )
        axes[1].set_title(
            f"Missingness Pattern (sample of {len(sample)} rows)",
            fontsize=_TITLE_SIZE,
        )
        axes[1].set_xlabel("Row index (sample)", fontsize=_LABEL_SIZE)
        axes[1].tick_params(axis="y", labelsize=7)
        axes[1].tick_params(axis="x", labelbottom=False)

        plt.tight_layout()
        path = self._save_figure(fig, "missing_values.png")
        plt.show()
        return fig

    # ------------------------------------------------------------------
    # 4. Feature distributions
    # ------------------------------------------------------------------

    def plot_distributions(
        self,
        cols:     Optional[List[str]] = None,
        bins:     int                 = 30,
        max_cols: int                 = 20,
    ) -> Optional[plt.Figure]:
        """
        Generate a histogram + KDE overlay for each numerical column.

        Columns are arranged in a grid layout. Each subplot shows:
            - Filled histogram (frequency / density)
            - Gaussian KDE curve
            - Vertical dashed lines for mean and median

        Args:
            cols (List[str], optional):
                Specific numerical columns to plot. Defaults to all :attr:`num_cols`.
            bins (int):
                Number of histogram bins. Defaults to 30.
            max_cols (int):
                Maximum columns to plot in a single call to avoid oversized figures.
                Defaults to 20.

        Returns:
            matplotlib.figure.Figure | None:
                The figure object, or ``None`` if no numerical columns exist.

        Side-effect:
            Saves the figure to ``{plots_dir}/feature_distributions.png``.
        """
        target_cols = cols or self.num_cols
        if not target_cols:
            print("  No numerical columns to plot distributions for.")
            return None

        target_cols = target_cols[:max_cols]
        n   = len(target_cols)
        ncols_grid = min(4, n)
        nrows_grid = (n + ncols_grid - 1) // ncols_grid

        fig, axes = plt.subplots(
            nrows_grid, ncols_grid,
            figsize=(ncols_grid * 4.5, nrows_grid * 3.2),
        )
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        fig.suptitle(
            f"Feature Distributions — {n} Numerical Column(s)",
            fontsize=_TITLE_SIZE + 1, fontweight="bold", y=1.01,
        )

        palette = sns.color_palette(_PALETTE, n)

        for i, (col, ax) in enumerate(zip(target_cols, axes_flat)):
            series = self.df[col].dropna()
            colour = palette[i % len(palette)]

            # Histogram + KDE
            ax.hist(
                series, bins=bins, density=True,
                color=colour, alpha=0.55, edgecolor="white", linewidth=0.4,
            )
            try:
                series.plot.kde(ax=ax, color=colour, linewidth=2)
            except Exception:
                pass  # KDE may fail for near-constant columns

            # Mean & median markers
            mean_val   = series.mean()
            median_val = series.median()
            ax.axvline(mean_val,   color="#E63946", linestyle="--",
                       linewidth=1.2, label=f"mean={mean_val:.2f}")
            ax.axvline(median_val, color="#2A9D8F", linestyle=":",
                       linewidth=1.2, label=f"med={median_val:.2f}")

            ax.set_title(col, fontsize=_LABEL_SIZE, fontweight="semibold")
            ax.set_xlabel("Value", fontsize=_TICK_SIZE)
            ax.set_ylabel("Density", fontsize=_TICK_SIZE)
            ax.legend(fontsize=6, framealpha=0.7)
            ax.grid(axis="y", alpha=_GRID_ALPHA)

        # Hide unused subplots
        for ax in axes_flat[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        path = self._save_figure(fig, "feature_distributions.png")
        plt.show()
        logger.info("Distribution plots saved (%d columns).", n)
        return fig

    # ------------------------------------------------------------------
    # 5. Box plots
    # ------------------------------------------------------------------

    def plot_boxplots(
        self,
        cols:     Optional[List[str]] = None,
        max_cols: int                 = 20,
    ) -> Optional[plt.Figure]:
        """
        Generate box plots for each numerical column to visualise spread and outliers.

        Each subplot shows the IQR box, whiskers (1.5 × IQR), median line,
        and individual outlier points.

        Args:
            cols (List[str], optional):
                Specific numerical columns to plot. Defaults to all :attr:`num_cols`.
            max_cols (int):
                Maximum columns to plot. Defaults to 20.

        Returns:
            matplotlib.figure.Figure | None:
                The figure object, or ``None`` if no numerical columns exist.

        Side-effect:
            Saves the figure to ``{plots_dir}/boxplots.png``.
        """
        target_cols = cols or self.num_cols
        if not target_cols:
            print("  No numerical columns to plot box plots for.")
            return None

        target_cols = target_cols[:max_cols]
        n          = len(target_cols)
        ncols_grid = min(4, n)
        nrows_grid = (n + ncols_grid - 1) // ncols_grid

        fig, axes = plt.subplots(
            nrows_grid, ncols_grid,
            figsize=(ncols_grid * 3.5, nrows_grid * 3.0),
        )
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        fig.suptitle(
            f"Box Plots — Spread & Outliers ({n} Column(s))",
            fontsize=_TITLE_SIZE + 1, fontweight="bold", y=1.01,
        )

        palette = sns.color_palette(_PALETTE, n)

        for i, (col, ax) in enumerate(zip(target_cols, axes_flat)):
            series = self.df[col].dropna()
            colour = palette[i % len(palette)]

            ax.boxplot(
                series,
                patch_artist=True,
                boxprops=dict(facecolor=colour, alpha=0.6),
                medianprops=dict(color="#E63946", linewidth=2),
                whiskerprops=dict(linestyle="--", linewidth=1.2),
                flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                markerfacecolor=colour),
                notch=False,
            )

            # Overlay IQR stats as text
            q1, med, q3 = series.quantile([0.25, 0.50, 0.75]).values
            iqr = q3 - q1
            ax.set_title(col, fontsize=_LABEL_SIZE, fontweight="semibold")
            ax.set_ylabel("Value", fontsize=_TICK_SIZE)
            ax.set_xticks([])
            ax.text(
                0.97, 0.97,
                f"IQR={iqr:.2f}\nmed={med:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )
            ax.grid(axis="y", alpha=_GRID_ALPHA)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        path = self._save_figure(fig, "boxplots.png")
        plt.show()
        logger.info("Box plots saved (%d columns).", n)
        return fig

    # ------------------------------------------------------------------
    # 6. Correlation heatmap
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(
        self,
        cols:      Optional[List[str]] = None,
        method:    Literal["pearson", "spearman", "kendall"] = "pearson",
        max_cols:  int   = 30,
        annot:     bool  = True,
    ) -> Optional[plt.Figure]:
        """
        Generate an annotated correlation matrix heatmap.

        Only numerical columns are included. When the column count exceeds
        *max_cols*, the columns with the highest mean absolute correlation
        to all other columns are selected to keep the figure readable.

        Args:
            cols (List[str], optional):
                Subset of columns to include. Defaults to all :attr:`num_cols`.
            method (str):
                Correlation method — ``"pearson"`` (default), ``"spearman"``,
                or ``"kendall"``.
            max_cols (int):
                Maximum columns to include before auto-selecting the most
                correlated subset. Defaults to 30.
            annot (bool):
                Annotate cells with their correlation coefficient.
                Automatically disabled when more than 20 columns are present
                to avoid clutter. Defaults to ``True``.

        Returns:
            matplotlib.figure.Figure | None:
                The figure object, or ``None`` if fewer than 2 numerical cols exist.

        Side-effect:
            Saves the figure to ``{plots_dir}/correlation_heatmap.png``.
        """
        target_cols = cols or self.num_cols
        if len(target_cols) < 2:
            print("  Need at least 2 numerical columns for a correlation heatmap.")
            return None

        # Prune to the most-correlated subset if too many columns
        if len(target_cols) > max_cols:
            corr_full  = self.df[target_cols].corr(method=method).abs()
            mean_corr  = corr_full.mean().sort_values(ascending=False)
            target_cols = mean_corr.head(max_cols).index.tolist()
            logger.info(
                "Correlation heatmap: auto-selected top %d of %d columns.",
                max_cols, len(self.num_cols),
            )

        corr = self.df[target_cols].corr(method=method)
        n    = len(target_cols)

        # Auto-disable annotations for large grids
        do_annot = annot and (n <= 20)
        fmt      = ".2f" if do_annot else ""

        fig_size = max(8, n * 0.55)
        fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        # Diverging colour map centred at 0
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        mask = np.triu(np.ones_like(corr, dtype=bool))   # upper-triangle mask

        sns.heatmap(
            corr,
            ax=ax,
            mask=mask,
            cmap=cmap,
            vmin=-1.0, vmax=1.0, center=0,
            annot=do_annot, fmt=fmt,
            annot_kws={"size": max(5, 9 - n // 5)},
            linewidths=0.4, linecolor="#E0E0E0",
            square=True,
            cbar_kws={"shrink": 0.75, "label": f"{method.capitalize()} r"},
        )

        ax.set_title(
            f"Correlation Heatmap ({method.capitalize()}) — {n} Features",
            fontsize=_TITLE_SIZE + 1, fontweight="bold", pad=14,
        )
        ax.tick_params(axis="x", rotation=45, labelsize=_TICK_SIZE)
        ax.tick_params(axis="y", rotation=0,  labelsize=_TICK_SIZE)

        plt.tight_layout()
        path = self._save_figure(fig, "correlation_heatmap.png")
        plt.show()
        logger.info("Correlation heatmap saved (%d × %d, method=%s).", n, n, method)
        return fig

    # ------------------------------------------------------------------
    # 7. Target distribution  (optional)
    # ------------------------------------------------------------------

    def plot_target_distribution(
        self,
        target_col: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Visualise the distribution of the target / label column.

        For binary or low-cardinality categorical targets, a styled bar chart
        is produced showing class counts and percentages.
        For continuous targets, a histogram + KDE is shown instead.

        Args:
            target_col (str, optional):
                Column to plot. Falls back to :attr:`target_col` set at
                construction time. If neither is set, returns ``None``.

        Returns:
            matplotlib.figure.Figure | None:
                The figure object, or ``None`` if no target column is available.

        Side-effect:
            Saves the figure to ``{plots_dir}/target_distribution.png``.
        """
        col = target_col or self.target_col
        if col is None:
            logger.info("No target_col set — target distribution plot skipped.")
            return None
        if col not in self.df.columns:
            logger.warning("target_col '%s' not found — skipping.", col)
            return None

        series = self.df[col]
        n_unique = series.nunique()

        fig, ax = plt.subplots(figsize=(max(6, n_unique * 0.8 + 3), 5))

        # --- Categorical / binary target ---
        if series.dtype == "object" or n_unique <= 20:
            counts  = series.value_counts().sort_index()
            total   = counts.sum()
            colours = sns.color_palette(_PALETTE, len(counts))
            bars = ax.bar(
                counts.index.astype(str), counts.values,
                color=colours, edgecolor="white", linewidth=0.8,
            )

            # Annotate with count + percentage
            for bar, (label, count) in zip(bars, counts.items()):
                pct = count / total * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.01,
                    f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=9,
                )

            ax.set_xlabel(col, fontsize=_LABEL_SIZE)
            ax.set_ylabel("Count", fontsize=_LABEL_SIZE)
            ax.set_title(
                f"Target Distribution — '{col}'  (n={total:,})",
                fontsize=_TITLE_SIZE + 1, fontweight="bold",
            )
            ax.grid(axis="y", alpha=_GRID_ALPHA)

        # --- Continuous target ---
        else:
            series_clean = series.dropna()
            ax.hist(series_clean, bins=40, density=True,
                    color="#457B9D", alpha=0.6, edgecolor="white")
            try:
                series_clean.plot.kde(ax=ax, color="#E63946", linewidth=2)
            except Exception:
                pass
            ax.set_xlabel(col, fontsize=_LABEL_SIZE)
            ax.set_ylabel("Density", fontsize=_LABEL_SIZE)
            ax.set_title(
                f"Target Distribution — '{col}'",
                fontsize=_TITLE_SIZE + 1, fontweight="bold",
            )
            ax.grid(axis="y", alpha=_GRID_ALPHA)

        plt.tight_layout()
        path = self._save_figure(fig, "target_distribution.png")
        plt.show()
        logger.info("Target distribution plot saved for column '%s'.", col)
        return fig

    # ------------------------------------------------------------------
    # Saved-paths summary
    # ------------------------------------------------------------------

    def display_saved_paths(self) -> None:
        """
        Print a summary of all plot files saved during the current session.
        """
        if not self._saved_paths:
            print("\n  No plots were saved in this session.\n")
            return

        sep = "─" * 65
        print(f"\n{sep}")
        print(f"  Saved Plots ({len(self._saved_paths)} file(s)) → {self.plots_dir}")
        print(sep)
        for i, path in enumerate(self._saved_paths, 1):
            size_kb = path.stat().st_size / 1024
            print(f"  {i:>2}. {path.name:<40} ({size_kb:.1f} KB)")
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_figure(
        self,
        fig:      plt.Figure,
        filename: str,
        dpi:      int = _FIG_DPI,
    ) -> Path:
        """
        Save *fig* to :attr:`plots_dir` / *filename* and record the path.

        Args:
            fig (plt.Figure): The figure to save.
            filename (str):   Output file name (e.g. ``"heatmap.png"``).
            dpi (int):        Resolution in dots-per-inch. Defaults to 150.

        Returns:
            Path: Absolute path of the saved file.
        """
        output_path = self.plots_dir / filename
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        self._saved_paths.append(output_path)
        logger.debug("Plot saved → %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EDAAnalyser("
            f"shape={self.df.shape}, "
            f"num_cols={len(self.num_cols)}, "
            f"cat_cols={len(self.cat_cols)}, "
            f"target='{self.target_col}', "
            f"plots_dir='{self.plots_dir}')"
        )

    def __str__(self) -> str:
        return (
            f"EDAAnalyser — {self.df.shape[0]:,} rows × {self.df.shape[1]:,} cols "
            f"| {len(self.num_cols)} numerical, {len(self.cat_cols)} categorical "
            f"| plots → {self.plots_dir}"
        )
