from typing import Dict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

system_order = [
    "redset",
    "duckdb",
]

config_lookup = {
    "result": "Result\nCache",
    "pred": "Pred.\nCache",
    "pred_result": "Pred.\n+ Result\nCache",
    "subresult": "Subres.\nCache",
    "subresult_result": "Subres.\n+ Result\nCache",
    "scan": "Scan\nCache",
    "scan_subresult_result": "Scan\n+ Subres.\n+ Result\nCache",
}

wl_gen_translation = {
    "matching": r"$\mathbf{Redbench}$" + "\n(Matching)",
    "generation": r"$\mathbf{Redbench}$" + "\n(Generation)",
    "baseline_round_robin": r"$\mathbf{Baseline}$" + "\n(Round-Robin)",
}


def _filter_df(
    df: pd.DataFrame, analyze_read_only: bool, system: str, config: str, wl_gen: str
):
    # only include configs that have relative_runtime computed
    assert isinstance(df, pd.DataFrame), f"df is not a DataFrame: {type(df)}"
    assert "relative_runtime" in df.columns, (
        f"df has no 'relative_runtime' column: {system}/{wl_gen}"
    )

    if analyze_read_only:
        filtered_df = df[df["query_type"] == "select"]
    else:
        filtered_df = df

    if system == "redset":
        tmp = filtered_df[["relative_runtime", "query_hash"]].copy()
        tmp["runtime_sec"] = filtered_df["runtime_ms"] / 1000
    else:
        tmp = filtered_df[["relative_runtime", "sql"]].copy()
        tmp["runtime_sec"] = filtered_df["runtime_sec"]
    tmp["baseline_runtime_sec"] = filtered_df["baseline_runtime_sec"]
    # drop NAs for plotting
    tmp = tmp[tmp["relative_runtime"].notna()].copy()
    if tmp.empty:
        return

    tmp["system"] = system

    if config == "baseline":
        return

    tmp["config"] = config_lookup[config]
    tmp["wl_gen"] = wl_gen_translation[wl_gen]
    return tmp


def plot1a(
    per_system_dict: Dict,
    anonymize_systems: bool = False,
    anonymize_dict: Dict[str, str] = None,
    system_color_dict: Dict[str, str] = None,
    boxplot: bool = True,
):
    plot_data_readonly = []
    plot_data_all = []

    if anonymize_systems:
        anon_order = sorted(anonymize_dict.values())

        # create tmp order based on anonymized names
        tmp_order = [""] * len(system_order)
        for system in system_order:
            anon_name = anonymize_dict[system]
            index = anon_order.index(anon_name)
            tmp_order[index] = system

        print("Tmp order:", tmp_order)
    else:
        tmp_order = system_order

    for system in tmp_order:
        if system not in per_system_dict:
            continue

        if system == "redset":
            continue

        # extract config dict
        configs = per_system_dict[system]

        # retrieve result cache config
        if system == "???":
            config_key = "subresult_result"
        else:
            config_key = "result"

        config_dict = configs[config_key]
        for wl_gen, df in config_dict.items():
            if "relative_runtime" not in df.columns:
                continue

            res_read = _filter_df(
                df,
                analyze_read_only=True,
                system=system,
                config=config_key,
                wl_gen=wl_gen,
            )
            res = _filter_df(
                df,
                analyze_read_only=False,
                system=system,
                config=config_key,
                wl_gen=wl_gen,
            )

            if res_read is not None:
                plot_data_readonly.append(res_read)
            if res is not None:
                plot_data_all.append(res)

    plot_data_readonly = pd.concat(plot_data_readonly, ignore_index=True)
    plot_data_all = pd.concat(plot_data_all, ignore_index=True)

    # build palette from system_color_dict (keys in plot_df are capitalized)
    palette = {
        k.capitalize(): v
        for k, v in system_color_dict.items()
        # if k.capitalize() in plot_data_all["system"].unique()
    }

    if anonymize_systems:
        # apply anonymization mapping
        plot_data_readonly["system"] = plot_data_readonly["system"].map(anonymize_dict)
        plot_data_all["system"] = plot_data_all["system"].map(anonymize_dict)

        # apply anonymized palette
        palette = {anonymize_dict[k]: v for k, v in system_color_dict.items()}

    # create multiplot with 2 columns
    fig, axes = plt.subplots(
        1, 2, figsize=(8, 3.5), sharey=True, gridspec_kw={"wspace": 0.05}
    )

    def create_plot(
        plot_df, ax, title: str, show_legend: bool = True, legend_pos=(-0.05, -0.3)
    ):
        # small cosmetic tweaks
        plot_df["system"] = plot_df["system"].astype(str).str.title()
        plot_df["config"] = plot_df["config"].astype(str)

        # create boxplot
        # Define order: alphabetically but baseline last
        wl_gen_values = plot_df["wl_gen"].unique()
        wl_gen_order = sorted([w for w in wl_gen_values if "Baseline" not in w])
        baseline_values = [w for w in wl_gen_values if "Baseline" in w]
        wl_gen_order.extend(sorted(baseline_values))

        if boxplot:
            ax = sns.boxplot(
                ax=ax,
                data=plot_df,
                x="wl_gen",
                y="relative_runtime",
                hue="system",
                showfliers=False,
                palette=palette,
                gap=0.1,
                legend="auto" if show_legend else None,
                order=wl_gen_order,
            )
        else:
            avg_speedups = True
            if avg_speedups:
                # barplot: compute total speedup (1 / mean relative_runtime)
                plot_df = (
                    plot_df.groupby(["system", "wl_gen"])["relative_runtime"]
                    .mean()
                    .reset_index()
                )
                plot_df["speedup"] = 1 / plot_df["relative_runtime"]
            else:
                # compute overall speedup for each group
                plot_df = (
                    plot_df.groupby(["system", "wl_gen"])
                    .agg({"baseline_runtime_sec": "sum", "runtime_sec": "sum"})
                    .reset_index()
                )
                plot_df["speedup"] = (
                    plot_df["baseline_runtime_sec"] / plot_df["runtime_sec"]
                )

            ax = sns.barplot(
                ax=ax,
                data=plot_df,
                x="wl_gen",
                y="speedup",
                hue="system",
                palette=palette,
                legend="auto" if show_legend else None,
                order=wl_gen_order,
            )

            # set y-limit to 6
            ylim_max = 5.5
            ax.set_ylim(0, ylim_max)

            # add data labels on top of bars
            for p in ax.patches:
                height = p.get_height()

                if height == 0:
                    continue

                # annotate below the top if bar exceeds y-limit
                ax.annotate(
                    f"{height:.2f}",
                    (
                        p.get_x() + p.get_width() / 2,
                        ylim_max if height > ylim_max else height,
                    ),
                    ha="center",
                    va="top" if height > ylim_max else "bottom",
                    fontsize=11,
                    rotation=90,
                    xytext=(0, -3) if height > ylim_max else (0, 3),
                    textcoords="offset points",
                )

        # relative runtimes are often skewed; a log scale helps visualization
        # try:
        #     ax.set_yscale("log")
        # except Exception:
        #     pass

        ax.set_title(title, fontfamily="monospace", fontweight="bold")
        ax.set_xlabel("")

        if boxplot:
            ax.set_ylabel("Relative runtime\n(runtime / baseline runtime)")
        else:
            ax.set_ylabel("Speedup")
        # plt.xticks(rotation=45, ha='right')

        # add hline at y=1
        ax.axhline(
            y=1,
            color="grey",
            linestyle="--",
            linewidth=1,
            zorder=1,
            label="No Speedup",
        )

        if show_legend:
            ax.legend(
                bbox_to_anchor=legend_pos, loc="upper left", frameon=False, ncol=5
            )

        # show a small summary table under the plot (counts & medians)
        print(f"Summary for {title}:")
        if boxplot:
            summary = (
                plot_df.groupby(["system", "wl_gen"])["relative_runtime"]
                .agg(["count", "median"])
                .reset_index()
            )
            print(summary.sort_values(["wl_gen", "system"], ascending=[False, True]))
        else:
            print(plot_df)

    create_plot(
        plot_data_readonly,
        axes[0],
        "SELECT",
        show_legend=True,
        legend_pos=(-0.1, -0.2),
    )
    create_plot(plot_data_all, axes[1], "SELECT + DML", show_legend=False)

    if boxplot:
        plot_type = "boxplot"
    else:
        plot_type = "barplot"
    plt.savefig(f"output/plot1a_{plot_type}.pdf", bbox_inches="tight")
    plt.show()
