import os
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
    # "matching": r"Redbench V2" + "\n(Matching)",
    "generation": r"Production" + "\n" + r"Workloads",
    "baseline_round_robin": r"Traditional" + "\n" + r"Benchmarks",
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
    else:
        tmp = filtered_df[["relative_runtime", "sql"]].copy()
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


def plot0(
    per_system_dict: Dict,
    anonymize_systems: bool = False,
    anonymize_dict: Dict[str, str] = None,
    system_color_dict: Dict[str, str] = None,
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

        if system in ["redset"]:
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
                print(f"Skipping {system}/{wl_gen} due to missing relative_runtime")
                continue

            if "matching" in wl_gen:
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

            assert len(res) > 0, f"No data for {system}/{wl_gen}"

            if res_read is not None:
                plot_data_readonly.append(res_read)
            if res is not None:
                plot_data_all.append(res)

    plot_data_all = pd.concat(plot_data_all, ignore_index=True)
    plot_data_readonly = pd.concat(plot_data_readonly, ignore_index=True)

    # compute total speedup (sum of baseline runtimes / sum of cached runtimes)
    def compute_speedup(plot_df):
        # speedup = 1 / mean(relative_runtime)
        # since relative_runtime = runtime / baseline_runtime
        # total_speedup = sum(baseline_runtime) / sum(runtime)
        #               = sum(baseline_runtime) / sum(relative_runtime * baseline_runtime)
        #               = 1 / mean(relative_runtime)
        speedup = (
            plot_df.groupby(["system", "wl_gen"])["relative_runtime"]
            .apply(lambda x: 1 / x.mean())
            .reset_index()
        )
        speedup.columns = ["system", "wl_gen", "speedup"]

        # sort by system
        speedup = speedup.sort_values(by=["system"], ascending=[True])

        return speedup

    speedup_readonly = compute_speedup(plot_data_readonly)
    speedup_all = compute_speedup(plot_data_all)

    # build palette from system_color_dict
    palette = {k.capitalize(): v for k, v in system_color_dict.items()}

    if anonymize_systems:
        # apply anonymization mapping
        speedup_readonly["system"] = speedup_readonly["system"].map(anonymize_dict)
        speedup_all["system"] = speedup_all["system"].map(anonymize_dict)

        # apply anonymized palette
        palette = {anonymize_dict[k]: v for k, v in system_color_dict.items()}

    # create multiplot with 2 columns
    fig, axes = plt.subplots(
        1, 2, figsize=(6, 2.2), sharey=True, gridspec_kw={"wspace": 0.05}
    )

    def create_plot(
        plot_df, ax, title: str, show_legend: bool = True, legend_pos=(-0.05, -0.3)
    ):
        # sort by system
        plot_df = plot_df.sort_values(by=["system"], ascending=[True])

        # small cosmetic tweaks
        plot_df["system"] = plot_df["system"].astype(str).str.title()

        # Define order: alphabetically but baseline last
        wl_gen_values = plot_df["wl_gen"].unique()
        wl_gen_order = sorted([w for w in wl_gen_values if "Production" not in w])
        baseline_values = [w for w in wl_gen_values if "Production" in w]
        wl_gen_order.extend(sorted(baseline_values))

        # create barplot for speedup
        ax = sns.barplot(
            ax=ax,
            data=plot_df,
            x="wl_gen",
            y="speedup",
            hue="system",
            palette=palette,
            gap=0.1,
            legend="auto" if show_legend else None,
            order=wl_gen_order,
        )

        ax.set_title(title, fontfamily="monospace")
        ax.set_xlabel("")
        ax.set_ylabel("Result-Caching Speedup")

        # add hline at y=1
        ax.axhline(
            y=1, color="grey", linestyle="--", linewidth=1, zorder=1, label="No Speedup"
        )

        if show_legend:
            ax.legend(
                bbox_to_anchor=legend_pos, loc="upper left", frameon=False, ncol=5
            )

        # show summary table
        print(f"Summary for {title}:")
        print(plot_df)

    create_plot(
        speedup_readonly,
        axes[0],
        "Read-Only Worklaod",
        show_legend=True,
        legend_pos=(0.1, -0.23),
    )
    create_plot(speedup_all, axes[1], "Read and Write", show_legend=False)

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/plot0.pdf", bbox_inches="tight")
    plt.show()
