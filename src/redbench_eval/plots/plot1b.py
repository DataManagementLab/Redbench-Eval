import matplotlib.pyplot as plt

system_list = ["duckdb", "duckdb"]


def get_runtime_sums(df):
    # get sum of total runtime (for all queries, and only select)
    total_runtime = df["runtime_sec"].sum()
    readonly_runtime = df[df["query_type"] == "select"]["runtime_sec"].sum()
    write_runtime = total_runtime - readonly_runtime
    return (
        total_runtime,
        readonly_runtime,
        write_runtime,
    )


config_lookup = {
    "baseline": "No\nCache",
    "result": "Result\nCache",
    "pred": "Pred.\nCache",
    "pred_result": "Pred.\n+ Res.\nCache",
    "subresult": "Subres.\nCache",
    "subresult_result": "Subres.\n+ Res.\nCache",
    "scan": "Scan\nCache",
    "scan_subresult_result": "Subres.\n+ Res.\n+ Scan\nCache",
}

config_order = [
    "baseline",
    "pred",
    "scan",
    "subresult",
    "result",
    "pred_result",
    "subresult_result",
    "scan_subresult_result",
]


def plot1b(
    per_system_dict, wl_gen="matching", anonymize_systems=False, anonymize_dict=None
):
    print(f"Plotting 1B for workload generation method: {wl_gen}")

    # Give the first subplot more room by adjusting the width ratios

    width_ratios = []
    for sys in system_list:
        # count how often wl_gen appears in the dicts
        count = sum(
            1
            for config in per_system_dict[sys]
            if wl_gen in per_system_dict[sys][config]
        )
        width_ratios.append(count if count > 0 else 1)  # avoid zero

    fig, axes = plt.subplots(
        1,
        len(system_list),
        figsize=(3.1 * len(system_list), 2.2),
        gridspec_kw={
            "width_ratios": width_ratios,
            "wspace": 0.035,
        },
        sharey=True,
    )

    for i, (target_system, ax) in enumerate(zip(system_list, axes)):
        if wl_gen not in per_system_dict[target_system]["baseline"]:
            print(
                f"Skipping {target_system} for workload generation method {wl_gen} as no data is available."
            )
            continue

        baseline = per_system_dict[target_system]["baseline"][wl_gen]

        baseline_rt, _, _ = get_runtime_sums(baseline)

        j = 0
        for config in config_order:
            if config not in per_system_dict[target_system]:
                continue
            df_dict = per_system_dict[target_system][config]

            if wl_gen not in df_dict:
                print(
                    f"Skipping {target_system} - {config} for workload generation method {wl_gen} as no data is available."
                )
                j += 1
                continue
            matching_df = df_dict[wl_gen]
            total_rt, readonly_rt, write_rt = get_runtime_sums(matching_df)
            ax.bar(
                config_lookup[config],
                readonly_rt / baseline_rt,
                label="SELECT" if j == 0 else None,
                color="skyblue",
            )
            ax.bar(
                config_lookup[config],
                write_rt / baseline_rt,
                bottom=readonly_rt / baseline_rt,
                label="DML" if j == 0 else None,
                color="orange",
                zorder=2,
            )

            j += 1

        if i == 0:
            ax.set_ylabel("Relative Runtime")

        if anonymize_systems:
            ax.set_title(f"{anonymize_dict[target_system].title()}", fontweight="bold")
        else:
            ax.set_title(f"{target_system.capitalize()}", fontweight="bold")

        # add horizontal line at y=1
        hline_key = "Runtime w/o Caches"
        ax.axhline(
            y=1, color="grey", linestyle="--", linewidth=1, label=hline_key, zorder=1
        )

        # reorder legend to show "Runtime w/o Caches" last
        handles, labels = ax.get_legend_handles_labels()
        if hline_key in labels:
            idx = labels.index(hline_key)
            handles.append(handles.pop(idx))
            labels.append(labels.pop(idx))

        if i == 0:
            ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(1.1, -0.45),
                ncol=len(system_list) + 1,
                fontsize=11,
            )

    plt.savefig(f"output/plot1b_{wl_gen}.pdf", bbox_inches="tight")

    plt.show()
