import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

system_list = [
    "redset",
    "duckdb",
]


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


def get_runtime_by_query_type(df):
    """
    Calculate runtime percentage for each query type.
    Returns a dict with query_type as key and percentage of total runtime as value.
    """
    total_runtime = df["runtime_sec"].sum()

    if total_runtime == 0:
        return {}

    # Get unique query types in the dataframe
    query_types = df["query_type"].unique()

    runtime_percentages = {}
    for query_type in query_types:
        query_runtime = df[df["query_type"] == query_type]["runtime_sec"].sum()
        runtime_percentages[query_type] = (query_runtime / total_runtime) * 100

    return runtime_percentages


def plot1e(
    per_system_dict,
    wl_gen="matching",
    anonymize_systems=False,
    anonymize_dict=None,
    system_color_dict=None,
):
    """
    Plot runtime percentage for each query type across all systems in one plot.
    Shows what percentage of the total runtime each query type (insert, update, delete, select, etc.) represents.
    Each query type is shown as a separate bar, grouped by system.

    Args:
        per_system_dict: Dictionary of systems and their configurations
        wl_gen: Workload generation method (e.g., "matching", "generation")
        anonymize_systems: Whether to anonymize system names
        anonymize_dict: Dictionary mapping system names to anonymized names
        system_color_dict: Dictionary mapping system names to colors
    """
    print(f"Plotting 1E for workload generation method: {wl_gen}")

    # Collect data from all systems
    system_data = {}
    all_query_types = set()

    for sys in system_list:
        if (
            sys in per_system_dict
            and "baseline" in per_system_dict[sys]
            and wl_gen in per_system_dict[sys]["baseline"]
        ):
            baseline = per_system_dict[sys]["baseline"][wl_gen]
            runtime_percentages = get_runtime_by_query_type(baseline)

            if runtime_percentages:
                system_data[sys] = runtime_percentages
                all_query_types.update(runtime_percentages.keys())

    if not system_data:
        print(f"No valid systems found for workload generation method {wl_gen}")
        return

    valid_systems = list(system_data.keys())

    # Sort systems alphabetically by their display name (anonymized or not)
    if anonymize_systems and anonymize_dict:
        # Sort by anonymized names
        valid_systems.sort(key=lambda sys: anonymize_dict.get(sys, sys))
    else:
        # Sort by regular system names
        valid_systems.sort()

    # Order query types: select, insert, update, delete, then any others alphabetically
    preferred_order = ["select", "insert", "update", "delete"]
    query_types = [qt for qt in preferred_order if qt in all_query_types]
    # Add any remaining query types not in preferred_order
    query_types += sorted([qt for qt in all_query_types if qt not in preferred_order])

    # 2x2 subplot: each query type in a separate box
    n_types = len(query_types)
    fig, axs = plt.subplots(2, 2, figsize=(6, 4))
    axs = axs.flatten()

    # Exclude 'redset' from bars; it will be shown as a dashed horizontal line instead
    systems_for_bars = [s for s in valid_systems if s != "redset"]
    has_redset = "redset" in system_data

    for i, qt in enumerate(query_types[:4]):
        ax = axs[i]
        # Compute bar positions and values (excluding 'redset')
        x = range(len(systems_for_bars))
        percentages = [system_data[sys].get(qt, 0) for sys in systems_for_bars]

        # Get colors for each system
        if system_color_dict:
            colors = [system_color_dict.get(sys, None) for sys in systems_for_bars]
        else:
            colors = ["tab:blue"] * len(systems_for_bars)

        # Draw 'redset' as a dashed horizontal line first, behind bars
        redset_pct = None
        redset_color = "#FF6B6B"  # Set to red
        if has_redset:
            redset_pct = system_data["redset"].get(qt, None)
            if redset_pct is not None:
                ax.axhline(
                    y=redset_pct,
                    linestyle="--",
                    color=redset_color,
                    linewidth=1.5,
                    alpha=0.9,
                    zorder=0.5,  # ensure it's behind bars
                )

        # Now draw the bars on top
        ax.bar(x, percentages, color=colors, alpha=0.8, zorder=2)

        # Annotate the redset hline on the right side (after bars so axes limits are set)
        if has_redset and redset_pct is not None:
            # Use blended transform: x in axes fraction, y in data coords
            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                1.01,
                redset_pct,
                f"{redset_pct:.1f}%",
                transform=trans,
                ha="left",
                va="center",
                # color=redset_color,
                fontsize=9,
                zorder=4,
                clip_on=False,
            )

        ax.set_xlabel("System", fontsize=11)
        ax.set_ylabel("Runtime %", fontsize=11)
        ax.set_title(
            f"{qt.upper()}", fontsize=13, fontweight="bold", fontfamily="monospace"
        )
        # Hide x-axis ticks and labels
        ax.set_xticks([])
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Hide unused subplots if <4 query types
    for j in range(n_types, 4):
        fig.delaxes(axs[j])

    # Add legend below the four plots for system colors
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    legend_handles = []

    # Add patch handles for systems shown as bars (excluding 'redset')
    for sys in systems_for_bars:
        if system_color_dict:
            color = system_color_dict.get(sys, "tab:blue")
        else:
            color = "tab:blue"
        if anonymize_systems and anonymize_dict:
            label = anonymize_dict.get(sys, sys)
        else:
            label = sys
        legend_handles.append(mpatches.Patch(color=color, label=label))

    # Add line handle for 'redset'
    if has_redset:
        redset_color = "#FF6B6B"  # Set to red for legend
        if anonymize_systems and anonymize_dict:
            redset_label = anonymize_dict.get("redset", "redset")
        else:
            redset_label = "Redset"
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=redset_color,
                linestyle="--",
                linewidth=1.5,
                label=redset_label,
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(3, len(legend_handles)),
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"output/plot1e_{wl_gen}_2x2.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    # Print summary statistics
    print("\nRuntime Percentage Summary:")
    for sys in valid_systems:
        runtime_percentages = system_data[sys]

        if anonymize_systems and anonymize_dict:
            display_name = anonymize_dict.get(sys, sys).upper()
        else:
            display_name = sys.upper()

        print(f"\n{display_name}:")
        for query_type, percentage in sorted(
            runtime_percentages.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {query_type}: {percentage:.2f}%")
