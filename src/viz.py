"""Plotting helpers used by EDA + dashboard."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "viridis"


def correlation_heatmap(df: pd.DataFrame, cols: list[str], title: str = "Feature correlations"):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[cols].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def violin_by_group(df: pd.DataFrame, value_col: str, group_col: str, title: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x=group_col, y=value_col, ax=ax, palette=PALETTE)
    ax.set_title(title or f"{value_col} by {group_col}")
    fig.tight_layout()
    return fig


def comorbidity_network(df: pd.DataFrame, conditions: list[str], save_path: Path | None = None):
    """Co-occurrence graph of binary condition columns."""
    G = nx.Graph()
    for c in conditions:
        G.add_node(c, prevalence=int(df[c].sum()))
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i + 1:]:
            cooccur = int(((df[c1] == 1) & (df[c2] == 1)).sum())
            if cooccur > 0:
                G.add_edge(c1, c2, weight=cooccur)

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    sizes = [G.nodes[n]["prevalence"] * 30 + 200 for n in G.nodes]
    weights = [G.edges[e]["weight"] / 5 for e in G.edges]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#0F766E", alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, edge_color="gray", ax=ax)
    edge_labels = {e: G.edges[e]["weight"] for e in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    ax.set_title("Comorbidity co-occurrence network")
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, G
