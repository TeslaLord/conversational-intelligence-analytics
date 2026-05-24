"""Stratified subset selection by outcome x primary_intent x language."""
from __future__ import annotations

import pandas as pd


def stratified_conv_sample(
    turns: pd.DataFrame,
    n: int,
    seed: int,
    exclude: set[str] | None = None,
) -> pd.DataFrame:
    """Return turns belonging to `n` sampled conv_ids, stratified by
    outcome x primary_intent x language at the conversation level.

    If `exclude` is given, those conv_ids are removed from the candidate pool
    before sampling (used for incremental batched runs).
    """
    if exclude:
        turns = turns[~turns["conv_id"].isin(exclude)]
        if turns.empty:
            return turns
    conv_meta = (
        turns.groupby("conv_id")
        .agg(
            outcome=("outcome", "first"),
            primary_intent=("primary_intent", "first"),
            language=("language", "first"),
        )
        .reset_index()
    )
    conv_meta["_stratum"] = (
        conv_meta["outcome"].astype(str)
        + "|" + conv_meta["primary_intent"].astype(str)
        + "|" + conv_meta["language"].astype(str)
    )
    total = len(conv_meta)
    if n >= total:
        chosen = conv_meta["conv_id"].tolist()
    else:
        # proportional with floor of 1 per stratum
        counts = conv_meta["_stratum"].value_counts()
        chosen = []
        rng_state = seed
        for stratum, group in conv_meta.groupby("_stratum"):
            share = max(1, round(n * counts[stratum] / total))
            take = min(share, len(group))
            chosen.extend(
                group.sample(n=take, random_state=rng_state)["conv_id"].tolist()
            )
            rng_state += 1
        # trim or pad to roughly n
        if len(chosen) > n:
            chosen = (
                pd.Series(chosen).sample(n=n, random_state=seed).tolist()
            )
    return turns[turns["conv_id"].isin(set(chosen))].copy()
