"""Shared code for the spectrum + formula models. Owner: Member 4.

``features``  numpy-only: spectrum binning, adduct/precursor features, formula counts.
``data``      parquet access (pyarrow) and Morgan fingerprints (RDKit).
``team``      adapter to Member 3's ``casmi_pipeline``: the team's split, pool, ranking and scorer.
``model``     the torch network.
"""
