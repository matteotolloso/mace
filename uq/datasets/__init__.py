"""Dataset builders. Each dataset is declared in ``data/<dataset>/dataset.yaml``
(fidelities, files, build arguments); ``uq data build <dataset>`` runs its builder
module for missing splits only. See docs/datasets.md.

    ani1x_system   ANI-1x split by molecular system (System-ID / System-OOD)
    ani1x_energy   ANI-1x split by within-system energy quantile (Energy-ID / Energy-OOD)
    water_n2p2     n2p2 ``input.data`` -> extxyz train/val/test per level of theory
    water_overlap  overlap of the (unaligned) BLYP and CCSD(T) water splits
"""
