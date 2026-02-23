###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# Modified for ensemble epistemic + MVE aleatoric uncertainty
###########################################################################################

import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import ase.io
import numpy as np
import torch
from e3nn import o3

from mace import data
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)

    # ### ENSEMBLE ###
    # Accept one or more model paths. Also allow repeating --model.
    # Examples:
    #   --model model1.model
    #   --model model1.model model2.model model3.model
    #   --model model1.model --model model2.model
    parser.add_argument(
        "--model",
        help="path(s) to model(s). Provide one or multiple paths.",
        required=True,
        nargs="+",
        action="append",
    )
    # ### /ENSEMBLE ###

    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--enable_cueq",
        help="enable cuequivariance acceleration",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compute_bec",
        help="compute BEC",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_contributions",
        help="model outputs energy contributions for each body order, only supported for MACE, not ScaleShiftMACE",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_descriptors",
        help="model outputs MACE descriptors",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--descriptor_num_layers",
        help="number of layers to take descriptors from",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--descriptor_aggregation_method",
        help="method for aggregating node features. None saves descriptors for each atom.",
        choices=["mean", "per_element_mean", None],
        default=None,
    )
    parser.add_argument(
        "--descriptor_invariants_only",
        help="save invariant (l=0) descriptors only",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--return_node_energies",
        help="model outputs MACE node energies",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--head",
        help="Model head used for evaluation",
        type=str,
        required=False,
        default=None,
    )
    return parser.parse_args()


def _flatten_model_args(model_arg: Sequence[Sequence[str]]) -> List[str]:
    model_args = model_arg[0]
    model_paths = []
    #split on spaces and commas, and flatten
    for arg in model_args:
        paths = [p.strip() for p in arg.replace(",", " ").split()]
        model_paths.extend(paths)
    return model_paths



def _select_head_if_needed(
    x: Optional[torch.Tensor],
    head_name: Optional[str],
    model_heads: Optional[List[str]],
) -> Optional[torch.Tensor]:
    """
    If x has a head dimension (typically [B, H] or [N, H]), select one head.
    If x is already [B] or [N], return unchanged.
    """
    if x is None:
        return None
    if x.ndim >= 2 and model_heads is not None and x.shape[1] == len(model_heads):
        idx = 0
        if head_name is not None:
            if head_name not in model_heads:
                raise ValueError(f"Requested head '{head_name}' not in model.heads={model_heads}")
            idx = model_heads.index(head_name)
        return x[:, idx, ...]
    return x


def get_model_output(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    compute_stress: bool,
    compute_bec: bool,
) -> Dict[str, torch.Tensor]:
    forward_args = {"compute_stress": compute_stress}
    if compute_bec:
        forward_args["compute_bec"] = compute_bec
    return model(batch, **forward_args)


def get_ensemble_output(
    models: List[torch.nn.Module],
    batch_dict: Dict[str, torch.Tensor],
    compute_stress: bool,
    compute_bec: bool,
    head_name: Optional[str],
    model_heads: Optional[List[str]],
    need_node_energies: bool,
) -> Dict[str, torch.Tensor]:
    """
    Returns a dict compatible with single-model output, but with:
      - energy_mean = ensemble mean
      - energy_aleatoric_var = mean over models of predicted var
      - energy_epistemic_var = var over models of predicted mean
      - energy_var = total var (ale + epi)
      - forces = mean forces over models
    Node-energy equivalents are returned if present and requested.
    """
    outs = [
        get_model_output(m, batch_dict, compute_stress=compute_stress, compute_bec=compute_bec)
        for m in models
    ]

    base_out = outs[0]  # used for optional extras (descriptors, contributions, BEC, etc.)

    # ---- Energies: means and (aleatoric) variances per model ----
    mus = []
    ale_vars = []
    for out in outs:
        mu = out.get("energy_mean", out.get("energy"))
        mu = _select_head_if_needed(mu, head_name, model_heads)
        if mu is None:
            raise KeyError("Model output has no 'energy' (or 'energy_mean') key.")
        mus.append(mu)

        var = out.get("energy_var", None)
        if var is None:
            logvar = out.get("energy_logvar", None)
            if logvar is not None:
                var = torch.exp(logvar)
        if var is None:
            var = torch.zeros_like(mu) # TODO check this logic
        else:
            var = _select_head_if_needed(var, head_name, model_heads)
        ale_vars.append(var)

    mus_t = torch.stack(mus, dim=0)              # [M, B] (or [M, B, ...])
    mu_ens = torch.mean(mus_t, dim=0)            # [B]
    epi_var = torch.var(mus_t, dim=0, unbiased=False)  # [B]
    ale_var = torch.mean(torch.stack(ale_vars, dim=0), dim=0)  # [B]
    total_var = ale_var + epi_var
    total_logvar = torch.log(total_var + EPS)

    # ---- Forces: average across models ----
    forces_stack = torch.stack([o["forces"] for o in outs], dim=0)  # [M, N, 3]
    forces_mean = torch.mean(forces_stack, dim=0)

    # ---- Stress: average across models if requested and available ----
    stress_mean = None
    if compute_stress and base_out.get("stress", None) is not None:
        stress_stack = torch.stack([o["stress"] for o in outs], dim=0)  # [M, B, 3, 3] typically
        stress_mean = torch.mean(stress_stack, dim=0)

    # Build final output dict
    out_final = dict(base_out)
    out_final["energy"] = mu_ens
    out_final["energy_mean"] = mu_ens

    # Decomposition
    out_final["energy_aleatoric_var"] = ale_var
    out_final["energy_epistemic_var"] = epi_var

    # For backward compatibility with your previous MVE logging:
    # energy_var/std/logvar represent TOTAL predictive uncertainty.
    out_final["energy_var"] = total_var # TODO maybe change names
    out_final["energy_logvar"] = total_logvar

    out_final["forces"] = forces_mean
    if stress_mean is not None:
        out_final["stress"] = stress_mean

    # ---- Node energies (optional) ----
    if need_node_energies:
        node_mus = []
        node_ale_vars = []
        for o in outs:
            nmu = o.get("node_energy_mean", o.get("node_energy"))
            nmu = _select_head_if_needed(nmu, head_name, model_heads)
            if nmu is None:
                raise KeyError(
                    "Requested --return_node_energies but model output lacks 'node_energy'/'node_energy_mean'."
                )
            node_mus.append(nmu)

            nvar = o.get("node_energy_var", None)
            if nvar is None:
                nlogvar = o.get("node_energy_logvar", None)
                if nlogvar is not None:
                    nvar = torch.exp(nlogvar)
            if nvar is None:
                nvar = torch.zeros_like(nmu)
            else:
                nvar = _select_head_if_needed(nvar, head_name, model_heads)
            node_ale_vars.append(nvar)

        node_mus_t = torch.stack(node_mus, dim=0)                 # [M, N]
        node_mu_ens = torch.mean(node_mus_t, dim=0)               # [N]
        node_epi_var = torch.var(node_mus_t, dim=0, unbiased=False)  # [N]
        node_ale_var = torch.mean(torch.stack(node_ale_vars, dim=0), dim=0)  # [N]
        node_total_var = node_ale_var + node_epi_var
        node_total_logvar = torch.log(node_total_var + EPS)

        out_final["node_energy"] = node_mu_ens
        out_final["node_energy_mean"] = node_mu_ens

        out_final["node_energy_aleatoric_var"] = node_ale_var
        out_final["node_energy_epistemic_var"] = node_epi_var

        out_final["node_energy_var"] = node_total_var
        out_final["node_energy_logvar"] = node_total_logvar

    return out_final


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # ### ENSEMBLE ###
    model_paths = _flatten_model_args(args.model)
    if len(model_paths) == 0:
        raise ValueError("No model paths provided.")
    # ### /ENSEMBLE ###

    # Load models
    models: List[torch.nn.Module] = []
    for p in model_paths:
        m = torch.load(f=p, map_location=args.device)
        if m.__class__.__name__ != "MACELES" and args.compute_bec:
            raise ValueError("BEC can only be computed with MACELES model.")
        if args.enable_cueq:
            print(f"Converting model to CuEq for acceleration: {p}")
            m = run_e3nn_to_cueq(m, device=device)
        m = m.to(args.device)
        m.eval()
        for param in m.parameters():
            param.requires_grad = False
        models.append(m)

    model0 = models[0]

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, index=":")
    if args.head is not None:
        for atoms in atoms_list:
            atoms.info["head"] = args.head
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model0.atomic_numbers])

    try:
        heads = model0.heads
    except AttributeError:
        heads = None

    # Basic compatibility checks (recommended)
    for mi, m in enumerate(models[1:], start=1):
        if hasattr(m, "atomic_numbers") and hasattr(model0, "atomic_numbers"):
            if not torch.equal(m.atomic_numbers.cpu(), model0.atomic_numbers.cpu()):
                raise ValueError(f"Model {model_paths[mi]} has different atomic_numbers than {model_paths[0]}")
        if float(m.r_max) != float(model0.r_max):
            raise ValueError(f"Model {model_paths[mi]} has different r_max than {model_paths[0]}")
        if getattr(m, "heads", None) != getattr(model0, "heads", None):
            raise ValueError(f"Model {model_paths[mi]} has different heads than {model_paths[0]}")

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model0.r_max), heads=heads
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    contributions_list = []
    descriptors_list = []
    node_energies_list = []
    stresses_list = []
    bec_list = []
    qs_list = []
    forces_collection = []

    # ### UNCERTAINTY ###
    # TOTAL (written also to legacy energy_var/std/logvar)
    energy_logvar_list = []
    energy_var_list = []

    # Decomposition
    energy_aleatoric_var_list = []
    energy_epistemic_var_list = []

    node_energy_logvar_list = []
    node_energy_var_list = []
    node_energy_aleatoric_var_list = []
    node_energy_epistemic_var_list = []
    # ### /UNCERTAINTY ###

    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()

        output = get_ensemble_output(
            models=models,
            batch_dict=batch_dict,
            compute_stress=args.compute_stress,
            compute_bec=args.compute_bec,
            head_name=args.head,
            model_heads=heads,
            need_node_energies=args.return_node_energies,
        )

        # ptr for splitting must be on CPU for numpy
        ptr = batch.ptr.detach().cpu().numpy()

        # Energies
        energy_mean = output.get("energy_mean", output["energy"])
        energies_list.append(torch_tools.to_numpy(energy_mean))

        # Total uncertainty (compat)
        if output.get("energy_logvar") is not None:
            energy_logvar_list.append(torch_tools.to_numpy(output["energy_logvar"]))
        if output.get("energy_var") is not None:
            energy_var_list.append(torch_tools.to_numpy(output["energy_var"]))

        # Decomposition
        if output.get("energy_aleatoric_var") is not None:
            energy_aleatoric_var_list.append(torch_tools.to_numpy(output["energy_aleatoric_var"]))
        if output.get("energy_epistemic_var") is not None:
            energy_epistemic_var_list.append(torch_tools.to_numpy(output["energy_epistemic_var"]))

        # Stress (ensemble mean if requested)
        if args.compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))

        # BEC: taken from base model output (ensemble BEC is not defined here)
        if args.compute_bec:
            becs = np.split(
                torch_tools.to_numpy(output["BEC"]),
                indices_or_sections=ptr[1:],
                axis=0,
            )
            bec_list.append(becs[:-1])

            qs = np.split(
                torch_tools.to_numpy(output["latent_charges"]),
                indices_or_sections=ptr[1:],
                axis=0,
            )
            qs_list.append(qs[:-1])

        # Contributions/descriptors: from base model output (not ensemble-averaged)
        if args.return_contributions:
            contributions_list.append(torch_tools.to_numpy(output["contributions"]))

        if args.return_descriptors:
            num_layers = args.descriptor_num_layers
            if num_layers == -1:
                num_layers = int(model0.num_interactions)
            irreps_out = o3.Irreps(str(model0.products[0].linear.irreps_out))
            l_max = irreps_out.lmax
            num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
            per_layer_features = [irreps_out.dim for _ in range(int(model0.num_interactions))]
            per_layer_features[-1] = num_invariant_features

            descriptors = output["node_feats"]
            if args.descriptor_invariants_only:
                descriptors = extract_invariant(
                    descriptors,
                    num_layers=num_layers,
                    num_features=num_invariant_features,
                    l_max=l_max,
                )

            to_keep = int(np.sum(per_layer_features[:num_layers]))
            descriptors = descriptors[:, :to_keep].detach().cpu().numpy()

            descriptors = np.split(descriptors, indices_or_sections=ptr[1:], axis=0)
            descriptors_list.extend(descriptors[:-1])

        # Node energies + node uncertainty decomposition (optional)
        if args.return_node_energies:
            node_energy_mean = output.get("node_energy_mean", output["node_energy"])
            node_energies_list.append(
                np.split(torch_tools.to_numpy(node_energy_mean), indices_or_sections=ptr[1:], axis=0)[:-1]
            )

            if output.get("node_energy_logvar") is not None:
                node_energy_logvar_list.append(
                    np.split(torch_tools.to_numpy(output["node_energy_logvar"]), indices_or_sections=ptr[1:], axis=0)[:-1]
                )
            if output.get("node_energy_var") is not None:
                node_energy_var_list.append(
                    np.split(torch_tools.to_numpy(output["node_energy_var"]), indices_or_sections=ptr[1:], axis=0)[:-1]
                )
            if output.get("node_energy_aleatoric_var") is not None:
                node_energy_aleatoric_var_list.append(
                    np.split(torch_tools.to_numpy(output["node_energy_aleatoric_var"]), indices_or_sections=ptr[1:], axis=0)[:-1]
                )
            if output.get("node_energy_epistemic_var") is not None:
                node_energy_epistemic_var_list.append(
                    np.split(torch_tools.to_numpy(output["node_energy_epistemic_var"]), indices_or_sections=ptr[1:], axis=0)[:-1]
                )

        # Forces: ensemble mean already
        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=ptr[1:],
            axis=0,
        )
        forces_collection.append(forces[:-1])

    # Concatenate
    energies = np.concatenate(energies_list, axis=0)

    energy_logvar = np.concatenate(energy_logvar_list, axis=0) if len(energy_logvar_list) > 0 else None
    energy_var = np.concatenate(energy_var_list, axis=0) if len(energy_var_list) > 0 else None

    energy_aleatoric_var = (
        np.concatenate(energy_aleatoric_var_list, axis=0) if len(energy_aleatoric_var_list) > 0 else None
    )
    energy_epistemic_var = (
        np.concatenate(energy_epistemic_var_list, axis=0) if len(energy_epistemic_var_list) > 0 else None
    )

    forces_list = [f for flist in forces_collection for f in flist]
    assert len(atoms_list) == len(energies) == len(forces_list)

    if args.compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(atoms_list) == stresses.shape[0]

    if args.compute_bec:
        bec_list = [becs for sublist in bec_list for becs in sublist]
        qs_list = [qs for sublist in qs_list for qs in sublist]

    if args.return_contributions:
        contributions = np.concatenate(contributions_list, axis=0)
        assert len(atoms_list) == contributions.shape[0]

    if args.return_descriptors:
        assert len(atoms_list) == len(descriptors_list)

    if args.return_node_energies:
        node_energies = np.concatenate(node_energies_list, axis=0)
        assert len(atoms_list) == node_energies.shape[0]

        node_energy_logvar = (
            np.concatenate(node_energy_logvar_list, axis=0) if len(node_energy_logvar_list) > 0 else None
        )
        node_energy_var = (
            np.concatenate(node_energy_var_list, axis=0) if len(node_energy_var_list) > 0 else None
        )
        node_energy_aleatoric_var = (
            np.concatenate(node_energy_aleatoric_var_list, axis=0)
            if len(node_energy_aleatoric_var_list) > 0
            else None
        )
        node_energy_epistemic_var = (
            np.concatenate(node_energy_epistemic_var_list, axis=0)
            if len(node_energy_epistemic_var_list) > 0
            else None
        )
    else:
        node_energies = None
        node_energy_logvar = None
        node_energy_var = None
        node_energy_aleatoric_var = None
        node_energy_epistemic_var = None

    # Store data in atoms objects
    for i, (atoms, energy, forces) in enumerate(zip(atoms_list, energies, forces_list)):
        atoms.calc = None  # crucial

        atoms.info[args.info_prefix + "energy"] = float(energy)
        atoms.arrays[args.info_prefix + "forces"] = forces

        # TOTAL uncertainty (compat keys)
        if energy_var is not None:
            atoms.info[args.info_prefix + "energy_var"] = float(energy_var[i])
            atoms.info[args.info_prefix + "energy_std"] = float(np.sqrt(max(energy_var[i], 0.0)))
        if energy_logvar is not None:
            atoms.info[args.info_prefix + "energy_logvar"] = float(energy_logvar[i])

        # Decomposition
        if energy_aleatoric_var is not None:
            atoms.info[args.info_prefix + "energy_aleatoric_var"] = float(energy_aleatoric_var[i])
            atoms.info[args.info_prefix + "energy_aleatoric_std"] = float(np.sqrt(max(energy_aleatoric_var[i], 0.0)))
        if energy_epistemic_var is not None:
            atoms.info[args.info_prefix + "energy_epistemic_var"] = float(energy_epistemic_var[i])
            atoms.info[args.info_prefix + "energy_epistemic_std"] = float(np.sqrt(max(energy_epistemic_var[i], 0.0)))

        if args.compute_stress:
            atoms.info[args.info_prefix + "stress"] = stresses[i]

        if args.compute_bec:
            atoms.arrays[args.info_prefix + "BEC"] = bec_list[i].reshape(-1, 9)
            atoms.arrays[args.info_prefix + "latent_charges"] = qs_list[i]

        if args.return_contributions:
            atoms.info[args.info_prefix + "BO_contributions"] = contributions[i]

        if args.return_descriptors:
            descriptors = descriptors_list[i]
            if args.descriptor_aggregation_method:
                if args.descriptor_aggregation_method == "mean":
                    descriptors = np.mean(descriptors, axis=0)
                elif args.descriptor_aggregation_method == "per_element_mean":
                    descriptors = {
                        element: np.mean(descriptors[atoms.symbols == element], axis=0).tolist()
                        for element in np.unique(atoms.symbols)
                    }
                atoms.info[args.info_prefix + "descriptors"] = descriptors
            else:
                atoms.arrays[args.info_prefix + "descriptors"] = np.array(descriptors)

        if args.return_node_energies and node_energies is not None:
            atoms.arrays[args.info_prefix + "node_energies"] = node_energies[i]

            if node_energy_var is not None:
                atoms.arrays[args.info_prefix + "node_energy_var"] = node_energy_var[i]
                atoms.arrays[args.info_prefix + "node_energy_std"] = np.sqrt(np.clip(node_energy_var[i], 0.0, None))
            if node_energy_logvar is not None:
                atoms.arrays[args.info_prefix + "node_energy_logvar"] = node_energy_logvar[i]

            if node_energy_aleatoric_var is not None:
                atoms.arrays[args.info_prefix + "node_energy_aleatoric_var"] = node_energy_aleatoric_var[i]
                atoms.arrays[args.info_prefix + "node_energy_aleatoric_std"] = np.sqrt(
                    np.clip(node_energy_aleatoric_var[i], 0.0, None)
                )
            if node_energy_epistemic_var is not None:
                atoms.arrays[args.info_prefix + "node_energy_epistemic_var"] = node_energy_epistemic_var[i]
                atoms.arrays[args.info_prefix + "node_energy_epistemic_std"] = np.sqrt(
                    np.clip(node_energy_epistemic_var[i], 0.0, None)
                )

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
