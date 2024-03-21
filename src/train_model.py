from .data_preparation import get_all_species
import os

import torch
import ase.io
import numpy as np
from tqdm import tqdm
from .utilities import ModelKeeper
import time
import pickle
from torch_geometric.nn import DataParallel

from .hypers import save_hypers, set_hypers_from_files, Hypers, hypers_to_dict
from .pet import PET, PETMLIPWrapper, PETUtilityWrapper
from .utilities import FullLogger, get_scheduler, load_checkpoint, get_data_loaders
from .utilities import get_rmse, get_loss, set_reproducibility, get_calc_names
from .utilities import get_optimizer
from .analysis import adapt_hypers
from .data_preparation import get_self_contributions, get_corrected_energies
import argparse
from .data_preparation import get_pyg_graphs, update_pyg_graphs, get_forces


def fit_pet(
    train_structures,
    val_structures,
    hypers_dict,
    name_of_calculation,
    device,
    output_dir,
):
    TIME_SCRIPT_STARTED = time.time()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    hypers = Hypers(hypers_dict)
    FITTING_SCHEME = hypers.FITTING_SCHEME
    MLIP_SETTINGS = hypers.MLIP_SETTINGS
    ARCHITECTURAL_HYPERS = hypers.ARCHITECTURAL_HYPERS

    if FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS:
        raise ValueError(
            "shift agnostic loss is intended only for general target training"
        )

    ARCHITECTURAL_HYPERS.D_OUTPUT = 1  # energy is a single scalar
    ARCHITECTURAL_HYPERS.TARGET_TYPE = "structural"  # energy is structural property
    ARCHITECTURAL_HYPERS.TARGET_AGGREGATION = (
        "sum"  # energy is a sum of atomic energies
    )

    set_reproducibility(FITTING_SCHEME.RANDOM_SEED, FITTING_SCHEME.CUDA_DETERMINISTIC)

    adapt_hypers(FITTING_SCHEME, train_structures)
    structures = train_structures + val_structures
    all_species = get_all_species(structures)

    name_to_load, NAME_OF_CALCULATION = get_calc_names(
        os.listdir(output_dir), name_of_calculation
    )

    os.mkdir(f"{output_dir}/{NAME_OF_CALCULATION}")
    np.save(f"{output_dir}/{NAME_OF_CALCULATION}/all_species.npy", all_species)
    hypers.UTILITY_FLAGS.CALCULATION_TYPE = "mlip"
    save_hypers(hypers, f"{output_dir}/{NAME_OF_CALCULATION}/hypers_used.yaml")

    print(len(train_structures))
    print(len(val_structures))

    train_graphs = get_pyg_graphs(
        train_structures,
        all_species,
        ARCHITECTURAL_HYPERS.R_CUT,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
    )
    val_graphs = get_pyg_graphs(
        val_structures,
        all_species,
        ARCHITECTURAL_HYPERS.R_CUT,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
    )

    if MLIP_SETTINGS.USE_ENERGIES:
        self_contributions = get_self_contributions(
            MLIP_SETTINGS.ENERGY_KEY, train_structures, all_species
        )
        np.save(
            f"{output_dir}/{NAME_OF_CALCULATION}/self_contributions.npy",
            self_contributions,
        )

        train_energies = get_corrected_energies(
            MLIP_SETTINGS.ENERGY_KEY, train_structures, all_species, self_contributions
        )
        val_energies = get_corrected_energies(
            MLIP_SETTINGS.ENERGY_KEY, val_structures, all_species, self_contributions
        )

        update_pyg_graphs(train_graphs, "y", train_energies)
        update_pyg_graphs(val_graphs, "y", val_energies)

    if MLIP_SETTINGS.USE_FORCES:
        train_forces = get_forces(train_structures, MLIP_SETTINGS.FORCES_KEY)
        val_forces = get_forces(val_structures, MLIP_SETTINGS.FORCES_KEY)

        update_pyg_graphs(train_graphs, "forces", train_forces)
        update_pyg_graphs(val_graphs, "forces", val_forces)

    train_loader, val_loader = get_data_loaders(
        train_graphs, val_graphs, FITTING_SCHEME
    )

    model = PET(ARCHITECTURAL_HYPERS, 0.0, len(all_species)).to(device)
    model = PETUtilityWrapper(model, FITTING_SCHEME.GLOBAL_AUG)

    model = PETMLIPWrapper(model, MLIP_SETTINGS.USE_ENERGIES, MLIP_SETTINGS.USE_FORCES)
    if FITTING_SCHEME.MULTI_GPU and torch.cuda.is_available():
        model = DataParallel(model)
        model = model.to(torch.device("cuda:0"))

    if FITTING_SCHEME.MODEL_TO_START_WITH is not None:
        model.load_state_dict(torch.load(FITTING_SCHEME.MODEL_TO_START_WITH))

    optim = get_optimizer(model, FITTING_SCHEME)
    scheduler = get_scheduler(optim, FITTING_SCHEME)

    if name_to_load is not None:
        load_checkpoint(
            model, optim, scheduler, f"{output_dir}/{name_to_load}/checkpoint"
        )

    history = []
    if MLIP_SETTINGS.USE_ENERGIES:
        energies_logger = FullLogger(FITTING_SCHEME.SUPPORT_MISSING_VALUES)

    if MLIP_SETTINGS.USE_FORCES:
        forces_logger = FullLogger(FITTING_SCHEME.SUPPORT_MISSING_VALUES)

    if MLIP_SETTINGS.USE_FORCES:
        val_forces = torch.cat(val_forces, dim=0)

        sliding_forces_rmse = get_rmse(val_forces.data.cpu().numpy(), 0.0)

        forces_rmse_model_keeper = ModelKeeper()
        forces_mae_model_keeper = ModelKeeper()

    if MLIP_SETTINGS.USE_ENERGIES:
        if FITTING_SCHEME.ENERGIES_LOSS == "per_structure":
            sliding_energies_rmse = get_rmse(val_energies, np.mean(val_energies))
        else:
            val_n_atoms = np.array([len(struc.positions) for struc in val_structures])
            val_energies_per_atom = val_energies / val_n_atoms
            sliding_energies_rmse = get_rmse(
                val_energies_per_atom, np.mean(val_energies_per_atom)
            )

        energies_rmse_model_keeper = ModelKeeper()
        energies_mae_model_keeper = ModelKeeper()

    if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
        multiplication_rmse_model_keeper = ModelKeeper()
        multiplication_mae_model_keeper = ModelKeeper()

    pbar = tqdm(range(FITTING_SCHEME.EPOCH_NUM))

    for epoch in pbar:

        model.train(True)
        for batch in train_loader:
            if not FITTING_SCHEME.MULTI_GPU:
                batch.to(device)

            predictions_energies, predictions_forces = model(
                batch, augmentation=True, create_graph=True
            )
            if FITTING_SCHEME.ENERGIES_LOSS == "per_atom":
                predictions_energies = predictions_energies / batch.n_atoms
                ground_truth_energies = batch.y / batch.n_atoms
            else:
                ground_truth_energies = batch.y

            if MLIP_SETTINGS.USE_ENERGIES:
                energies_logger.train_logger.update(
                    predictions_energies, ground_truth_energies
                )
                loss_energies = get_loss(
                    predictions_energies,
                    ground_truth_energies,
                    FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                    FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
                )
            if MLIP_SETTINGS.USE_FORCES:
                forces_logger.train_logger.update(predictions_forces, batch.forces)
                loss_forces = get_loss(
                    predictions_forces,
                    batch.forces,
                    FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                    FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
                )

            if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
                loss = FITTING_SCHEME.ENERGY_WEIGHT * loss_energies / (
                    sliding_energies_rmse**2
                ) + loss_forces / (sliding_forces_rmse**2)
                loss.backward()

            if MLIP_SETTINGS.USE_ENERGIES and (not MLIP_SETTINGS.USE_FORCES):
                loss_energies.backward()
            if MLIP_SETTINGS.USE_FORCES and (not MLIP_SETTINGS.USE_ENERGIES):
                loss_forces.backward()

            if FITTING_SCHEME.DO_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=FITTING_SCHEME.GRADIENT_CLIPPING_MAX_NORM,
                )
            optim.step()
            optim.zero_grad()

        model.train(False)
        for batch in val_loader:
            if not FITTING_SCHEME.MULTI_GPU:
                batch.to(device)

            predictions_energies, predictions_forces = model(
                batch, augmentation=False, create_graph=False
            )

            if FITTING_SCHEME.ENERGIES_LOSS == "per_atom":
                predictions_energies = predictions_energies / batch.n_atoms
                ground_truth_energies = batch.y / batch.n_atoms
            else:
                ground_truth_energies = batch.y

            if MLIP_SETTINGS.USE_ENERGIES:
                energies_logger.val_logger.update(
                    predictions_energies, ground_truth_energies
                )
            if MLIP_SETTINGS.USE_FORCES:
                forces_logger.val_logger.update(predictions_forces, batch.forces)

        now = {}

        if FITTING_SCHEME.ENERGIES_LOSS == "per_structure":
            energies_key = "energies per structure"
        else:
            energies_key = "energies per atom"

        if MLIP_SETTINGS.USE_ENERGIES:
            now[energies_key] = energies_logger.flush()

        if MLIP_SETTINGS.USE_FORCES:
            now["forces"] = forces_logger.flush()
        now["lr"] = scheduler.get_last_lr()
        now["epoch"] = epoch
        now["elapsed_time"] = time.time() - TIME_SCRIPT_STARTED

        if MLIP_SETTINGS.USE_ENERGIES:
            sliding_energies_rmse = (
                FITTING_SCHEME.SLIDING_FACTOR * sliding_energies_rmse
                + (1.0 - FITTING_SCHEME.SLIDING_FACTOR)
                * now[energies_key]["val"]["rmse"]
            )

            energies_mae_model_keeper.update(
                model, now[energies_key]["val"]["mae"], epoch
            )
            energies_rmse_model_keeper.update(
                model, now[energies_key]["val"]["rmse"], epoch
            )

        if MLIP_SETTINGS.USE_FORCES:
            sliding_forces_rmse = (
                FITTING_SCHEME.SLIDING_FACTOR * sliding_forces_rmse
                + (1.0 - FITTING_SCHEME.SLIDING_FACTOR) * now["forces"]["val"]["rmse"]
            )
            forces_mae_model_keeper.update(model, now["forces"]["val"]["mae"], epoch)
            forces_rmse_model_keeper.update(model, now["forces"]["val"]["rmse"], epoch)

        if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
            multiplication_mae_model_keeper.update(
                model,
                now["forces"]["val"]["mae"] * now[energies_key]["val"]["mae"],
                epoch,
                additional_info=[
                    now[energies_key]["val"]["mae"],
                    now["forces"]["val"]["mae"],
                ],
            )
            multiplication_rmse_model_keeper.update(
                model,
                now["forces"]["val"]["rmse"] * now[energies_key]["val"]["rmse"],
                epoch,
                additional_info=[
                    now[energies_key]["val"]["rmse"],
                    now["forces"]["val"]["rmse"],
                ],
            )

        val_mae_message = "val mae/rmse "
        train_mae_message = "train mae/rmse "

        if MLIP_SETTINGS.USE_ENERGIES:
            val_mae_message += energies_key + ": "
            train_mae_message += energies_key + ": "
            val_mae_message += f" {now[energies_key]['val']['mae']}/{now[energies_key]['val']['rmse']};"
            train_mae_message += f" {now[energies_key]['train']['mae']}/{now[energies_key]['train']['rmse']};"
        if MLIP_SETTINGS.USE_FORCES:
            val_mae_message += "forces per component: "
            train_mae_message += "forces per component: "
            val_mae_message += (
                f" {now['forces']['val']['mae']}/{now['forces']['val']['rmse']}"
            )
            train_mae_message += (
                f" {now['forces']['train']['mae']}/{now['forces']['train']['rmse']}"
            )

        pbar.set_description(
            f"lr: {scheduler.get_last_lr()}; " + val_mae_message + train_mae_message
        )

        history.append(now)
        scheduler.step()
        elapsed = time.time() - TIME_SCRIPT_STARTED
        if FITTING_SCHEME.MAX_TIME is not None:
            if elapsed > FITTING_SCHEME.MAX_TIME:
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        f"{output_dir}/{NAME_OF_CALCULATION}/checkpoint",
    )
    with open(f"{output_dir}/{NAME_OF_CALCULATION}/history.pickle", "wb") as f:
        pickle.dump(history, f)

    def save_model(model_name, model_keeper):
        torch.save(
            model_keeper.best_model.state_dict(),
            f"{output_dir}/{NAME_OF_CALCULATION}/{model_name}_state_dict",
        )

    summary = ""
    if MLIP_SETTINGS.USE_ENERGIES:
        if FITTING_SCHEME.ENERGIES_LOSS == "per_structure":
            postfix = "per structure"
        if FITTING_SCHEME.ENERGIES_LOSS == "per_atom":
            postfix = "per atom"
        save_model("best_val_mae_energies_model", energies_mae_model_keeper)
        summary += f"best val mae in energies {postfix}: {energies_mae_model_keeper.best_error} at epoch {energies_mae_model_keeper.best_epoch}\n"

        save_model("best_val_rmse_energies_model", energies_rmse_model_keeper)
        summary += f"best val rmse in energies {postfix}: {energies_rmse_model_keeper.best_error} at epoch {energies_rmse_model_keeper.best_epoch}\n"

    if MLIP_SETTINGS.USE_FORCES:
        save_model("best_val_mae_forces_model", forces_mae_model_keeper)
        summary += f"best val mae in forces: {forces_mae_model_keeper.best_error} at epoch {forces_mae_model_keeper.best_epoch}\n"

        save_model("best_val_rmse_forces_model", forces_rmse_model_keeper)
        summary += f"best val rmse in forces: {forces_rmse_model_keeper.best_error} at epoch {forces_rmse_model_keeper.best_epoch}\n"

    if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
        save_model("best_val_mae_both_model", multiplication_mae_model_keeper)
        summary += f"best both (multiplication) mae in energies {postfix}: {multiplication_mae_model_keeper.additional_info[0]} in forces: {multiplication_mae_model_keeper.additional_info[1]} at epoch {multiplication_mae_model_keeper.best_epoch}\n"

        save_model("best_val_rmse_both_model", multiplication_rmse_model_keeper)
        summary += f"best both (multiplication) rmse in energies {postfix}: {multiplication_rmse_model_keeper.additional_info[0]} in forces: {multiplication_rmse_model_keeper.additional_info[1]} at epoch {multiplication_rmse_model_keeper.best_epoch}\n"

    with open(f"{output_dir}/{NAME_OF_CALCULATION}/summary.txt", "w") as f:
        print(summary, file=f)

    print("total elapsed time: ", time.time() - TIME_SCRIPT_STARTED)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "train_structures_path",
        help="Path to an xyz file with train structures",
        type=str,
    )
    parser.add_argument(
        "val_structures_path",
        help="Path to an xyz file with validation structures",
        type=str,
    )
    parser.add_argument(
        "provided_hypers_path",
        help="Path to a YAML file with provided hypers",
        type=str,
    )
    parser.add_argument(
        "default_hypers_path", help="Path to a YAML file with default hypers", type=str
    )
    parser.add_argument(
        "name_of_calculation", help="Name of this calculation", type=str
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_structures = ase.io.read(args.train_structures_path, index=":")
    val_structures = ase.io.read(args.val_structures_path, index=":")

    hypers = set_hypers_from_files(args.provided_hypers_path, args.default_hypers_path)
    name_of_calculation = args.name_of_calculation

    output_dir = "results"

    hypers_dict = hypers_to_dict(hypers)
    fit_pet(
        train_structures,
        val_structures,
        hypers_dict,
        name_of_calculation,
        device,
        output_dir,
    )


if __name__ == "__main__":
    main()
