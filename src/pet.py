import torch
import numpy as np
import torch_geometric
from torch import nn
from typing import Dict, Optional

from .transformer import TransformerLayer, Transformer
from .molecule import batch_to_dict
from .utilities import get_rotations, NeverRun


class CentralSplitter(torch.nn.Module):
    def __init__(self):
        super(CentralSplitter, self).__init__()

    def forward(self, features, central_species):
        all_species = np.unique(central_species)
        all_species = [str(specie) for specie in all_species]

        result = {}
        for specie in all_species:
            result[specie] = {}

        for key, value in features.items():
            for specie in all_species:
                mask_now = central_species == int(specie)
                result[specie][key] = value[mask_now]
        return result


class CentralUniter(torch.nn.Module):
    def __init__(self):
        super(CentralUniter, self).__init__()

    def forward(self, features, central_species):
        all_species = np.unique(central_species)
        all_species = [str(specie) for specie in all_species]
        specie = all_species[0]

        shapes = {}
        for key, value in features[specie].items():
            now = list(value.shape)
            now[0] = 0
            shapes[key] = now

        device = None
        for specie in all_species:
            for key, value in features[specie].items():
                num = features[specie][key].shape[0]
                device = features[specie][key].device
                shapes[key][0] += num

        result = {
            key: torch.empty(shape, dtype=torch.get_default_dtype()).to(device)
            for key, shape in shapes.items()
        }

        for specie in features.keys():
            for key, value in features[specie].items():
                mask = int(specie) == central_species
                result[key][mask] = features[specie][key]

        return result


def cutoff_func(grid: torch.Tensor, r_cut: float, delta: float):
    mask_bigger = grid >= r_cut
    mask_smaller = grid <= r_cut - delta
    grid = (grid - r_cut + delta) / delta
    f = 1 / 2.0 + torch.cos(np.pi * grid) / 2.0

    f[mask_bigger] = 0.0
    f[mask_smaller] = 1.0
    return f


def get_activation(hypers):
    if hypers.ACTIVATION == "mish":
        return nn.Mish()
    if hypers.ACTIVATION == "silu":
        return nn.SiLU()
    raise ValueError("unknown activation")


class CartesianTransformer(torch.nn.Module):
    def __init__(
        self,
        hypers,
        d_model,
        n_head,
        dim_feedforward,
        n_layers,
        dropout,
        n_atomic_species,
        add_central_token,
        is_first,
    ):

        super(CartesianTransformer, self).__init__()
        self.hypers = hypers
        self.is_first = is_first
        self.trans_layer = TransformerLayer(
            d_model=d_model,
            n_heads=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=get_activation(hypers),
            transformer_type=hypers.TRANSFORMER_TYPE,
        )
        self.trans = Transformer(self.trans_layer, num_layers=n_layers)

        if hypers.USE_ONLY_LENGTH:
            input_dim = 1
        else:
            input_dim = 3
            if hypers.USE_LENGTH:
                input_dim += 1

        if hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES:
            input_dim += hypers.SCALAR_ATTRIBUTES_SIZE

        if hypers.R_EMBEDDING_ACTIVATION:
            self.r_embedding = nn.Sequential(
                nn.Linear(input_dim, d_model), get_activation(hypers)
            )
        else:
            self.r_embedding = nn.Linear(input_dim, d_model)

        if hypers.BLEND_NEIGHBOR_SPECIES and (not is_first):
            n_merge = 3
        else:
            n_merge = 2

        self.compress = None
        if hypers.COMPRESS_MODE == "linear":
            self.compress = nn.Linear(n_merge * d_model, d_model)
        if hypers.COMPRESS_MODE == "mlp":
            self.compress = nn.Sequential(
                nn.Linear(n_merge * d_model, d_model),
                get_activation(hypers),
                nn.Linear(d_model, d_model),
            )
        if self.compress is None:
            raise ValueError("unknown compress mode")

        self.neighbor_embedder = NeverRun()  # for torchscript
        if hypers.BLEND_NEIGHBOR_SPECIES and (not is_first):
            self.neighbor_embedder = nn.Embedding(n_atomic_species + 1, d_model)

        self.add_central_token = add_central_token

        self.central_embedder = NeverRun()  # for torchscript
        self.central_scalar_embedding = NeverRun()  # for torchscript
        self.central_compress = NeverRun()  # for torchscript

        if add_central_token:
            self.central_embedder = nn.Embedding(n_atomic_species + 1, d_model)
            if hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES:
                if hypers.R_EMBEDDING_ACTIVATION:
                    self.central_scalar_embedding = nn.Sequential(
                        nn.Linear(hypers.SCALAR_ATTRIBUTES_SIZE, d_model),
                        get_activation(hypers),
                    )
                else:
                    self.central_scalar_embedding = nn.Linear(
                        hypers.SCALAR_ATTRIBUTES_SIZE, d_model
                    )

                if hypers.COMPRESS_MODE == "linear":
                    self.central_compress = nn.Linear(2 * d_model, d_model)
                if hypers.COMPRESS_MODE == "mlp":
                    self.central_compress = nn.Sequential(
                        nn.Linear(2 * d_model, d_model),
                        get_activation(hypers),
                        nn.Linear(d_model, d_model),
                    )

        # assign hypers one by one for torch.script
        self.USE_LENGTH = hypers.USE_LENGTH
        self.BLEND_NEIGHBOR_SPECIES = hypers.BLEND_NEIGHBOR_SPECIES
        self.USE_ADDITIONAL_SCALAR_ATTRIBUTES = hypers.USE_ADDITIONAL_SCALAR_ATTRIBUTES
        self.USE_ONLY_LENGTH = hypers.USE_ONLY_LENGTH
        self.R_CUT = hypers.R_CUT
        self.CUTOFF_DELTA = hypers.CUTOFF_DELTA

    def forward(self, batch_dict: Dict[str, torch.Tensor]):

        x = batch_dict["x"]

        if self.USE_LENGTH:
            neighbor_lengths = torch.sqrt(torch.sum(x**2, dim=2) + 1e-15)[:, :, None]
        else:
            neighbor_lengths = torch.empty(
                0, dtype=x.dtype, device=x.device
            )  # for torch script

        central_species = batch_dict["central_species"]
        neighbor_species = batch_dict["neighbor_species"]
        input_messages = batch_dict["input_messages"]
        mask = batch_dict["mask"]
        batch = batch_dict["batch"]
        nums = batch_dict["nums"]

        if self.BLEND_NEIGHBOR_SPECIES and (not self.is_first):
            neighbor_embedding = self.neighbor_embedder(neighbor_species)
        else:
            neighbor_embedding = torch.empty(
                0, dtype=x.dtype, device=x.device
            )  # for torch script

        if self.USE_ADDITIONAL_SCALAR_ATTRIBUTES:
            neighbor_scalar_attributes = batch_dict["neighbor_scalar_attributes"]
            central_scalar_attributes = batch_dict["central_scalar_attributes"]
        else:
            neighbor_scalar_attributes = torch.empty(
                0, dtype=x.dtype, device=x.device
            )  # for torch script
            central_scalar_attributes = torch.empty(
                0, dtype=x.dtype, device=x.device
            )  # for torch script

        initial_n_tokens = x.shape[1]
        max_number = int(torch.max(nums))

        if self.USE_ONLY_LENGTH:
            coordinates = [neighbor_lengths]
        else:
            coordinates = [x]
            if self.USE_LENGTH:
                coordinates.append(neighbor_lengths)

        if self.USE_ADDITIONAL_SCALAR_ATTRIBUTES:
            coordinates.append(neighbor_scalar_attributes)
        coordinates = torch.cat(coordinates, dim=2)
        coordinates = self.r_embedding(coordinates)

        if self.BLEND_NEIGHBOR_SPECIES and (not self.is_first):
            tokens = torch.cat([coordinates, neighbor_embedding, input_messages], dim=2)
        else:
            tokens = torch.cat([coordinates, input_messages], dim=2)

        tokens = self.compress(tokens)

        if self.add_central_token:
            central_specie_embedding = self.central_embedder(central_species)
            if self.USE_ADDITIONAL_SCALAR_ATTRIBUTES:
                central_scalar_embedding = self.central_scalar_embedding(
                    central_scalar_attributes
                )
                central_token = torch.cat(
                    [central_specie_embedding, central_scalar_embedding], dim=1
                )
                central_token = self.central_compress(central_token)
            else:
                central_token = central_specie_embedding

            tokens = torch.cat([central_token[:, None, :], tokens], dim=1)

            submask = torch.zeros(mask.shape[0], dtype=torch.bool).to(mask.device)
            total_mask = torch.cat([submask[:, None], mask], dim=1)

            lengths = torch.sqrt(torch.sum(x * x, dim=2) + 1e-16)
            multipliers = cutoff_func(lengths, self.R_CUT, self.CUTOFF_DELTA)
            sub_multipliers = torch.ones(mask.shape[0], device=mask.device)
            multipliers = torch.cat([sub_multipliers[:, None], multipliers], dim=1)
            multipliers[total_mask] = 0.0

            multipliers = multipliers[:, None, :]
            multipliers = multipliers.repeat(1, multipliers.shape[2], 1)

            output_messages = self.trans(
                tokens[:, : (max_number + 1), :],
                multipliers=multipliers[:, : (max_number + 1), : (max_number + 1)],
            )
            if max_number < initial_n_tokens:
                padding = torch.zeros(
                    output_messages.shape[0],
                    initial_n_tokens - max_number,
                    output_messages.shape[2],
                    device=output_messages.device,
                )
                output_messages = torch.cat([output_messages, padding], dim=1)

            return {
                "output_messages": output_messages[:, 1:, :],
                "central_token": output_messages[:, 0, :],
            }
        else:

            lengths = torch.sqrt(torch.sum(x * x, dim=2) + 1e-16)

            multipliers = cutoff_func(lengths, self.R_CUT, self.CUTOFF_DELTA)
            multipliers[mask] = 0.0

            multipliers = multipliers[:, None, :]
            multipliers = multipliers.repeat(1, multipliers.shape[2], 1)

            output_messages = self.trans(
                tokens[:, :max_number, :],
                multipliers=multipliers[:, :max_number, :max_number],
            )
            if max_number < initial_n_tokens:
                padding = torch.zeros(
                    output_messages.shape[0],
                    initial_n_tokens - max_number,
                    output_messages.shape[2],
                    device=output_messages.device,
                )
                output_messages = torch.cat([output_messages, padding], dim=1)

            return {"output_messages": output_messages}


class CentralSpecificModel(torch.nn.Module):
    def __init__(self, models):
        super(CentralSpecificModel, self).__init__()
        self.models = torch.nn.ModuleDict(models)
        self.splitter = CentralSplitter()
        self.uniter = CentralUniter()

    def forward(self, batch_dict):
        central_indices = batch_dict["central_species"].data.cpu().numpy()
        splitted = self.splitter(batch_dict, central_indices)

        result = {}
        for key in splitted.keys():
            result[str(key)] = self.models[str(key)](splitted[key])

        result = self.uniter(result, central_indices)
        return result


class Head(torch.nn.Module):
    def __init__(self, hypers, n_in, n_neurons):
        super(Head, self).__init__()
        self.hypers = hypers
        self.nn = nn.Sequential(
            nn.Linear(n_in, n_neurons),
            get_activation(hypers),
            nn.Linear(n_neurons, n_neurons),
            get_activation(hypers),
            nn.Linear(n_neurons, hypers.D_OUTPUT),
        )

    def forward(self, batch_dict: Dict[str, torch.Tensor]):
        pooled = batch_dict["pooled"]
        outputs = self.nn(pooled)
        return {"atomic_predictions": outputs}


class CentralTokensPredictor(torch.nn.Module):
    def __init__(self, hypers, head):
        super(CentralTokensPredictor, self).__init__()
        self.head = head
        self.hypers = hypers

    def forward(self, central_tokens: torch.Tensor, central_species: torch.Tensor):
        predictions = self.head(
            {"pooled": central_tokens, "central_species": central_species}
        )["atomic_predictions"]
        return predictions


class MessagesPredictor(torch.nn.Module):
    def __init__(self, hypers, head):
        super(MessagesPredictor, self).__init__()
        self.head = head
        self.AVERAGE_POOLING = hypers.AVERAGE_POOLING

    def forward(
        self,
        messages: torch.Tensor,
        mask: torch.Tensor,
        nums: torch.Tensor,
        central_species: torch.Tensor,
        multipliers: torch.Tensor,
    ):
        messages_proceed = messages * multipliers[:, :, None]

        messages_proceed[mask] = 0.0
        if self.AVERAGE_POOLING:
            pooled = messages_proceed.sum(dim=1) / nums[:, None]
        else:
            pooled = messages_proceed.sum(dim=1)

        predictions = self.head({"pooled": pooled, "central_species": central_species})[
            "atomic_predictions"
        ]
        return predictions


class MessagesBondsPredictor(torch.nn.Module):
    def __init__(self, hypers, head):
        super(MessagesBondsPredictor, self).__init__()
        self.head = head
        self.AVERAGE_BOND_ENERGIES = hypers.AVERAGE_BOND_ENERGIES

    def forward(
        self,
        messages: torch.Tensor,
        mask: torch.Tensor,
        nums: torch.Tensor,
        central_species: torch.Tensor,
    ):
        predictions = self.head(
            {"pooled": messages, "central_species": central_species}
        )["atomic_predictions"]

        mask_expanded = mask[..., None].repeat(1, 1, predictions.shape[2])
        predictions = torch.where(mask_expanded, 0.0, predictions)
        if self.AVERAGE_BOND_ENERGIES:
            result = predictions.sum(dim=1) / nums
        else:
            result = predictions.sum(dim=1)
        return result


class PET(torch.nn.Module):
    def __init__(self, hypers, transformer_dropout, n_atomic_species):
        super(PET, self).__init__()
        self.hypers = hypers
        transformer_d_model = hypers.TRANSFORMER_D_MODEL
        transformer_n_head = hypers.TRANSFORMER_N_HEAD
        transformer_dim_feedforward = hypers.TRANSFORMER_DIM_FEEDFORWARD
        transformer_n_layers = hypers.N_TRANS_LAYERS
        n_gnn_layers = hypers.N_GNN_LAYERS
        head_n_neurons = hypers.HEAD_N_NEURONS
        transformers_central_specific = hypers.TRANSFORMERS_CENTRAL_SPECIFIC
        heads_central_specific = hypers.HEADS_CENTRAL_SPECIFIC

        add_central_tokens = []
        for _ in range(hypers.N_GNN_LAYERS - 1):
            add_central_tokens.append(hypers.ADD_TOKEN_FIRST)
        add_central_tokens.append(hypers.ADD_TOKEN_SECOND)

        self.embedding = nn.Embedding(n_atomic_species + 1, transformer_d_model)
        gnn_layers = []
        if transformers_central_specific:
            for layer_index in range(n_gnn_layers):
                if layer_index == 0:
                    is_first = True
                else:
                    is_first = False
                models = {
                    str(i): CartesianTransformer(
                        hypers,
                        transformer_d_model,
                        transformer_n_head,
                        transformer_dim_feedforward,
                        transformer_n_layers,
                        transformer_dropout,
                        n_atomic_species,
                        add_central_tokens[layer_index],
                        is_first,
                    )
                    for i in range(len(all_species))
                }

                gnn_layers.append(CentralSpecificModel(models))
        else:
            for layer_index in range(n_gnn_layers):
                if layer_index == 0:
                    is_first = True
                else:
                    is_first = False
                model = CartesianTransformer(
                    hypers,
                    transformer_d_model,
                    transformer_n_head,
                    transformer_dim_feedforward,
                    transformer_n_layers,
                    transformer_dropout,
                    n_atomic_species,
                    add_central_tokens[layer_index],
                    is_first,
                )
                gnn_layers.append(model)

        self.gnn_layers = torch.nn.ModuleList(gnn_layers)

        heads = []
        if heads_central_specific:
            for _ in range(n_gnn_layers):
                models = {
                    str(i): Head(hypers, transformer_d_model, head_n_neurons)
                    for i in range(len(all_species))
                }
                heads.append(CentralSpecificModel(models))

            models = {
                str(i): Head(hypers, transformer_d_model, head_n_neurons)
                for i in range(len(all_species))
            }
        else:
            for _ in range(n_gnn_layers):
                heads.append(Head(hypers, transformer_d_model, head_n_neurons))

        self.heads = torch.nn.ModuleList(heads)
        self.central_tokens_predictors = torch.nn.ModuleList(
            [CentralTokensPredictor(hypers, head) for head in heads]
        )
        self.messages_predictors = torch.nn.ModuleList(
            [MessagesPredictor(hypers, head) for head in heads]
        )

        if hypers.USE_BOND_ENERGIES:
            bond_heads = []
            if heads_central_specific:
                for _ in range(n_gnn_layers):
                    models = {
                        str(i): Head(hypers, transformer_d_model, head_n_neurons)
                        for i in range(len(all_species))
                    }
                    bond_heads.append(CentralSpecificModel(models))

                models = {
                    str(i): Head(hypers, transformer_d_model, head_n_neurons)
                    for i in range(len(all_species))
                }
            else:
                for _ in range(n_gnn_layers):
                    bond_heads.append(Head(hypers, transformer_d_model, head_n_neurons))

            self.bond_heads = torch.nn.ModuleList(bond_heads)
            self.messages_bonds_predictors = torch.nn.ModuleList(
                [MessagesBondsPredictor(hypers, head) for head in bond_heads]
            )
        else:
            self.messages_bonds_predictors = torch.nn.ModuleList(
                [NeverRun() for _ in range(n_gnn_layers)]
            )

        self.R_CUT = hypers.R_CUT
        self.CUTOFF_DELTA = hypers.CUTOFF_DELTA
        self.USE_BOND_ENERGIES = hypers.USE_BOND_ENERGIES
        self.TARGET_TYPE = hypers.TARGET_TYPE
        self.TARGET_AGGREGATION = hypers.TARGET_AGGREGATION
        self.N_GNN_LAYERS = hypers.N_GNN_LAYERS

    def get_predictions(self, batch_dict: Dict[str, torch.Tensor]):

        x = batch_dict["x"]
        central_species = batch_dict["central_species"]
        neighbor_species = batch_dict["neighbor_species"]
        batch = batch_dict["batch"]
        mask = batch_dict["mask"]
        nums = batch_dict["nums"]

        lengths = torch.sqrt(torch.sum(x * x, dim=2) + 1e-16)
        multipliers = cutoff_func(lengths, self.R_CUT, self.CUTOFF_DELTA)

        neighbors_index = batch_dict["neighbors_index"]
        neighbors_pos = batch_dict["neighbors_pos"]

        batch_dict["input_messages"] = self.embedding(neighbor_species)
        atomic_predictions = torch.tensor([0]).to(x.device)

        for layer_index, (
            central_tokens_predictor,
            messages_predictor,
            gnn_layer,
            messages_bonds_predictor,
        ) in enumerate(
            zip(
                self.central_tokens_predictors,
                self.messages_predictors,
                self.gnn_layers,
                self.messages_bonds_predictors,
            )
        ):

            result = gnn_layer(batch_dict)
            output_messages = result["output_messages"]

            # batch_dict['input_messages'] = output_messages[neighbors_index, neighbors_pos]
            new_input_messages = output_messages[neighbors_index, neighbors_pos]
            batch_dict["input_messages"] = 0.5 * (
                batch_dict["input_messages"] + new_input_messages
            )

            if "central_token" in result.keys():
                atomic_predictions = atomic_predictions + central_tokens_predictor(
                    result["central_token"], central_species
                )
            else:
                atomic_predictions = atomic_predictions + messages_predictor(
                    output_messages, mask, nums, central_species, multipliers
                )

            if self.USE_BOND_ENERGIES:
                atomic_predictions = atomic_predictions + messages_bonds_predictor(
                    output_messages, mask, nums, central_species
                )

        if self.TARGET_TYPE == "structural":
            if self.TARGET_AGGREGATION == "sum":
                return torch_geometric.nn.global_add_pool(
                    atomic_predictions, batch=batch_dict["batch"]
                )
            if self.TARGET_AGGREGATION == "mean":
                return torch_geometric.nn.global_mean_pool(
                    atomic_predictions, batch=batch_dict["batch"]
                )
            raise ValueError("unknown target aggregation")
        if self.TARGET_TYPE == "atomic":
            return atomic_predictions
        raise ValueError("unknown target type")

    def forward(
        self,
        batch_dict: Dict[str, torch.Tensor],
        rotations: Optional[torch.Tensor] = None,
    ):
        if rotations is not None:
            x_initial = batch_dict["x"]
            batch_dict["x"] = torch.bmm(x_initial, rotations)
            predictions = self.get_predictions(batch_dict)
            batch_dict["x"] = x_initial
            return predictions
        else:
            return self.get_predictions(batch_dict)


class PETUtilityWrapper(torch.nn.Module):
    """Uncoupling torch.unscriptable logic from the main PET class"""

    def __init__(self, pet_model, global_aug):
        super(PETUtilityWrapper, self).__init__()
        self.pet_model = pet_model
        self.global_aug = global_aug

    def forward(self, batch, augmentation):
        batch_dict = batch_to_dict(batch)
        rotations = None
        if augmentation:
            indices = batch.batch.cpu().data.numpy()
            rotations = torch.tensor(
                get_rotations(indices, global_aug=self.global_aug), dtype=batch.x.dtype
            ).to(batch.x.device)
        return self.pet_model(batch_dict, rotations)


class PETMLIPWrapper(torch.nn.Module):
    def __init__(self, model, use_energies, use_forces):
        super(PETMLIPWrapper, self).__init__()
        self.model = model
        self.use_energies = use_energies
        self.use_forces = use_forces
        if self.model.pet_model.hypers.D_OUTPUT != 1:
            raise ValueError("D_OUTPUT should be 1 for MLIP; energy is a single scalar")
        if self.model.pet_model.hypers.TARGET_TYPE != "structural":
            raise ValueError("TARGET_TYPE should be structural for MLIP")
        if self.model.pet_model.hypers.TARGET_AGGREGATION != "sum":
            raise ValueError("TARGET_AGGREGATION should be sum for MLIP")

    def get_predictions(self, batch, augmentation):
        predictions = self.model(batch, augmentation=augmentation)
        if predictions.shape[-1] != 1:
            raise ValueError("D_OUTPUT should be 1 for MLIP; energy is a single scalar")
        # if predictions.shape[0] != batch.num_graphs:
        #    raise ValueError("model should return a single scalar per structure")
        return predictions[..., 0]

    def forward(self, batch, augmentation, create_graph):

        if self.use_forces:
            batch.x.requires_grad = True
            predictions = self.get_predictions(batch, augmentation)
            grads = torch.autograd.grad(
                predictions,
                batch.x,
                grad_outputs=torch.ones_like(predictions),
                create_graph=create_graph,
            )[0]
            neighbors_index = batch.neighbors_index.transpose(0, 1)
            neighbors_pos = batch.neighbors_pos
            grads_messaged = grads[neighbors_index, neighbors_pos]
            grads[batch.mask] = 0.0
            first = grads.sum(dim=1)
            grads_messaged[batch.mask] = 0.0
            second = grads_messaged.sum(dim=1)
        else:
            predictions = self.get_predictions(batch, augmentation)

        result = []
        if self.use_energies:
            result.append(predictions)
        else:
            result.append(None)

        if self.use_forces:
            result.append(first - second)
        else:
            result.append(None)

        return result
