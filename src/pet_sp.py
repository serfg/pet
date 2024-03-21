import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from .torch_geometric.data import Batch


class PETSP(torch.nn.Module):
    def __init__(
        self,
        model_main,
        model_aux,
        r_cut,
        use_energies,
        use_forces,
        sp_frames_calculator,
        batch_size_sp,
        num_species,
        epsilon=1e-10,
        show_progress=False,
        max_num=None,
        n_aug=None,
    ):
        super(PETSP, self).__init__()
        self.show_progress = show_progress
        self.r_cut = r_cut
        self.use_energies = use_energies
        self.use_forces = use_forces
        self.n_aug = n_aug

        self.max_num = max_num
        self.model_main = model_main
        self.model_aux = model_aux
        self.model_main.task = "energies"
        if self.model_aux is not None:
            self.model_aux.task = "energies"

        self.sp_frames_calculator = sp_frames_calculator
        self.batch_size_sp = batch_size_sp

        self.epsilon = epsilon
        self.num_species = num_species

    def get_all_frames(self, batch):
        all_envs = []
        for env_index in range(batch.x.shape[0]):
            mask_now = torch.logical_not(batch.mask[env_index])
            env_now = batch.x[env_index][mask_now]
            neighbor_species_now = batch.neighbor_species[env_index][mask_now]
            # print('env now shape: ', env_now.shape, 'neighbor_species now shape: ', neighbor_species_now.shape)
            # print(neighbor_species_now)
            central_specie = batch.central_species[env_index]

            env_now = [env_now, neighbor_species_now, central_specie]
            all_envs.append(env_now)

        return self.sp_frames_calculator.get_all_frames_global(
            all_envs, self.r_cut, self.num_species
        )

    def get_all_contributions(self, batch, additional_rotations):
        x_initial = batch.x
        x_initial.requires_grad = True

        batch.x = x_initial
        frames, weights, weight_aux = self.get_all_frames(batch)
        if self.max_num is not None:
            if len(frames) > self.max_num:
                raise ValueError(
                    f"number of frames ({len(frames)}) is bigger than the upper bound provided"
                )

        if self.show_progress:
            print("number of frames now: ", len(frames))
        weight_accumulated = 0.0
        for weight in weights:
            weight_accumulated = weight_accumulated + weight

        total_main_weight = weight_accumulated
        weight_accumulated = weight_accumulated * len(additional_rotations)
        if self.model_aux is not None:
            weight_accumulated = weight_accumulated + weight_aux

        strucs_minibatch_sp = []
        weights_minibatch_sp = []

        num_handled = 0
        for additional_rotation in additional_rotations:
            additional_rotation = additional_rotation.to(batch.x.device)
            for index in range(len(frames)):
                frame = frames[index]
                weight = weights[index]
                # print(x_initial[0, 0, 0], batch.x[0, 0, 0])
                frame = torch.matmul(frame, additional_rotation)
                frame = frame[None]
                frame = frame.repeat(x_initial.shape[0], 1, 1)

                batch_now = batch.clone()
                batch_now.x = torch.bmm(x_initial, frame)

                strucs_minibatch_sp.append(batch_now)
                weights_minibatch_sp.append(weight[None])

                num_handled += 1
                if num_handled == self.batch_size_sp:

                    batch_sp = Batch.from_data_list(strucs_minibatch_sp)
                    weights_minibatch_sp = torch.cat(weights_minibatch_sp)

                    predictions = self.model_main(batch_sp)
                    # print("shapes: ", predictions.shape, weights_minibatch_sp.shape)
                    predictions_accumulated = torch.sum(
                        predictions * weights_minibatch_sp
                    )[None]

                    result = predictions_accumulated / weight_accumulated
                    result.backward()
                    grads = x_initial.grad
                    x_initial.grad = None
                    yield result, grads, len(frames), weight_aux, total_main_weight

                    frames, weights, weight_aux = self.get_all_frames(batch)

                    weight_accumulated = 0.0
                    for weight in weights:
                        weight_accumulated = weight_accumulated + weight
                    weight_accumulated = weight_accumulated * len(additional_rotations)
                    if self.model_aux is not None:
                        weight_accumulated = weight_accumulated + weight_aux

                    strucs_minibatch_sp = []
                    weights_minibatch_sp = []
                    num_handled = 0

        if num_handled > 0:

            batch_sp = Batch.from_data_list(strucs_minibatch_sp)
            weights_minibatch_sp = torch.cat(weights_minibatch_sp)

            predictions = self.model_main(batch_sp)
            # print("shapes: ", predictions.shape, weights_minibatch_sp.shape)
            predictions_accumulated = torch.sum(predictions * weights_minibatch_sp)[
                None
            ]

            result = predictions_accumulated / weight_accumulated
            result.backward()
            grads = x_initial.grad
            x_initial.grad = None

            yield result, grads, len(frames), weight_aux, total_main_weight

        if weight_aux > self.epsilon:
            if self.model_aux is not None:

                frames, weights, weight_aux = self.get_all_frames(batch)

                weight_accumulated = 0.0
                for weight in weights:
                    weight_accumulated = weight_accumulated + weight
                weight_accumulated = weight_accumulated * len(additional_rotations)
                if self.model_aux is not None:
                    weight_accumulated = weight_accumulated + weight_aux

                batch.x = x_initial
                predictions_accumulated = self.model_aux(batch) * weight_aux

                result = predictions_accumulated / weight_accumulated
                result.backward()
                grads = x_initial.grad
                x_initial.grad = None

                yield result, grads, len(frames), weight_aux, total_main_weight

    def forward(self, batch):
        if self.n_aug is None:
            additional_rotations = [torch.eye(3)]
        else:
            additional_rotations = [
                torch.tensor(el) for el in Rotation.random(self.n_aug).as_matrix()
            ]

        predictions_total, forces_predicted_total = 0.0, 0.0
        n_frames = None
        for predictions, grads, n_frames, weight_aux, total_main_weight in tqdm(
            self.get_all_contributions(batch, additional_rotations),
            disable=not self.show_progress,
        ):
            predictions_total += predictions
            if self.use_forces:
                neighbors_index = batch.neighbors_index.transpose(0, 1)
                neighbors_pos = batch.neighbors_pos
                grads_messaged = grads[neighbors_index, neighbors_pos]
                grads[batch.mask] = 0.0
                first = grads.sum(dim=1)
                grads_messaged[batch.mask] = 0.0
                second = grads_messaged.sum(dim=1)
                forces_predicted = first - second
                forces_predicted_total += forces_predicted

        if n_frames is None:
            raise ValueError(
                "all collinear problem happened, but aux model was not provided"
            )

        result = [n_frames, weight_aux, total_main_weight]
        if self.use_energies:
            result.append(predictions_total)
            result.append(batch.y)
        else:
            result.append(None)
            result.append(None)

        if self.use_forces:
            result.append(forces_predicted_total)
            result.append(batch.forces)
        else:
            result.append(None)
            result.append(None)

        return result
