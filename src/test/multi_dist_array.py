from typing import Dict
import unittest
from EinsumNetwork.ExponentialFamilyArray import (
    ExponentialFamilyArray,
    MultiDistArray,
    NormalArray,
    CategoricalArray,
)
from EinsumNetwork import Graph, EinsumNetwork
import torch
import numpy as np
import random


def _construct_multi_dist_args(num_var, K):
    # Construct scope subsets
    scope_normal = []
    scope_categorical = []
    # Ensure, that neither array is empty
    while len(scope_normal) == 0 or len(scope_normal) == num_var:
        scope_normal = [i for i in range(num_var) if random.random() > 0.5]
        scope_categorical = [i for i in range(num_var) if i not in scope_normal]

    array_normal = NormalArray
    array_normal_args = dict()
    array_categorical = CategoricalArray
    array_categorical_args = dict(
        K=K,
    )

    array_to_scope_dict = {
        array_normal: scope_normal,
        array_categorical: scope_categorical,
    }
    array_args_dict = {
        array_normal: array_normal_args,
        array_categorical: array_categorical_args,
    }

    multi_dist_args_dict = dict(
        exponential_family_to_scope_dict=array_to_scope_dict,
        exponential_family_args_dict=array_args_dict,
    )

    return multi_dist_args_dict, scope_normal, scope_categorical


def _construct_random_data(N, scope_normal, scope_categorical, num_var, num_dims, K, device):
    # Construct random data
    x_n = torch.randn(N, len(scope_normal), num_dims, device=device)
    x_c = torch.randint(0, K, (N, len(scope_categorical), num_dims), device=device).float()
    x = torch.empty(N, num_var, num_dims, device=device)
    # Place input data at correct position
    x[:, scope_normal] = x_n
    x[:, scope_categorical] = x_c
    return x, x_n, x_c


class TestMultiDistArray(unittest.TestCase):
    def test_log_prob(self):
        """
        Compare log probabilities computed from MultiDistArray with the logprobs from the
        array subsets.
        """

        def _run(device, use_em, num_var, num_dims, array_shape, K):
            kwargs, scope_normal, scope_categorical = _construct_multi_dist_args(
                num_var, K
            )
            multi_dist = MultiDistArray(
                use_em=use_em,
                num_var=num_var,
                num_dims=num_dims,
                array_shape=array_shape,
                **kwargs
            )
            multi_dist.initialize()
            multi_dist.to(device)

            # Random data
            x, x_n, x_c = _construct_random_data(
                10, scope_normal, scope_categorical, num_var, num_dims, K, device
            )

            actual = multi_dist(x)

            # Obtain result subsets
            input_dict: Dict[type, torch.Tensor] = {
                NormalArray: x_n,
                CategoricalArray: x_c,
            }
            for array, scope in multi_dist.arrays_scope_dict.items():
                expected = array(input_dict[array.__class__])
                actual_subset = actual[:, scope]

                diff = (expected - actual_subset).abs()
                self.assertTrue((diff < 1e-5).all())

        # Creat subtests for some config combinations
        for use_em in [True, False]:
            for num_var in [2, 5]:
                for num_dims in [1, 5]:
                    for K in [1, 5]:
                        for array_shape in [(1,), (5,)]:
                            devices = [torch.device("cpu")]
                            if torch.cuda.is_available():
                                devices.append(torch.device("cuda"))
                            for device in devices:
                                with self.subTest(
                                    use_em=use_em,
                                    num_var=num_var,
                                    num_dims=num_dims,
                                    array_shape=array_shape,
                                    K=K,
                                    device=device
                                ):
                                    _run(device, use_em, num_var, num_dims, array_shape, K)

    def test_optimization(self):
        """Check if any configuration throws an error during forward/backward passes."""

        def _run(device, use_em, num_var, num_dims, K_rat, K_cat):
            # Some args
            pd_num_pieces = [4]
            width = int(np.sqrt(num_var))
            height = width
            online_em_frequency = 1
            online_em_stepsize = 0.05

            kwargs, scope_normal, scope_categorical = _construct_multi_dist_args(
                num_var, K_cat
            )

            exponential_family = MultiDistArray
            exponential_family_args = kwargs

            # Setup
            if structure == "poon-domingos":
                pd_delta = [[height / d, width / d] for d in pd_num_pieces]
                graph = Graph.poon_domingos_structure(
                    shape=(height, width), delta=pd_delta
                )
            elif structure == "binary-trees":
                graph = Graph.random_binary_trees(
                    num_var=num_var,
                    depth=depth,
                    num_repetitions=num_repetitions,
                )

            # Construct EiNet
            args = EinsumNetwork.Args(
                num_var=num_var,
                num_dims=num_dims,
                num_classes=1,
                num_sums=K_rat,
                num_input_distributions=K_rat,
                exponential_family=exponential_family,
                exponential_family_args=exponential_family_args,
                online_em_frequency=online_em_frequency,
                online_em_stepsize=online_em_stepsize,
                use_em=use_em,
            )

            einet = EinsumNetwork.EinsumNetwork(graph, args)
            einet.initialize()
            einet.to(device)

            if not use_em:
                optimizer = torch.optim.SGD(einet.parameters(), lr=1e-3)

            # Sample/MPE
            samples = einet.sample(10)
            samples_mpe = einet.mpe(10)

            # Random data
            x, x_n, x_c = _construct_random_data(
                10, scope_normal, scope_categorical, num_var, num_dims, K_cat, device
            )

            # Compute log LLs
            EinsumNetwork.eval_loglikelihood_batched(einet, samples, batch_size=10)

            # Perform single optimization step
            outputs = einet.forward(x)
            ll_sample = EinsumNetwork.log_likelihoods(outputs)
            log_likelihood = ll_sample.sum()
            log_likelihood.backward()
            if use_em:
                einet.em_process_batch()
                einet.em_update()
            else:
                optimizer.step()

        # Creat subtests for some config combinations
        for K_rat in [1, 5]:
            for K_cat in [2, 5]:
                for structure in ["poon-domingos", "binary-trees"]:
                    for num_var in [16, 36]:
                        for depth in [1, 3]:
                            for num_repetitions in [1, 3]:
                                for num_dims in [1, 3]:
                                    for use_em in [True, False]:
                                        devices = [torch.device("cpu")]
                                        if torch.cuda.is_available():
                                            devices.append(torch.device("cuda"))
                                        for device in devices:
                                            with self.subTest(
                                                use_em=use_em,
                                                num_var=num_var,
                                                num_dims=num_dims,
                                                K_rat=K_rat,
                                                K_cat=K_cat,
                                                device=device,
                                            ):
                                                _run(
                                                    device,
                                                    use_em,
                                                    num_var,
                                                    num_dims,
                                                    K_rat,
                                                    K_cat,
                                                )


if __name__ == "__main__":
    unittest.main()
