import unittest
from EinsumNetwork.ExponentialFamilyArray import PoissonArray
from EinsumNetwork import Graph, EinsumNetwork
from torch.distributions import Poisson
import torch
import numpy as np


def _poisson_array_to_torch_dist(poisson_array: PoissonArray) -> Poisson:
    """
    Construct an equivalent Torch Poisson object with the same parameters.
    Args:
      poisson_array (PoissonArray): Input array which is to be converted into a torch Poisson object.

    Returns:
        torch.distributed.Poisson: Torch Poisson object.

    """
    params = poisson_array.params
    if not poisson_array._use_em:
        params = poisson_array.reparam(params)
    return Poisson(params.squeeze(1))


class TestPoissonArray(unittest.TestCase):
    """
    This test case compares the implementation of PoissonArray with the Torch reference implementation in
    torch.distributions.Poisson.
    """

    def test_log_prob_torch(self):
        """Compare log probabilities computed from PoissonArray with the Torch reference implementation."""
        for use_em in [True, False]:
            for num_var in [2, 5, 10]:
                for num_dims in [1, 5, 10]:
                    # Initialize
                    poisson_array = PoissonArray(
                        num_var=num_var,
                        num_dims=num_dims,
                        array_shape=(1,),
                        use_em=use_em,
                    )
                    poisson_array.initialize()

                    # Obtain an equivalent torch distribution object
                    poisson_torch = _poisson_array_to_torch_dist(poisson_array)

                    # Generate some samples
                    samples = torch.poisson(torch.rand(100, num_var, num_dims))

                    # Compute the log probabilities with the poisson implementation and the torch reference
                    log_prob_actual = poisson_array.forward(samples)
                    log_prob_expected = poisson_torch.log_prob(samples).sum(
                        -1, keepdim=True
                    )  # Sum over num_dims axis

                    # Ensure, that the computed probabilities are equal
                    diff = (log_prob_expected.exp() - log_prob_actual.exp()).abs()
                    self.assertTrue((diff < 1e-5).all())

    def test_sampling_mean(self):
        """Compare sampling means computed from PoissonArray with the Torch reference implementation."""
        N = int(1e5)
        for use_em in [True, False]:
            for num_var in [2, 5, 10]:
                for num_dims in [1, 5, 10]:
                    # Initialize
                    poisson_array = PoissonArray(
                        num_var=num_var,
                        num_dims=num_dims,
                        array_shape=(1,),
                        use_em=use_em,
                        min_lambda=0.01,
                        max_lambda=0.1,
                    )
                    poisson_array.initialize()

                    # Obtain an equivalent torch distribution object
                    poisson_torch = _poisson_array_to_torch_dist(poisson_array)

                    # Compute samples from both distribution implementations
                    samples_poisson_array = poisson_array.sample(N)
                    samples_torch = poisson_torch.sample((N,))

                    # Compute sample means
                    mean_actual = samples_poisson_array.mean().item()
                    mean_expected = samples_torch.mean().item()

                    # Ensure, that the sample means are approximately equal
                    # (NOTE: increasing N makes this precise but will also take longer)
                    self.assertAlmostEqual(mean_actual, mean_expected, places=2)

    def test_einsum_network_with_poisson_array(self):
        """
        Check that EiNet construction, likelihood computation, sampling, mpe and optimization steps
        do not cause any error when using PoissonArray.
        """
        exponential_family = EinsumNetwork.PoissonArray
        for K in [1, 5]:
            for structure in ["poon-domingos", "binary-trees"]:
                for num_var in [16, 25, 36]:
                    for depth in [1, 3]:
                        for num_repetitions in [1, 3]:
                            for num_dims in [1, 3]:
                                # Some args
                                pd_num_pieces = [4]
                                width = int(np.sqrt(num_var))
                                height = width
                                exponential_family_args = {
                                    "min_lambda": 1e-6,
                                    "max_lambda": 1.0,
                                }
                                online_em_frequency = 1
                                online_em_stepsize = 0.05


                                # Setup
                                if structure == "poon-domingos":
                                    pd_delta = [
                                        [height / d, width / d] for d in pd_num_pieces
                                    ]
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
                                    num_sums=K,
                                    num_input_distributions=K,
                                    exponential_family=exponential_family,
                                    exponential_family_args=exponential_family_args,
                                    online_em_frequency=online_em_frequency,
                                    online_em_stepsize=online_em_stepsize,
                                )

                                einet = EinsumNetwork.EinsumNetwork(graph, args)
                                einet.initialize()

                                # Sample/MPE
                                samples = einet.sample(10)
                                samples_mpe = einet.mpe(10)

                                # Compute log LLs
                                einet.forward(
                                    torch.poisson(torch.rand(100, num_var, num_dims))
                                )
                                EinsumNetwork.eval_loglikelihood_batched(
                                    einet, samples, batch_size=10
                                )

                                # Perform single optimization step
                                outputs = einet.forward(samples)
                                ll_sample = EinsumNetwork.log_likelihoods(outputs)
                                log_likelihood = ll_sample.sum()
                                log_likelihood.backward()
                                einet.em_process_batch()
                                einet.em_update()


if __name__ == "__main__":
    unittest.main()
