import unittest
from EinsumNetwork.ExponentialFamilyArray import PoissonArray
from torch.distributions import Poisson
import torch


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
                    log_prob_expected = poisson_torch.log_prob(samples).sum(-1, keepdim=True)  # Sum over num_dims axis

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
                        max_lambda=1.0
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

if __name__ == "__main__":
    unittest.main()
