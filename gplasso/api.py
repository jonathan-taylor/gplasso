from .kernels import (covariance_structure,
                      gaussian_kernel,
                      discrete_structure,
                      SVDSampler,
                      GSToolsSampler)
from .base import LASSOInference
from .peaks import default_clusters

from .discrete import DiscreteLASSOInference
from .gridded import GridLASSOInference
