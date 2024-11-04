import os
import random
import numpy as np
import torch
import yaml

def set_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 

    print(f"Reproducibility mode is set with seed {seed}. Note that this might impact performance.")

def set_seed_and_determinism(seed, num_threads=1):
    """
    Set the seed for all random number generators and enforce deterministic behavior.

    Args:
    seed (int): The seed for random number generators.
    num_threads (int, optional): The number of threads for operations. Default is 1.
    """

    # Python random module
    random.seed(seed)

    # Numpy module
    np.random.seed(seed)

    # PyTorch module
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Also sets the seed for all GPUs.

    # Controlling the number of threads with environment variable
    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    # For operations that support parallelism, use only num_threads
    torch.set_num_threads(num_threads)

    # Ensure PyTorch uses deterministic algorithms (may reduce performance)
    torch.backends.cudnn.deterministic = True

    # Do not use the benchmark mode for convolution operations
    torch.backends.cudnn.benchmark = False

    # Print a warning or information message to let the user know about potential performance impact
    print(f"Reproducibility mode is set with seed {seed}. Note that this might impact performance.")