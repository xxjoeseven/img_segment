def set_seed(seed):
    
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)

    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
