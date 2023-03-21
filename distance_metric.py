class DistanceMetric:
    """Distance metric computed for look-up inference
    """

    def __init__(self):
        """Distance metric computed for look-up inference
        """
        
    def compute_no_clustering(self):
        """
        """
        pass

def compute_closest_outputs(df, config, starting_dir, p_t, h_t, l_t, device, a, b, c):
    if config.clustering:
        return closest_outputs_clustering(config, starting_dir, p_t, h_t, l_t, device, a, b, c)
    else:
        return closest_outputs_no_clustering(df, config, p_t, h_t, l_t, device, a, b, c)

def closest_outputs_no_clustering(df, config, p_t: torch.Tensor, h_t: torch.Tensor, l_t: torch.Tensor, device, a: float, b: float, c: float) -> torch.Tensor:
    # Check if the weights for the Euclidean distance make sense
    if a+b+c <= 0:
        print("Error: The weights a, b, and c must sum up to a number greater than zero")
        sys.exit(1)
    
    # Convert the dataframe to a tensor
    data = torch.tensor(df.values, dtype=torch.float32).to(device)
    
    # Extract the first three tensors for each row into separate tensors
    h_arr = data[:, :len(h_t[-1])]
    l_arr = data[:, len(h_t[-1]):len(h_t[-1])+len(l_t[-1])]
    phi_arr = data[:, len(h_t[-1])+len(l_t[-1]):len(h_t[-1])+len(l_t[-1])+len(p_t[-1])]
    
    # Calculate the Manhattan distance between the target values and the values in each row for each tensor
    p_diff = (p_t.unsqueeze(1) - phi_arr.unsqueeze(0)).abs().sum(dim=-1)
    h_diff = (h_t.unsqueeze(1) - h_arr.unsqueeze(0)).abs().sum(dim=-1)
    l_diff = (l_t.unsqueeze(1) - l_arr.unsqueeze(0)).abs().sum(dim=-1)

    # Add Gaussian noise to the calculation of the weighted Manhattan distance with mean 0 and standard deviation equal to the noise_coeff
    p_diff = p_diff + torch.randn(p_diff.shape).to(device) * config.noise_coeff
    h_diff = h_diff + torch.randn(h_diff.shape).to(device) * config.noise_coeff
    l_diff = l_diff + torch.randn(l_diff.shape).to(device) * config.noise_coeff

    # Calculate the total difference for each row for each tensor
    diff = (a*p_diff + b*h_diff + c*l_diff) / (a+b+c)
    
    # Find the index of the row with the minimum difference for each tensor
    min_index = diff.argmin(dim=-1)
    
    # Create a tensor with the 3 outputs of the closest row (`ht1`, `lt1`, and `a` if present)
    closest_outputs = data[min_index, len(h_t[-1])+len(l_t[-1])+len(p_t[-1]):]
    
    return closest_outputs  # (B,M-114)

def closest_outputs_clustering(config, starting_dir, p_t, h_t, l_t, device, a, b, c):
    """
    Args:
        config (object): The configuration object with the path to the CSV file.
        p_t (torch.Tensor): The target vectors for the `phi` component, with shape (B,48).
        h_t (torch.Tensor): The target vectors for the `h` component, with shape (B,64).
        l_t (torch.Tensor): The target vectors for the `l` component, with shape (B,2).
    """
    B = p_t.shape[0]
    closest_outputs = torch.empty(0, 67).to(device)

    for sample in range(B):
        closest_file_path = search_file(config, starting_dir, p_t[sample], h_t[sample], l_t[sample], device, a, b, c)
        closest_output = _closest_output(closest_file_path, config, p_t[sample], h_t[sample], l_t[sample], device, a, b, c)  # (M-114,)
        closest_outputs = torch.vstack((closest_outputs, closest_output))  # (B,M-114)

    return closest_outputs  # (B,M-114)

def _closest_output(path, config, p_t: torch.Tensor, h_t: torch.Tensor, l_t: torch.Tensor, device, a: float, b: float, c: float) -> torch.Tensor:
    """
    Args:
        config (object): The configuration object with the path to the CSV file.
        p_t (torch.Tensor): The target vector for the `phi` component, with shape (48,).
        h_t (torch.Tensor): The target vector for the `h` component, with shape (64,).
        l_t (torch.Tensor): The target vector for the `l` component, with shape (2,).
        device (torch.device): The device to run the computation on.
        a (float): The weight for the `phi` component in the distance calculation.
        b (float): The weight for the `h` component in the distance calculation.
        c (float): The weight for the `l` component in the distance calculation.

    Returns:
        torch.Tensor: A tensor with the output of the closest row for `ht1`, `lt1`, and `a`.
    """
    # Check if the weights for the Euclidean distance make sense
    if a+b+c <= 0:
        print("Error: The weights a, b, and c must sum up to a number greater than zero")
        sys.exit(1)

    # Load the data from the csv file into a pandas dataframe
    df = pd.read_csv(path, header=None)
    
    # Convert the dataframe to a tensor
    data = torch.tensor(df.values).to(device)  # (N,M)
    
    # Extract the first three tensors for each row into separate tensors
    h_arr = data[:, :len(h_t[-1])]  # (N,64)
    l_arr = data[:, len(h_t[-1]):len(h_t[-1])+len(l_t[-1])]
    phi_arr = data[:, len(h_t[-1])+len(l_t[-1]):len(h_t[-1])+len(l_t[-1])+len(p_t[-1])]
    
    # Calculate the Manhattan distance between the target values and the values in each row for each tensor
    p_diff = (phi_arr - p_t.unsqueeze(0)).abs().sum(dim=-1)    # |(N,48) - (1,48)|.sum(dim=-1) = (N,) --> target vector distance from each vector of the cluster
    h_diff = (h_arr - h_t.unsqueeze(0)).abs().sum(dim=-1)
    l_diff = (l_arr - l_t.unsqueeze(0)).abs().sum(dim=-1)

    # Add Gaussian noise to the calculation of the weighted Manhattan distance with mean 0 and standard deviation equal to the noise_coeff
    p_diff = p_diff + torch.randn(p_diff.shape).to(device) * config.noise_coeff    # (N,) + scalar = (N,) --> noised target vector distance from each vector of the cluster
    h_diff = h_diff + torch.randn(h_diff.shape).to(device) * config.noise_coeff
    l_diff = l_diff + torch.randn(l_diff.shape).to(device) * config.noise_coeff

    # Calculate the total difference for each row for each tensor
    diff = (a*p_diff + b*h_diff + c*l_diff) / (a+b+c)   # (N,) --> weighted target vector distance from each vector of the cluster
    
    # Find the index of the row with the minimum difference for each tensor
    min_index = diff.argmin(dim=-1) # (N,).argmin(dim=-1) = scalar --> returns the index of the closest vector (minimum distance)
    
    # Create a tensor with the 3 outputs of the closest row (`ht1`, `lt1`, and `a` if present)
    closest_output = data[min_index, len(h_t[-1])+len(l_t[-1])+len(p_t[-1]):] # (N,M)[scalar, 114:] = (1, M-114) --> closest output
    
    return closest_output

def search_file(config, current_dir, p_t, h_t, l_t, device, a, b, c):
    """
    Args:
        config (object): The configuration object with the path to the CSV file.
        p_t (torch.Tensor): The target vector for the `phi` component, with shape (48,).
        h_t (torch.Tensor): The target vector for the `h` component, with shape (64,).
        l_t (torch.Tensor): The target vector for the `l` component, with shape (2,).
    """

    while True:
        
        centers_df = pd.read_csv(os.path.join(current_dir, 'centers.csv'), header=None)

        # Convert the dataframe to a tensor
        data = torch.tensor(centers_df.values).to(device)  # (K,M) --> K centroids

        # Extract the features from the vectors
        h_arr = data[:, :len(h_t[-1])]
        l_arr = data[:, len(h_t[-1]):len(h_t[-1])+len(l_t[-1])]
        phi_arr = data[:, len(h_t[-1])+len(l_t[-1]):len(h_t[-1])+len(l_t[-1])+len(p_t[-1])]  # (K,M)[:48] = (K,48)
        
        # Calculate the Manhattan distance between the target values and the values in each row for each tensor
        p_dist = (phi_arr - p_t).abs().sum(dim=-1) # |(K,48) - (K,48)|.sum(dim=-1) = (K,) --> distance from each centroid
        h_dist = (h_arr - h_t).abs().sum(dim=-1)
        l_dist = (l_arr - l_t).abs().sum(dim=-1)

        # Add Gaussian noise to the calculation of the weighted Manhattan distance with mean 0 and standard deviation equal to the noise_coeff
        p_dist = p_dist + torch.randn(p_dist.shape).to(device) * config.noise_coeff    # (K,48) + scalar: noised distance from the centroid
        h_dist = h_dist + torch.randn(h_dist.shape).to(device) * config.noise_coeff
        l_dist = l_dist + torch.randn(l_dist.shape).to(device) * config.noise_coeff

        # Calculate the total difference for each row for each tensor
        distances = (a*p_dist + b*h_dist + c*l_dist) / (a+b+c)  # (K,M) --> weighted distance from the centroid

        closest_cluster = distances.argmin(dim=-1)    # (K,M).argmin(dim=-1) --> index of centroid closest to target vector
        

        # Una volta che ho trovato il closest_cluster, se c'e' la tabella corrispondente ci entro, altrimenti mi prendo la closest row nel closest cluster
        closest_cluster_dir = os.path.join(current_dir, str(closest_cluster.item()))

        if os.path.isdir(closest_cluster_dir):
            current_dir = closest_cluster_dir
        else:
            break
    
    closest_cluster_filename = str(closest_cluster.item()) + "_cluster.csv"
    return os.path.join(current_dir, closest_cluster_filename)