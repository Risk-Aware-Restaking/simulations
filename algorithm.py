import numpy as np



def get_split_choice(Degrees):
    """
    Identifies unique degrees, sorts them, and generates masks for elements
    whose degree is no less than each unique degree.

    Args:
        Degrees (list or numpy.ndarray): A vector of floats or integers
                                        representing degrees.

    Returns:
        tuple: A tuple containing:
            - unique_degrees (list): A list of sorted unique degree values
                                     (from low to high).
            - strategy_masks (list of lists): A list where each inner list contains
                                            the indices from the original 'Degrees'
                                            vector whose corresponding degree is
                                            greater than or equal to the unique degree
                                            at the same position in 'unique_degrees'.
    """
    if not isinstance(Degrees, (list, np.ndarray)):
        print("Input 'Degrees' must be a list or a NumPy array.")
        return [], []

    # Convert to a NumPy array for easier indexing and unique operations
    degrees_array = np.array(Degrees)

    # 1. Identify unique values and sort them
    unique_degrees = sorted(list(np.unique(degrees_array)))

    # 2. Calculate strategy_masks
    strategy_masks = []
    for unique_deg in unique_degrees:
        # Get indices where the original degree is no less than the current unique degree
        # np.where returns a tuple, we need the first element which is the array of indices
        indices_for_mask = np.where(degrees_array >= unique_deg)[0].tolist()
        strategy_masks.append(indices_for_mask)

    return unique_degrees, strategy_masks


def calc_split_strategy(d, mask, rewards):
    """
    Calculates an allocation vector by distributing a degree 'd' proportionally
    among specified rewards, iteratively adjusting for values that would exceed 1.

    The process works as follows:
    1. For the currently active (unallocated) indices in the mask, calculate
       their proportional share of the remaining 'd' based on their rewards.
    2. If any calculated share is greater than or equal to 1:
       a. Set that index's allocation to 1.
       b. Reduce the remaining 'd' by 1.
       c. Remove that index from the active mask.
    3. Repeat steps 1-2 until no calculated share exceeds 1, or until no
       'd' remains, or no active indices are left.
    4. Distribute any remaining 'd' proportionally among the final active indices.

    Args:
        d (float or int): The total 'degree' value to be allocated. Must be non-negative.
        mask (list or numpy.ndarray): A list or array of integer indices from the
                                      original 'rewards' vector that are eligible
                                      for allocation.
        rewards (list or numpy.ndarray): The full vector of reward values.
                                         The length of the final 'allocate'
                                         vector will match the length of 'rewards'.

    Returns:
        numpy.ndarray: The final allocation vector. Its length is the same as
                       'rewards'. Values are capped at 1.0 and are non-negative.
                       Returns an empty NumPy array if input is invalid.
    """
    # --- Input Validation ---
    if not isinstance(d, (int, float)) or d < 0:
        print("Error: Input 'd' must be a non-negative number.")
        return np.array([])
    if not isinstance(mask, (list, np.ndarray)):
        print("Error: Input 'mask' must be a list or a NumPy array of indices.")
        return np.array([])
    if not isinstance(rewards, (list, np.ndarray)):
        print("Error: Input 'rewards' must be a list or a NumPy array.")
        return np.array([])

    # Convert rewards to a NumPy array for efficient numeric operations,
    # ensuring float type for allocation calculations.
    rewards_array = np.array(rewards, dtype=float)

    # If rewards array is empty, no allocation is possible.
    if rewards_array.size == 0:
        return np.array([])

    # Initialize the 'allocate' vector with zeros. This will store the final allocations.
    # Its size matches the original 'rewards' array.
    allocate = np.zeros_like(rewards_array, dtype=float)

    # Create a mutable list of active indices from the input mask.
    # This list will be modified as indices get 'capped' at 1.
    active_mask = list(mask)

    # Initialize the remaining 'd' to be distributed.
    remaining_d = float(d)

    # --- Iterative Allocation Loop ---
    # The loop continues as long as there's 'd' left to distribute,
    # and there are active indices, and we've recently capped values
    # (meaning we need to re-evaluate remaining values) or it's the first run.
    # The 'True' ensures at least one iteration for initial distribution.
    while True:
        # If no active indices or no 'd' left, break the loop.
        if not active_mask or remaining_d <= 0:
            break

        # Calculate the sum of rewards for the currently active indices.
        # This is the denominator for proportional distribution.
        active_rewards_sum = sum(rewards_array[idx] for idx in active_mask)

        # If the sum of active rewards is zero, no further proportional distribution
        # is possible among the remaining active indices.
        if active_rewards_sum == 0:
            break

        # List to store indices that will be capped at 1.0 in this iteration.
        newly_capped_indices = []

        # Calculate potential allocations for the current iteration based on `remaining_d`.
        # We temporarily store these to check which ones exceed 1.
        current_iteration_temp_allocations = {}
        for idx in active_mask:
            # Calculate the proportional share for the current index.
            # Handle potential division by zero if active_rewards_sum is 0, though
            # checked above.
            share = (rewards_array[idx] / active_rewards_sum) * remaining_d
            current_iteration_temp_allocations[idx] = share

            # If this share is 1.0 or greater, it needs to be capped.
            if share >= 1.0:
                newly_capped_indices.append(idx)

        # --- Check for Capped Values and Update ---
        if not newly_capped_indices:
            # If no values exceeded 1.0 in this iteration, it means the remaining
            # 'd' can be distributed proportionally among the `active_mask` without
            # any further capping. We are done.
            for idx in active_mask:
                allocate[idx] = current_iteration_temp_allocations[idx]
            break
        else:
            # If there are newly capped indices:
            # 1. Set their allocation to 1.0 in the final 'allocate' array.
            # 2. Reduce the 'remaining_d' by 1.0 for each newly capped index.
            # 3. Remove these indices from the 'active_mask' for the next iteration.
            for idx in newly_capped_indices:
                # Ensure we only cap if it hasn't been capped in a previous iteration
                # (e.g., if an index appeared in two different masks or scenarios).
                if allocate[idx] < 1.0:
                    allocate[idx] = 1.0
                    remaining_d -= 1.0
            
            # Rebuild active_mask by removing the newly capped indices
            active_mask = [idx for idx in active_mask if idx not in newly_capped_indices]
            active_mask.sort() # Keep the mask sorted for consistency

    # --- Final Clipping ---
    # Ensure all values in the final allocation are between 0.0 and 1.0.
    # This handles any floating-point inaccuracies and explicitly caps values.
    allocate = np.clip(allocate, 0.0, 1.0)

    return allocate


def calc_capital_efficiency(rewards, allocation):
    """
    Calculates the capital efficiency as the sum of the element-wise
    product of rewards and their corresponding allocations.

    Args:
        rewards (list or numpy.ndarray): A vector of reward values.
        allocation (list or numpy.ndarray): A vector of allocation values,
                                            corresponding to the rewards.

    Returns:
        float: The calculated capital efficiency. Returns 0.0 if inputs are invalid
               or empty.
    """
    # --- Input Validation ---
    if not isinstance(rewards, (list, np.ndarray)) or \
       not isinstance(allocation, (list, np.ndarray)):
        print("Error: Both 'rewards' and 'allocation' must be lists or NumPy arrays.")
        return 0.0

    rewards_array = np.array(rewards, dtype=float)
    allocation_array = np.array(allocation, dtype=float)

    if rewards_array.size == 0 or allocation_array.size == 0:
        print("Warning: One or both input arrays are empty. Capital efficiency is 0.")
        return 0.0

    if rewards_array.shape != allocation_array.shape:
        print("Error: 'rewards' and 'allocation' arrays must have the same shape.")
        return 0.0

    # Calculate capital efficiency: sum of (reward * allocation) for corresponding elements
    capital_efficiency = np.sum(rewards_array * allocation_array)

    return capital_efficiency