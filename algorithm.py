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
    print(f"Initial active indices: {active_mask}")

    # Initialize the remaining 'd' to be distributed.
    remaining_d = float(d)
    print(f"Initial remaining d: {remaining_d}")

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
            print(f"Share for index {idx} (reward {rewards_array[idx]}): {share:.4f}")
            current_iteration_temp_allocations[idx] = share

            # If this share is 1.0 or greater, it needs to be capped.
            if share > 1.0:
                print("Warning : Share exceeds 1.0, capping at 1.0.")
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
            active_mask = [
                idx for idx in active_mask if idx not in newly_capped_indices
            ]
            active_mask.sort()  # Keep the mask sorted for consistency

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
    if not isinstance(rewards, (list, np.ndarray)) or not isinstance(
        allocation, (list, np.ndarray)
    ):
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


def calculate_split_algorithm(degrees, rewards):
    """
    Calculates an allocation strategy for various splits (based on unique degrees)
    such that the total stake is distributed, and the ratio of (split amount / base_capital_efficiency)
    is identical for all splits. The total stake is assumed to be 1.0.

    The "base_capital_efficiency" for each split's mask is defined as the sum of
    the original reward values of items included in that mask. This interpretation
    allows for direct calculation without a bisection search.

    Args:
        degrees (list or numpy.ndarray): A vector of floats or integers representing degrees.
        rewards (list or numpy.ndarray): The full vector of reward values.

    Returns:
        tuple: A tuple containing:
            - final_splits (list): A list of 'd' values (split amounts) for each unique degree strategy.
            - final_allocations (list of numpy.ndarray): A list of allocation vectors, one for each strategy.
            - final_efficiencies (list): A list of actual capital efficiencies for each strategy,
                                         calculated using `calc_capital_efficiency` after allocation.
            - constant_K_ratio (float): The calculated constant K = split / base_capital_efficiency.
                                        Note: Due to the capping in calc_split_strategy, the
                                        (split / actual_efficiency) may not be strictly identical
                                        across all splits, but (split / base_efficiency) will be.
    """
    # --- Input Validation ---
    if not isinstance(degrees, (list, np.ndarray)) or not isinstance(
        rewards, (list, np.ndarray)
    ):
        print("Error: 'degrees' and 'rewards' must be lists or NumPy arrays.")
        return [], [], [], 0.0

    # total_stake is hardcoded to 1.0 as per request.
    total_stake = 1.0

    rewards_array = np.array(rewards, dtype=float)

    # Step 1: Get unique degrees and their corresponding strategy masks
    unique_degrees, strategy_masks = get_split_choice(degrees)

    if not unique_degrees:
        print("No unique degrees found. Cannot calculate splits.")
        return [], [], [], 0.0

    # Step 2: Calculate a 'base capital efficiency' for each split.
    # This is defined as the sum of original rewards for items included in that split's mask.
    base_efficiencies = []
    for mask_indices in strategy_masks:
        # Sum of rewards for items included in this specific mask
        mask_rewards_sum = np.sum(rewards_array[mask_indices]) if mask_indices else 0.0
        base_efficiencies.append(mask_rewards_sum)

    # Calculate the total sum of these base efficiencies across all strategies
    total_base_efficiency_sum = sum(base_efficiencies)

    # Handle case where all base efficiencies sum to zero.
    if total_base_efficiency_sum <= 1e-9:  # Effectively zero sum of base efficiencies
        # If total_stake is positive (which it is, hardcoded to 1.0), and base efficiencies are zero,
        # it's impossible to proportionally distribute.
        print(
            "Warning: All calculated base efficiencies are zero. Cannot proportionally distribute a positive total_stake."
        )
        num_strategies = len(unique_degrees)
        final_splits = [0.0] * num_strategies
        final_allocations = [
            np.zeros_like(rewards_array) for _ in range(num_strategies)
        ]
        final_efficiencies = [0.0] * num_strategies
        return final_splits, final_allocations, final_efficiencies, 0.0

    # Step 3: Calculate the constant ratio (K) for (split / base_capital_efficiency)
    # K = (total stake) / (sum of all base efficiencies)
    constant_K_ratio = total_stake / total_base_efficiency_sum

    final_splits = []
    final_allocations = []
    final_efficiencies = []

    # Step 4: Calculate final splits (d values) and their resulting allocations/efficiencies
    for i, mask_indices in enumerate(strategy_masks):
        # Calculate the split amount (d_val) for this strategy based on proportionality
        # d_val = K_ratio * base_efficiency_for_this_mask
        split_Value = constant_K_ratio * base_efficiencies[i]
        final_splits.append(split_Value)

        # Calculate the actual allocation using calc_split_strategy with this d_val
        alloc = calc_split_strategy(unique_degrees[i], mask_indices, rewards_array)
        final_allocations.append(alloc)

        # Calculate the actual capital efficiency using calc_capital_efficiency
        eff = calc_capital_efficiency(rewards_array, alloc)
        final_efficiencies.append(eff)

    return final_splits, final_allocations, final_efficiencies, constant_K_ratio


def calculate_equilibrium_algorithm(degrees, rewards, stakes):
    """
    For each stake in the provided stakes vector, calculates the optimal splits
    and their corresponding allocations (scaled by the split amount) according
    to the defined proportionality rule.

    Args:
        degrees (list or numpy.ndarray): A vector of floats or integers representing degrees.
        rewards (list or numpy.ndarray): The full vector of reward values.
        stakes (list or numpy.ndarray): A vector of total stake amounts for which
                                        to calculate the splits and allocations.

    Returns:
        tuple: A tuple containing:
            - all_final_splits_per_stake (list of lists): A list where each inner list
                                                       contains the 'd' values (splits)
                                                       for a given total_stake from `stakes`.
            - all_final_allocations_per_stake (list of 2D numpy.ndarray):
                                                       A list where each inner NumPy array is
                                                       a 2D array of allocation vectors for all splits
                                                       for a given total_stake from `stakes`.
                                                       Each row of the 2D array corresponds to
                                                       an allocation vector for a specific split.
    """
    # --- Input Validation ---
    if not isinstance(stakes, (list, np.ndarray)):
        print("Error: 'stakes' must be a list or a NumPy array.")
        return [], []
    if not isinstance(degrees, (list, np.ndarray)) or not isinstance(
        rewards, (list, np.ndarray)
    ):
        print("Error: 'degrees' and 'rewards' must be lists or NumPy arrays.")
        return [], []

    rewards_array = np.array(rewards, dtype=float)

    all_final_splits_per_stake = []
    all_final_allocations_per_stake = []

    for current_stake in stakes:
        # Call the main split calculation algorithm for each stake
        splits_for_unit_stake, allocations_for_unit_stake, _, _ = (
            calculate_split_algorithm(degrees, rewards_array)
        )

        # Scale the splits by the current_stake
        scaled_splits = [s * current_stake for s in splits_for_unit_stake]
        scaled_allocations = []

        for j, alloc_vec in enumerate(allocations_for_unit_stake):
            # Each alloc_vec needs to be multiplied by its corresponding scaled_split
            scaled_allocations.append(alloc_vec * scaled_splits[j])
        # Scale the allocations by the current_stake
        # `allocations_for_unit_stake` is a list of 1D numpy arrays.
        # np.array() converts it to a 2D numpy array, then we multiply by current_stake.
        # scaled_allocations = np.array(allocations_for_unit_stake) * current_stake

        all_final_splits_per_stake.append(scaled_splits)
        all_final_allocations_per_stake.append(scaled_allocations)

    return all_final_splits_per_stake, all_final_allocations_per_stake


# --- Example Usage (for testing the new function) ---
if __name__ == "__main__":
    print("--- Testing calculate_equilibrium_algorithm ---")

    # degrees_example = [1.5, 2, 1.5, 2, 2, 1.5]
    # rewards_example = [5.0, 15.0, 7.0, 15.0, 12.0, 10.0]
    # stakes_example = [1.0, 0.5, 2.0] # Test with different total stake values
    degrees_example = [1.5, 1.5, 1]
    rewards_example = [2, 1, 3]
    stakes_example = [1.0, 0.5, 2.0] # Test with different total stake values


    all_splits, all_allocations = calculate_equilibrium_algorithm(
        degrees_example, rewards_example, stakes_example
    )

    print(f"Original Degrees: {degrees_example}")
    print(f"Original Rewards: {rewards_example}")
    print(f"Stakes to test: {stakes_example}\n")

    for i, stake_val in enumerate(stakes_example):
        print(f"--- Results for Total Stake = {stake_val:.2f} ---")

        current_final_splits = all_splits[i]
        current_final_allocations = all_allocations[i]

        print(f"  Splits : {current_final_splits}")
        print(f"  Sum of Splits: {np.sum(current_final_splits):.4f}")

        print("  Allocations (within each split):")
        # Now, current_final_allocations is a 2D NumPy array.
        # Iterating over it directly yields its rows (which are the 1D allocation vectors).
        for j, alloc_vec in enumerate(current_final_allocations):
            unique_degrees, _ = get_split_choice(
                degrees_example
            )  # Re-get unique degrees for printing
            print(
                f"    Split {j+1} (Degree Threshold: {unique_degrees[j]}): {alloc_vec.round(4).tolist()}"
            )
        print("-" * 40)
