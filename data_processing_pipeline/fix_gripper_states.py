#!/usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm

DATA_ROOT = "/media/skills/RRC HDD A/cross-emb/real-lab-data/processed_data"

def fix_gripper_actions(states):
    """
    Derive gripper actions from state deltas.
    Logic:
    - delta = next_state - current_state
    - if delta > 0.05: action = 30
    - if delta < -0.05: action = -30
    - else: action = 0
    """
    n_frames = states.shape[0]
    gripper_states = states[:, 6]
    
    # Calculate deltas (last frame delta = 0)
    deltas = np.zeros(n_frames)
    deltas[:-1] = gripper_states[1:] - gripper_states[:-1]
    
    new_actions_col = np.zeros(n_frames)
    
    # Apply thresholds
    # Assuming symmetric thresholds based on standard robotics logic:
    # > 0.05 implies opening -> 30
    # < -0.05 implies closing -> -30
    # Between -0.05 and 0.05 -> 0 (Hold)
    
    mask_open = deltas > 0.05
    mask_close = deltas < -0.05
    
    new_actions_col[mask_open] = 30.0
    new_actions_col[mask_close] = -30.0
    
    return new_actions_col

def fix_gripper_states(actions, states):
    """
    Correct gripper states based on action sequences.
    Logic:
    - Before any 30 action: force 0.
    - During a 30 command block: keep original states (interpolated).
    - Post 30 command block and before next -30: force 1.
    - During a -30 command block: keep original states (interpolated).
    - Post -30 command block and before next 30: force 0.
    """
    n_frames = actions.shape[0]
    gripper_actions = actions[:, 6]
    original_gripper = states[:, 6]
    correct_gripper = original_gripper.copy()
    
    current_forced_val = 0.0
    i = 0
    while i < n_frames:
        # Check for 30 block (Opening/Open)
        if np.isclose(gripper_actions[i], 30.0):
            # During the block, we keep the original states
            while i < n_frames and np.isclose(gripper_actions[i], 30.0):
                # correct_gripper[i] = original_gripper[i] 
                i += 1
            current_forced_val = 1.0
            
        # Check for -30 block (Closing/Close)
        elif np.isclose(gripper_actions[i], -30.0):
            # During the block, we keep the original states
            while i < n_frames and np.isclose(gripper_actions[i], -30.0):
                # correct_gripper[i] = original_gripper[i]
                i += 1
            current_forced_val = 0.0
            
        else:
            # Stationary phase (0 action): Force to the last relevant binary state
            correct_gripper[i] = current_forced_val
            i += 1
            
    # Update states array
    new_states = states.copy()
    if new_states.shape[1] > 6:
        new_states[:, 6] = correct_gripper
    return new_states

def main():
    # Find all subdirectories in DATA_ROOT
    if not os.path.exists(DATA_ROOT):
        print(f"Error: DATA_ROOT {DATA_ROOT} does not exist.")
        return

    trajectories = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    
    print(f"Found {len(trajectories)} trajectories to process.")
    
    for traj in tqdm(trajectories):
        # The structure is processed_data/<bag>/extracted_data/<bag>/[actions.txt, states.txt]
        path = os.path.join(DATA_ROOT, traj, "extracted_data", traj)
        action_file = os.path.join(path, "actions.txt")
        state_file = os.path.join(path, "states.txt")
        
        # Check primary path
        if not (os.path.exists(action_file) and os.path.exists(state_file)):
            # Try alternative path (flat structure)
            alt_path = os.path.join(DATA_ROOT, traj)
            action_file = os.path.join(alt_path, "actions.txt")
            state_file = os.path.join(alt_path, "states.txt")
            if not (os.path.exists(action_file) and os.path.exists(state_file)):
                # print(f"Skipping {traj}: Files not found.")
                continue
            path = alt_path
            
        try:
            actions = np.loadtxt(action_file)
            states = np.loadtxt(state_file)
            
            # Handle shapes
            if actions.ndim == 1: actions = actions.reshape(1, -1)
            if states.ndim == 1: states = states.reshape(1, -1)
            
            if actions.shape[0] == 0:
                continue
            
            # 1. Fix Actions based on State Deltas
            new_gripper_actions = fix_gripper_actions(states)
            
            corrected_actions = actions.copy()
            if corrected_actions.shape[1] > 6:
                corrected_actions[:, 6] = new_gripper_actions
                
            # 2. Fix States based on New Actions
            corrected_states = fix_gripper_states(corrected_actions, states)
            
            # Save
            save_action_path = os.path.join(path, "actions_correct.txt")
            save_state_path = os.path.join(path, "states_correct.txt")
            
            np.savetxt(save_action_path, corrected_actions, fmt="%.6f")
            np.savetxt(save_state_path, corrected_states, fmt="%.6f")
            
        except Exception as e:
            print(f"Error processing {traj}: {e}")

if __name__ == "__main__":
    main()
