"""
Save Trajectory Videos Script

This script samples trajectories from a trained StreamVLN model and saves them as videos.
It reuses the training infrastructure from streamvln_grpo_train.py.

Usage:
    python streamvln/save_trajectory_videos.py \
        --model_path checkpoints/your_model \
        --habitat_config_path config/vln_r2r.yaml \
        --output_path results/videos \
        --num_videos 10 \
        --deterministic
"""

import os
import sys
import argparse
import numpy as np
import torch
import transformers
import cv2
from pathlib import Path

# Import the training infrastructure
from streamvln_grpo_train import (
    StreamVLNGRPOTrainer,
    GRPOTrainingConfig,
)

from streamvln.model.stream_video_vln import StreamVLNForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from habitat.utils.visualizations.utils import images_to_video
from habitat.utils.visualizations import maps as habitat_maps
from streamvln.habitat_extensions.maps import (
    get_top_down_map,
    colorize_top_down_map,
    draw_source_and_target,
    AGENT_SPRITE,
)


# Action names for display (simplified)
ACTION_NAMES = {
    0: "STOP",
    1: "FWD",
    2: "LEFT", 
    3: "RIGHT"
}


def add_text_to_frame(frame: np.ndarray, text: str, position: str = 'bottom-right') -> np.ndarray:
    """
    Add text overlay to a frame.
    
    Args:
        frame: RGB image (H, W, 3)
        text: Text to display
        position: 'bottom-right' or 'bottom-left'
    
    Returns:
        Frame with text overlay
    """
    # Make a copy to avoid modifying original
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    # Text settings (smaller and more compact)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black background
    padding = 5
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Calculate position (bottom-right corner by default)
    if position == 'bottom-right':
        text_x = w - text_w - padding - 5
        text_y = h - padding - 5
        bg_x1 = w - text_w - padding * 2 - 5
        bg_y1 = h - text_h - padding * 2 - 5
        bg_x2 = w - 5
        bg_y2 = h - 5
    else:  # bottom-left
        text_x = padding + 5
        text_y = h - padding - 5
        bg_x1 = 5
        bg_y1 = h - text_h - padding * 2 - 5
        bg_x2 = text_w + padding * 2 + 5
        bg_y2 = h - 5
    
    # Draw semi-transparent background rectangle
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    # Blend with original frame (70% background, 30% original)
    cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)
    
    # Draw text
    cv2.putText(frame_copy, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return frame_copy


def save_topdown_map_video(trajectory: dict, env, output_dir: str, filename: str):
    """
    Save animated top-down map video showing agent movement with orientation.
    
    Args:
        trajectory: Trajectory dict with 'states' containing positions and rotations
        env: Habitat environment
        output_dir: Directory to save video
        filename: Video filename (without extension)
    """
    try:
        sim = env.sim
        episode = env.current_episode
        
        # Get base top-down map
        map_resolution = 1024
        meters_per_pixel = 0.025
        top_down_map = get_top_down_map(sim, map_resolution, meters_per_pixel)
        base_map = colorize_top_down_map(top_down_map)
        
        # Draw start and goal positions (static)
        draw_source_and_target(base_map, sim, episode, meters_per_pixel)
        
        # Generate frames for each step
        map_frames = []
        actions = trajectory.get('actions', [])
        
        # Extract all positions and rotations
        agent_states = []
        for state in trajectory['states']:
            pos = state.get('agent_position', None)
            if pos is not None:
                # Get rotation from environment state or estimate from trajectory
                agent_states.append({
                    'position': pos,
                    'rotation': state.get('agent_rotation', None)  # Will be None, we'll handle it
                })
        
        # Generate a frame for each step
        for step_idx in range(len(agent_states)):
            # Start with a fresh copy of the base map
            frame_map = base_map.copy()
            
            # Draw historical trajectory up to current step (in blue)
            for i in range(step_idx):
                pos1 = agent_states[i]['position']
                pos2 = agent_states[i+1]['position']
                
                grid_x1, grid_y1 = habitat_maps.to_grid(
                    pos1[2], pos1[0], frame_map.shape[0:2], sim
                )
                grid_x2, grid_y2 = habitat_maps.to_grid(
                    pos2[2], pos2[0], frame_map.shape[0:2], sim
                )
                
                cv2.line(frame_map, (grid_y1, grid_x1), (grid_y2, grid_x2),
                         (255, 0, 0), thickness=2)  # Blue line for history
            
            # Draw current agent position with orientation indicator
            current_pos = agent_states[step_idx]['position']
            grid_x, grid_y = habitat_maps.to_grid(
                current_pos[2], current_pos[0], frame_map.shape[0:2], sim
            )
            
            # Estimate heading from movement direction (if not first step)
            if step_idx > 0:
                prev_pos = agent_states[step_idx - 1]['position']
                dx = current_pos[2] - prev_pos[2]
                dz = current_pos[0] - prev_pos[0]
                heading = np.arctan2(dx, dz)  # Heading in radians
            else:
                heading = 0.0  # Default heading
            
            # Draw agent sprite with orientation
            # Agent sprite is a small arrow/triangle showing direction
            agent_radius = int(0.3 / meters_per_pixel)  # 30cm agent size
            
            # Draw circle for agent body
            cv2.circle(frame_map, (grid_y, grid_x), agent_radius, (0, 255, 0), -1)  # Green filled circle
            
            # Draw direction arrow
            arrow_length = int(agent_radius * 1.5)
            end_x = int(grid_x + arrow_length * np.sin(heading))
            end_y = int(grid_y + arrow_length * np.cos(heading))
            cv2.arrowedLine(frame_map, (grid_y, grid_x), (end_y, end_x),
                           (255, 255, 0), thickness=2, tipLength=0.4)  # Yellow arrow
            
            # Add step and action text
            if step_idx < len(actions):
                action_id = actions[step_idx]
                action_name = ACTION_NAMES.get(action_id, f"A{action_id}")
                text = f"{step_idx}:{action_name}"
                frame_map = add_text_to_frame(frame_map, text, position='bottom-right')
            
            map_frames.append(frame_map)
        
        # Save as video
        if len(map_frames) > 0:
            video_filename = f"{filename}_topdown"
            images_to_video(map_frames, output_dir, video_filename, fps=4)
            print(f"  ✅ Saved top-down map video: {video_filename}.mp4 ({len(map_frames)} frames)")
            return True
        else:
            print(f"  ⚠️  No map frames generated")
            return False
        
    except Exception as e:
        print(f"  ❌ Failed to save top-down map video: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_oracle_topdown_map_videos(trajectory: dict, env, output_dir: str, filename: str):
    """
    Save animated top-down map videos for oracle demonstrations.
    
    Args:
        trajectory: Trajectory dict with oracle_demonstrations
        env: Habitat environment
        output_dir: Directory to save videos
        filename: Base filename (without extension)
    """
    oracle_demonstrations = trajectory.get('oracle_demonstrations', [])
    if len(oracle_demonstrations) == 0:
        return
    
    print(f"  📹 Generating {len(oracle_demonstrations)} oracle top-down map video(s)")
    
    for demo_idx, demo in enumerate(oracle_demonstrations):
        try:
            sim = env.sim
            episode = env.current_episode
            
            # Get base top-down map
            map_resolution = 1024
            meters_per_pixel = 0.025
            top_down_map = get_top_down_map(sim, map_resolution, meters_per_pixel)
            base_map = colorize_top_down_map(top_down_map)
            
            # Draw start and goal positions (static)
            draw_source_and_target(base_map, sim, episode, meters_per_pixel)
            
            # Get demo data
            demo_positions = demo.get('positions', [])
            demo_actions = demo.get('actions', [])
            
            if len(demo_positions) == 0:
                continue
            
            # Generate frames for each oracle step
            map_frames = []
            for step_idx in range(len(demo_positions)):
                # Start with a fresh copy of the base map
                frame_map = base_map.copy()
                
                # Draw historical trajectory up to current step (in orange)
                for i in range(step_idx):
                    pos1 = demo_positions[i]
                    pos2 = demo_positions[i+1]
                    
                    grid_x1, grid_y1 = habitat_maps.to_grid(
                        pos1[2], pos1[0], frame_map.shape[0:2], sim
                    )
                    grid_x2, grid_y2 = habitat_maps.to_grid(
                        pos2[2], pos2[0], frame_map.shape[0:2], sim
                    )
                    
                    cv2.line(frame_map, (grid_y1, grid_x1), (grid_y2, grid_x2),
                             (0, 165, 255), thickness=2)  # Orange line for oracle history
                
                # Draw current agent position with orientation
                current_pos = demo_positions[step_idx]
                grid_x, grid_y = habitat_maps.to_grid(
                    current_pos[2], current_pos[0], frame_map.shape[0:2], sim
                )
                
                # Estimate heading from movement direction
                if step_idx > 0:
                    prev_pos = demo_positions[step_idx - 1]
                    dx = current_pos[2] - prev_pos[2]
                    dz = current_pos[0] - prev_pos[0]
                    heading = np.arctan2(dx, dz)
                else:
                    heading = 0.0
                
                # Draw agent sprite
                agent_radius = int(0.3 / meters_per_pixel)
                
                # Draw circle for agent body (orange for oracle)
                cv2.circle(frame_map, (grid_y, grid_x), agent_radius, (0, 200, 255), -1)  # Orange filled circle
                
                # Draw direction arrow
                arrow_length = int(agent_radius * 1.5)
                end_x = int(grid_x + arrow_length * np.sin(heading))
                end_y = int(grid_y + arrow_length * np.cos(heading))
                cv2.arrowedLine(frame_map, (grid_y, grid_x), (end_y, end_x),
                               (255, 255, 0), thickness=2, tipLength=0.4)  # Yellow arrow
                
                # Add step and action text
                if step_idx < len(demo_actions):
                    action_id = demo_actions[step_idx]
                    action_name = ACTION_NAMES.get(action_id, f"A{action_id}")
                    text = f"O{step_idx}:{action_name}"
                    frame_map = add_text_to_frame(frame_map, text, position='bottom-right')
                
                map_frames.append(frame_map)
            
            # Save as video
            if len(map_frames) > 0:
                video_filename = f"{filename}_oracle{demo_idx+1}_topdown"
                images_to_video(map_frames, output_dir, video_filename, fps=4)
                print(f"  ✅ Saved oracle {demo_idx+1} top-down map video: {video_filename}.mp4 ({len(map_frames)} frames)")
        
        except Exception as e:
            print(f"  ❌ Failed to save oracle {demo_idx+1} top-down map video: {e}")
            import traceback
            traceback.print_exc()


def save_topdown_map(trajectory: dict, env, output_dir: str, filename: str):
    """
    Save static top-down map of the scene with agent trajectory.
    
    Args:
        trajectory: Trajectory dict with 'states' containing positions
        env: Habitat environment
        output_dir: Directory to save map
        filename: Map filename (without extension)
    """
    try:
        sim = env.sim
        episode = env.current_episode
        
        # Get top-down map
        map_resolution = 1024
        meters_per_pixel = 0.025
        top_down_map = get_top_down_map(sim, map_resolution, meters_per_pixel)
        
        # Colorize the map
        map_img = colorize_top_down_map(top_down_map)
        
        # Draw start and goal positions
        draw_source_and_target(map_img, sim, episode, meters_per_pixel)
        
        # Draw agent trajectory (policy)
        agent_positions = []
        for state in trajectory['states']:
            pos = state.get('agent_position', None)
            if pos is not None:
                # Convert position to grid coordinates
                grid_x, grid_y = habitat_maps.to_grid(
                    pos[2], pos[0], map_img.shape[0:2], sim
                )
                agent_positions.append((grid_y, grid_x))
        
        # Draw the trajectory path in blue
        if len(agent_positions) > 1:
            for i in range(len(agent_positions) - 1):
                cv2.line(map_img, agent_positions[i], agent_positions[i+1], 
                         (255, 0, 0), thickness=2)  # Blue line
        
        # Draw oracle demonstration paths if they exist (in orange)
        oracle_demonstrations = trajectory.get('oracle_demonstrations', [])
        for demo_idx, demo in enumerate(oracle_demonstrations):
            demo_positions = demo.get('positions', [])
            demo_grid_positions = []
            for pos in demo_positions:
                grid_x, grid_y = habitat_maps.to_grid(
                    pos[2], pos[0], map_img.shape[0:2], sim
                )
                demo_grid_positions.append((grid_y, grid_x))
            
            # Draw demo path in orange
            if len(demo_grid_positions) > 1:
                for i in range(len(demo_grid_positions) - 1):
                    cv2.line(map_img, demo_grid_positions[i], demo_grid_positions[i+1],
                             (0, 165, 255), thickness=2)  # Orange line
        
        # Save the map image
        output_path = os.path.join(output_dir, f"{filename}_topdown.png")
        cv2.imwrite(output_path, cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR))
        
        print(f"  ✅ Saved top-down map: {filename}_topdown.png")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to save top-down map: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_topdown_frames(agent_positions, actions, sim, episode, is_oracle=False):
    """
    Generate top-down map frames for trajectory.
    
    Args:
        agent_positions: List of agent 3D positions
        actions: List of actions
        sim: Habitat simulator
        episode: Episode info
        is_oracle: Whether this is oracle demonstration
    
    Returns:
        List of RGB frames (H, W, 3) uint8
    """
    # Get base top-down map (smaller resolution, larger scale for better overview)
    map_resolution = 512  # Reduced from 1024 for smaller output
    meters_per_pixel = 0.05  # Increased from 0.025 to show larger area
    top_down_map = get_top_down_map(sim, map_resolution, meters_per_pixel)
    base_map = colorize_top_down_map(top_down_map)
    
    # Draw start and goal positions (static)
    draw_source_and_target(base_map, sim, episode, meters_per_pixel)
    
    # Calculate bounding box for cropping (based on start, goal, and all agent positions)
    # Get start and goal positions in grid coordinates
    start_pos = episode.start_position
    goal_pos = episode.goals[0].position
    
    start_grid_x, start_grid_y = habitat_maps.to_grid(
        start_pos[2], start_pos[0], base_map.shape[0:2], sim
    )
    goal_grid_x, goal_grid_y = habitat_maps.to_grid(
        goal_pos[2], goal_pos[0], base_map.shape[0:2], sim
    )
    
    # Get all trajectory positions in grid coordinates
    all_grid_x = [start_grid_x, goal_grid_x]
    all_grid_y = [start_grid_y, goal_grid_y]
    
    for pos in agent_positions:
        grid_x, grid_y = habitat_maps.to_grid(
            pos[2], pos[0], base_map.shape[0:2], sim
        )
        all_grid_x.append(grid_x)
        all_grid_y.append(grid_y)
    
    # Calculate crop boundaries with padding
    padding = 150  # pixels padding around the trajectory
    crop_x_min = max(0, min(all_grid_x) - padding)
    crop_x_max = min(base_map.shape[0], max(all_grid_x) + padding)
    crop_y_min = max(0, min(all_grid_y) - padding)
    crop_y_max = min(base_map.shape[1], max(all_grid_y) + padding)
    
    # Generate frames for each step
    map_frames = []
    for step_idx in range(len(agent_positions)):
        # Start with a fresh copy of the base map
        frame_map = base_map.copy()
        
        # Draw historical trajectory up to current step
        line_color = (0, 165, 255) if is_oracle else (255, 0, 0)  # Orange for oracle, blue for policy
        for i in range(step_idx):
            pos1 = agent_positions[i]
            pos2 = agent_positions[i+1]
            
            grid_x1, grid_y1 = habitat_maps.to_grid(
                pos1[2], pos1[0], frame_map.shape[0:2], sim
            )
            grid_x2, grid_y2 = habitat_maps.to_grid(
                pos2[2], pos2[0], frame_map.shape[0:2], sim
            )
            
            cv2.line(frame_map, (grid_y1, grid_x1), (grid_y2, grid_x2),
                     line_color, thickness=3)  # Increased thickness
        
        # Draw current agent position with orientation
        current_pos = agent_positions[step_idx]
        grid_x, grid_y = habitat_maps.to_grid(
            current_pos[2], current_pos[0], frame_map.shape[0:2], sim
        )
        
        # Estimate heading from movement direction
        if step_idx > 0:
            prev_pos = agent_positions[step_idx - 1]
            dx = current_pos[2] - prev_pos[2]
            dz = current_pos[0] - prev_pos[0]
            heading = np.arctan2(dx, dz)
        else:
            heading = 0.0
        
        # Draw agent sprite (larger)
        agent_radius = int(0.4 / meters_per_pixel)  # Increased from 0.3
        agent_color = (0, 200, 255) if is_oracle else (0, 255, 0)  # Orange for oracle, green for policy
        cv2.circle(frame_map, (grid_y, grid_x), agent_radius, agent_color, -1)
        
        # Draw direction arrow (much larger and thicker)
        arrow_length = int(agent_radius * 3.5)  # Increased from 2.5 to 3.5 for even better visibility
        end_x = int(grid_x + arrow_length * np.sin(heading))
        end_y = int(grid_y + arrow_length * np.cos(heading))
        cv2.arrowedLine(frame_map, (grid_y, grid_x), (end_y, end_x),
                       (255, 255, 0), thickness=5, tipLength=0.35)  # Thicker arrow (5 instead of 4)
        
        # Crop the frame to focus on the region of interest
        cropped_frame = frame_map[crop_x_min:crop_x_max, crop_y_min:crop_y_max]
        
        # Note: Action text is only shown on first-person view, not on top-down map
        map_frames.append(cropped_frame)
    
    return map_frames


def combine_frames_vertical(top_frames, bottom_frames, target_width=640):
    """
    Combine two sets of frames vertically (top and bottom).
    Resizes frames to same width and stacks vertically.
    
    Args:
        top_frames: List of top frames (RGB uint8)
        bottom_frames: List of bottom frames (RGB uint8)
        target_width: Target width for both frames
    
    Returns:
        List of combined frames
    """
    combined = []
    for top, bottom in zip(top_frames, bottom_frames):
        # Resize top frame
        h_top, w_top = top.shape[:2]
        new_h_top = int(h_top * target_width / w_top)
        top_resized = cv2.resize(top, (target_width, new_h_top))
        
        # Resize bottom frame
        h_bot, w_bot = bottom.shape[:2]
        new_h_bot = int(h_bot * target_width / w_bot)
        bottom_resized = cv2.resize(bottom, (target_width, new_h_bot))
        
        # Stack vertically
        combined_frame = np.vstack([top_resized, bottom_resized])
        combined.append(combined_frame)
    
    return combined


def save_trajectory_video(trajectory: dict, output_dir: str, filename: str, env=None):
# def save_trajectory_video(trajectory: dict, output_dir: str, filename: str, env=None):
    """
    Save trajectory as combined video (first-person view + top-down map).
    
    Args:
        trajectory: Trajectory dict with 'states' containing RGB observations
        output_dir: Directory to save video
        filename: Video filename (without extension)
        env: Habitat environment (for generating top-down map)
    """
    try:
        # Extract RGB observations and actions from states (policy trajectory)
        rgb_frames = []
        actions = trajectory.get('actions', [])
        agent_positions = []
        
        for idx, state in enumerate(trajectory['states']):
            rgb = state['rgb']
            # Convert to uint8 if needed
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8)
            
            # Add action text overlay
            if idx < len(actions):
                action_id = actions[idx]
                action_name = ACTION_NAMES.get(action_id, f"A{action_id}")
                text = f"{idx}:{action_name}"
                rgb = add_text_to_frame(rgb, text, position='bottom-right')
            
            rgb_frames.append(rgb)
            
            # Collect agent positions for top-down map
            pos = state.get('agent_position', None)
            if pos is not None:
                agent_positions.append(pos)
        
        if len(rgb_frames) == 0:
            print(f"  ⚠️  No frames in policy trajectory")
            return False
        
        # Create video directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate top-down map frames if environment provided
        if env is not None and len(agent_positions) == len(rgb_frames):
            topdown_frames = generate_topdown_frames(
                agent_positions, actions, env.sim, env.current_episode, is_oracle=False
            )
            
            # Combine RGB and top-down frames vertically
            combined_frames = combine_frames_vertical(rgb_frames, topdown_frames)
            
            # Save combined video
            images_to_video(combined_frames, output_dir, filename, fps=4)
        else:
            # Fallback: save only RGB video if no environment
            images_to_video(rgb_frames, output_dir, filename, fps=4)
        
        # Print stats
        total_reward = sum(trajectory['rewards'])
        success = trajectory.get('final_metrics', {}).get('success', 0.0)
        spl = trajectory.get('final_metrics', {}).get('spl', 0.0)
        
        print(f"  ✅ Saved policy video: {filename}.mp4")
        print(f"     Frames: {len(rgb_frames)}, Reward: {total_reward:.2f}, Success: {success:.0f}, SPL: {spl:.3f}")
        print(f"     Actions: {[ACTION_NAMES.get(a, f'A{a}') for a in actions]}")
        
        # 🔥 Save oracle demonstration videos if they exist
        oracle_demonstrations = trajectory.get('oracle_demonstrations', [])
        if len(oracle_demonstrations) > 0:
            print(f"  📹 Found {len(oracle_demonstrations)} oracle demonstration(s)")
            
            for demo_idx, demo in enumerate(oracle_demonstrations):
                demo_rgbs = demo.get('rgbs', [])
                demo_actions = demo.get('actions', [])
                demo_positions = demo.get('positions', [])
                
                if len(demo_rgbs) == 0:
                    continue
                
                # Convert oracle demo frames and add action overlays
                demo_frames = []
                for frame_idx, rgb in enumerate(demo_rgbs):
                    if rgb.dtype != np.uint8:
                        rgb = (rgb * 255).astype(np.uint8)
                    
                    # Add action text overlay
                    if frame_idx < len(demo_actions):
                        action_id = demo_actions[frame_idx]
                        action_name = ACTION_NAMES.get(action_id, f"A{action_id}")
                        text = f"O{frame_idx}:{action_name}"
                        rgb = add_text_to_frame(rgb, text, position='bottom-right')
                    
                    demo_frames.append(rgb)
                
                # Generate top-down map frames for oracle if positions available
                if env is not None and len(demo_positions) == len(demo_frames):
                    demo_topdown_frames = generate_topdown_frames(
                        demo_positions, demo_actions, env.sim, env.current_episode, is_oracle=True
                    )
                    
                    # Combine oracle RGB and top-down frames
                    demo_combined_frames = combine_frames_vertical(demo_frames, demo_topdown_frames)
                    
                    # Save combined oracle video
                    demo_filename = f"{filename}_oracle{demo_idx+1}"
                    images_to_video(demo_combined_frames, output_dir, demo_filename, fps=4)
                else:
                    # Fallback: save only RGB video
                    demo_filename = f"{filename}_oracle{demo_idx+1}"
                    images_to_video(demo_frames, output_dir, demo_filename, fps=4)
                
                rollback_step = demo.get('rollback_step_id', 0)
                rollback_waypoint = demo.get('rollback_waypoint_idx', 0)
                print(f"  ✅ Saved oracle demo {demo_idx+1}: {demo_filename}.mp4")
                print(f"     Frames: {len(demo_frames)}, Rollback from step {rollback_step} (waypoint {rollback_waypoint})")
                print(f"     Actions: {[ACTION_NAMES.get(a, f'A{a}') for a in demo_actions]}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to save video: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Save trajectory videos from StreamVLN model")
    
    # Model args
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to LoRA checkpoint (if using LoRA)")
    
    # Environment args
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml",
                        help="Path to Habitat config")
    
    # Output args
    parser.add_argument("--output_path", type=str, default="results/trajectory_videos",
                        help="Output directory for videos")
    parser.add_argument("--num_videos", type=int, default=10,
                        help="Number of videos to generate")
    parser.add_argument("--episode_ids", type=str, default=None,
                        help="Comma-separated list of specific episode IDs to generate videos for (e.g., 'r2r_9552,rxr_70713')")
    
    # Sampling args
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic sampling (greedy)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (if not deterministic)")
    
    # Training config args (needed for initialization)
    parser.add_argument("--phase", type=str, default="phase1_stop")
    parser.add_argument("--lora_enable", action="store_true",
                        help="Enable LoRA")
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    
    args = parser.parse_args()
    
    print("="*80)
    print("🎥 StreamVLN Trajectory Video Generator")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Config: {args.habitat_config_path}")
    print(f"Output: {args.output_path}")
    print(f"Sampling: {'Deterministic (greedy)' if args.deterministic else f'Stochastic (temp={args.temperature})'}")
    if args.episode_ids:
        target_episodes = [ep.strip() for ep in args.episode_ids.split(',')]
        print(f"Target episodes: {target_episodes}")
    else:
        print(f"Videos to generate: {args.num_videos}")
    print("="*80)
    
    # Create training config (minimal setup for video generation)
    config = GRPOTrainingConfig(
        model_path=args.model_path,
        habitat_config_path=args.habitat_config_path,
        output_path=args.output_path,
        phase=args.phase,
        num_episodes_per_update=1,  # One episode at a time
        group_size=1,  # Single trajectory per instruction
        lora_enable=args.lora_enable,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        sampling_temperature=args.temperature,
    )
    
    # Load tokenizer
    print("\n📦 Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=4096,
        padding_side="right",
    )
    
    # Load model
    print("📦 Loading model...")
    model_config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        config=model_config,
        low_cpu_mem_usage=False,
    )
    
    # Setup model
    model.model.num_history = 8
    model.reset(1)
    model.to('cuda')
    
    # Apply LoRA if enabled
    if config.lora_enable:
        if args.resume_from:
            print(f"🔧 Loading LoRA from {args.resume_from}")
            model = PeftModel.from_pretrained(
                model, 
                args.resume_from, 
                is_trainable=False  # Inference only
            )
            # 🔥 For inference, we need at least one parameter to be trainable for optimizer
            # Set a dummy parameter to be trainable (won't be used)
            for param in model.parameters():
                param.requires_grad = True
                break  # Just make one parameter trainable
        else:
            print(f"🔧 Adding LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # 🔥 No LoRA - make one parameter trainable for optimizer
        for param in model.parameters():
            param.requires_grad = True
            break
    
    # Set model to eval mode
    model.eval()
    
    # Initialize trainer
    print("📦 Setting up trainer...")
    trainer = StreamVLNGRPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )
    
    # 🔥 Manually create training environment (normally done in trainer.train())
    print("📦 Creating Habitat environment...")
    import habitat
    trainer._train_env = habitat.Env(config=trainer.habitat_config)
    trainer._episode_iterator = iter(trainer._train_env.episode_iterator)
    
    # Initialize ShortestPathFollower for GT actions
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    trainer.shortest_path_follower = ShortestPathFollower(
        sim=trainer._train_env.sim,
        goal_radius=3.0,
        return_one_hot=False
    )
    trainer.shortest_path_follower.mode = 'geodesic_path'
    
    print("✅ Environment ready for trajectory collection")
    
    # Create output directory
    video_dir = os.path.join(args.output_path, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"\n🎬 Starting video generation...")
    print(f"Saving to: {video_dir}\n")
    
    # Parse target episode IDs if provided
    target_episodes = None
    if args.episode_ids:
        target_episodes = set(ep.strip() for ep in args.episode_ids.split(','))
        print(f"🎯 Looking for specific episodes: {target_episodes}\n")
        
        # 🔥 Build episode ID -> episode mapping for direct lookup
        episode_map = {ep.episode_id: ep for ep in trainer._train_env.episodes}
        print(f"📊 Total episodes in dataset: {len(episode_map)}\n")
    
    # Generate videos
    success_count = 0
    
    # 🔥 Direct episode lookup if specific IDs provided
    if target_episodes:
        found_episodes = set()
        for target_id in target_episodes:
            print(f"Processing target episode: {target_id}")
            
            if target_id not in episode_map:
                print(f"  ❌ Episode {target_id} not found in dataset!")
                continue
            
            try:
                # Get the specific episode
                episode = episode_map[target_id]
                
                # Reset environment with this specific episode
                trainer._train_env.episode_iterator._episodes = [episode]
                trainer._train_env.episode_iterator._iterator = iter([episode])
                
                # Collect trajectory
                trajectories = trainer.collect_trajectory_group(update_id=0)
                
                if len(trajectories) == 0:
                    print(f"  ⚠️  No trajectory collected")
                    continue
                
                trajectory = trajectories[0]
                instruction = trajectory['instruction']
                
                print(f"  ✅ Collected trajectory")
                print(f"  Instruction: {instruction[:80]}...")
                
                # Create episode-specific directory
                episode_dir = os.path.join(video_dir, target_id)
                os.makedirs(episode_dir, exist_ok=True)
                
                filename = f"trajectory"
                
                # Save video
                if save_trajectory_video(trajectory, episode_dir, filename, env=trainer._train_env):
                    success_count += 1
                    found_episodes.add(target_id)
                    print(f"  🎥 Video saved!")
                
                # Save static map
                save_topdown_map(trajectory, trainer._train_env, episode_dir, filename)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            print()
        
        # Summary for targeted mode
        print("="*80)
        print(f"✅ Video generation complete!")
        print(f"   Target episodes: {len(target_episodes)}")
        print(f"   Found episodes: {len(found_episodes)}")
        print(f"   Successfully saved: {success_count} videos")
        if found_episodes != target_episodes:
            missing = target_episodes - found_episodes
            print(f"   ⚠️  Missing episodes: {missing}")
        print(f"   Output directory: {video_dir}")
        print("="*80)
        
    else:
        # 🔥 Random sampling mode (original behavior)
        max_attempts = args.num_videos
        
        for video_idx in range(max_attempts):
            print(f"Video {video_idx+1}/{max_attempts}:")
            
            try:
                # Collect a trajectory group (contains 1 trajectory due to group_size=1)
                trajectories = trainer.collect_trajectory_group(update_id=video_idx)
                
                if len(trajectories) == 0:
                    print(f"  ⚠️  No trajectory collected")
                    continue
                
                # Get the first (and only) trajectory
                trajectory = trajectories[0]
                
                # Get episode info
                episode_id = trainer._train_env.current_episode.episode_id
                instruction = trajectory['instruction']
                
                print(f"  Episode: {episode_id}")
                print(f"  Instruction: {instruction[:80]}...")
                
                # Create episode-specific directory (e.g., r2r_179)
                episode_dir = os.path.join(video_dir, episode_id)
                os.makedirs(episode_dir, exist_ok=True)
                
                # Generate filename (simple name since folder is already episode-specific)
                filename = f"trajectory"
                
                # Save combined video (first-person + top-down map)
                if save_trajectory_video(trajectory, episode_dir, filename, env=trainer._train_env):
                    success_count += 1
                
                # Also save static top-down map image for reference
                save_topdown_map(trajectory, trainer._train_env, episode_dir, filename)
                
            except Exception as e:
                print(f"  ❌ Error collecting trajectory: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            print()
        
        # Summary for random mode
        print("="*80)
        print(f"✅ Video generation complete!")
        print(f"   Successfully saved: {success_count}/{args.num_videos} videos")
        print(f"   Output directory: {video_dir}")
        print("="*80)
    
    # Close environment
    if hasattr(trainer, '_train_env') and trainer._train_env is not None:
        trainer._train_env.close()


if __name__ == "__main__":
    main()
