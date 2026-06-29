"""
Script to evaluate the integrated system: DAgger-trained CNN agent + YOLO object detection + OpenCV line detection.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024"

import time
import os
import logging
import numpy as np
import argparse
import cv2
import tensorflow as tf

from config.config import find_and_load_config_by_seed, update_config
from config.curriculum import *
from duckietown_utils.env import launch_and_wrap_env
from duckietown_utils.utils import seed
from duckietown_utils.duckietown_world_evaluator import DuckietownWorldEvaluator, DEFAULT_EVALUATION_MAP
from duckietown_utils.rllib_callbacks import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# ------------------------------- Computer Vision & YOLO -------------------------------

def detect_stop_sign(net, image):
    # 1. Safeguard against empty or invalid images from the simulator
    if image is None or image.size == 0 or len(image.shape) < 3:
        return None
        
    h, w, _ = image.shape
    
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    try:
        # 2. Explicitly request the correct output layer to prevent OpenCV graph crashes
        layer_names = net.getUnconnectedOutLayersNames()
        outputs = net.forward(layer_names)
        predictions = outputs[0]
    except Exception as e:
        print(f"[YOLO Inference Error]: {e}")
        return None
           
    predictions = predictions.T   
    
    max_conf_found = 0.0
    best_class_found = -1

    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence > max_conf_found:
            max_conf_found = confidence
            best_class_found = class_id

        if confidence > 0.15:
            cx = pred[0] * (w / 416.0)
            cy = pred[1] * (h / 416.0)
            box_w = pred[2] * (w / 416.0)
            box_h = pred[3] * (h / 416.0)
            
            x = int(cx - (box_w / 2.0))
            y = int(cy - (box_h / 2.0))
            
            boxes.append([x, y, int(box_w), int(box_h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.15, 0.45)
        # Class 12 represents the Stop Sign in your YOLO mapping
        if len(indices) > 0 and best_class_found == 12:
            best_idx = indices[0]
            if isinstance(best_idx, (list, np.ndarray)):
                best_idx = best_idx[0]
            
            print(f"    -> [MATCH DETECTED] Class: {class_ids[best_idx]} Conf: {confidences[best_idx]:.2f}")
            return boxes[best_idx]
            
    return None

def is_at_red_stop_line(image):
    h, w, _ = image.shape
    y_start, y_end = int(h * 0.75), int(h * 0.95)
    x_start, x_end = int(w * 0.55), int(w * 0.95)
    
    roi = image[y_start:y_end, x_start:x_end]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    lower_red1, upper_red1 = np.array([0, 120, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 120, 100]), np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    red_line_mask = cv2.bitwise_or(mask1, mask2)
    
    red_pixel_count = cv2.countNonZero(red_line_mask)
    return red_pixel_count > 400

# ------------------------------- Integrated Agent Wrapper -------------------------------

class IntegratedAgentWrapper:
    """
    Combines the RL Keras model, the YOLO ONNX model, and OpenCV heuristics.
    """
    def __init__(self, rl_model, yolo_net, env):
        self.model = rl_model
        self.yolo_net = yolo_net
        self.env = env
        self.cooldown = 0  # Used to prevent re-triggering the stop logic immediately after resuming

    def compute_action(self, obs, explore=False):
        if self.cooldown > 0:
            self.cooldown -= 1

        # 1. Grab the un-altered, high-res RGB image from the simulator
        raw_rgb = self.env.unwrapped.render_obs()
        
        # Safeguard: if the simulator returns an invalid frame (e.g., during episode resets)
        # gracefully skip OpenCV processing and drive straight for this tiny tick.
        if raw_rgb is not None and raw_rgb.size > 0:
            bgr_image = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)

            # 2. Check for red stop line
            if self.cooldown == 0 and is_at_red_stop_line(bgr_image):
                # 3. Check for stop sign via YOLO
                sign_box = detect_stop_sign(self.yolo_net, bgr_image)
                
                if sign_box is not None:
                    print("\n>>> [STOP SIGN DETECTED AT RED LINE] Pausing control thread for 2 seconds...")
                    time.sleep(2)
                    print(">>> Resuming driving...\n")
                    
                    # Prevent triggering the same stop sign/line over the next ~30 frames
                    self.cooldown = 30 

        # 4. Standard RL execution
        obs_batch = np.expand_dims(obs, axis=0)
        obs_batch = obs_batch.astype(np.float32)
        action = self.model.predict(obs_batch)[0]
        return action

# ------------------------------- Main Loop -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed-model-id', default=42, type=int, help='Unique experiment identifier for the config.')
    parser.add_argument('--analyse-trajectories', action='store_true', help='Calculate metrics and create trajectory plots.')
    parser.add_argument('--map-name', default='LF-norm-zigzag', help="Specify the map (defaulting to zigzag to test stop signs)")
    parser.add_argument('--top-view', action='store_true', help="View the simulation from a fixed bird's eye view")
    
    parser.add_argument('--curriculum-base', dest='curriculum_base', action='store_true', default=True)
    parser.add_argument('--no-curriculum-base', dest='curriculum_base', action='store_false')
    parser.add_argument('--curriculum', dest='curriculum', action='store_true', default=True)
    parser.add_argument('--no-curriculum', dest='curriculum', action='store_false')
    parser.add_argument('-e', '--experiment-idx', default=0, type=int)
    parser.add_argument('-c', '--checkpoint-idx', default=0, type=int)
    parser.add_argument('-m', '--model-path', default='', type=str)
    parser.add_argument('--yolo-path', default='artifacts/model_yolo/best.onnx', type=str, help='Path to YOLO ONNX model')
    parser.add_argument('--grayscale', dest='grayscale', default=True, action='store_true')
    parser.add_argument('--no-grayscale', dest='grayscale', action='store_false')
    args = parser.parse_args()

    render_mode = 'top_down' if args.top_view else 'human'
    test_map = args.map_name
    seed(1234)
    SEED = args.seed_model_id
    filt = None

    if args.model_path != '':
        model_path = args.model_path
        results_path = 'eval_results_' + model_path.split('.h5')[0]
    elif args.curriculum_base:
        filt = 'Curriculum'
        if args.curriculum:
            results_path = f"artifacts/eval_results_full_cur_both"
            model_path = f"artifacts/dagger_model_seed_{SEED}_cur_both_best.h5"
        else:
            results_path = f"artifacts/eval_results_full_cur_ppo"
            model_path = f"artifacts/dagger_model_seed_{SEED}_cur_ppo_best.h5"
    else:
        if args.curriculum:
           results_path = f"artifacts/eval_results_full_cur_dag" 
           model_path = f"artifacts/dagger_model_seed_{SEED}_cur_dag_best.h5"
        else:
            results_path = f"artifacts/eval_results_full"
            model_path = f"artifacts/dagger_model_seed_{SEED}_best.h5"

    ###########################################################
    # Load Models
    print(f"\n>>> Loading DAgger model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}.")
    novice_model = tf.keras.models.load_model(model_path)

    print(f">>> Loading YOLO model from: {args.yolo_path}")
    if not os.path.exists(args.yolo_path):
        raise FileNotFoundError(f"YOLO model not found at {args.yolo_path}.")
    yolo_net = cv2.dnn.readNetFromONNX(args.yolo_path)

    ###########################################################
    # Load experiment config
    config, _ = find_and_load_config_by_seed(SEED, preselected_experiment_idx=args.experiment_idx, preselected_checkpoint_idx=args.checkpoint_idx, experiment_name_filter=filt)
    
    update_config(config, {
        'env_config': {
            'mode': 'inference',
            'training_map': test_map,
            "domain_rand": True,
            "dynamics_rand": True,
            "camera_rand": True,
            "grayscale_image": args.grayscale,
            "spawn_obstacles": True,
            "spawn_forward_obstacle": False,
            "obstacles": {
                "duckie": {"density": 0.5, "static": False}
            }
        }
    })

    ###########################################################
    # Init Env and Integrated Agent
    env = launch_and_wrap_env(config["env_config"])
    integrated_agent = IntegratedAgentWrapper(novice_model, yolo_net, env)

    ###########################################################
    # Visual Loop
    if not args.analyse_trajectories:
        print(">>> Starting Visual Loop...")
        for i in range(5):
            obs = env.reset()
            env.render(render_mode)
            done = False
            
            while not done:
                action = integrated_agent.compute_action(obs, explore=False)
                obs, reward, done, info = env.step(action)
                
                # Show observation camera
                cv2.imshow("Observation", cv2.cvtColor(cv2.resize(obs[..., -3:].astype('float32'), (300, 300)), cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
                
                # Render simulator window
                orig_distortion = env.unwrapped.distortion
                env.unwrapped.distortion = False
                env.render(render_mode)
                env.unwrapped.distortion = orig_distortion
                
                if env.unwrapped.frame_skip > 1:
                    time.sleep(env.unwrapped.delta_time * env.unwrapped.frame_skip)
                    
        env.close()

    ###########################################################
    # Trajectory Evaluator
    if args.analyse_trajectories:
        config['env_config']['spawn_forward_obstacle'] = False 
        evaluator = DuckietownWorldEvaluator(config['env_config'], eval_lenght_sec=15, eval_map=test_map)
        
        print(f"\n>>> Running Trajectory Analysis. Results will be saved to {results_path}...")
        # Overwrite the evaluator's env reference so the IntegratedAgent can access it inside the evaluator loop
        integrated_agent.env = evaluator.env
        evaluator.evaluate(integrated_agent, results_path, episodes=25)