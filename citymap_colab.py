"""
===============================================================================
ASSIGNMENT: FIX THE AUTONOMOUS CAR NAVIGATION (COLAB VERSION)
===============================================================================

Welcome Students!

This is the Google Colab version of the assignment. 
The GUI has been replaced with Matplotlib plots, but the core logic
and the parameters you need to fix are EXACTLY THE SAME.

YOUR TASK:
Find and fix all parameters marked with "FIX ME" comments.

HINTS & GRADING CRITERIA:
Same as the original file. See the comments below.

GOOD LUCK! 🚗💨
===============================================================================
"""

import math
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time

# --- PYTORCH ---
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 0. MOCKED PYQT CLASSES (For Colab)
# ==========================================
class QColor:
    def __init__(self, r, g, b, a=255):
        # Handle string hex codes if passed (e.g., "#2E3440")
        if isinstance(r, str):
            h = r.lstrip('#')
            self._r = int(h[0:2], 16)
            self._g = int(h[2:4], 16)
            self._b = int(h[4:6], 16)
            self._a = 255
        else:
            self._r = int(r)
            self._g = int(g)
            self._b = int(b)
            self._a = int(a)

    def red(self): return self._r
    def green(self): return self._g
    def blue(self): return self._b
    def name(self): return f"#{self._r:02x}{self._g:02x}{self._b:02x}"

class QPointF:
    def __init__(self, x, y):
        self._x = float(x)
        self._y = float(y)
    def x(self): return self._x
    def y(self): return self._y

class QImage:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        # Initialize white background
        self.data = np.ones((height, width, 3), dtype=np.uint8) * 255

    def width(self): return self.w
    def height(self): return self.h

    def pixel(self, x, y):
        # Clamp to bounds
        x = max(0, min(x, self.w - 1))
        y = max(0, min(y, self.h - 1))
        rgb = self.data[int(y), int(x)]
        return QColor(rgb[0], rgb[1], rgb[2])
    
    # Helper to draw on the mock image
    def draw_ellipse(self, cx, cy, rx, ry, color):
        # updates the numpy array
        y, x = np.ogrid[:self.h, :self.w]
        mask = ((x - cx)**2 / rx**2 + (y - cy)**2 / ry**2) <= 1
        self.data[mask] = [color.red(), color.green(), color.blue()]

# ==========================================
# 1. PHYSICS PARAMETERS - FIX ME!
# ==========================================
# NOTE: These are the exact same parameters as the GUI version.
CAR_WIDTH = 14     
CAR_HEIGHT = 8   
SENSOR_DIST = 200   # Increased range
SENSOR_ANGLE = 60    # Wider angle
SPEED = 3.0          # Moderate speed (reduced from 5.0)
TURN_SPEED = 3.0     # Matched turn speed
SHARP_TURN = 6.0      # Adjusted sharp turn

# ==========================================
# 2. RL HYPERPARAMETERS - FIX ME!
# ==========================================
BATCH_SIZE = 256      # Increased from 2
                    # Hint: Typically 32-512

GAMMA = 0.99        # Increased from 0.01
                    # Hint: Usually 0.9-0.99

LR = 0.002            # Decreased from 1.0 (which was broken)
                    # Hint: Usually 0.0001 to 0.01

TAU = 0.005           # Decreased from 0.9
                    # Hint: Usually 0.001 to 0.01

MAX_CONSECUTIVE_CRASHES = 10  # Reduced from 100
                               # Hint: Usually 2-10

# ==========================================
# 3. NEURAL NETWORK
# ==========================================
class DrivingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DrivingDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

# ==========================================
# 4. PHYSICS & LOGIC (CarBrain)
# ==========================================
class CarBrain:
    def __init__(self, map_image: QImage):
        self.map = map_image
        self.w, self.h = map_image.width(), map_image.height()
        
        # RL Init
        self.input_dim = 9
        self.n_actions = 5
        self.policy_net = DrivingDQN(self.input_dim, self.n_actions)
        self.target_net = DrivingDQN(self.input_dim, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        
        self.memory = deque(maxlen=10000)
        self.priority_memory = deque(maxlen=3000)
        self.current_episode_buffer = []
        self.episode_scores = deque(maxlen=100)
        
        self.steps = 0
        self.epsilon = 1.0  # Starts at 1.0 for full exploration 
                             # Hint: Usually starts at 1.0
        self.consecutive_crashes = 0
        
        self.start_pos = QPointF(100, 100) 
        self.car_pos = QPointF(100, 100)   
        self.car_angle = 0
        self.target_pos = QPointF(200, 200)
        
        self.targets = []
        self.current_target_idx = 0
        self.targets_reached = 0
        self.alive = True
        self.score = 0
        self.sensor_coords = [] 
        self.prev_dist = None

    def set_start_pos(self, point):
        self.start_pos = point
        self.car_pos = point

    def reset(self):
        self.alive = True
        self.score = 0
        self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
        self.car_angle = random.randint(0, 360)
        self.current_target_idx = 0
        self.targets_reached = 0
        if len(self.targets) > 0:
            self.target_pos = self.targets[0]
        state, dist = self.get_state()
        self.prev_dist = dist
        return state
    
    def add_target(self, point):
        self.targets.append(QPointF(point.x(), point.y()))
        if len(self.targets) == 1:
            self.target_pos = self.targets[0]
            self.current_target_idx = 0
    
    def switch_to_next_target(self):
        if self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1
            self.target_pos = self.targets[self.current_target_idx]
            self.targets_reached += 1
            return True
        return False

    def get_state(self):
        sensor_vals = []
        self.sensor_coords = []
        angles = [-45, -30, -15, 0, 15, 30, 45]
        
        for a in angles:
            rad = math.radians(self.car_angle + a)
            sx = self.car_pos.x() + math.cos(rad) * SENSOR_DIST
            sy = self.car_pos.y() + math.sin(rad) * SENSOR_DIST
            self.sensor_coords.append(QPointF(sx, sy))
            
            val = 0.0
            if 0 <= sx < self.w and 0 <= sy < self.h:
                c = self.map.pixel(int(sx), int(sy))
                brightness = (c.red() + c.green() + c.blue()) / 3.0
                val = brightness / 255.0
            sensor_vals.append(val)
            
        dx = self.target_pos.x() - self.car_pos.x()
        dy = self.target_pos.y() - self.car_pos.y()
        dist = math.sqrt(dx*dx + dy*dy)
        if dist == 0: dist = 0.001 # prevent div by zero
        
        rad_to_target = math.atan2(dy, dx)
        angle_to_target = math.degrees(rad_to_target)
        
        angle_diff = (angle_to_target - self.car_angle) % 360
        if angle_diff > 180: angle_diff -= 360
        
        norm_dist = min(dist / 800.0, 1.0)
        norm_angle = angle_diff / 180.0
        
        state = sensor_vals + [norm_angle, norm_dist]
        return np.array(state, dtype=np.float32), dist

    def step(self, action):
        turn = 0
        if action == 0:   turn = -TURN_SPEED
        elif action == 1: turn = 0
        elif action == 2: turn = TURN_SPEED
        elif action == 3: turn = -SHARP_TURN
        elif action == 4: turn = SHARP_TURN
        
        self.car_angle += turn
        rad = math.radians(self.car_angle)
        
        new_x = self.car_pos.x() + math.cos(rad) * SPEED
        new_y = self.car_pos.y() + math.sin(rad) * SPEED
        self.car_pos = QPointF(new_x, new_y)
        
        next_state, dist = self.get_state()
        
        reward = -0.1
        done = False
        
        car_center_val = self.check_pixel(self.car_pos.x(), self.car_pos.y())
        
        if car_center_val < 0.4: # OFF ROAD
            reward = -100
            done = True
            self.alive = False
        elif dist < 20:          # REACHED TARGET
            reward = 100
            has_next = self.switch_to_next_target()
            if has_next:
                done = False
                _, new_dist = self.get_state()
                self.prev_dist = new_dist
            else:
                done = True
        else:
            reward += (1.0 - next_state[4]) * 20
            if self.prev_dist is not None and dist > self.prev_dist:
                reward -= 10
            self.prev_dist = dist
            
        self.score += reward
        return next_state, reward, done

    def check_pixel(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            c = self.map.pixel(int(x), int(y))
            return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        return 0.0

    def optimize(self):
        total_memory_size = len(self.memory) + len(self.priority_memory)
        if total_memory_size < BATCH_SIZE: return 0
        
        success_rate = len(self.priority_memory) / max(total_memory_size, 1)
        priority_ratio = 0.3 + (success_rate * 0.4)
        priority_samples = int(BATCH_SIZE * priority_ratio)
        regular_samples = BATCH_SIZE - priority_samples
        
        batch = []
        if len(self.priority_memory) >= priority_samples:
            batch.extend(random.sample(self.priority_memory, priority_samples))
        else:
            batch.extend(list(self.priority_memory))
            regular_samples += priority_samples - len(self.priority_memory)
        
        if len(self.memory) >= regular_samples:
            batch.extend(random.sample(self.memory, regular_samples))
        else:
            batch.extend(list(self.memory))
        
        if len(batch) < BATCH_SIZE // 2: return 0
        
        s, a, r, ns, d = zip(*batch)
        s = torch.FloatTensor(np.array(s))
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        ns = torch.FloatTensor(np.array(ns))
        d = torch.FloatTensor(d).unsqueeze(1)
        
        q = self.policy_net(s).gather(1, a)
        next_q = self.target_net(ns).max(1)[0].detach().unsqueeze(1)
        target = r + GAMMA * next_q * (1 - d)
        
        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > 0.001: self.epsilon *= 0.999
        return loss.item()
    
    def store_experience(self, experience):
        self.current_episode_buffer.append(experience)
    
    def finalize_episode(self, episode_reward):
        if len(self.current_episode_buffer) == 0: return
        self.episode_scores.append(episode_reward)
        if not self.alive:
            self.consecutive_crashes += 1
        else:
            self.consecutive_crashes = 0
        
        if episode_reward > 0:
            for exp in self.current_episode_buffer: self.priority_memory.append(exp)
        else:
            for exp in self.current_episode_buffer: self.memory.append(exp)
        self.current_episode_buffer = []

# ==========================================
# 5. COLAB RUNNER
# ==========================================

def create_mock_map():
    # Creates a 1000x800 map with a track
    img = QImage(1000, 800)
    # Dark grey background
    img.data[:, :] = [46, 52, 64] # #2E3440
    
    # Draw track (White ellipse)
    img.draw_ellipse(500, 400, 400, 300, QColor(255, 255, 255))
    # Draw inner island (Dark grey)
    img.draw_ellipse(500, 400, 250, 150, QColor(46, 52, 64))
    
    return img

def run_training(max_episodes=50, max_steps=1000, render_interval=5):
    map_img = create_mock_map()
    brain = CarBrain(map_img)
    
    # 1. Setup Map -> CAR
    start_pos = QPointF(500 + 350, 400) # Right side of track
    brain.set_start_pos(start_pos)
    
    # 2. Setup Map -> TARGETS
    # 4 targets around the ellipse
    brain.add_target(QPointF(500, 400 - 325)) # Top
    brain.add_target(QPointF(500 - 325, 400)) # Left
    brain.add_target(QPointF(500, 400 + 325)) # Bottom
    brain.add_target(QPointF(500 + 325, 400)) # Right (near start)
    
    print("Starting Training...")
    
    all_scores = []
    
    for episode in range(1, max_episodes + 1):
        state = brain.reset()
        done = False
        step = 0
        episode_path = []
        
        while not done and step < max_steps:
            # Action selection
            if random.random() < brain.epsilon:
                action = random.randint(0, 4)
            else:
                with torch.no_grad():
                    q = brain.policy_net(torch.FloatTensor(state).unsqueeze(0))
                    action = q.argmax().item()
                    
            next_state, reward, done = brain.step(action)
            brain.store_experience((state, action, reward, next_state, done))
            loss = brain.optimize()
            
            # Target Update
            for target_param, policy_param in zip(brain.target_net.parameters(), brain.policy_net.parameters()):
                target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
                
            state = next_state
            step += 1
            brain.steps += 1
            
            # Store path for visualization
            episode_path.append((brain.car_pos.x(), brain.car_pos.y()))
            
            # --- VISUALIZATION ---
            if step % render_interval == 0:
                clear_output(wait=True)
                
                # Setup plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # MAP VIEW
                # Show background
                ax1.imshow(brain.map.data)
                
                # Draw path
                if len(episode_path) > 1:
                    px, py = zip(*episode_path)
                    ax1.plot(px, py, color='cyan', linewidth=1, alpha=0.6)
                
                # Draw car
                car_circle = plt.Circle((brain.car_pos.x(), brain.car_pos.y()), 
                                      CAR_WIDTH, color='yellow')
                ax1.add_patch(car_circle)
                
                # Draw targets
                for i, t in enumerate(brain.targets):
                    col = 'lime' if i == brain.current_target_idx else 'white'
                    t_circ = plt.Circle((t.x(), t.y()), 20, color=col, fill=False, linewidth=2)
                    ax1.add_patch(t_circ)
                    ax1.text(t.x(), t.y(), str(i+1), color='white', ha='center', va='center')

                ax1.set_title(f"Episode {episode} | Step {step}")
                ax1.axis('off')

                # SCORE CHECK
                ax2.set_title("Training Progress (Scores)")
                ax2.set_xlabel("Episode")
                ax2.set_ylabel("Score")
                if len(all_scores) > 0:
                    ax2.plot(all_scores, marker='o')
                
                # Stats Text
                stats = (
                    f"Epsilon: {brain.epsilon:.4f}\n"
                    f"Current Score: {brain.score:.1f}\n"
                    f"Crashes: {brain.consecutive_crashes}\n"
                    f"Targets: {brain.targets_reached}"
                )
                ax2.text(0.05, 0.95, stats, transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

                plt.tight_layout()
                plt.show()

        brain.finalize_episode(brain.score)
        all_scores.append(brain.score)
        
        # Reset if too many crashes (Mocking the UI reset logic)
        if brain.consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
            print(f"⚠️ {MAX_CONSECUTIVE_CRASHES} consecutive crashes! Resetting...")
            brain.consecutive_crashes = 0
            # Resetting car pos score etc is done at start of loop by brain.reset()
            # but we might want to reset the episode buffer or something? 
            # In UI it calls full_reset sometimes or brain.reset(). 
            # Here we just proceed to next episode.

if __name__ == "__main__":
    # If running in Colab/Notebook
    try:
        run_training()
    except KeyboardInterrupt:
        print("Training stopped.")
