import cv2
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

class AUVNavigationEnv(gym.Env):
    """Enhanced AUV Navigation Environment with Realistic Visualization"""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(AUVNavigationEnv, self).__init__()

        # Action space: Forward, Backward, Left, Right, Ascend, Descend
        self.action_space = spaces.Discrete(6)

        # Observation space: Position (x, y, z), Battery, Distance to goal
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([100, 100, 50, 100, 50]),
            dtype=np.float32
        )

        # Obstacles and environment objects
        self.num_obstacles = 10
        self.num_objects = 5

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.auv_position = np.array([10, 10, 5])
        self.goal_position = np.array([90, 90, 30])
        self.battery = 200
        self.steps = 0

        self.obstacles = np.random.randint(0, 100, size=(self.num_obstacles, 3))
        self.objects = np.random.randint(0, 100, size=(self.num_objects, 3))

        distance = np.linalg.norm(self.goal_position - self.auv_position)
        obs = np.concatenate((self.auv_position, [self.battery], [distance]))
        return obs, {}

    def step(self, action):
        movement = {
            0: np.array([1, 0, 0]),  # Forward
            1: np.array([-1, 0, 0]),  # Backward
            2: np.array([0, 1, 0]),  # Move Left
            3: np.array([0, -1, 0]), # Move Right
            4: np.array([0, 0, 1]),  # Ascend
            5: np.array([0, 0, -1])  # Descend
        }

        prev_distance = np.linalg.norm(self.goal_position - self.auv_position)
        new_position = np.clip(self.auv_position + movement[action], [0, 0, 0], [100, 100, 50])

        for obs in self.obstacles:
            if np.array_equal(new_position, obs):
                return np.concatenate((self.auv_position, [self.battery], [prev_distance])), -10, False, False, {}

        self.auv_position = new_position
        self.battery = max(0, self.battery - 1)
        distance = np.linalg.norm(self.goal_position - self.auv_position)
        
        reward = (prev_distance - distance) * 10
        done = distance < 5 or self.battery <= 0
        obs = np.concatenate((self.auv_position, [self.battery], [distance]))
        return obs, reward, done, False, {}

    def render(self, mode="human"):
        """Renders a high-quality, visually rich environment with OpenCV and Matplotlib."""
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255  # White background
        
        # Draw underwater gradient
        for i in range(500):
            img[i, :, :] = [255 - i//2, 255 - i//3, 255]  # Blue gradient
        
        scale = 5
        
        # Draw goal
        goal_x, goal_y = self.goal_position[:2] * scale
        cv2.circle(img, (int(goal_x), int(goal_y)), 10, (0, 255, 0), -1)
        cv2.putText(img, "Goal", (int(goal_x) + 10, int(goal_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw AUV
        auv_x, auv_y = self.auv_position[:2] * scale
        cv2.circle(img, (int(auv_x), int(auv_y)), 10, (255, 0, 0), -1)
        cv2.putText(img, "AUV", (int(auv_x) + 10, int(auv_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw obstacles
        for obs in self.obstacles:
            obs_x, obs_y = obs[:2] * scale
            cv2.rectangle(img, (int(obs_x) - 5, int(obs_y) - 5), (int(obs_x) + 5, int(obs_y) + 5), (0, 0, 255), -1)
        
        # Draw environmental objects
        for obj in self.objects:
            obj_x, obj_y = obj[:2] * scale
            cv2.circle(img, (int(obj_x), int(obj_y)), 5, (128, 128, 128), -1)
        
        # Add HUD Information
        cv2.putText(img, f"Battery: {self.battery}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Distance: {np.linalg.norm(self.goal_position - self.auv_position):.2f}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display Title
        cv2.putText(img, "AUV Navigation Environment", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        
        #cv2_imshow("AUV Navigation", img)
        cv2.waitKey(1)
        return img

    def close(self):
        cv2.destroyAllWindows()