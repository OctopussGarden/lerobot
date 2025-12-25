from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from lerobot.teleoperators.xlerobot_joycon import (
    XLerobotJoyconTeleop, 
    XLerobotJoyconTeleopConfig
)

# Create robot
robot_config = XLerobotConfig(id="my_xlerobot")
robot = XLerobot(robot_config)
robot.connect()

# Create Joycon teleoperator
joycon_config = XLerobotJoyconTeleopConfig(kp=1.0)
teleop = XLerobotJoyconTeleop(joycon_config)
teleop.connect(calibrate=True)

# Main control loop
while True:
    robot_obs = robot.get_observation()
    action = teleop.get_action(robot_obs, robot)
    robot.send_action(action)
    
    events = teleop.get_joycon_events()
    if events['exit_early']:
        break
