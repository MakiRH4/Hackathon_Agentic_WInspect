import time
import numpy as np
import cv2  # OpenCV for image processing and saving
import os   # To handle file system paths
import matplotlib.pyplot as plt

# Import the necessary modules from Isaac Sim:
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController

# Camera and utility imports
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils

###############################################
# 1. Set Up the Robot (WheeledRobot) Instance #
###############################################

# Define the prim path for the robot.
limo_prim_path = "/World/Limo_WInspect"

# Create (or get) the WheeledRobot instance.
limo_robot = WheeledRobot(
    prim_path=limo_prim_path,
    name="my_limo",
    wheel_dof_names=["FL_wheel_link", "FR_wheel_link", "RL_wheel_link", "RR_wheel_link"],
    create_robot=False,  # The asset is already in the stage.
    usd_path="",
    position=np.array([100.80479, 20.08091, 27.31036]),
    # orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, -180]), degrees=True),
    
)

# Set internal wheel properties if they don't exist.
if not hasattr(limo_robot, "_num_wheel_dof") or limo_robot._num_wheel_dof is None:
    if hasattr(limo_robot, "_wheel_dof_names") and limo_robot._wheel_dof_names is not None:
        limo_robot._num_wheel_dof = len(limo_robot._wheel_dof_names)
    else:
        limo_robot._num_wheel_dof = 4
    print(f"Set _num_wheel_dof to {limo_robot._num_wheel_dof}")

if not hasattr(limo_robot, "_wheel_dof_indices") or limo_robot._wheel_dof_indices is None:
    limo_robot._wheel_dof_indices = list(range(limo_robot._num_wheel_dof))
    print(f"Set _wheel_dof_indices to {limo_robot._wheel_dof_indices}")

try:
    limo_robot.initialize()
except Exception as e:
    print("Initialization error (may be expected if already in stage):", e)

###############################################
# 2. Set Up the Differential Controller       #
###############################################

diff_controller = DifferentialController(
    name="limo_diff_controller",
    wheel_radius=0.045,
    wheel_base=0.43
)
diff_controller.reset()

################################################
# 3. Set Up the RealSense RSD455 Camera Sensor #
################################################

camera = Camera(
    prim_path="/World/Limo_WInspect/base_link/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
    frequency=20,
    resolution=(256, 256),
)
camera.initialize()
camera.add_motion_vectors_to_frame()

# # (Optional) Display one image to verify the camera output.
# image_sample = camera.get_rgba()
# plt.imshow(image_sample[:, :, :3])
# plt.title("Camera Sample")
# plt.show()
# print("Motion Vectors:", camera.get_current_frame()["motion_vectors"])

##############################
# 4. Set Up image processing #
##############################

def process_image_and_compute_command(image):
    """
    Processes the input image to detect a blue line and computes a control command.
    Converts the image to HSV, thresholds for blue, computes the centroid of the detected region,
    and calculates an error relative to the image center.
    
    :param image: The RGB image as a NumPy array.
    :return: Tuple (linear_velocity, angular_velocity, line_detected)
    """
    # Convert from RGB to HSV.
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define blue color range in HSV (tune these values as needed).
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create a mask for blue.
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Optional: Clean the mask.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Compute moments to find the centroid.
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        line_detected = True
    else:
        cx = image.shape[1] // 2
        line_detected = False

    # Calculate error relative to image center.
    error = cx - (image.shape[1] // 2)
    
    # Proportional controller (tune gain as needed).
    angular_velocity = -error / 100.0
    linear_velocity = 1.0  # Constant forward speed.
    
    # # Optional: Display the mask.
    # try:
    #     cv2.imshow("Blue Mask", mask)
    #     cv2.waitKey(1)
    # except cv2.error:
    #     pass

    return linear_velocity, angular_velocity, line_detected

######################
# 3. Set Up movement #
######################

def send_command(linear_velocity, angular_velocity):
    """
    Sends a command to the robot using the differential controller.
    """
    wheel_action = diff_controller.forward(command=[linear_velocity, angular_velocity])
    joint_vels = np.atleast_1d(wheel_action.joint_velocities)
    
    if joint_vels.size == 1:
        joint_vels = np.array([joint_vels[0], joint_vels[0]])
    elif joint_vels.size == 2:
        pass
    elif joint_vels.size == 4:
        pass
    else:
        print("Unexpected joint_velocities shape:", joint_vels.shape)
    
    if joint_vels.size == 2:
        wheel_action.joint_velocities = np.hstack((joint_vels, joint_vels))
    else:
        wheel_action.joint_velocities = joint_vels

    limo_robot.apply_wheel_actions(wheel_action)
    print(f"Command sent: Linear {linear_velocity} m/s, Angular {angular_velocity} rad/s")

######################################################################
# 6. Main Control Loop: Retrieve Image and Control with image saving #
######################################################################

# Designate the folder where frames will be saved.
output_dir = r"C:\Hackathon_Agentic_WInspect\frames"  # Change this path as needed.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0  # Counter for saved frames

# Main control loop:
print("Starting control loop.")
try:
    while frame_count <= 30:
    # while True:
        
        # Get the current camera image.
        image = camera.get_rgba()  # Returns an RGBA image (shape: H x W x 4)

        # Save the frame as a JPEG.
        # Extract the RGB channels and convert from RGB to BGR for cv2.imwrite.
        frame_rgb = image[:, :, :3]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(filename, frame_bgr)
        frame_count += 1

        # Process the image to compute control commands.
        linear_vel, angular_vel, line_detected = process_image_and_compute_command(frame_rgb)

        # Stop the robot if the blue line is lost (for example, end of line).
        if not line_detected:
            print("Blue line lost. Stopping robot.")
            send_command(0.0, 0.0)
            break
        else:
            send_command(linear_vel, angular_vel)

        # Adjust the sleep time for your desired update rate.
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Control loop terminated by user.")

#########################################################################
# 6. Main Control Loop: Retrieve Image and Control WITHOUT image saving #
#########################################################################

# frame_count = 0

# print("Starting control loop. Press Ctrl+C to terminate.")
# try:
#     # while True:
#     while frame_count <=15:
#         # Get the current camera image.
#         image = camera.get_rgba()

#         # Process the image to compute control commands.
#         linear_vel, angular_vel, line_detected = process_image_and_compute_command(image)

#         # If the blue line is no longer detected (for example, end of line), stop the robot.
#         if not line_detected:
#             print("Blue line lost. Stopping robot.")
#             send_command(0.0, 0.0)
#             break
#         else:
#             send_command(linear_vel, angular_vel)

#         frame_count+=1
#         # Adjust the sleep time as needed for your update rate.
#         time.sleep(0.1)

# except KeyboardInterrupt:
#     print("Control loop terminated by user.")

######################################################################
# 6. Main Control Loop // NOT WORKING#
######################################################################

# Designate the folder where frames will be saved.
# output_dir = r"C:\Hackathon_Agentic_WInspect\frams"  # Change this path as needed.
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# frame_count = 0  # Counter for saved frames

# # Main control loop:
# print("Starting control function.")


# def gets_frame():
#     # Get the current camera image.
#     image = camera.get_rgba()  # Returns an RGBA image (shape: H x W x 4)
    
#     # Extract the RGB channels and convert from RGB to BGR for cv2.imwrite.
#     frame_rgb = image[:, :, :3]
#     frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
#     # Define filename and save the frame
#     filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
#     cv2.imwrite(filename, frame_bgr)
    
#     # Pause before capturing the next frame
#     # time.sleep(0.5)
    
#     return frame_rgb  # Fix indentation issue

    
# while True:
#     # frame = gets_frame()
#     # Process the image to compute control commands.
#     linear_vel, angular_vel, line_detected = process_image_and_compute_command(gets_frame())
#     # Stop the robot if the blue line is lost (for example, end of line).
#     if not line_detected:
#         print("Blue line lost. Stopping robot.")
#         send_command(0.0, 0.0)
#         break
#     else:
#         send_command(linear_vel, angular_vel)

#     # Adjust the sleep time for your desired update rate.
#     # time.sleep(0.5)
    
# print("Blue line lost. Stopping robot.")
# send_command(0.0, 0.0)