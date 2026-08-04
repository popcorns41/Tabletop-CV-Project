# Vision-Guided Tabletop Robotics System

A real-time computer vision project that uses a standard webcam to detect, localise, and track objects within a tabletop workspace.

The goal is to build the perception pipeline independently before integrating it with ROS2 and a simulated or physical robot. The completed system will convert webcam detections from pixel coordinates into real-world workspace coordinates that can be used for robotic pick-and-place operations.

This project is being developed as a practical introduction to computer vision for robotics, covering classical image processing, camera geometry, object tracking, machine learning, and ROS2 integration.

## Project Goals

The system will:

* Capture and process video from a standard webcam
* Calibrate the camera and correct lens distortion
* Define a measurable tabletop workspace
* Detect and track objects in real time
* Convert pixel coordinates into physical workspace coordinates
* Estimate object position and orientation
* Compare classical OpenCV methods with learned segmentation models
* Measure localisation accuracy, latency, and robustness
* Publish detected objects through ROS2
* Provide perception data to a simulated or physical robot

## Planned Pipeline

```text
Webcam
  ↓
Camera calibration and undistortion
  ↓
Workspace detection
  ↓
Object detection or segmentation
  ↓
Position and orientation estimation
  ↓
Object tracking and temporal filtering
  ↓
Pixel-to-world coordinate conversion
  ↓
Visualisation and metrics
  ↓
ROS2 publishing
  ↓
Robot pick-and-place integration
```

## Technology Stack

* Python
* OpenCV
* NumPy
* ArUco or ChArUco markers
* PyTorch
* Ultralytics YOLO segmentation
* pytest
* ROS2
* RViz, Gazebo, or PyBullet

The initial perception pipeline will remain independent of ROS2. ROS2 support will be added through a separate adapter once the vision system is stable.

## Project Milestones

### Milestone 1 — Computer Vision Fundamentals

Build the theoretical foundation required to understand the perception pipeline.

Topics include:

* Images as numerical arrays
* Pixel and image coordinate systems
* Homogeneous coordinates
* Perspective projection
* Camera intrinsics and extrinsics
* Lens distortion
* Camera calibration
* Homographies
* Colour spaces
* Image filtering
* Segmentation
* Contours
* Object tracking

**Status:** In progress

---

### Milestone 2 — Webcam Capture and Project Structure

Create the initial application and establish a modular project structure.

Tasks:

* Capture live webcam frames using OpenCV
* Configure image resolution and frame rate
* Display and save frames
* Record reusable test videos
* Add configuration and logging
* Separate camera, perception, geometry, and visualisation components

Expected output:

```text
Live webcam feed with stable frame capture and diagnostic information.
```

---

### Milestone 3 — Camera Calibration

Estimate the webcam’s intrinsic parameters and lens distortion.

Tasks:

* Generate or print a chessboard or ChArUco calibration board
* Capture calibration images from multiple positions and angles
* Detect calibration points
* Estimate the camera matrix
* Estimate distortion coefficients
* Undistort live webcam frames
* Calculate calibration reprojection error

Expected output:

```text
Camera matrix
Distortion coefficients
Mean reprojection error
Undistorted webcam feed
```

---

### Milestone 4 — Workspace Calibration

Define the physical tabletop workspace and map image pixels into real-world coordinates.

Tasks:

* Place ArUco markers around the workspace
* Detect marker positions
* Define known workspace dimensions
* Calculate a planar homography
* Generate a bird’s-eye workspace view
* Convert pixel coordinates into centimetres
* Validate the mapping using measured test points

Expected output:

```text
Pixel coordinate:      (814, 392)
Workspace coordinate:  (32.4 cm, 18.1 cm)
```

Evaluation:

* Mean localisation error
* Maximum localisation error
* Error across different areas of the workspace

---

### Milestone 5 — Classical Object Detection

Implement an initial object detector using classical OpenCV techniques.

Tasks:

* Convert frames into HSV colour space
* Segment objects using colour thresholds
* Apply morphological operations
* Extract contours
* Reject invalid detections using size and shape constraints
* Calculate object centroids
* Estimate object dimensions and orientation

Expected output:

```text
Object: Red block
Position: (32.4 cm, 18.1 cm)
Orientation: 42°
```

This implementation will provide a transparent and lightweight baseline for comparison with a learned model.

---

### Milestone 6 — Object Tracking and Filtering

Track objects across video frames and reduce measurement noise.

Tasks:

* Assign persistent object IDs
* Match detections between frames
* Handle temporary missed detections
* Reduce object ID switching
* Estimate object velocity
* Apply low-pass or Kalman filtering
* Measure stationary position jitter

Expected output:

```text
Object ID: 3
Position: (32.2 cm, 18.3 cm)
Velocity: (1.4 cm/s, -0.2 cm/s)
Tracking state: Active
```

---

### Milestone 7 — Dataset Collection and Annotation

Create a custom dataset for learned object detection or segmentation.

Tasks:

* Capture images under varied lighting conditions
* Include multiple backgrounds and object arrangements
* Record partial occlusions and overlapping objects
* Annotate object masks or bounding boxes
* Define training, validation, and test splits
* Prevent dataset leakage
* Document annotation rules

Dataset conditions should include:

* Bright and dim lighting
* Shadows
* Different object rotations
* Partial occlusion
* Multiple simultaneous objects
* Objects close to workspace boundaries
* Distractor objects

---

### Milestone 8 — Learned Instance Segmentation

Train a segmentation model to detect individual tabletop objects.

Tasks:

* Fine-tune a pretrained segmentation model
* Run inference on live webcam frames
* Extract masks, labels, and confidence scores
* Calculate object centroids and orientations from masks
* Compare learned segmentation against the OpenCV baseline
* Analyse failure cases

Comparison metrics:

| Metric               | Classical OpenCV | Learned Segmentation |
| -------------------- | ---------------: | -------------------: |
| Detection precision  |              TBD |                  TBD |
| Detection recall     |              TBD |                  TBD |
| Localisation error   |              TBD |                  TBD |
| Position jitter      |              TBD |                  TBD |
| Processing latency   |              TBD |                  TBD |
| Frame rate           |              TBD |                  TBD |
| Lighting robustness  |              TBD |                  TBD |
| Occlusion robustness |              TBD |                  TBD |

---

### Milestone 9 — System Evaluation

Evaluate the complete perception pipeline using repeatable experiments.

Metrics:

* Mean localisation error in centimetres
* Maximum localisation error
* Orientation error in degrees
* Detection precision and recall
* False-detection rate
* Detection dropout rate
* Position jitter
* End-to-end latency
* Processing frame rate
* Tracking stability
* Performance under poor lighting
* Performance during partial occlusion

Recorded test videos will be used to ensure that different versions of the system are evaluated under identical conditions.

---

### Milestone 10 — ROS2 Integration

Integrate the completed perception system with ROS2.

Tasks:

* Create a ROS2 perception package
* Capture or subscribe to camera images
* Convert ROS image messages using `cv_bridge`
* Publish detected object poses
* Publish visualisation markers
* Define camera, table, and robot coordinate frames
* Broadcast transforms using TF2
* Visualise detections in RViz

Planned object output:

```text
tracking_id
class_label
confidence
x
y
z
yaw
timestamp
```

The underlying vision pipeline will remain usable without ROS2.

---

### Milestone 11 — Robot Integration

Use the detected object positions to control a simulated or physical robot.

Tasks:

* Transform object poses into the robot base frame
* Select a target object
* Generate a grasp candidate
* Check workspace and joint limits
* Plan a pick-and-place operation
* Visualise the intended motion
* Execute the task in simulation
* Optionally deploy to a physical robot

Potential robot platforms:

* Interbotix PincherX-100
* Simulated robotic manipulator
* Custom tabletop gantry
* PyBullet or Gazebo robot model

---

## Proposed Project Structure

```text
vision-guided-tabletop-robotics/
├── config/
│   ├── camera.yaml
│   ├── workspace.yaml
│   └── detector.yaml
├── data/
│   ├── calibration/
│   ├── images/
│   ├── videos/
│   └── annotations/
├── models/
├── src/
│   ├── camera/
│   ├── calibration/
│   ├── workspace/
│   ├── detection/
│   ├── tracking/
│   ├── geometry/
│   ├── visualisation/
│   └── evaluation/
├── ros2_ws/
├── tests/
├── scripts/
├── docs/
├── requirements.txt
└── README.md
```

## Initial Detection Representation

```python
from dataclasses import dataclass


@dataclass
class DetectedObject:
    tracking_id: int
    label: str
    confidence: float

    x_metres: float
    y_metres: float
    z_metres: float
    yaw_radians: float

    timestamp: float
```

This representation is independent of OpenCV, the machine-learning model, and ROS2. Each integration layer can convert its own output into the shared detection format.

## Design Principles

### Modular architecture

Camera capture, detection, geometry, tracking, evaluation, and ROS2 integration should remain separate components.

### Recorded-data testing

Algorithms should be tested using recorded videos as well as the live webcam. This allows changes to be evaluated against identical inputs.

### Measurable performance

The project should report real accuracy and latency results rather than relying only on visually convincing demonstrations.

### Classical baseline first

A classical OpenCV detector will be implemented before the learned segmentation model. This will provide a meaningful baseline and help determine whether the neural model offers enough improvement to justify its computational cost.

### Robotics-focused evaluation

Model accuracy alone is not sufficient. The final system must produce stable and timely position estimates that are useful for robot control.

## Stretch Goals

* Human intrusion detection for workspace monitoring
* Gesture-based target selection
* Automatic grasp-point estimation
* Multi-camera localisation
* Depth-camera support
* C++ inference implementation
* ONNX model export
* GPU inference
* Hand-eye calibration
* Dynamic object interception
* Physical robot deployment

## Current Status

The project is currently in the fundamentals and planning stage.

Immediate priorities:

1. Complete the required camera geometry fundamentals
2. Establish the Python and OpenCV project structure
3. Capture stable webcam video
4. Calibrate the webcam
5. Implement tabletop coordinate mapping
6. Build the classical OpenCV detection baseline

ROS2 integration will begin once the standalone perception pipeline is stable and measurable.

## Motivation

This project builds on previous experience developing robotic systems while focusing specifically on independently implementing the computer vision pipeline.

The aim is not merely to run an existing object-detection model. The project will explore the complete perception process, from camera calibration and geometry through object detection, tracking, evaluation, and robot integration.
