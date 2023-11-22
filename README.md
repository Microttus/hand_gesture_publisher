# Camera Hand Gesture Recognition

This is a ROS2 package which by the use of a machine trained model
is able to use a camera feed to recognise different hand gestures.

This package provides a single executable which uses the build in camera
to track a hand, and post a string with the gesture in a topic. 

## How to use

To use the hand gesture package clone this repository and build your
project. To launch the executable:

```text
ros2 run hand_gesture_pub hand_gest_pub
```

