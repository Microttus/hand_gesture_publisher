from setuptools import find_packages, setup

package_name = 'hand_gesture_pub'

setup(
    name=package_name,
    version='0.0.   ',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['resource/CNN_LSTM_Latest.h5']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='oktma',
    maintainer_email='martino@uia.no',
    description='This package use a simple Machine Learing Model to recognice different basic hand gestures and post them to a topic',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_gest_pub = hand_gesture_pub.hand_gesture_publisher:main',
            'palm_pos_pub = hand_gesture_pub.hand_palm_pub:main',
            'marker_pos_pub = hand_gesture_pub.marker_finder:main'
        ],
    },
)
