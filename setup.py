from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ros2_maze'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jungho',
    maintainer_email='email@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'getObjectRange = ros2_maze.getObjectRange:main',
            'imageClassifier = ros2_maze.imageClassifier:main',
            'goToGoal = ros2_maze.goToGoal:main',  
        ],
    },
)
