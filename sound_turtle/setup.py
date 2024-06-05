from setuptools import find_packages, setup
from glob import glob

package_name = 'sound_turtle'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.launch.py')),
        ('share/' + package_name, glob('weights/*.pt')),
        ('share/' + package_name, glob('weights/*.ckpt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='humble',
    maintainer_email='hirekatsu0523@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'control_node = sound_turtle.control_node:main',
            'doa_node = sound_turtle.doa_node:main',
            'wrap_node = sound_turtle.wrap_node:main',
        ],
    },
)
