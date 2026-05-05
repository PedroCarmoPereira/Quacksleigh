# Quacksleigh

## Important Links

* https://sites.google.com/g.uporto.pt/feupsduckietown/simulation/tutorial?authuser=4

* https://docs.duckietown.com/daffy/opmanual-duckiebot/setup/setup_laptop/index.html

* https://github.com/duckietown/gym-duckietown

## Gym Requirements

* Python 3.6
* Change `setup.py` `install_requires` (line 26) to:
```
install_requires = [
    "gym>=0.17.1",
    "numpy>=1.10.0,<=1.20.0",
    "pyglet==1.5.11",
    "pyzmq>=16.0.0",
    "opencv-python>=3.4,<4.7; python_version<'3.7'",
    "opencv-python>=3.4; python_version>='3.7'",
    "PyYAML>=3.11",
    f"duckietown-world-{line}",
    "PyGeometry-z6",
    "carnivalmirror==0.6.2",
    "zuper-commons-z6",
    "typing_extensions",
    "Pillow",
]
```

