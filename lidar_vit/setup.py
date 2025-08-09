from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lidar-vit",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Vision Transformer for 3D Object Detection using LIDAR in Autonomous Driving",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/3d-object-detection-using-LIDAR-for-self-driving-car",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "lidar-vit-train=lidar_vit.train_vit:main",
            "lidar-vit-inference=lidar_vit.inference:main",
        ],
    },
)