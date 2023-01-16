<h1 align="center">
    One Eye is All You Need: Accurate Gaze Estimation
</h1>

This repository contains the code to reproduce the results of our paper _One Eye is All You Need: Lightweight CNN Encoders for Accurate Zero-Calibration Gaze Estimation_, submitted to the 14<sup>th</sup> ACM Symposium on Eye Tracking Research & Applications in Tübingen, Germany from May 30-June 2, 2023.

<!-- TABLE OF CONTENTS -->

## Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Contact](#issues-and-contact)

<!-- ABOUT THE PROJECT -->

## About The Project

Many current gaze estimation models not only fail to utilize robust computer vision (CV) algorithms but also require the use of either both eyes or the entire face, where high-resolution real-world data may not be available. Thus, we propose a gaze estimation model that implements lightweight ResNet, Inception, and SqueezeNet models as encoders for eye-image data and makes predictions using only one eye, which prioritizes both accuracy and speed. We attain high performance on the GazeCapture dataset with these models without calibration; with two eyes, we achieve a base prediction error of 1.471 cm on the test set, and with just a single eye, we achieve a base prediction error of 2.312 cm on the test set. These results surpass those of other uncalibrated gaze tracking models and demonstrate that gaze predictors can still achieve exceptional results when only considering a single eye.

<!-- Getting Started -->

## Getting Started

First, create a Conda environment, and then use the command `conda install --file requirements.txt --channel conda-forge --channel comet_ml` in the terminal to install the necessary packages and dependencies for this project. Note that this will only install the necessary libraries and dependencies required for this project. Due to the large size of the dataset, you will be unable to run this code on your machine.

<!-- Repository Structure -->

## Repository Structure

| Folder                                               | Description                                                                                                                                                                                          |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `full`                                    | Models and training files where both eyes are used in model training.                                                                                     |
| `one_eye`                                    | Models and training files where only one eye is used in model training.                                                                                       |
| `utils`                  | Data loading files from the GazeCapture dataset. There are two files, based on what type of data the model requests.                                                               |

<!-- Contact -->

## Issues and Contact

For any issues pertaining to the code, please use the Issues tab. For questions regarding the research paper, please email any of the first authors of this paper (Rishi Athavale, Lakshmi Sritan Motati, Anish Susarla, and Rohan Kalahasty; email addresses in the paper), and we will get back to you to address your questions.
