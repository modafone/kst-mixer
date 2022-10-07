# KST-Mixer

## About KST-Mixer
**KST-Mixer: Kinematic Spatio-Temporal Data Mixer For Colon Shape Estimation**

Masahiro Oda 1, Kazuhiro Furukawa 1, Nassir Navab 2, Kensaku Mori 1

1 Nagoya University, Japan

2 Technical University of Munich, Germany


Presented in [AE-CAI | CARE | OR 2.0 Joint MICCAI Workshop 2022](https://workshops.ap-lab.ca/aecai2022/)

Abstract:

We propose a spatio-temporal mixing kinematic data estimation method to estimate the shape of the colon with deformations caused by colonoscope insertion.
Endoscope tracking or a navigation system that navigates physicians to target positions is needed to reduce such complications as organ perforations.
Although many previous methods focused to track bronchoscopes and surgical endoscopes, few number of colonoscope tracking methods were proposed.
This is because the colon largely deforms during colonoscope insertion.
The deformation causes significant tracking errors.
Colon deformation should be taken into account in the tracking process.
We propose a colon shape estimation method using a Kinematic Spatio-Temporal data Mixer (KST-Mixer) that can be used during colonoscope insertions to the colon.
Kinematic data of a colonoscope and the colon, including positions and directions of their centerlines, are obtained using electromagnetic and depth sensors.
The proposed method separates the data into sub-groups along the spatial and temporal axes.
The KST-Mixer extracts kinematic features and mix them along the spatial and temporal axes multiple times.
We evaluated colon shape estimation accuracies in phantom studies.
The proposed method achieved 11.92 mm mean Euclidean distance error, the smallest of the previous methods.
Statistical analysis indicated that the proposed method significantly reduced the error compared to the previous methods.


## How to use code
To run training and testing processes of KST-Mixer, please run the code colon_code5.py.
It performs leave-one-colonoscope-insertion-out cross validation of eight cases.
