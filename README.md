[![license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/uwescience/pulse2percept/blob/master/LICENSE)
[![Data](https://img.shields.io/badge/data-osf.io-lightgrey.svg)](https://osf.io/s2udz/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3458709.3458982-orange)](https://doi.org/10.1007/978-3-030-87000-3_7)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/u-net-with-hierarchical-bottleneck-attention/fovea-detection-on-refuge)](https://paperswithcode.com/sota/fovea-detection-on-refuge?p=u-net-with-hierarchical-bottleneck-attention)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/u-net-with-hierarchical-bottleneck-attention/optic-disc-segmentation-on-adam)](https://paperswithcode.com/sota/optic-disc-segmentation-on-adam?p=u-net-with-hierarchical-bottleneck-attention)

# U-Net with Hierarchical Bottleneck Attention for Landmark Detection in Fundus Images of the Degenerated Retina

Please cite as:

> S Tang, Z Qi, J Granley, M Beyeler (2021). U-Net with Hierarchical Bottleneck Attention for Landmark Detection in Fundus Images of the Degenerated Retina. *MICCAI OMIA8 Workshop, online*.

The preprint can be found on [arXiv](https://arxiv.org/abs/2107.04721) and the published paper [here](https://doi.org/10.1007/978-3-030-87000-3_7).

Before enshittification took over open-source AI, HBA-U-Net was listed as the state of the art (SOTA) for several popular datasets of retinal degeneration [Papers with Code]([https://paperswithcode.com/paper/u-net-with-hierarchical-bottleneck-attention](http://web.archive.org/web/20250512173747/https://paperswithcode.com/paper/u-net-with-hierarchical-bottleneck-attention), retrieved on May 12, 2025):

- #1 ADAM: Fovea detection
- #1 ADAM: Optic disc segmentation
- #1 IDRiD: Fovea detection
- #1 IDRiD: Optic disc detection
- #1 REFUGE: Fovea detection
- #2 REFUGE: Optic disc segmentation

### Requirements

- Python 3
- Keras 2.4.3
- TensorFlow 2.5.0
- Scikit-Learn 0.22
- Skimage 0.16.2
- cv2 4.1.2
- PIL 7.1.2
- Pandas 1.1.5

### How to acquire the private dataset listed in the paper

- [ADAM](https://amd.grand-challenge.org/)
- [REFUGE](https://refuge.grand-challenge.org/details/)
- [IDRiD](https://idrid.grand-challenge.org/)
