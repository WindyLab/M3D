<div align="center">
  <h1>Domain Adaptive Detection of MAVs: A Benchmark and Noise Suppression Network</h1>
</div>
<p align="center">
  <a href="https://arxiv.org">
    <img src="https://img.shields.io/badge/comming soon-paper?style=socia&logo=arxiv&logoColor=white&labelColor=grey&color=blue"></a>
  <a href="https://arxiv.org">
    <img src="https://img.shields.io/badge/coming soon-blue?logo=googledocs&logoColor=white&labelColor=grey&color=blue"></a>
  <a href="https://westlakeu-my.sharepoint.com/:f:/g/personal/zhao_lab_westlake_edu_cn/Er96hmAJKZdKrjlBAMPLuFoBp3Gnuwy7k0Phqv8RZkO5sw?e=6FIzeZ">
    <img src="https://img.shields.io/badge/Dataset-blue?logo=microsoftsharepoint&logoColor=white&labelColor=grey&color=blue"></a>
  <a href="https://www.youtube.com">
    <img src="https://img.shields.io/badge/coming soon-blue?logo=youtube&logoColor=white&labelColor=grey&color=blue"></a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

This is the repository for the paper "Domain Adaptive Detection of MAVs: A Benchmark and Noise Suppression Network". This paper has been officially accepted by **IEEE Transactions on Automation Science and Engineering**.

This paper benchmarks the cross-domain MAV detection problem. We first propose a **Multi-MAV-Multi-Domain (M3D)** dataset and construct a novel domain adaptive MAV detection benchmark consisting of three representative domain adaptation tasks, i.e., simulation-to-real adaptation, cross-scene adaptation, and cross-camera adaptation. Moreover, we propose a novel noise suppression network with a prior-guided curriculum learning module, a masked copypaste augmentation module, and a large-to-small model training procedure. 

### Dataset

The link to the dataset present in the paper: [Dataset](https://westlakeu-my.sharepoint.com/:f:/g/personal/zhao_lab_westlake_edu_cn/Er96hmAJKZdKrjlBAMPLuFoBp3Gnuwy7k0Phqv8RZkO5sw?e=6FIzeZ)

This dataset includes the simulation images and realistic images. All the labels are in the YOLO format. Please refer https://github.com/ultralytics/yolov5 for details. 

The code for mAP evaluation adopts a standard evaluation tool:https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py instead of the code provided by yolov5 for fairness.

### Contact
If you have any problem when using this dataset, please feel free to contact: [zhangyin@westlake.edu.cn](mailto:zhangyin@westlake.edu.cn).
