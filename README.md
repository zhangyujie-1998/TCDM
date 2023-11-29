# [TCDM: Transformational Complexity Based Distortion Metric for Perceptual Point Cloud Quality Assessment (TVCG2023)](https://arxiv.org/abs/2210.04671)
by Yujie Zhang, Qi Yang, Yifei Zhou, Xiaozhong Xu, Le Yang, Yiling Xu

This respository is about a full-reference point cloud quality metric based on the transformational complexity. The basic motivation is that the quality of one distorted point cloud can be quantified as the complexity or the amount of effort of transforming it into its corresponding referenc

## Introduction
![image text](https://github.com/zyj1318053/TCDM/blob/main/fig/Framework.png)

The goal of objective point cloud quality assessment (PCQA) research is to develop quantitative metrics that measure point cloud quality in a perceptually consistent manner. Merging the research of cognitive science and intuition of the human visual system (HVS), in this paper, we evaluate the point cloud quality by measuring the complexity of transforming the distorted point cloud back to its reference, which in practice can be approximated by the code length of one point cloud when the other is given. For this purpose, we first make space segmentation for the reference and distorted point clouds based on a 3D Voronoi diagram to obtain a series of local patch pairs. Next, inspired by the predictive coding theory, we utilize a space-aware vector autoregressive (SA-VAR) model to encode the geometry and color channels of each reference patch with and without the distorted patch, respectively. Assuming that the residual errors follow the multi-variate Gaussian distributions, the self-complexity of the reference and transformational complexity between the reference and distorted samples are computed using covariance matrices. Additionally, the prediction terms generated by SA-VAR are introduced as one auxiliary feature to promote the final quality prediction. The effectiveness of the proposed transformational complexity based distortion metric (TCDM) is evaluated through extensive experiments conducted on five public point cloud quality assessment databases. The results demonstrate that TCDM achieves state-of-the-art (SOTA) performance, and further analysis confirms its robustness in various scenarios. 

## Results
- Dataset

- Quantitative comparison
![image text](https://github.com/zyj1318053/TCDM/blob/main/fig/table.png)


- Results reproduction

## Citation
If you find this work is helpful, please cite our paper
```
@misc{zhang2023tcdm,
      title={TCDM: Transformational Complexity Based Distortion Metric for Perceptual Point Cloud Quality Assessment}, 
      author={Yujie Zhang and Qi Yang and Yifei Zhou and Xiaozhong Xu and Le Yang and Yiling Xu},
      year={2023},
      eprint={2210.04671},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
