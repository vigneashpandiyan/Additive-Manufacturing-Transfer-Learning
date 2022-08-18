# Additive-Manufacturing-Transfer-Learning
This repo hosts the codes that were used in journal work "Deep transfer learning of additive manufacturing mechanisms across materials in metal-based laser powder bed fusion process".

# Journal link
https://doi.org/10.1016/j.jmatprotec.2022.117531

# Overview

So far, Knowledge gained for real-time monitoring of a particular material composition from sensor signatures cannot be re-used to monitor another material composition in Additive Manufacturing (AM). A topic rarely researched in AM. Inspired by the idea of Transfer learning, we, in the article recently published, demonstrate the Knowledge learnt by the two native Deep Learning (DL) networks on four Laser Powder Bed Fusion (LPBF) process mechanisms in stainless steel (316 L) such as balling, Lack of Fusion (LoF) pores, conduction mode, and keyhole pores can be transferred to bronze (CuSn8). 

![Figure 8](https://user-images.githubusercontent.com/39007209/185372473-a13f63b2-f795-4ed1-b965-5ecfd83bc1e0.jpg)

# Transfer learning

Traditional machine learning algorithms are trained based on a particular feature space to solve specific tasks. With a change in feature distribution or with the introduction of a new task, the algorithm might fail to adapt. In this case, the algorithm has to be re-trained from scratch. Transfer learning is a paradigm where a model already trained on a similar task is reused with minimum training to accomplish the second task. With neural architectures built with deep layers, the pretrained weights in them can be reused with minimum training and usage of computing resources. But, it is also to be noted that transfer learning is handy in deep learning if the features learned by the pre-trained model from the first task are general. Figure below presents different strategies adapted based on the complexity of the second task. In the case of tasks with higher complexity, the whole network is trained from the saved weights, as shown in Figure below. For a similar task, the few convolution layers or classification layers are trained as illustrated in Figure below. The training time is directly proportional to the number of learnable parameters to be updated during training.  Apart from image recognition and segmentation applications, the transfer learning paradigm has also been applied to fault diagnosis in locomotive bearings , identify remaining useful life prediction of tool in manufacturing processes, which prompted us to exploit this technique towards AM.

![Methodology](https://user-images.githubusercontent.com/39007209/185372898-5aed3829-ca18-4b8c-864a-0a26bded9097.jpg)

# Code
```bash
git clone https://github.com/vigneashpandiyan/Additive-Manufacturing-Transfer-Learning
cd Additive-Manufacturing-Transfer-Learning

```

# Citation
```
@article{pandiyan2022deep,
  title={Deep transfer learning of additive manufacturing mechanisms across materials in metal-based laser powder bed fusion process},
  author={Pandiyan, Vigneashwara and Drissi-Daoudi, Rita and Shevchik, Sergey and Masinelli, Giulio and Le-Quang, Tri and Log{\'e}, Roland and Wasmer, Kilian},
  journal={Journal of Materials Processing Technology},
  volume={303},
  pages={117531},
  year={2022},
  publisher={Elsevier}
}
```

