# SecureGaze: Defending Gaze Estimation Against Backdoor Attacks

This repository contains the introductions, the codes, and the demos for **ACM SenSys 2025** paper **SecureGaze: Defending Gaze Estimation Against Backdoor Attacks** by [Lingyu Du](https://github.com/LingyuDu), [Yupei Liu](https://liu00222.github.io/), [Jinyuan Jia](https://jinyuan-jia.github.io/), and [Guohao Lan](https://guohao.netlify.app/). If you have any questions, please send an email to Lingyu.Du AT tudelft.nl.

## Description
Gaze estimation models are widely used in applications such as driver attention monitoring and human-computer interaction. In essence, gaze estimation is a **regression** task that uses either eye or facial images to predict gaze direction. Similar to other computer vision tasks, deep learning advancements have greatly enhanced gaze estimation performance, but expose gaze estimation models to **backdoor attacks**. As is shown in the figure below, in such attacks, adversaries inject backdoor triggers by poisoning the training data, creating a backdoor vulnerability: the model performs normally with benign inputs, but produces manipulated gaze directions when a specific trigger is present. This compromises the security of many gaze-based applications, such as causing the model to fail in tracking the driver's attention. To date, there is no defense that addresses backdoor attacks on gaze estimation models. In response, we introduce SecureGaze, the first solution designed to protect gaze estimation models from such attacks. Unlike classification models, defending gaze estimation poses unique challenges due to its continuous output space and globally activated backdoor behavior. By identifying distinctive characteristics of backdoored gaze estimation models, we develop a novel and effective approach to reverse-engineer the trigger function for reliable backdoor detection. Extensive evaluations in both digital and physical worlds demonstrate that SecureGaze effectively counters a range of backdoor attacks.

<div align=center>
<img src="https://github.com/LingyuDu/SecureGaze/blob/main/figures/Backdoor_attack.png" alt="My Image" width="850"/>
  <br>Backdoor attacks on gaze estimation model.
</div>


## Real-time Video Demo for Physical World Attack
We demonstrate the threat of backdoor attacks on gaze estimation models in a real-world setting. Using a simple piece of white paper tape, an attacker can activate the backdoor in a compromised model, manipulating it to produce an attacker-chosen gaze direction instead of the actual one. The backdoored model was trained on the [GazeCapture dataset](https://gazecapture.csail.mit.edu/). We set the center of the screen as the attacker-chosen gaze direction. More training details can be found in our paper.

### Setup for the physical world attack
The participant is instructed to track a red square stimulus that appears sequentially at each corner of a 24-inch desktop monitor. The stimulus follows this order: top-left, top-right, bottom-right, and bottom-left. It remains visible at each corner for four seconds before disappearing and reappearing at the next position. During this process, a webcam captures full-face images of the participant at 25 Hz, which are used as inputs for gaze estimation. The setup is illustrated in the figure below.

<div align=center>
<img src="https://github.com/LingyuDu/SecureGaze/blob/main/figures/gaze_pipeline.png" alt="My Image" width="450"/>
</div>

### Demo of backdoor attack on gaze estimation model
The red arrow represents the gaze directions estimated by the backdoored gaze estimation model. As shown in Video (a), in the absence of the trigger, the arrow accurately follows the subject's gaze directions. However, when the trigger is present, as shown in Video (b), the arrow consistently points to the center of the screen (the attacker-chosen gaze direction), ignoring the subject's actual gaze. **This video demonstrates that a simple trigger, such as a piece of white paper tap, can effectively activate the backdoor behavior.**

<table style="border: none;">
  <tr>
    <td align="center">
      <img src="https://github.com/LingyuDu/SecureGaze/blob/main/figures/backdoor_benign.gif" alt="Benign Image" width="250"/>
      <br><em>(a) Estimated gaze without the trigger.</em>
    </td>
    <td align="center">
      <img src="https://github.com/LingyuDu/SecureGaze/blob/main/figures/backdoor_poisoned.gif" alt="Poisoned Image" width="250"/>
      <br><em>(b) Estimated gaze when the trigger is present. </em>
    </td>
    <td align="center">
      <img src="https://github.com/LingyuDu/SecureGaze/blob/main/figures/backdoor_both.gif" alt="Both Images" width="250"/>
      <br><em>(c) Estimated gaze when subject puts on and removes the trigger .</em>
    </td>
  </tr>
</table>

## Fundamental Difference between Backdoored Classification Models and Backdoored Gaze Estimation Models

While countermeasures have been developed to combat backdoor attacks in various classification tasks, no solution has been proposed for gaze estimation, which differs as it is a regression task. We reveal the following two inherent differences between backdoored gaze estimation and classification models that make existing defenses ineffective for gaze estimation.

* **Specific *vs.* Global Activation in Feature Space**. In backdoored classification models, the backdoor behavior is often triggered by the activation of *a specific set of compromised neurons* in the feature space. This characteristic allows existing feature-space defenses to distinguish compromised and benign neurons for backdoor detection. However, we reveal that backdoor behavior in gaze estimation models is driven by *the activation of all neurons in the feature space*, rather than a specific subset. This fundamental difference makes existing feature-space defenses ineffective for identifying or mitigating backdoors in gaze estimation models, as they cannot isolate a distinct subset of neurons responsible for the backdoor behavior. 

* **Discrete *vs.* Continuous Output Space**. 
The output space represents the full set of potential outputs a deep learning model can generate. Many existing defenses leverage the output-space characteristics of backdoored classification models for backdoor detection. These approaches require an exhaustive enumeration of all possible output labels. This strategy is feasible for classification models, such as face recognition, which have *a discrete output space limited to finite class labels*, e.g., a set of possible identities. By contrast, gaze estimation models have a *continuous output space* that spans *an infinite number of possible output vectors*. Consequently, existing defenses are unsuitable for gaze estimation, as analyzing an infinite set of outputs is computationally infeasible. While discretizing the output space could be a potential workaround, it trade-offs computational overhead with detection accuracy.

<div align=center>
<img src="https://github.com/LingyuDu/SecureGaze/blob/main/figures/difference_DCM_DRM.png" alt="My Image" width="450"/>
  <br>
</div>

## Overview of Reverse-engineering the Trigger Function
We propose SecureGaze to identify backdoored gaze estimation models by reverse-engineering the trigger function $\mathcal{A}$. The overview of reverse-engineering the trigger function is shown in the figure below. Our approach uses a generative model, $M_{\theta}$, to approximate $\mathcal{A}$. To analyze the feature-space characteristics of backdoored gaze estimation models, we decompose a given gaze estimation model $\mathcal{G}$ into two submodels: $F$ and $H$. Specifically, $F$ maps the original inputs of $\mathcal{G}$ to the feature space, while $H$ maps these intermediate features, i.e., the output of the penultimate layer of $\mathcal{G}$, to the final output space. We train $M_{\theta}$ to generate reverse-engineered poisoned images that can lead to the feature and output spaces characteristics of backdoored gaze estimation models.
**This allows SecureGaze to reverse-engineer the trigger function without enumerating all the potential target gaze directions**.

<div align=center>
<img src="https://github.com/LingyuDu/SecureGaze/blob/main/figures/over_view.png" alt="My Image" width="600"/>
  <br>
</div>


## Real-time Video Demo for Physical World Backdoor Mitigation
We now demonstrate the effectiveness of SecureGaze in mitigating backdoor attacks. The red and blue arrows represent the gaze directions estimated by the backdoored and backdoor-mitigated gaze estimation models, respectively. As shown in video (d), in the absence of the trigger, both the red and blue arrows accurately reflect the subject's actual gaze directions. However, when the trigger is present, as shown in video (e), the red arrow consistently points to the center of the screen, ignoring the subject's true gaze. By contrast, the blue arrow from the backdoor-mitigated model continues to accurately follow the subject's gaze. **This video demonstrates that SecureGaze can effectively mitigate the backdoor behavior of the backdoored gaze estimation model.**

<table>
  <tr>
    <td align="center" valign="top">
      <img src="https://github.com/LingyuDu/SecureGaze/blob/main/figures/mitigated_benign.gif" alt="Benign Image" width="250"/>
      <br><em>(d) Estimated gaze without the trigger.</em>
    </td>
    <td align="center" valign="top">
      <img src="https://github.com/LingyuDu/SecureGaze/blob/main/figures/mitigated_poisoned.gif" alt="Poisoned Image" width="250"/>
      <br><em>(e) Estimated gaze when the trigger is present.</em>
    </td>
    <td align="center" valign="top">
      <img src="https://github.com/LingyuDu/SecureGaze/blob/main/figures/mitigated_both.gif" alt="Both Images" width="250"/>
      <br><em>(f) Estimated gaze when subject puts on and removes the trigger.</em>
    </td>
  </tr>
</table>

## Codes and Backdoored Gaze Estimation Models
We develop SecureGaze using TensorFlow-gpu 2.9.0.

* The backdoored gaze estimation model with a piece of white tape as the trigger is available at [physical-world backdoored model](https://drive.google.com/drive/folders/1fr41I6Y3moDnNB6XCytBWUXq8Q-5FS5a?usp=sharing).
* CL_model.py defines the architecture of the generative model and the gaze estimator.
* 
## Citation 

Please cite the following paper in your publications if the code helps your research.

<div style="border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
<pre>
@article{du2025securegaze,
  title={SecureGaze: Defending Gaze Estimation Against Backdoor Attacks},
  author={Du, Lingyu and Liu, Yupei and Jia, Jinyuan and Lan, Guohao},
  journal={Proceedings of the 23rd ACM Conference on Embedded Networked Sensor Systems},
  year={2025},
  publisher={ACM New York, NY, USA}
}
</pre>
</div>


