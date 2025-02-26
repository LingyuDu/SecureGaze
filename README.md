# SecureGaze: Defending Gaze Estimation Against Backdoor Attacks

This repository contains the introductions, the codes, and the demos for SenSys 2025 paper **SecureGaze: Defending Gaze Estimation Against Backdoor Attacks** by [Lingyu Du](https://github.com/LingyuDu), [Yupei Liu](https://liu00222.github.io/), [Jinyuan Jia](https://jinyuan-jia.github.io/), and [Guohao Lan](https://guohao.netlify.app/). If you have any questions, please send an email to Lingyu.Du AT tudelft.nl.

## Description
Gaze estimation models are widely used in applications such as driver attention monitoring and human-computer interaction. In essence, gaze estimation is a **regression** task that uses either eye or facial images to predict gaze direction. Similar to other computer vision tasks, deep learning advancements have greatly enhanced gaze estimation performance, but expose gaze estimation models to **backdoor attacks**. In such attacks, adversaries inject backdoor triggers by poisoning the training data, creating a backdoor vulnerability: the model performs normally with benign inputs, but produces manipulated gaze directions when a specific trigger is present. This compromises the security of many gaze-based applications, such as causing the model to fail in tracking the driver's attention. To date, there is no defense that addresses backdoor attacks on gaze estimation models. In response, we introduce SecureGaze, the first solution designed to protect gaze estimation models from such attacks. Unlike classification models, defending gaze estimation poses unique challenges due to its continuous output space and globally activated backdoor behavior. By identifying distinctive characteristics of backdoored gaze estimation models, we develop a novel and effective approach to reverse-engineer the trigger function for reliable backdoor detection. Extensive evaluations in both digital and physical worlds demonstrate that SecureGaze effectively counters a range of backdoor attacks.

## Real-time Video Demo for Physical World Attack
We demonstrate the threat of backdoor attacks on gaze estimation models in a real-world setting. Using a simple piece of white paper tape, an attacker can activate the backdoor in a compromised model, manipulating it to produce an attacker-chosen gaze direction instead of the actual one. The backdoored model was trained on the GazeCapture dataset, and the demonstration involved four invited participants. We set the center of the screen as the attacker-chosen gaze direction. More training details can be found in our paper.

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




### Demo of backdoor mitigation on the gaze estimation model
We now demonstrate the effectiveness of SecureGaze in mitigating backdoor attacks. The red and blue arrows represent the gaze directions estimated by the backdoored and backdoor-mitigated gaze estimation models, respectively. As shown in video (d), in the absence of the trigger, both the red and blue arrows accurately reflect the subject's actual gaze directions. However, when the trigger is present, shown in video (e), the red arrow consistently points to the center of the screen, ignoring the subject's true gaze. By contrast, the blue arrow from the backdoor-mitigated model continues to accurately follow the subject's gaze. **This video demonstrates that SecureGaze can effectively mitigate the backdoor behavior of the backdoored gaze estimation model.**

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

