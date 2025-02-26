# SecureGaze: Defending Gaze Estimation Against Backdoor Attacks

This repository contains the demo for the SenSys 2025 submission **SecureGaze: Defending Gaze Estimation Against Backdoor Attacks**. We will make the implementation of SecureGaze publicly available upon paper acceptance.

## Real-time Video Demo
We demonstrate the threat of backdoor attacks on gaze estimation models in a real-world setting. Using a simple piece of white paper tape, an attacker can activate the backdoor in a compromised model, manipulating it to produce an attacker-chosen gaze direction instead of the actual one. The backdoored model was trained on the GazeCapture dataset, and the demonstration involved four invited participants. We set the center of the screen as the attacker-chosen gaze direction.

### Setup for the physical world attack
The participant is instructed to track a red square stimulus that appears sequentially at each corner of a 24-inch desktop monitor. The stimulus follows this order: top-left, top-right, bottom-right, and bottom-left. It remains visible at each corner for four seconds before disappearing and reappearing at the next position. During this process, a webcam captures full-face images of the participant at 25 Hz, which are used as inputs for gaze estimation. The setup is illustrated in the figure below.

<div align=center>
<img src="https://github.com/SecureGaze/SecureGaze/blob/main/figures/gaze_pipeline.png" alt="My Image" width="450"/>
</div>

### Demo of backdoor attack on gaze estimation model
The red arrow represents the gaze directions estimated by the backdoored gaze estimation model. As shown in Video (a), in the absence of the trigger, the arrow accurately follows the subject's gaze directions. However, when the trigger is present, as shown in Video (b), the arrow consistently points to the center of the screen (the attacker-chosen gaze direction), ignoring the subject's actual gaze. **This video demonstrates that a simple trigger, such as a piece of white paper tap, can effectively activate the backdoor behavior.**

<table style="border: none;">
  <tr>
    <td align="center">
      <img src="https://github.com/SecureGaze/SecureGaze/blob/main/figures/backdoor_benign.gif" alt="Benign Image" width="250"/>
      <br><em>(a) Estimated gaze without the trigger.</em>
    </td>
    <td align="center">
      <img src="https://github.com/SecureGaze/SecureGaze/blob/main/figures/backdoor_poisoned.gif" alt="Poisoned Image" width="250"/>
      <br><em>(b) Estimated gaze when the trigger is present. </em>
    </td>
    <td align="center">
      <img src="https://github.com/SecureGaze/SecureGaze/blob/main/figures/backdoor_both.gif" alt="Both Images" width="250"/>
      <br><em>(c) Estimated gaze when subject puts on and removes the trigger .</em>
    </td>
  </tr>
</table>




### Demo of backdoor mitigation on the gaze estimation model
We now demonstrate the effectiveness of SecureGaze in mitigating backdoor attacks. The red and blue arrows represent the gaze directions estimated by the backdoored and backdoor-mitigated gaze estimation models, respectively. As shown in video (d), in the absence of the trigger, both the red and blue arrows accurately reflect the subject's actual gaze directions. However, when the trigger is present, shown in video (e), the red arrow consistently points to the center of the screen, ignoring the subject's true gaze. By contrast, the blue arrow from the backdoor-mitigated model continues to accurately follow the subject's gaze. **This video demonstrates that SecureGaze can effectively mitigate the backdoor behavior of the backdoored gaze estimation model.**

<table>
  <tr>
    <td align="center" valign="top">
      <img src="https://github.com/SecureGaze/SecureGaze/blob/main/figures/mitigated_benign.gif" alt="Benign Image" width="250"/>
      <br><em>(d) Estimated gaze without the trigger.</em>
    </td>
    <td align="center" valign="top">
      <img src="https://github.com/SecureGaze/SecureGaze/blob/main/figures/mitigated_poisoned.gif" alt="Poisoned Image" width="250"/>
      <br><em>(e) Estimated gaze when the trigger is present.</em>
    </td>
    <td align="center" valign="top">
      <img src="https://github.com/SecureGaze/SecureGaze/blob/main/figures/mitigated_both.gif" alt="Both Images" width="250"/>
      <br><em>(f) Estimated gaze when subject puts on and removes the trigger.</em>
    </td>
  </tr>
</table>

