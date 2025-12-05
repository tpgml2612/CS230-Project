
# Grasp Pressure Prediction Using Haptic & Robot Teleoperation Data

This repository presents a deep-learning framework to estimate the **average pressure of a final successful grasp** using **haptic signals** and **robot end-effector trajectories**.  
We evaluate multiple neural architectures—**DNN**, **RNN**, and **Dilated 1D-CNN**—and demonstrate that **1D-CNNs best capture short, local temporal patterns**, which are critical for predicting grasp pressure.

---

## 📌 Overview

Teleoperated robotic grasping requires understanding how humans apply pressure during a successful grasp.  
We collected synchronized pressure and motion data from **27 participants** across **20 scenarios**, each containing multiple failed grasps followed by a successful trial.

This project explores:
- How different neural architectures learn grasp-related temporal features  
- Which signals (haptics vs. robot motion) contribute most to prediction  
- Why local temporal cues dominate grasp–pressure estimation  

---

## 📊 Dataset Description

- **795 trials total**  
- **251 trials** include synchronized robot trajectory  
- **Signals collected:**  
  - Thumb pressure  
  - Index pressure  
  - End-effector X (depth), Y (horizontal), Z (vertical)  
- **Target:** Mean pressure of the *final successful grasp*  
- Each sequence includes several failures followed by one successful grasp  

(Example figure: `figs/data.png`)

---

## 🧠 Methods

### 1. Feedforward Neural Networks (DNN / ResNet-DNN)
- Flattened time-series input → >1M parameters  
- High bias and poor generalization  
- Unable to capture temporal structure  

### 2. Recurrent Neural Networks (LSTM / GRU)
- Trained with various window/step sizes and depths  
- Long training time due to sequential computation  
- Underfitting persists across configurations  
- Long-term dependencies are *not* important for this task  

### 3. Dilated 1D-CNN (Best Performing Model)
- Captures short-range and local grasp events (pressure spikes, micro-slips)  
- Dilations expand receptive field efficiently  
- Trains quickly and generalizes well  
- Performs best on all datasets  

**Best Architectures**

| Dataset | CNN Structure |
|--------|---------------|
| Haptic Only | 5 conv layers, filters [32, 32, 64, 64, 64], kernel=10, dilation=[1,2,4,4,4] |
| Haptic + Robot | 3 conv layers, 64 filters each, kernel=2, dilation=[1,4,16] |

(Architecture figure: `figs/architecture_hr.pdf`)

---

## 🏆 Results

### 📈 Quantitative Comparison

| Model | Params | Train MSE | Test MSE |
|-------|--------|-----------|----------|
| DNN (H) | 1,213,762 | 0.025 | 0.071 |
| DNN (H+R) | 1,689,218 | 0.017 | 0.084 |
| ResNet-DNN | 2,385,938 | 0.031 | 0.120 |
| LSTM (H) | 5,427,746 | 0.085 | 0.075 |
| **1D-CNN (H)** | 174,658 | **0.0041** | **0.0048** |
| **1D-CNN (H+R)** | **26,274** | **0.0011** | **0.0056** |

### 🔑 Key Takeaways
- **1D-CNN is the most effective architecture**, outperforming DNN/RNN by >10×  
- Robot motion data significantly improves prediction even with fewer samples  
- DNN and RNN fail to capture the localized patterns crucial for grasp events  

(Outputs shown in: `figs/CNN_output_train.png`, `figs/CNN_output_test.png`)

---

## 💬 Discussion

### Why 1D-CNN Works
- Grasp success depends on **short, localized temporal events**  
- Dilated convolutions capture both short-range and broader context  
- Efficient training and fewer parameters → less overfitting  

### Why DNN Fails
- Flattening destroys temporal structure  
- Excessively high parameter count  

### Why RNN Fails
- Long-term dependencies unnecessary  
- Sequential computation → slow training + underfitting  
- Unable to isolate quick pressure/motion transitions  

---

## 🔮 Future Work
- Expand dataset with more diverse teleoperation trials  
- Incorporate joint torque, velocity, or tactile array data  
- Explore transformers or hybrid CNN-RNN approaches  
- Build real-time grasp-pressure adaptation for teleoperation  


---

## Acknowledgment
This README was edited with the assistance of ChatGPT for grammar and style.
