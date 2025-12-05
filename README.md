🖐️ Grasp Pressure Prediction from Haptic & Robot Teleoperation Data
This repository contains a framework to predict the average pressure of successful grasps in a teleoperated robotic-hand system using haptic signals and robot end-effector trajectories.
We evaluate multiple architectures (DNN, RNN, 1D-CNN) and show that Dilated 1D-CNNs most effectively capture the short, local temporal patterns that define grasp success.
🚀 Overview
Predicting the correct pressure for a robot grasp requires understanding how humans grasp objects during teleoperation.
We collected synchronized haptic and motion data from participants performing grasp tasks, then trained neural networks to estimate the pressure at the final successful grasp.
Key findings:
Local temporal features, not long-range dependencies, are the dominant cues for grasp prediction.
Dilated 1D-CNNs outperform DNNs and RNNs by a large margin.
Adding robot trajectory significantly improves performance despite a smaller dataset.
📊 Dataset
Participants: 27
Scenarios: 20 teleoperated manipulation tasks
Trials: 795 total sequences (251 with valid robot trajectories)
Signals Collected:
Thumb pressure
Index finger pressure
End-effector X, Y, Z
Target: Mean pressure during the successful grasp at the end of each trial
Example dataset visualization:
figs/data.png
Each trial includes:
Multiple failed attempts
One final successful grasp
Synchronized pressure + robot motion signals
🧠 Methods
We implemented and compared three major neural architectures:
1. Feedforward Neural Network (DNN / ResNet-DNN)
Uses flattened time-series inputs → extremely high dimensional
1–2M parameters
Cannot capture temporal structure
High train & test errors
➡️ Serves only as a baseline.
2. Recurrent Neural Networks (LSTM / GRU)
Trained with various window sizes, step sizes, and depths
Applied dropout & batch norm
Training slow due to sequential nature
Underfitting observed consistently
➡️ Long-term dependency modeling is not useful for this task.
3. Dilated 1D Convolutional Neural Network (Best Model)
Benefits:
Captures local pressure events (spikes, micro-slip patterns)
Expands receptive field with dilation
Trains efficiently
Requires far fewer parameters
Best architectures:
Dataset	CNN Structure
Haptic Only	5 CNN layers, filters [32, 32, 64, 64, 64], kernel=10, dilation=[1,2,4,4,4]
Haptic + Robot	3 CNN layers, 64 filters each, dilation=[1,4,16], kernel=2
Architecture figure:
figs/architecture_hr.pdf
🏆 Results
Performance Comparison Table
Model	Params	Train MSE	Test MSE
DNN (H)	1,213,762	0.025	0.071
DNN (H+R)	1,689,218	0.017	0.084
ResNet-DNN	2,385,938	0.031	0.12
LSTM (H)	5,427,746	0.085	0.075
1D-CNN (H)	174,658	0.0041	0.0048
1D-CNN (H+R)	26,274	0.0011	0.0056
🔑 Key Results
The 1D-CNN (H+R) is the best performing and most compact model.
Robot trajectory provides strong complementary information.
DNN and RNN underperform due to:
loss of temporal structure (DNN)
inability to focus on short-range events (RNN)
Prediction examples:
figs/ho_prediction_original_20251203_124232_Test.png
figs/hr_prediction_original_20251203_095144_Test.png
💬 Discussion
Why 1D-CNN Works Best
Grasp success signals are short, local, and distinctive.
Dilated convolutions allow:
Local feature extraction
Broader temporal context
Efficient parallel training
Why DNN Fails
Flattening removes all temporal structure
Too many parameters → overfits quickly
Why RNN Fails
Long-term dependency modeling is not relevant
Sequential operations → slow & memory-heavy
Underfitting observed even with tuning
🔮 Future Work
Expand dataset size and diversity
Integrate more sensing modalities (e.g., joint torque)
Explore multi-task learning
Real-time deployment in teleoperation
🧪 How to Run
Installation
pip install -r requirements.txt
Training
python train.py --config configs/cnn_haptic.yaml
python train.py --config configs/cnn_haptic_robot.yaml
Evaluation
python evaluate.py --model checkpoint/best_model.pt
👥 Contributors
Name	Contribution
Juhyun Jung	Implemented RNN & ResNet models, visualizations, literature review
Sehui Jeong	Proposed idea, provided dataset, implemented DNN & 1D-CNN framework
Joint	Analysis, discussion, report writing
📚 References
Murali et al., Learning to Grasp Without Seeing (2018)
Zhuwawu & El-Hussieny, Grasp Prediction via LSTM and Haptic Feedback (2025)
