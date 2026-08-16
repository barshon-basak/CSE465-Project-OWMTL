# Possible Papers

## 1. Validating Your OOD Dataset Strategy
Deep Learning-Driven Early Diagnosis of Respiratory Diseases using CNN-RNN Fusion on Lung Sound Data (Published: Nov 2025)

The Core Focus: This paper develops a CNN-RNN fusion model for lung sound analysis.

Why it matters for your project: It explicitly uses both the ICBHI and Coswara datasets to handle respiratory diseases. This directly validates your pivot to use Coswara as a large-N external dataset. You can cite this to justify combining these specific datasets in modern respiratory ML pipelines.

## 2. Supporting the Edge-Deployment (CQKD) Narrative
A Computationally Efficient Fully Convolutional Network for Respiratory Sound Classification (Presented at: EUSIPCO 2025)

The Core Focus: Formulating a simple, low-computational-cost deep learning architecture that achieves state-of-the-art results for real-world medical applications.

Why it matters for your project: The authors validate their lightweight model using the SPRSound dataset (another one of your OOD targets) and ICBHI. This paper is an excellent anchor for your Related Work section when you introduce Dr. Khan's CQKD method. You can contrast their standard architectural efficiency with your novel cluster-quantized distillation approach.

## 3. Backbone Ablation & Architectural Justification
MFITN-E2NetGA: a transformer-based multi-level fusion framework for multi-class respiratory disease classification from lung sounds (Published: Taylor & Francis, 2025)

The Core Focus: Utilizing a transformer-based multi-level fusion framework to classify multi-class respiratory diseases.

Why it matters for your project: In Section 8 of your guideline, you plan to ablate the backbone choice (CNN vs. transformer-based audio encoder). This 2025 paper gives you the perfect recent reference for the transformer side of that baseline comparison.

## 4. Refining the Primary Sound-Event Task
ADFF-Net: An Attention-Based Dual-Stream Feature Fusion Network for Respiratory Sound Classification (Published: MDPI, Jan 2026)

The Core Focus: Introducing an attention mechanism and dual-stream feature fusion to capture both global and local acoustic patterns in respiratory sounds.

Why it matters for your project: Your primary, statistically defensible task is sound-event classification (Normal/Crackle/Wheeze/Both) using 6,898 cycles. This paper highlights the current state-of-the-art for feature extraction on that exact cycle-level task, which you can use to benchmark your shared backbone's performance.

## 5. Baseline Modeling & Multi-Architecture Comparisons
End-to-End Acoustic Classification of Respiratory Sounds Using Multi-Architecture Deep Neural Networks (Published: MDPI, March 2026)

The Core Focus: A holistic comparative analysis of distinct modeling paradigms (like 1D-CNNs and CNN-LSTMs) for classifying respiratory sounds into normal, crackles, and wheezes.

Why it matters for your project: This provides excellent, highly recent comparative baselines for the ICBHI dataset. When reviewers ask why you chose your specific shared-backbone architecture before applying the CQKD consistency signal, you can point to the foundational analyses in this paper.
