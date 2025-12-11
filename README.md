# HEDGE: A Gradient-Aware Hybrid Ensemble Defense Against Transferable Adversarial Evasion in Intrusion Detection and Prevention Systems
Codes and Datasets for A Gradient-Aware Hybrid Ensemble Defense Against Transferable Adversarial Evasion in Intrusion Detection and Prevention Systems

This repository contains the source code, experiments, and processed datasets for the paper: "HEDGE: A Gradient-Aware Hybrid Ensemble Defense Against Transferable Adversarial Evasion in Intrusion Detection and Prevention Systems."

HEDGE implements a hybrid ensemble defense that strengthens the robustness of modern Intrusion Detection and Prevention Systems (IDPSs) against transferable adversarial evasion attacks, especially those that exploit asymmetric vulnerabilities between deep learning (DL) and gradient-boosting tree (GBT) models.

# Key Features
## 1. Gradient-Aware Transferability Analysis

Measures asymmetric adversarial transfer between DL → GBT and GBT → DL.

Identifies structural factors enabling high cross-model attack success.

Includes attack-space visualization, Jacobian-based similarity metrics, and feature-sensitivity maps.

## 2. Hybrid Adversarial Training (HAT)

Generates both model-specific and cross-model adversarial examples.

Integrates them into a unified robustness training pipeline.

Supports multiple attack algorithms: FGSM, PGD, BIM, MIM, T-PGD, and tree-specific decision-boundary perturbations.

## 3. Lightweight Feature-Squeezing Layer

Implements controlled feature-space compression.

Enhances input stability with negligible inference cost.

Works as a drop-in layer for any PyTorch model.

## 4. DL–GBT Ensemble Inference

Joint decision mechanism using complementary boundaries of DL and GBT models.

Resistant to instances that evade one model but not the other.

Provides robust confidence fusion and threat-level scoring.

# Results Summary

HEDGE achieves:

- Attack Success Rate: 99–100% → < 1% on most adversarial attacks

- Clean Accuracy: 99% (NSL-KDD) and 95% (CSE-CIC-IDS2018)

- Large robustness gains from Hybrid Adversarial Training
