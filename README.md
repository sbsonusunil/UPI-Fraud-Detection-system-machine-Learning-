UPI Fraud Detection System

An end-to-end, industry-ready UPI fraud detection system built using real transaction data, machine learning, and a production-style pipeline.
Designed to simulate how Indian fintech and banking systems detect fraudulent UPI transactions in both real-time and batch scenarios.

Project Overview

UPI fraud is a growing challenge in India due to high transaction volumes, diverse user behavior, and evolving fraud patterns.

This project demonstrates:

A complete ML lifecycle (data → model → deployment)

Training- serving parity using a shared preprocessor

Real-time fraud risk scoring via a Streamlit dashboard

Batch inference pipeline similar to bank back-office systems

Business Problem

Objective:
Predict whether a UPI transaction is fraudulent or legitimate using historical transaction patterns.

Why it matters:

Prevents financial losses

Improves customer trust

Enables proactive fraud operations

Solution Approach
1.️ Data Processing

Cleaned and processed raw UPI transaction data

Engineered time-based and behavioral features

Handled categorical and numerical variables properly

2. Feature Engineering

Transaction amount normalization (amount_log)

Cyclical time features:

hour_sin, hour_cos

day_of_week_sin, day_of_week_cos

Date-based features:

year, month, day, minute

3. Model Training

Algorithm: XGBoost Classifier

Handles class imbalance and non-linear fraud patterns

Evaluated using ROC-AUC

4. Preprocessing Strategy (Industry-Grade)

A single shared preprocessor.pkl

Used in:

Training pipeline

Inference pipeline

Streamlit app

Prevents training–serving skew