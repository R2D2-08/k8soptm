# Optimizing Cluster Management

This repository outlines the approach taken to optimize cluster management, predict pod failures and further decrease downtime.

## Phase 1: Prediction
Phase 1 of this project involves the use of an AI model to predict the following:
- Pod failures
- Resource exhaustion
- Network & connectivity issues
- Service Disruption
To achieve this we split the phase into a set of sub-phases:
### Data collection for training
- Prometheus and Grafana for real time metrics
- Simulation of requests using Apache Jmeter
- Form a Base Dataset
- Use the above to create a larger dataset
### Data Pre-Processing
- Pre-process and clean the data
- Prepare for training
### Training the model
- Train
### Testing the model
- Ensure the availability of a test dataset when creating the entire dataset
- Test and evaluate the model
### Re-train and Re-test with different models
### Anomaly detection based approach
- Instead of relying on prediction which heavily depends on the dataset the model was trained on
- Use an anomaly detection based approach which 'learns' what normal behavior is
### Finalize the model
- Design an approach to either use the prediction model or the anomaly detection model
- Finalize the workflow (Technically a part of phase 2)
### Containerize
- Load the architecture in the server and copy the params into that directory
- Set up the Dockerfile 
- Build the image and test
## Phase 2: Remediation
### Re-train over incoming data
- Re-train over set intervals in time with incoming data
- Gives the model the ability to better suit to its user-server model

