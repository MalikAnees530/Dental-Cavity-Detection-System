# Dental Cavity Detection System

## Overview
The **Dental Cavity Detection System** is an advanced AI-powered tool designed to analyze dental X-ray images and detect cavities with high accuracy. Leveraging a Convolutional Neural Network (CNN), this project provides an efficient and reliable diagnostic solution for dental professionals.

## Features
- **AI-Driven Analysis:** Utilizes a custom-trained CNN to identify cavities in X-ray images.
- **Input Support:** Accepts JPG, PNG, and DICOM file formats.
- **User-Friendly Interface:** Features a web-based platform with drag-and-drop upload and an "Analyze Image" button.
- **Confidence Scoring:** Provides a confidence score with each detection result.
- **Real-Time Results:** Delivers instant feedback to enhance workflow efficiency.

## Technical Details
- **Backend:** Developed using Python with the Flask framework and PyTorch for model inference.
- **Model Architecture:** A CNN with convolutional and fully connected layers, optimized for X-ray analysis.
- **Preprocessing:** Includes image resizing and tensor transformation for model compatibility.
- **Deployment:** Hosted locally with a debug port (5000) for testing.

## How to Use
1. Upload a dental X-ray image (JPG, PNG, or DICOM) via the web interface.
2. Click the "Analyze Image" button to initiate detection.
3. Receive a detailed result indicating the presence of a cavity and confidence level.
4. To open the frontend page, navigate to the `index` file in VS Code and run it to display the interface.

## Installation
- Clone the repository:  
  ```bash
  https://github.com/malikanees530/dental-cavity-detection.git
  ```
- Install dependencies:  
  ```bash
  pip install -r requirements.txt
  ```
- Run the application:  
  ```bash
  python app.py
  ```

## Future Enhancements
- Integration with cloud-based storage for scalability.
- Expansion to detect additional dental conditions.
- Development of a mobile app for on-the-go analysis.

## Author
- **Name:** Malik Anees Ahmed  
- **Role:** AI Developer  
- **Location:** Islamabad, Pakistan  
- **Email:** malikaneesahmed530@gmail.com  
- **LinkedIn:** [https://www.linkedin.com/in/malik-anees-ahmed-1a3307291/](https://www.linkedin.com/in/malik-anees-ahmed-1a3307291/)


## Acknowledgments
- Thanks to the open-source community for tools like PyTorch and Flask.

