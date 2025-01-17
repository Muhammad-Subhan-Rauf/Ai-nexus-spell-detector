# Spell Detection AI: Learn and Master Your Spells ✨  

This repository contains the code for a **spell detection application** that won the **AI Nexus** category of the prestigious **PsiFi** event, organized by the **Spades** student society at **LUMS University**. The competition’s theme revolved around assisting wizards in training, and this app is a magical solution to a truly enchanting problem!  

## The Problem  
In the magical realm, novice wizards often struggle to perfect the intricate wand movements required to cast spells. To address this, the wise and legendary wizard **DumbleMore** (a playful homage to Dumbledore) has tasked us with creating an AI-powered app that teaches wizards how to perform their spell movements correctly.  

The challenge was to:  
1. Track the wand’s **green chroma-colored tip** as it moves through the air during a spell gesture.  
2. Capture and normalize the wand's **x, y coordinates** over **100 frames (approximately 3 seconds)**.  
3. Train an AI model to classify the movement patterns into one of **25 predefined spells**.  
4. Provide real-time feedback, making it both engaging and effective for wizards-in-training.  

## Unique Features  
### 1. **Voice Feedback with a Powerful Wizard’s Voice**  
The app takes spell training to the next level with **AI-generated voice feedback**. As a wizard performs a spell, the app speaks the spell's name in the voice of a commanding, powerful wizard, making the learning experience truly immersive and magical.  

### 2. **Pre-Prepared Dataset**  
To make the application accessible and easy to use, the **entire dataset** we used for training (including both organizer-provided data and manually created samples) is included in this repository. You don’t need to spend time collecting data—just focus on training the model and running the application!  

## How It Works  
1. **Data Preprocessing**  
   - The wand’s green chroma tip is isolated and tracked using computer vision techniques, capturing its movements as a sequence of **x, y coordinates** normalized to a range of `0-1`.  

2. **AI Model**  
   - A robust **LSTM-based model** with three layers is used to capture the temporal dependencies of the wand movements.  
   - The model predicts the spell being performed with high accuracy.  

3. **Real-Time Detection**  
   - Once trained, the application tracks the wand in real-time, predicts the spell, and provides both visual and voice feedback.  

## Achievements  
This project won the **AI Nexus** event under PsiFi, earning a **75,000 PKR prize**. Its combination of accuracy, creativity, and user engagement set it apart from the competition.  

## Getting Started  
1. **Install Dependencies**  
   Run the following command to install all required packages:  
   ```bash
   pip install -r requirements.txt
   ```  

2. **Train the Model**  
   Train the model using the provided dataset:  
   ```bash
   python main_train.py
   ```  

3. **Run the Application**  
   After training is complete, launch the application for real-time spell detection:  
   ```bash
   python main.py
   ```  

## Open Source Contribution  
This repository is open to all wizards and AI enthusiasts! Whether you’re exploring sequence modeling, experimenting with spell detection, or simply looking to enhance your wizarding skills, this project is here to inspire.  

**Let the spell training begin! ✨**
