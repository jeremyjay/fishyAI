# Fishy AI


## Overview
This is an AI that plays the old flash game "Fishy". The objective of the game is to eat smaller fish, and avoid getting eaten by bigger fish. The whole game is played by pressing the four arrow keys

## How to Use This AI
1. Clone this repository
2. Run the fishy.swf file using a Shockwave File Player (I have been using Elmedia Video Player)
3. When the Player makes it to the title screen, run the command
`
python3 main.py
`
4. Click on the Elmedia Video Player to focus on that window
5. Sometimes the AI doesn't recognize that the game is playing, so you will have click "Play Game", and maybe click on the screen a couple more times. It will usually recognize that it's playing within a few seconds.

## Current Status
- **Code Base**: Implemented in Python.
- **AI Strategy**: Utilizes Deep Q-Learning.
- **Libraries & Tools**: Employs Quartz and OpenCV for screen capture and processing. Leverages EasyOCR for Optical Character Recognition (OCR) to decipher game state data. Utilizes the Keyboard library for key operations.


## Known Areas for Improvement
1. **Reward System**: 
   - Create a mechanism to reward the AI for consuming smaller fish. 
2. **Processing Speed**:
   - Presently operating on a CPU, achieving approximately 3 iterations/second. This needs enhancement.
3. **Game State Detection**:
   - Improve AI's ability to detect when the game has started without manual clicks.



