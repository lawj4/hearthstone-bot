# 🃏 Hearthstone Bot

An automated system for detecting game state and simulating player interactions in Hearthstone. This bot uses computer vision and image processing to read in-game data and execute strategic actions.

## 📄 Development Process

Detailed documentation of the development steps, design decisions, and challenges can be found here:  
👉 [Hearthstone Bot Development Process](https://docs.google.com/document/d/1F02ZN9schgOm8mOwWUKTzamb2eIfqOuHPYmD4_-tSNw/edit?usp=sharing)

## 🚀 Features

- Detects and reads mana crystals, board state, and card regions.
- Uses OpenCV for image region extraction and contour analysis.
- Designed to handle tilted or partially obscured cards.
- Modular architecture for easy testing and scaling.

## 🛠️ Technologies Used

- Python 3
- OpenCV
- NumPy

## 🧠 How It Works

1. Captures screenshots of the Hearthstone game window.
2. Isolates regions of interest (e.g., mana crystals, hand cards).
3. Applies thresholding and contour detection to identify elements.
4. Uses shape and position analysis to interpret game state.
