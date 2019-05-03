# chrome_dino_ai
Simple deep learning bot to play chrome dinosaur game, written in python.

## Requirements
python 3.4-3.6
tensorflow
numpy
pynput (for keyboard input simulation)
opencv (for handling images)
mss (for capturing screen)
pyxhook (for logging jump input)

## How to use
1. Use info.py to define constant values such as image size, screenshot position or model's path.
2. Run save_data.py to create dataset. Press \`(grave) to start recording and \` again to stop recording.
3. Run preprocess.py to simplify images.
4. Run trainer.py to load model(if available), train it and save model back.
5. Run program.py to test the model to game. press \` to start and \` again to stop.
