import keyboard
import cv2 as cv
from time import time, sleep
from windowcapture import WindowCapture
from ocrProcessor import EasyOCRProcessor
import numpy as np
import os
import model as Qmodel


# Limit the FPS
TARGET_FPS = 30
FRAME_DURATION = 1 / TARGET_FPS


# Create the deep-q neural network model
# Create the model
input_shape = (84, 84, 4)
n_actions = 4  # Number of arrow keys
dqn_model = Qmodel.create_dqn_model(input_shape, n_actions)

# Training loop
agent = Qmodel.DQNAgent(dqn_model)

if os.path.exists('dqn_weights.h5'):
    agent.load_weights('dqn_weights.h5')


# initialize the WindowCapture class
wincap = WindowCapture('fishy.swf')

ocr_processor = EasyOCRProcessor()

true_screen_width = wincap.get_window_width()
true_screen_height = wincap.get_window_height()
print (true_screen_height)
print (true_screen_width)
screen_width = 84
screen_height = 84

# death box
deathbox_left_edge = int(true_screen_width/2 - true_screen_width/4)
deathbox_top_edge = int(true_screen_height/4)
deathbox_box_width = int(true_screen_width/2)
deathbox_box_height = int(true_screen_height/6)

# score box
scorebox_left_edge = int(screen_width/2 - screen_width/8)
scorebox_top_edge = int(screen_height/12)
scorebox_box_width = int(screen_width/4)
scorebox_box_height = int(screen_height/20)

loop_time = time()
button_press_time = time()
check_alive_time = time()
alive = 0
can_punish_for_dying = False
restart_step = 1
while True:
    start_time = time()

    # get an updated image of the game
    true_screen = wincap.get_image_from_window()
    # Resize the image to 84x84 pixels
    screen = cv.resize(true_screen, (screen_width, screen_height))

    reward = 0


    # only do this stuff when the game is running
    # automatically start game
    if not alive and restart_step == 0:
        keyboard.press('tab')
        keyboard.release('tab')
        keyboard.press('enter')
        keyboard.release('enter')
        restart_step = 1
    elif not alive and restart_step == 1:
        keyboard.press('tab')
        keyboard.release('tab')
        keyboard.press('enter')
        keyboard.release('enter')
        alive = 1
        restart_step = 0
    else:
        can_punish_for_dying = True
        roi_coords = (scorebox_left_edge, scorebox_top_edge, scorebox_box_width, scorebox_box_height)
        current_score = ocr_processor.extract_score(screen, roi_coords)
        x, y, w, h = roi_coords
        # Draw a yellow rectangle around the region
        # cv.rectangle(screen, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Store the current score for the next iteration
        # Compute reward based on change in score
        reward += 1
    
        # Preprocess the screen for neural network
        screen_gray = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)
        screen_normalized = screen_gray / 255.0
        state = np.stack([screen_normalized] * 4, axis=2)  # Initial state with 4 repeated frames
        action_elapsed_time = time() - button_press_time
        # Let the agent decide an action
        current_actions = agent.act(state)
    
        
        # Press or release each key based on the agent's decision
        keys_to_press = current_actions - agent.last_action
        keys_to_release = agent.last_action - current_actions

        for key in keys_to_press:
            keyboard.press(key)
            
        for key in keys_to_release:
            keyboard.release(key)

        agent.last_action = current_actions

        next_screen = wincap.get_image_from_window()
        # Determine if game over or not
        roi_coords = (deathbox_left_edge, deathbox_top_edge, deathbox_box_width, deathbox_box_height)
        alive = ocr_processor.extract_death(next_screen, roi_coords) # i use true screen because the regular screen is too small for OCR
        x, y, w, h = roi_coords
        # Draw a yellow rectangle around the region
        # cv.rectangle(screen, (x, y), (x+w, y+h), (0, 255, 255), 2)
        if not alive:
            reward -= 100

        print(reward)
        # display the processed image
        cv.imshow('Computer Vision', screen)

        next_screen = cv.resize(next_screen, (screen_width, screen_height))
        next_screen_gray = cv.cvtColor(next_screen, cv.COLOR_BGR2GRAY)
        next_screen_normalized = next_screen_gray / 255.0
        next_state = np.stack([state[:,:,1], state[:,:,2], state[:,:,3], next_screen_normalized], axis=2)

        agent.memory.push(state, current_actions, reward, next_state, not alive)
        agent.train()

    
    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()
    
    elapsed_time = time() - start_time
    sleep_time = FRAME_DURATION - elapsed_time
    if sleep_time > 0:
        sleep(sleep_time)
    
    # hold 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        keyboard.release('up')
        keyboard.release('down')
        keyboard.release('left')
        keyboard.release('right')
        keyboard.release('tab')
        keyboard.release('enter')
        agent.save_weights('dqn_weights.h5')
        cv.destroyAllWindows()
        break

