import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import time
from collections import deque


# CONFIGURATION
GREEN = '\033[32m'
YELLOW = '\033[33m' 
BLUE = '\033[34m'
RED = '\033[31m '
RESET = '\033[0m'

# model config
MODEL_PATH = '../STAGE1/cnn_mediapipe/asl_final_cnn_mediapipe_model.keras'
IMG_SIZE = 64
CLASS_NAMES = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

# webcam resolution config
CAMERA_INDEX = 0
CAMERA_WIDTH = 640  
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# mediapipe config
MIN_DETECTION_CONFIDENCE = 0.4 # mp will ignore hands detected with confidence below this 
MIN_TRACKING_CONFIDENCE = 0.4  
# mp uses temporal info (landmarks from previous frames) and is faster. but if confidence drops below this, it will run detection again. 
BBOX_PADDING = 0.15  

# prediction stability config
CONFIDENCE_THRESHOLD = 0.5  
STABILITY_FRAMES = 3         
MAX_PREDICTION_HISTORY = 5   

# process EVERY frame for responsiveness
SKIP_FRAMES = 1  

# UI config 
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_PRED = 1.0
FONT_SCALE_CONF = 0.6
FONT_THICKNESS_BOLD = 2
FONT_THICKNESS_NORMAL = 2
FONT_THICKNESS_THIN = 1

# colors 
COLOR_BBOX = (0, 255, 0)
COLOR_PRED_STABLE = (0, 255, 0)
COLOR_PRED_DETECTING = (0, 165, 255)
COLOR_TEXT_BG = (210, 210, 207)
COLOR_INFO_TEXT = (255, 255, 255)
COLOR_NO_HAND = (0, 0, 255)

INFO_PANEL_HEIGHT = 70
BBOX_THICKNESS = 2
TEXT_PADDING = 8

print(f"\n{'='*40}{BLUE} A S L    R E A L - T I M E    R E C O G N I T I O N {RESET}{'='*40}")

print(f"\n{YELLOW}---> LOADING THE MODEL . . . {RESET}\n")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"\n{GREEN}MODEL LOADED !{RESET}")
except Exception as e:
    print(f"{RED}Error: {e}{RESET}")
    exit()

# WARM UP MODEL (WHICH IS IMPORTANT FOR FIRST PREDICTION SPEED!)
print(f"\n{YELLOW}---> WARMING UP THE MODEL . . . {RESET}")
dummy_input = np.random.rand(1, 64, 64, 3).astype('float32')
_ = model.predict(dummy_input, verbose=0)
print(f"\n{GREEN}MODEL IS READY !{RESET}")

print(f"\n{YELLOW}---> INITIALIZING MEDIAPIPE . . . {RESET}")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands= 1,
    min_detection_confidence= MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    model_complexity= 0  # use lighter model (0=lite, 1=full)
)

print(f"\n{GREEN}MEDIAPIPE INITIALIZED !{RESET}")

# HELPER FUNCTIONS:
def get_bbox_from_landmarks(landmarks, img_width, img_height):

    try:
        # get coordinates directly in pixel space
        x_coords = [lm.x * img_width for lm in landmarks.landmark]
        y_coords = [lm.y * img_height for lm in landmarks.landmark]
        
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))
        
        # padding definition
        width = x_max - x_min
        height = y_max - y_min
        
        padding_x = int(width * BBOX_PADDING)
        padding_y = int(height * BBOX_PADDING)
        
        x_min = max(0, x_min - padding_x)
        x_max = min(img_width, x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(img_height, y_max + padding_y)
        
        # validate bbox
        if x_max <= x_min or y_max <= y_min:
            return None
        
        return (x_min, y_min, x_max, y_max)
        
    except Exception as e:
        print(f"{RED}Bbox error: {e}{RESET}")
        return None

# same as in training
def preprocess_hand_crop(hand_crop):
    try:
        # the input must be RGB, which is already so from frame_rgb
        if hand_crop.shape[2] == 3:
            pass
        # resize to 64x64
        resized = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype('float32') / 255.0
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
        
    except Exception as e:
        print(f"{RED}Preprocessing error: {e}{RESET}")
        return None

# PREDICTION VOTING LOGIC:
def get_stable_prediction(prediction_history): 

    if len(prediction_history) == 0:
        return None, 0.0, False
    
    # if we have enough history
    if len(prediction_history) >= STABILITY_FRAMES:
        # get recent predictions
        recent = list(prediction_history)[-STABILITY_FRAMES:]
        labels = [p[0] for p in recent]

        # check if last prediction appears multiple times
        last_label = labels[-1]
        count = labels.count(last_label)
        
        if count >= 2:  # but if appears at least twice in recent history, get the confidence average 
            confs = [p[1] for p in recent if p[0] == last_label]
            avg_conf = np.mean(confs)
            return last_label, avg_conf, True
    
    # here it returns most recent prediction (not stable yet)
    return prediction_history[-1][0], prediction_history[-1][1], False


def draw_text_with_background(frame, text, position, font_scale, color, thickness):
    x, y = position
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)

    # the background
    cv2.rectangle(frame, (x - TEXT_PADDING, y - text_h - TEXT_PADDING), (x + text_w + TEXT_PADDING, y + baseline + TEXT_PADDING), 
                COLOR_TEXT_BG, -1)
    
    # text
    cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)
    
    return text_h


def draw_prediction_on_bbox(frame, bbox, prediction, confidence, is_stable):
    x_min, y_min, x_max, y_max = bbox
    
    color = COLOR_PRED_STABLE if is_stable else COLOR_PRED_DETECTING
    status = "" if is_stable else " ..."
    
    pred_text = f"{prediction}{status}"
    combined_text = f"{pred_text} ({confidence*100:.1f}%)"

    text_x = x_min
    text_y = max(50, y_min - 10)

    cv2.putText(frame, combined_text, (text_x, text_y), FONT, FONT_SCALE_PRED, color, FONT_THICKNESS_BOLD, cv2.LINE_AA)


def draw_info_panel(frame, fps, hand_detected):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, INFO_PANEL_HEIGHT), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), FONT, 0.5, COLOR_INFO_TEXT, FONT_THICKNESS_THIN, cv2.LINE_AA)
    
    # hand status
    status = "Hand: YES" if hand_detected else "Hand: NO"
    status_color = COLOR_PRED_STABLE if hand_detected else COLOR_NO_HAND
    
    cv2.putText(frame, status, (10, 50), FONT, 0.5, status_color, FONT_THICKNESS_THIN, cv2.LINE_AA)
    
    cv2.putText(frame, "Q: Quit", (w - 110, 25), FONT,  0.4, COLOR_INFO_TEXT, FONT_THICKNESS_THIN, cv2.LINE_AA)
    cv2.putText(frame, "SPACE: Pause", (w - 110, 50), FONT, 0.4, COLOR_INFO_TEXT, FONT_THICKNESS_THIN, cv2.LINE_AA)

# MAIN FUNCTION
def main():
    print(f"\n{BLUE}---> S T A R T I N G   W E B C A M{RESET}")
    print("\nCONTROL: Q/ESC=Quit, SPACE=Pause\n")
    
    # initialize webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize buffer lag!
    
    if not cap.isOpened():
        print(f"{RED}Error: Could not open webcam!{RESET}")
        return
    
    # state variables
    prediction_history = deque(maxlen=MAX_PREDICTION_HISTORY)
    prev_time = time.time()
    paused = False
    frame_count = 0
    
    # current state
    current_prediction = None
    current_confidence = 0.0
    is_stable = False
    current_bbox = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print(f"{RED}Error: Could not read frame!{RESET}")
            break
        
        # flip immediately for mirrror view/effect + convert
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 0.0001)
        prev_time = current_time
        
        hand_detected = False
        
        if not paused:
            # STEP 1: MediaPipe detection 
            results = hands.process(frame_rgb) # starts mp preprocessing here
            
            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]

                # draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=( 0, 0, 255), thickness=2, circle_radius=2), # BGR
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # STEP 2: Get bbox IMMEDIATELY
                h, w, _ = frame.shape
                bbox = get_bbox_from_landmarks(hand_landmarks, w, h)
                
                if bbox is not None:
                    x_min, y_min, x_max, y_max = bbox
                    
                    # update bbox IMMEDIATELY (no lag!)
                    current_bbox = bbox
                    
                    # STEP 3: Crop hand
                    hand_crop = frame_rgb[y_min:y_max, x_min:x_max]
                    
                    if hand_crop.size > 0:
                        try:
                            # STEP 4: Preprocess 
                            preprocessed = preprocess_hand_crop(hand_crop)
                            
                            if preprocessed is not None:
                                # STEP 5: Predict 
                                pred_probs = model.predict(preprocessed, verbose=0)[0]
                                pred_idx = np.argmax(pred_probs)
                                prediction = CLASS_NAMES[pred_idx]
                                confidence = pred_probs[pred_idx]

                                # STEP 6: Add to history if confident
                                if confidence >= CONFIDENCE_THRESHOLD:
                                    prediction_history.append((prediction, confidence))
                                    
                                    # STEP 7: Get stable prediction 
                                    stable_pred, stable_conf, stable = get_stable_prediction(prediction_history)
                                    
                                    # update display IMMEDIATELY
                                    current_prediction = stable_pred
                                    current_confidence = stable_conf
                                    is_stable = stable
                                else:
                                    # low confidence - show but mark unstable
                                    current_prediction = prediction
                                    current_confidence = confidence
                                    is_stable = False
                        
                        except Exception as e:
                            print(f"{RED}Error: {e}{RESET}")
            
            else:
                # if no hand detected, clear immediately
                prediction_history.clear()
                current_prediction = None
                current_confidence = 0.0
                is_stable = False
                current_bbox = None
        
        # DRAW (always draw current state immediately!)
        if current_bbox is not None and hand_detected:
            x_min, y_min, x_max, y_max = current_bbox

            bbox_color = COLOR_PRED_STABLE if is_stable else COLOR_PRED_DETECTING
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bbox_color, BBOX_THICKNESS)
            
            if current_prediction:
                draw_prediction_on_bbox( frame, current_bbox, current_prediction, current_confidence, is_stable )
        
        # info panel
        draw_info_panel(frame, fps, hand_detected)
        
        # pause indicator
        if paused:
            h, w = frame.shape[:2]
            cv2.putText(frame, "PAUSED", (w//2 - 70, h//2), FONT, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        
        cv2.imshow('ASL Recognition - Stage 2', frame)
        
        # keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        
        frame_count += 1
    
    # cleanup
    print(f"\nCLEANING UP...")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print(f"Total frames: {GREEN}{frame_count}{RESET}")
    print(f"{RED}\nSHUTTING DOWN . . . \n\n{BLUE}S T A G E   2   E N D E D ! {RESET}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()