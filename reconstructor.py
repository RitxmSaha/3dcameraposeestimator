import cv2
import numpy as np

cap =  cv2.VideoCapture(0)
margin = 20

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow("Life Video Feed", cv2.WINDOW_NORMAL)

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        height, width = frame.shape[:2]

        small_frame = cv2.flip(cv2.resize(frame, (width // 4, height // 4)), 1)

        height, width, _ = small_frame.shape
        combined_width = 2 * width + margin
        combined_frame = np.zeros((height,combined_width, 3), dtype=np.uint8)

        combined_frame[:, :width] = small_frame
        combined_frame[:, width + margin:] = small_frame

        cv2.imshow('Live Video Feed', combined_frame)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    #When program is quit, release the capture
    cap.release()
    cv2.destroyAllWindows()