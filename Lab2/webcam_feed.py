import cv2 as cv
import numpy as np


# Key values
ESC_KEY = 27
RET_KEY = 13
SPC_KEY = 32

# Default capture id is 0
# If you use other webcam than the laptop camera try 1 or 2
CAPTURE_DEVICE_ID = 0

# The name of the window you want the image to be shown in
WINDOW_NAME = 'Webcam Feed'
CONTROL_WINDOW_NAME = 'Control'


status = {
    'mirror': False,
    'invert': False,
    'gray': False,
    'binary': False,
    'bthresh': 0,
    'width': 1.0,
    'height': 1.0,
    'alpha': 1.0,
    'beta': 0
}


def update_bthresh(value):
    global status
    status['bthresh'] = value


def update_alpha(value):
    global status
    status['alpha'] = value/100


def update_height(value):
    global status
    status['height'] = value/100


def update_width(value):
    global status
    status['width'] = value/100


def update_beta(value):
    global status
    status['beta'] = value - 100


def update_status(key):
    global status

    if key in [ord('m'), ord('M')]:
        status['mirror'] = not status['mirror']
    if key in [ord('i'), ord('I')]:
        status['invert'] = not status['invert']
    if key in [ord('g'), ord('G')]:
        status['gray'] = not status['gray']
    if key in (ord('b'), ord('B')):
        status['binary'] = not status['binary']


def something(frame):
    # You could do something to the frame here
    if status['mirror']:
        frame = frame[:, ::-1]
    if status['invert']:
        frame = 255 - frame
    if status['gray']:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame = status['alpha']*frame + status['beta']
    frame = np.clip(frame, 0, 255)
    frame = frame.astype(np.uint8)

    h, w = frame.shape[:2]
    new_h = int(h*status['height'])
    new_w = int(w*status['width'])
    frame = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_LANCZOS4)

    if status['binary']:
        thresh = status['bthresh']
        if len(frame.shape) > 2:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, frame = cv.threshold(frame, thresh, 255, cv.THRESH_BINARY)

    return frame


def main():
    # Initialize a capture object
    cap = cv.VideoCapture(CAPTURE_DEVICE_ID)
    cv.namedWindow(CONTROL_WINDOW_NAME)

    control_image = np.zeros((1, int(cap.get(cv.CAP_PROP_FRAME_WIDTH))), np.uint8)
    cv.imshow(CONTROL_WINDOW_NAME, control_image)

    cv.createTrackbar('Alpha', CONTROL_WINDOW_NAME, 0, 200, update_alpha)
    cv.setTrackbarPos('Alpha', CONTROL_WINDOW_NAME, 100)
    cv.createTrackbar('Beta', CONTROL_WINDOW_NAME, 0, 200, update_beta)
    cv.setTrackbarPos('Beta', CONTROL_WINDOW_NAME, 100)
    cv.createTrackbar('Height', CONTROL_WINDOW_NAME, 0, 400, update_height)
    cv.setTrackbarPos('Height', CONTROL_WINDOW_NAME, 100)
    cv.setTrackbarMin('Height', CONTROL_WINDOW_NAME, 1)
    cv.createTrackbar('Width', CONTROL_WINDOW_NAME, 0, 400, update_width)
    cv.setTrackbarPos('Width', CONTROL_WINDOW_NAME, 100)
    cv.setTrackbarMin('Width', CONTROL_WINDOW_NAME, 1)
    cv.createTrackbar('Binary Threshold', CONTROL_WINDOW_NAME, 0, 255, update_bthresh)

    # Capture a first frame
    success, frame = cap.read()
    # If for some reason success is False then the capture
    # doesn't work

    while success:

        # If you wanna do something to the frame you could
        # do it in the something function before the imshow call
        frame = something(frame)

        # Display the captured frame
        cv.imshow(WINDOW_NAME, frame)

        # Wait for one milisecond for a keypress
        key = cv.waitKeyEx(1)
        # waitKeyEx() is used instead of waitKey() as it returns
        # also the special keys like the arrow keys
        # If a value of 0 or less is passed as a parameter it will
        # wait forever for a keypress and stop the loop

        # Check if Escape or Q is pressed to end the pain
        if key in (ESC_KEY, ord('q'), ord('Q')):
            # Cleanup
            cv.destroyAllWindows()
            # And release the capture
            cap.release()
            # This is good practice as an unrealesed capture might render
            # the camera in use after termination and thus making it...
            # unusable

            # and kill the loop
            break

        update_status(key)

        # Get the next frame
        success, frame = cap.read()


if __name__ == '__main__':  # Please have this in your main python files
    main()  # to have it in a function is optional
