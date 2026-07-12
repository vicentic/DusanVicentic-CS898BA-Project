import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the video and subtract the background

video_path = 'Shot.mov'
cap = cv2.VideoCapture(video_path)

backSub = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 50, detectShadows = False)

ball_coordinates = []

print ('Processing video...')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply gmm background subtraction
    fg_mask = backSub.apply(frame)

    # remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # 2nd itteration: added color mask (isolate basketball color)
    # 3rd itteration: adjusted range to better match basketball hsv [3, 146, 196]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 80, 130])
    upper_orange = np.array([15, 220, 255])
    color_mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # 2nd itt: combine masks (must be moving and orange)
    combined_mask = cv2.bitwise_and(fg_mask, color_mask)

    # 2. Hough Circle Transform to detect the ball (2nd itt: tightened parameters)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp = 1.2,
        minDist = 30,
        param1 = 100, #increased from 50
        param2 = 30, # increased from 30, decreased from 45 to 30
        minRadius = 10,
        maxRadius = 20 # 3rd itt: colomeasured ball radius to ~ 15
    )

    # Filter circles using GMM mask
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]

            # 2nd itt: check if the detected circle is in combined mask
            if combined_mask[y, x] == 255:
                ball_coordinates.append((x, y))

                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)


    cv2.imshow('Tracking Baseline', frame)
    cv2.imshow('Combined Mask', combined_mask) # 2nd itt: show combined mask

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# 3. Polynomial regression arc

if len(ball_coordinates) > 3:
    print(f'Successfully extracted {len(ball_coordinates)} ball coordinates.')

    # split coordinates into x and y
    x = np.array([pt[0] for pt in ball_coordinates])
    y = np.array([pt[1] for pt in ball_coordinates])

    # fit a 2nd degree polynomial
    coefficients = np.polyfit(x, y, 2)
    a, b, c = coefficients

    # generate fitted curve
    x_line = np.linspace(min(x), max(x), 100)
    y_line = a * x_line**2 + b * x_line + c

    # plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='orange', label='Extracted Ball Coordinates')
    plt.plot(x_line, y_line, color='blue', label='Fitted Parabola')

    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates

    plt.title('Baseline Implementation:Ball Trajectory Fitting')
    plt.xlabel('X Coordinate (pixels)')
    plt.ylabel('Y Coordinate (pixels)')
    plt.legend()
    plt.grid()
    plt.savefig('ball_trajectory_baseline.png', dpi=300)
    print('Saved baseline plot to ball_trajectory_baseline.png')
    plt.show()
else:
    print('Not enough ball coordinates adjust Hough Circle parameters.')