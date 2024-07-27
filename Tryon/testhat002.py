import cv2
import numpy as np

# Load models
face_detection_model = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml")

# Load hat image with alpha channel
hat_image = cv2.imread("hat_image/santahat.png", cv2.IMREAD_UNCHANGED)

# Function to overlay image with transparency
def overlay_image_alpha(background, overlay, pos):
    x, y = pos

    # Get the dimensions of the overlay image
    h, w = overlay.shape[0], overlay.shape[1]

    # Extract the alpha mask of the RGBA image, convert to RGB
    overlay_image = overlay[:, :, :3]
    mask = overlay[:, :, 3:]

    # Calculate the region to overlay
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)

    y1o, y2o = max(0, -y), min(h, background.shape[0] - y)
    x1o, x2o = max(0, -x), min(w, background.shape[1] - x)

    # Overlay the images
    roi = background[y1:y2, x1:x2]
    roi_masked = roi * (1 - mask[y1o:y2o, x1o:x2o] / 255)
    overlay_masked = overlay_image[y1o:y2o, x1o:x2o] * (mask[y1o:y2o, x1o:x2o] / 255)
    background[y1:y2, x1:x2] = roi_masked + overlay_masked

    return background

vid = cv2.VideoCapture(0)
while True:
    ret, image = vid.read()
    if ret:
        final_image = image.copy()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_detection_model.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(200, 200))

        for (face_x, face_y, face_w, face_h) in faces:
            hat_width_resize = int(1.5 * face_w)
            scale_factor = hat_width_resize / hat_image.shape[1]
            resized_hat = cv2.resize(hat_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            # Get position to place hat
            hat_x = face_x - int(0.25 * resized_hat.shape[1])
            hat_y = face_y - int(0.75 * resized_hat.shape[0])

            # Overlay hat
            final_image = overlay_image_alpha(final_image, resized_hat, (hat_x, hat_y))

        cv2.imshow("Hat Overlay", final_image)

    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
