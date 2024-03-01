import cv2
import os
from time import sleep

# Set the camera index (0 for the default camera, adjust if using an external camera)
camera_index = 2

def take_picture(id):

    # Open the camera
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Set the resolution (optional, adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Capture and save an image
    ret, frame = cap.read()

    if ret:
        # Display the captured image (optional)
        cv2.imshow('Captured Image', frame)

        # Save the image to a file (adjust the filename as needed)
        filename = '_captured_image.jpg'
        filepath = '../images/'
        img_id = id
        file_str = str(img_id) + filename
        filename = os.path.join(filepath, file_str)
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")

        # Wait for a key press and then close the window
        cv2.waitKey(500)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not capture an image.")

    # Release the camera
    cap.release()




def main():
    for i in range(0,20):
        take_picture(i)
        print("Imgae taken with id: {}".format(i))


if __name__ == '__main__':
    main()
