import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

colormap = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera is not accessible")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if colormap is None:
        display_frame = frame.copy()
    else:
        display_frame = cv2.applyColorMap(gray, colormap)

    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(display_frame,
                    "Face Detected",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    cv2.putText(display_frame,
                "Faces : " + str(len(faces)),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2)

    cv2.putText(display_frame,
                "Press 0-9 to Change Filters",
                (20, 360),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    cv2.putText(display_frame,
                "Press Q to Quit | Press S to Save",
                (20, 385),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    cv2.imshow("Face Detection Camera", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('0'):
        colormap = None

    elif key == ord('1'):
        colormap = cv2.COLORMAP_AUTUMN

    elif key == ord('2'):
        colormap = cv2.COLORMAP_BONE

    elif key == ord('3'):
        colormap = cv2.COLORMAP_JET

    elif key == ord('4'):
        colormap = cv2.COLORMAP_WINTER

    elif key == ord('5'):
        colormap = cv2.COLORMAP_RAINBOW

    elif key == ord('6'):
        colormap = cv2.COLORMAP_OCEAN

    elif key == ord('7'):
        colormap = cv2.COLORMAP_SUMMER

    elif key == ord('8'):
        colormap = cv2.COLORMAP_PINK

    elif key == ord('9'):
        colormap = cv2.COLORMAP_HOT

    elif key == ord('s'):
        cv2.imwrite("myimage.png", display_frame)
        print("Image Save Successfully")

cap.release()
cv2.destroyAllWindows()