cap = cv2.VideoCapture('saint.mp4')
if not cap.isOpened():
    print("is not opened")
    cv2.VideoCapture.open(cap, 'saint.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()