import cv2

cap = cv2.VideoCapture("/dev/video3", cv2.CAP_V4L2)

# 设置 MJPG 格式
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

ret, frame = cap.read()
if ret:
    print("成功读取一帧！")
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
else:
    print("读取失败！请检查权限或设备")
cap.release()
cv2.destroyAllWindows()
