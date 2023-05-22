import cv2
import dlib
import numpy as np

def nothing(x):
	pass
# set up the 68 point facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cv2.namedWindow('controls')
    #create trackbar in 'controls' window with name 'r''
cv2.createTrackbar('r','controls',15,255,nothing)
cv2.createTrackbar('b','controls',15,255,nothing)

# bring in the input image
while True:
    img = cv2.imread('j.jpg', 1)
    mask = np.zeros_like(img)

    shape = img.shape
    c_mask = np.zeros(shape, np.uint8)

    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # faces = face_detection(img_gray)

    # Detect all the point on face using dlib
    # landmarks = face_point(img_gray, faces)
    # lip(landmarks, mask, image_hsv)
    # detect faces in the image
    faces_in_image = detector(img_gray, 0)

    #size = cv2.getTrackbarPos('Blur', 'Track')
    #size= int(cv2.getTrackbarPos('Blur','Track'))
    radius= int(cv2.getTrackbarPos('r','controls'))
    blur_t= int(cv2.getTrackbarPos('b','controls'))
    
    # size = cv2.getTrackbarPos('Size', 'Main Window',)
    for face in faces_in_image:
        #size = cv2.getTrackbarPos('Size', 'Main Window',)
        # assign the facial landmarks
        landmarks = predictor(img_gray, face)

        # unpack the 68 landmark coordinates from the dlib object into a list 
        landmarks_list = []
        for i in range(0, landmarks.num_parts):
            landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))


####################################################################### Extacking left and right Cheek center points for blush ###########################################################

        # for each landmark, plot and write number
    for landmark_num4, xy4 in enumerate(landmarks_list, start = 1):
        #cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
        if str(landmark_num4) == '49':
            # cir = cv2.circle(img, (int(xy4[0]), int(xy4[1])), radius, (168, 100, 50), -1)
            ep = xy4[0]
            ep2 = xy4[1]
            break
    for landmark_num5, xy5 in enumerate(landmarks_list, start = 1):
        #cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
        if str(landmark_num5) == '37':
            # cir = cv2.circle(img, (int(xy5[0]), int(xy5[1])), radius, (168, 100, 50), -1)
            sp = xy5[0]
            sp2 = xy5[1]
            break

    for landmark_num2, xy2 in enumerate(landmarks_list, start = 1):
        #cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
        if str(landmark_num2) == '46':
            r_sp = xy2[0]
            r_sp2 = xy2[1]
            break
    for landmark_num3, xy3 in enumerate(landmarks_list, start = 1):
        #cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
        if str(landmark_num3) == '55':
            r_ep = xy3[0]
            r_ep2 = xy3[1]
            break
        
    # for landmark_num, xy in enumerate(landmarks_list, start = 1):
    #     cv2.circle(img, (xy[0], xy[1]), 5, (168, 0, 20), -1)
        # if str(landmark_num) == '36':
        #     c_xy = xy[0]
        #     c_xy2 = xy[1]
            #cv2.putText(img, str(landmark_num),(xy[0]-7,xy[1]+5), cv2.FONT_HERSHEY_SIMPLEX, .7,(0,0,0), 2)



################################################## Extracking eye points for eyeshadow ##########################################################

    # for landmarks in landmarks_list:
    #     points = []
    #     for n in range(23, 28):
    #         x = landmarks.part(n).x
    #         y = landmarks.part(n).y
    #         points.append((x, y))
    #     for n2 in range(43, 47):
    #         x2 = landmarks.part(n2).x
    #         y2 = landmarks.part(n2).y
    #         points.append((x2, y2))
    # arr = np.array(points)

    #     # if lips existed draw those point
    # if points:
    #     cv2.fillPoly(c_mask, [arr], (255, 255, 255))

    points = []
    # for landmark_num3, xy3 in enumerate(landmarks_list, start = 1):
    #     #cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
    #     if str(landmark_num3) >= '23' and str(landmark_num3) <= '27':
    #         x = xy3[0]
    #         y = xy3[1]
    #         points.append((x,y))
    # for landmark_num33, xy33 in enumerate(landmarks_list, start = 1):
    #     #cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
    #     if str(landmark_num33) >= '43' and str(landmark_num33) <= '46':
    #         x3 = xy33[0]
    #         y3 = xy33[1]
    #         points.append((x3,y3))
    # arr = np.array(points)    
    # if points:
    #     cv2.fillPoly(c_mask, [arr], (255, 255, 255))


    # for landmark_num3, xy3 in enumerate(landmarks_list, start = 1):
    #     #cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
    #     if str(landmark_num3) == '25':
    #         r_eye_sp = xy3[0]
    #         r_eye_sp2 = xy3[1]
    #         break

    # for landmark_num3, xy3 in enumerate(landmarks_list, start = 1):
    #     #cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
    #     if str(landmark_num3) == '45':
    #         r_eye_ep = xy3[0]
    #         r_eye_ep2 = xy3[1]
    #         break


           
    #h, w, c = img.shape
    # cv2.line(img, (sp,sp2), (ep,ep2), (0, 255, 0), 8)
    #cv2.rectangle(img, (r_eye_sp,r_eye_sp2), (r_eye_ep,r_eye_ep2), (0, 255, 0), -1)
    # cv2.line(img, (r_sp,r_sp2), (r_ep,r_ep2), (0, 255, 0), 8)

    def midpoint(x1, x2, y1, y2):
 
        return ((x1 + x2) // 2 ,(y1 + y2) // 2)

    #x1, y1, x2, y2 = -1, 2, 3, -6
    c,c2 = midpoint(sp, ep, sp2, ep2)
    r_c,r_c2 = midpoint(r_sp, r_ep, r_sp2, r_ep2)
    
    #cv2.line(image, (x+w//2, y), (x+w//2, y+h), (0, 0, 255), 2)
    center = (c, c2)
    r_center = (r_c, r_c2)
    # cv2.circle(img, center, 15, (255, 255, 0), -1)
    # cv2.circle(img, r_center, 15, (255, 255, 0), -1)

    # shape = img.shape
    # c_mask = np.zeros(shape, np.uint8)
    cir = cv2.circle(c_mask, (c, c2), radius, (168, 0, 20), -1)
    cir = cv2.circle(c_mask, ((r_c), r_c2), radius, (168, 0, 20), -1)
    if blur_t < 2:
        blur_t = 2
    blur = cv2.blur(cir,(blur_t, blur_t))
    dst = cv2.addWeighted(blur, 1, img, 1, 0.0)
    


    # visualise the image with landmarksqqq
    
    # cv2.imshow('img', img)
    # cv2.imshow('img2', blur)
    cv2.imshow('img3', dst)

    # cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()