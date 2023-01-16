import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
from matplotlib import pyplot as plt

joint_list = [[4,0,8]]
joint_list1 = [[8,6,5]]
joint_list2 = [[12,10,9]]
joint_list3 = [[16,14,13]]
joint_list4 = [[20,18,17]]

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# mpDraw = mp.solutions.drawing_utils
mp_hands.HandLandmark.WRIST

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
        # print(results)



        #for angle in fingers
        def draw_finger_angles(image, results, joint_list):
    
    # Loop through hands
            for hand in results.multi_hand_landmarks:

                for id, lm in enumerate(hand.landmark):
                        # print(id,lm)
                        h,w,c = image.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        
                        if id == 8:
                            cv2.circle(image, (cx,cy),15,(255,0,255), cv2.FILLED)
                            a = cy
                            # print(a)
                        if id == 4:
                            cv2.circle(image, (cx,cy),15,(255,0,255), cv2.FILLED)
                            
                        if id == 0:
                            cv2.circle(image, (cx,cy),15,(255,0,255), cv2.FILLED)

                        if id == 5:
                            cv2.circle(image, (cx,cy),15,(255,0,255), cv2.FILLED)
                            b = cy
                            # print(b)

                        if id == 12:
                            cv2.circle(image, (cx,cy),15,(255,0,255), cv2.FILLED)
                            cb = cy
                            # print(cb)
                        if id == 9:
                            cv2.circle(image, (cx,cy),15,(255,0,255), cv2.FILLED)
                            d = cy
                            # print(d)
                        if id == 16:
                            cv2.circle(image, (cx,cy),15,(255,0,255), cv2.FILLED)
                            e = cy
                            # print(e)
                        if id == 13:
                            cv2.circle(image, (cx,cy),15,(255,0,255), cv2.FILLED)
                            f = cy
                            # print(f)
                        if id == 20:
                            cv2.circle(image, (cx,cy),15,(255,0,255), cv2.FILLED)
                            g = cy
                            # print(e)
                        if id == 17:
                            cv2.circle(image, (cx,cy),15,(255,0,255), cv2.FILLED)
                            g2 = cy
                            # print(f)
                #fore finger        
                print(a,b)
                if (a > b):
                    print("FORE FINGER FOLDED")
                    cv2.putText(image, 
                "Fore finger folded", 
                (300, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
                else:
                    print("NOT FOLDED - REST POSITION(FORE FINGER)")
                #middle finger
                print(cb,d)
                if (cb > d):
                    print("MIDDLE FINGER FOLDED")
                    cv2.putText(image, 
                "Middle finger folded", 
                (300, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
                else:
                    print("NOT FOLDED - REST POSITION (MIDDLE FINGER)")
                #ring finger
                print(e,f)
                if (e > f):
                    print("RING FINGER FOLDED")
                    cv2.putText(image, 
                "Ring finger folded", 
                (300, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
                else:
                    print("NOT FOLDED - REST POSITION (RING FINGER)")
                #pinky finger
                print(g,g2)
                if (g > g2):
                    print("PINKY FINGER FOLDED")
                    cv2.putText(image, 
                "Pinky finger folded", 
                (300, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
                else:
                    print("NOT FOLDED - REST POSITION (PINKY FINGER)")
                
                        # if id == 0:
                        #     cv2.circle(img, (cx,cy),15,(255,0,255), cv2.FILLED)
                        #     print(id, cx, cy)
                    if True:
                        #Draw dots and connect them
                        mp_drawing.draw_landmarks(image,hand,
                                                mp_hands.HAND_CONNECTIONS)
                #Loop through joint sets 
                # for joint in joint_list:
                #     a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
                #     b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
                #     c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
                    
                #     radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                #     angle = np.abs(radians*180.0/np.pi)
                    
                #     cv2.putText(image, "Angle in fore-finger"+
                # str(round(angle, 2)), 
                # (50, 330), 
                # cv2.FONT_HERSHEY_SIMPLEX, 1, 
                # (0, 255, 255), 
                # 2, 
                # cv2.LINE_4)  
                #     cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  
                #fore finger      
                for joint in joint_list1:
                    a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
                    b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
                    c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
                    
                    radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                    angle = np.abs(radians*180.0/np.pi)
                    
                    cv2.putText(image, "Angle in fore-finger"+
                str(round(angle, 2)), 
                (50, 360), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)  
                    cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)     
                #middle finger   
                for joint in joint_list2:
                    a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
                    b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
                    c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
                    
                    radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                    angle = np.abs(radians*180.0/np.pi)
                    
                    cv2.putText(image, "Angle in middle-finger"+
                str(round(angle, 2)), 
                (50, 390), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)  
                    cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)   

                #ring finger     
                for joint in joint_list3:
                    a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
                    b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
                    c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
                    
                    radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                    angle = np.abs(radians*180.0/np.pi)
                    
                    cv2.putText(image, "Angle in ring-finger"+
                str(round(angle, 2)), 
                (50, 420), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)  
                    cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  

                #pinky finger      
                for joint in joint_list4:
                    a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
                    b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
                    c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
                    
                    radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                    angle = np.abs(radians*180.0/np.pi)
                    
                    cv2.putText(image, "Angle in pinky-finger"+
                str(round(angle, 2)), 
                (50, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)  
                    cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)        
      
            return image
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                
                # Render left or right detection
                
            
            # Draw angles to image from joint list
            draw_finger_angles(image, results, joint_list)
            draw_finger_angles(image, results, joint_list1)
            draw_finger_angles(image, results, joint_list2)
            draw_finger_angles(image, results, joint_list3)
            draw_finger_angles(image, results, joint_list4)
            
        # Save our image    
        #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


