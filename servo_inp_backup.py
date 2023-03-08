import serial                                           # import serial library
import cv2

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)

def load_faceslist():
    if device == 'cpu':
        embeds = torch.load(DATA_PATH+'/faceslistCPU.pth')
    else:
        embeds = torch.load(DATA_PATH+'/faceslist.pth')
    names = np.load(DATA_PATH+'/usernames.npy')
    return embeds, names

def inference(model, face, local_embeds, threshold = 0.3):
    #local: [n,512] voi n la so nguoi trong faceslist
    embeds = []
    # print(trans(face).unsqueeze(0).shape)
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds) #[1,512]
    # print(detect_embeds.shape)
                    #[1,512,1]                                      [1,512,n]
    norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    # print(norm_diff)
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1) #(1,n), moi cot la tong khoang cach euclide so vs embed moi
    
    min_dist, embed_idx = torch.min(norm_score, dim = 1)
    print(min_dist*power, names[embed_idx])
    # print(min_dist.shape)
    if min_dist*power > threshold:
        return -1, -1
    else:
        return embed_idx, min_dist.double()

def extract_face(box, img, margin=20):
    face_size = 160
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ] #tạo margin bao quanh box cũ
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face

frame_size = (640, 480)
IMG_PATH = './data/test_images'
DATA_PATH = './data'


OPEN_POS = 180
CLOSED_POS = 90

def get_number_in_string(string):
        return int(''.join(filter(str.isdigit, string)))
 
if __name__ == "__main__":
    arduino = serial.Serial()  # create serial object named arduino
    arduino.baudrate = 9600                                 # set baud rate
    arduino.port = 'COM11'
    arduino.open()                              # set COM port
    prev_frame_time = 0
    new_frame_time = 0
    power = pow(10, 6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)
    model.eval()

    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)
    USER = input("Enter your username: ")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    embeddings, names = load_faceslist()

    DISTANCE = 40
    VERIFY = False
    OPEN = False
    arduino.write(str.encode(str(CLOSED_POS)))

    while cap.isOpened():                                             # create loop
        # cmd = str(input ("Open or close? "))       # query servo position
        # if cmd == "open":
        #         arduino.write(str.encode(str(OPEN_POS)))                          # write position to serial port
        #         reachedPos = str(arduino.readline())            # read serial port for arduino echo
        #         print(reachedPos)                               # print arduino echo to console
        # elif cmd == "close":
        #         arduino.write(str.encode(str(CLOSED_POS)))                          # write position to serial port
        #         reachedPos = str(arduino.readline())

        # command = str(input ("Servo position: "))       # query servo position
        # arduino.write(str.encode(command))                          # write position to serial port
        # reachedPos = str(arduino.readline())            # read serial port for arduino echo
        # print(reachedPos)                               # print arduino echo to console
        distance = str(arduino.readline())
        int_distance = get_number_in_string(distance)
        print(int_distance)
        if int_distance < DISTANCE:
            
                isSuccess, frame = cap.read()
                if isSuccess:
                        boxes, _ = mtcnn.detect(frame)
                if boxes is not None:
                        for box in boxes:
                                bbox = list(map(int,box.tolist()))
                                face = extract_face(bbox, frame)
                                idx, score = inference(model, face, embeddings)
                        if idx != -1:
                                frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                                score = torch.Tensor.cpu(score[0]).detach().numpy()*power
                                frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                                if names[idx] == USER:
                                        VERIFY = True
                                        print("Verified")
                                else:
                                        VERIFY = False
                                        print("Unverified")
                        else:
                                frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                                frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)

                new_frame_time = time.time() 
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                fps = str(int(fps))
                cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1)&0xFF == 27:
                        break

                # write position to serial port
                if not OPEN and VERIFY:
                        arduino.write(str.encode(str(OPEN_POS)))
                        OPEN = True
                        print("Open")
                # read serial port for arduino echo
                # reachedPos = str(arduino.readline())
                # print arduino echo to console
                # print(reachedPos)
        elif int_distance > DISTANCE:
                # write position to serial port
                if OPEN:
                        arduino.write(str.encode(str(CLOSED_POS)))
                        OPEN = False
                        print("Close")
                # reachedPos = str(arduino.readline())
                # cap.release()
                # cv2.destroyAllWindows()
