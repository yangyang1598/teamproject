import myopencv as my
import cv2, copy
import numpy as np

def binary(thresold, binary_img):
    binary = np.zeros_like(binary_img)
    binary[(binary_img >= thresold[0]) & (binary_img <= thresold[1])] = 255
    return binary
    pass

def babo(img,src,dst):
    height, width = img.shape[:2]
    dst_size = (width, height)
    src = src * np.float32([width, height])  ## width, height 비율 값
    dst = dst * np.float32(dst_size)  ## 이미지를 적용할 화면 비율
    M = cv2.getPerspectiveTransform(src, dst)  ## 자를 이미지 좌표값
    img_src = cv2.warpPerspective(img, M, dst_size)  ## 잘라낼 이미지, 잘라낼 이미지 영역값, 잘라낼 이미지를 붙일 영역 사이즈
    return img_src
    pass

def Yolo(img, score, nms):
    height, width = img.shape[:2]
    yolo_net = cv2.dnn.readNet('YOLO/yolov3.weights',
                               'YOLO/yolov3.cfg')
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in
                     yolo_net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0),
                                 True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # 검출 신뢰도
            if confidence > 0.5:
                # Object detected
                # 검출기의 경계상자 좌표는 0 ~ 1로 정규화되어있으므로 다시 전처리
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                dw = int(detection[2] * width)
                dh = int(detection[3] * height)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score, nms)
    for i in range(len(boxes)):

        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]
            text = f'{label} {score:.2f}'
            # 경계상자와 클래스 정보 투영
            cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), 2)
            cv2.rectangle(img, (x, y - 19), (int(x + len(text) * 2 * 4.5), y - 18 + len(text) * 2), (50, 50, 50), -1)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 2)
    return img
    pass

classes = ["person", "bicycle", "car", "motorcycle",
           "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
           "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
           "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
           "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
           "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

margin = 150
nwindows = 9
minpix = 1

img_ = []
count = 0
frame_start = 0
frame_count = 0
frame = 0
A = 0
step = 1 # 한번에 넘어갈 프레임수
line_len = 100
road_width = 2.5
_ = True

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

leftx_base_ = []
rightx_base_ = []

leftx_base_step = []
rightx_base_step = []

leftx_ = []
lefty_ = []
rightx_ = []
righty_ = []

left_fit_ = np.empty(3)
right_fit_ = np.empty(3)

name = ['REC_2021_10_09_10_16_45_F', 'REC_2021_10_09_10_17_45_F', 'REC_2021_10_09_10_18_45_F']
Video = cv2.VideoCapture(f'video/1/{name[count]}.mp4')
Out = cv2.VideoWriter('show_orginal_1.mp4', cv2.VideoWriter_fourcc(*'FMP4'), 30, (1920, 1080 + 270))

frame_end = int(Video.get(cv2.CAP_PROP_FRAME_COUNT))
pass
while True:
    frame_count += step
    _, img = Video.read()

    img_blur = cv2.blur(img, (5, 5), 0)

    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    img = my.Undistort(img, 'wide_dist_pickle.p')

    height, width = img.shape[:2]

    img_cut = np.zeros_like(img)
    if A == 0:
        A = 1
        pass
    ## 블랙박스
    TopLeft = (.4, .65)
    TopRight = (.55, .65)
    BottomLeft = (.0, .95)
    BottomRight = (.9, .95)
    ## 블랙박스

    ## 흰색
    white_lower = (15, 5, 90)
    white_upper = (255, 255, 255)
    ## 흰색

    factor = np.float32([width, height])
    src = np.float32([TopLeft, TopRight, BottomLeft, BottomRight])
    dst = np.float32([(.0, .0), (1., .0), (.0, 1.), (1., 1.)])
    cut = np.float32([TopLeft, TopRight, BottomRight, BottomLeft])
    cut = np.int_(cut * factor)
    cv2.fillPoly(img_cut, [cut], (255, 255, 255))

    img_src = cv2.bitwise_and(img_hsv, img_cut)

    img_temp = babo(img_src, src, dst)
    img_perspect = babo(img, src, dst)

    temp = np.zeros_like(img)

    temp_1 = copy.deepcopy(img_perspect)
    temp_2 = copy.deepcopy(img_perspect)
    temp_3 = copy.deepcopy(img_perspect)
    temp_4 = copy.deepcopy(img_perspect)
    temp_5 = copy.deepcopy(img_perspect)

    Yolo(img, 0.1, 0.4)

    img_white = cv2.inRange(img_temp, white_lower, white_upper)
    img_white_line = cv2.bitwise_and(img_perspect, img_perspect, mask=img_white)

    img_white_line_blur = cv2.blur(img_white_line, (3, 3), 1)

    img_hls = cv2.cvtColor(img_white_line_blur, cv2.COLOR_BGR2HLS)

    img_hls_h, img_hls_l, img_hls_s = cv2.split(img_hls)

    s_thresold = (120, 255)
    h_thresold = (150, 255)

    img_binary_1 = binary(h_thresold, img_hls_h)
    img_binary_2 = binary(s_thresold, img_hls_l)

    img_binary = cv2.addWeighted(img_binary_1, 1., img_binary_2, 1., 0)

    ## 네모 상자 그리기
    histogram = np.sum(img_binary[height // 2:, :], axis=0)
    ## histogram 설명 : numpy.sum 설명 읽기
    ## x,y 2차원 배열을 y축 반틈 아래 부분을 y축을 제거 하고
    ## x축의 값만 더한 값
    midpoint = int(histogram.shape[0] / 2)
    ## midpoint 설명 : histogram의 반틈 길이 좌,우 나누기 위함
    leftx_base = np.argmax(histogram[:midpoint])
    ## leftx_base 설명 : histogram의 0 ~ midpoint 중 제일 큰값
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    ## rightx_base 설명 : histogram의 midpoint ~ end 중 제일 큰값

    leftx_base_.append(leftx_base)
    rightx_base_.append(rightx_base)

    leftx_base = np.int64(np.mean(leftx_base_[-10:]))
    rightx_base = np.int64(np.mean(rightx_base_[-10:]))

    window_height = int(height / nwindows)
    ## window_height 설명 : 사각형 범위 높이 겟수
    nonzero = img_binary.nonzero()
    ## nonzero 설명 : 0이 아닌 값인 x, y 인덱스값 분리
    nonzero_y = np.array(nonzero[0])
    ## nonzero_y 설명 : y축의 0이 아닌 인덱스
    nonzero_x = np.array(nonzero[1])
    ## nonzero_x 설명 : x축의 0이 아닌 인덱스

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    # 네모 상자 그리기
    for window in range(nwindows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(temp, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (100, 255, 255), 3)
        cv2.rectangle(temp, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (100, 255, 255), 3)

        cv2.rectangle(temp_1, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (100, 255, 255), 3)
        cv2.rectangle(temp_1, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (100, 255, 255), 3)

        good_left_inds = ((nonzero_y >= win_y_low) &
                          (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xleft_low) &
                          (nonzero_x < win_xleft_high)).nonzero()[0]
        ## good_left_inds 설명 : 왼쪽 사각형 범위 안에 x값의 0이 아닌 값 추출

        good_right_inds = ((nonzero_y >= win_y_low) &
                           (nonzero_y < win_y_high) &
                           (nonzero_x >= win_xright_low) &
                           (nonzero_x < win_xright_high)).nonzero()[0]
        ## good_right_inds 설명 : 오른쪽 사각형 범위 안에 x값의 0이 아닌 값 추출

        left_lane_inds.append(good_left_inds)
        ## left_lane_inds 설명 : good_left_inds 값을 left_lane_inds에 저장
        right_lane_inds.append(good_right_inds)
        ## right_lane_inds 설명 : good_right_inds 값을 right_lane_inds 저장

        leftx_base_step.append(leftx_current)
        rightx_base_step.append(rightx_current)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzero_x[good_left_inds]))
            ## leftx_current 설명 : nonzero_x 안에 good_left_inds 인덱스 값의 평균 값
            pass
        elif len(good_left_inds) == 0:
            leftx_current = int(np.mean(leftx_base_step[-10:]))
            pass
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzero_x[good_right_inds]))
            ## rightx_current 설명 : nonzero_x 안에 good_right_inds 인덱스 값의 평균 값
            pass
        elif len(good_right_inds) == 0:
            rightx_current = int(np.mean(rightx_base_step[-10:]))
            pass
        # 네모 상자 그리기
    # 네모 상자 그리기

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    ## left_lane_inds, right_lane_inds 설명 : 2차 배열을 1차 배열로 합침


    leftx = nonzero_x[left_lane_inds]
    ## leftx 설명 : nonzero_x 안에 left_lane_inds 인덱스 값
    lefty = nonzero_y[left_lane_inds]
    ## lefty 설명 : nonzero_y 안에 left_lane_inds 인덱스 값
    rightx = nonzero_x[right_lane_inds]
    ## rightx 설명 : nonzero_x 안에 right_lane_inds 인덱스 값
    righty = nonzero_y[right_lane_inds]
    ## righty 설명 : nonzero_y 안에 right_lane_inds 인덱스 값

    leftx_.append(leftx)
    lefty_.append(lefty)
    rightx_.append(rightx)
    righty_.append(righty)

    if len(left_lane_inds) == 0:
        for i in range(len(leftx_)):
            if len(leftx_[-(i + 1)]) != 0:
                leftx = leftx_[-(i + 1)]
                lefty = lefty_[-(i + 1)]
                print("왼쪽 에러")
                break
                pass
            pass
    if len(right_lane_inds) == 0:
        for i in range(len(rightx_)):
            if len(rightx_[-(i + 1)]) != 0:
                rightx = rightx_[-(i + 1)]
                righty = righty_[-(i + 1)]
                print("오른쪽 에러")
                break
                pass
            pass
        pass

    left_fit = np.polyfit(lefty, leftx, 2)
    ## left_fit 설명 : np.polyfit를 통해 lefty, leftx 값에 대한 차수가 2인 값을 반환 (곡선)
    right_fit = np.polyfit(righty, rightx, 2)
    ## right_fit 설명 : np.polyfit를 통해 righty, rightx 값에 대한 차수가 2인 값을 반환 (곡선)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    ## 설명 :  아래와 같다
    # left_fit_[0] = left_fit[0]
    # left_fit_[1] = left_fit[1]
    # left_fit_[2] = left_fit[2]
    # right_fit_[0] = right_fit[0]
    # right_fit_[1] = right_fit[1]
    # right_fit_[2] = right_fit[2]

    ploty = np.linspace(0, height - 1, height)
    ## ploty 설명 : np.linspace 통해 0 ~ 이미지 height 값 까지 값을 순서대로 배열 생성

    left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
    ## left_fitx 설명 : X = a[0] * y^2 + a[1] * y + a[2] // 2차 다항식 회귀 공식
    right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]
    ## right_fitx 설명 : X = a[0] * y^2 + a[1] * y + a[2] // 2차 다항식 회귀 공식

    mid_fitx = (left_fitx + right_fitx) // 2

    temp[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 100]
    temp[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 100, 255]

    temp_2[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 100]
    temp_2[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 100, 255]

    left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    mid = np.array([np.transpose(np.vstack([mid_fitx, ploty]))])
    points = np.hstack((left, right))

    left_center = np.int_(np.mean(left_fitx))
    right_center = np.int_(np.mean(right_fitx))
    mid_center = np.int_(np.mean(mid_fitx))
    road_center = width // 2

    road_pixel = road_width / (right_center - left_center)
    error = mid_center - road_center
    error_pixel = error * road_pixel

    cv2.line(temp_3, (left_center + 10, height // 2 + line_len), (left_center - 10, height // 2 - line_len), (255, 0, 255), 30)
    cv2.line(temp_3, (right_center + 10, height // 2 + line_len), (right_center - 10, height // 2 - line_len), (255, 0, 255), 30)
    cv2.line(temp_3, (mid_center + 10, height // 2 + line_len), (mid_center - 10, height // 2 - line_len), (255, 0, 255), 30)
    cv2.line(temp_3, (width // 2 - 10, height // 2), (width // 2 + 20, height), (255, 0, 255), 10)
    cv2.line(temp_3, (mid_center, height // 2), (width // 2 - 10, height // 2), (255, 255, 255), 30)

    cv2.polylines(temp, np.int_(points), False, (0, 255, 255), 10)
    cv2.polylines(temp_3, np.int_(points), False, (0, 255, 255), 10)
    cv2.polylines(temp_3, np.int_(mid), False, (0, 255, 255), 10)

    cv2.fillPoly(temp, np.int_(points), (0, 255, 0))
    cv2.fillPoly(temp_4, np.int_(points), (0, 255, 0))

    cv2.line(temp, (left_center + 10, height // 2 + line_len), (left_center - 10, height // 2 - line_len), (255, 0, 255), 30)
    cv2.line(temp, (right_center + 10, height // 2 + line_len), (right_center - 10, height // 2 - line_len), (255, 0, 255), 30)
    cv2.line(temp, (mid_center + 10, height // 2 + line_len), (mid_center - 10, height // 2 - line_len), (255, 0, 255), 30)
    cv2.line(temp, (width // 2 - 10, height // 2), (width // 2 + 20, height), (255, 0, 255), 10)
    cv2.line(temp, (mid_center, height // 2), (width // 2 - 10, height // 2), (255, 255, 255), 30)

    temp = babo(temp, dst, src)

    img_result = cv2.addWeighted(img, 1., temp, 0.4, 0)
    if error > 0:
        cv2.putText(img_result, f'right : {abs(error_pixel):.2f}m', (width // 2 - 50, height // 2 + 200), cv2.FONT_ITALIC, 0.5,
                    (0, 0, 255), 2)
    elif error < 0:
        cv2.putText(img_result, f'left : {abs(error_pixel):.2f}m', (width // 2 - 50, height // 2 + 200), cv2.FONT_ITALIC, 0.5,
                    (0, 0, 255), 2)
    elif error == 0:
        cv2.putText(img_result, f'center', (width // 2, height // 2), cv2.FONT_ITALIC, 0.5,
                    (0, 0, 255), 2)
    cv2.putText(img_result, f'{frame} / {frame_end}', (20, height - 40), cv2.FONT_ITALIC, 0.5,
                (255, 255, 255), 2)
    cv2.putText(img_result, f'{frame_count} / {frame_end * 3}', (20, height - 20), cv2.FONT_ITALIC, 0.5,
                (255, 255, 255), 2)

    a = cv2.hconcat([temp_1, temp_2, temp_3, temp_4])
    _, top = my.imgsize(a, 1920)
    result = cv2.vconcat([top, img_result])
    _, result = my.imgsize(result, 1920)

    my.imgshow("result", result, 1000)
    Out.write(result)

    key = cv2.waitKey(1)
    frame += step
    Video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    if Video.get(cv2.CAP_PROP_POS_FRAMES) ==\
            Video.get(cv2.CAP_PROP_FRAME_COUNT):
        if count < 2:
            count += 1
            frame = 0
            # frame_count = 0
            Video = cv2.VideoCapture(f'video/1/{name[count]}.mp4')
            print(name[count])
            Video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = frame_end * count
            frame_end = int(Video.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"{count}")
        elif count > 2:
            print(f"저장 완료")
            break
        pass
    if key == 115: ## s
        cv2.waitKey(0)
        pass
    elif key == 102: ## f
        frame_start = int(input(""))
        frame_count += frame_start
        frame = frame_start
        Video.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        cv2.waitKey(0)
    elif key == 114: ## r
        Video.set(cv2.CAP_PROP_POS_FRAMES, frame_end - 1)
        frame = frame_end - 1
    elif key == 27:
        Video.release()
        Out.release()
        cv2.destroyAllWindows()
        break
        pass
    pass
Video.release()
Out.release()
cv2.destroyAllWindows()
pass
