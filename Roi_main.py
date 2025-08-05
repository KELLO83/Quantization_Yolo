from ultralytics import YOLO
import cv2
import logging
import openpyxl
import argparse
from datetime import datetime
import os
import shutil
import openvino as ov
import torch
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class main():
    def __init__(self,config):
        self.config = config
        self.model = YOLO(f'{config.version}.pt')
        if config.device == 'cuda':
            flag = torch.cuda.is_available()
            if not flag:
                logging.info("Gpu 추론을 지원하지않습니다")
        config.device = 'cpu'
        self.model.to(f'{config.device}')  
        logging.info(f"Model loaded: {config.version} on {config.device}")
        
        self.roi_x1 = config.roi_x1 if hasattr(config, 'roi_x1') and config.roi_x1 is not None else 200
        self.roi_y1 = config.roi_y1 if hasattr(config, 'roi_y1') and config.roi_y1 is not None else 150
        self.roi_x2 = config.roi_x2 if hasattr(config, 'roi_x2') and config.roi_x2 is not None else 400
        self.roi_y2 = config.roi_y2 if hasattr(config, 'roi_y2') and config.roi_y2 is not None else 350
        self.roi_initialized = False
        
        self.mouse_x = 0
        self.mouse_y = 0

        if config.Quantization == 'intel' and config.device == 'cpu':
            ov_model_path = f"{config.version}_openvino_model/"
            device = self.get_available_openvino_device()
            if device == 'CPU':
                logging.info(f"Intel GPU NPU 추론을 지원히지않습니다 !!!!! {config.Quantization} -----> None 설정으로 실행해주세요")
                exit(1)
            if not hasattr(self,'device'):
                self.device = device

            if not os.path.exists(ov_model_path):
                logging.info("Exporting to OpenVINO format...")
                try:
                    self.model.export(format='openvino' , half = True , simplify= True , dynamic = False , verbose = True , optimize = True)
                except Exception as e:
                    logging.info(f"OpenVINO EXPORT Fail.... 일반버전으로 실행해주세요 ({config.Quantization} ---> None설정)")
                    logging.info(f"{e}")
                    exit(1)
            else:
                logging.info("OpenVINO loadded..")
            
            ov_model = YOLO(ov_model_path, task='detect')
            logging.info(f"Model type : {type(ov_model)}")
            logging.info(f'Intel OpenVINO model loaded from: {ov_model_path}')
            self.model = ov_model

        
        elif config.Quantization == 'mac' and config.device =='cpu':
            coreml_path = f"{config.version}.mlpackage/"
            if not os.path.exists(coreml_path):
                logging.info("MacOS에서 사용가능한 가속모드입니다... Windows환경 사용불가")
                logging.info("Exporting to CoreML format...")
                try:
                    self.model.export(format="coreml", half=True) 
                except Exception as e:
                    logging.info(f"CoreML EXPORT Fail.... 일반버전으로 실행해주세요 ({config.Quantization} ---> None설정)")
                    logging.info(f"{e}")
                    exit(1)

            coreml_model = YOLO(f"{config.version}.mlpackage/" , task='detect')
            logging.info(f"Model type : {type(coreml_model)}")
            logging.info(f"CoreML model loaded from: {coreml_path}")
            self.model = coreml_model

        

        start_time = datetime.now().strftime("%d_%H_%M_%S")
        self.excel_file = os.path.join(config.save_path, f'{start_time}_results.xlsx')
        self._init_excel()  
        self.last_save_time = datetime.now().timestamp()
        self.total_people_count = 0

        if self.config.save_image:
            self.image_save_path = os.path.join(config.save_path, 'images')

            if os.path.exists(self.image_save_path):
                shutil.rmtree(self.image_save_path)

            os.makedirs(self.image_save_path, exist_ok=True)

    def is_in_roi(self, x1, y1, x2, y2, frame_width, frame_height):
        """바운딩 박스가 ROI 영역과 겹치는지 확인"""
        roi_x1_pixel = self.roi_x1
        roi_y1_pixel = self.roi_y1
        roi_x2_pixel = self.roi_x2
        roi_y2_pixel = self.roi_y2
        
        # 바운딩 박스 중심점이 ROI 안에 있는지 확인
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return (roi_x1_pixel <= center_x <= roi_x2_pixel and 
                roi_y1_pixel <= center_y <= roi_y2_pixel)
    
    def draw_roi(self, image):
        """ROI 영역을 녹색 박스로 그리기"""
        height, width = image.shape[:2]
        if not self.roi_initialized:
            self.roi_initialized = True
            logging.info(f"ROI 영역 설정: ({self.roi_x1}, {self.roi_y1}) to ({self.roi_x2}, {self.roi_y2})")
        
        x1 = int(self.roi_x1)
        y1 = int(self.roi_y1)
        x2 = int(self.roi_x2)
        y2 = int(self.roi_y2)
        
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, 'ROI', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백 함수 - 마우스 좌표 추적"""
        self.mouse_x = x
        self.mouse_y = y
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"마우스 좌표: x={x}, y={y}")

    def run(self ):
        cv2.namedWindow('w', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('w', self.mouse_callback)
        cap = cv2.VideoCapture(config.cam)
            
        if cap.isOpened():
            logging.info("Camera opened successfully.")
        else:
            logging.error("Failed to open camera.")
            return
        
        count = 0
        fps = 0
        prev_time = datetime.now().timestamp()
        while True:
            ret, frame = cap.read()
            if not ret :
                logging.info("동영상 버퍼 오버플로우 발생 동영상 읽는속도 >>>>>>>>>> 모델 추론속도")
                logging.info(f"{config.version} 모델이 현재 환경에 적합하지않습니다 모델 버전을 낮춰주세요")
                cap.release()
                cv2.destroyAllWindows()
                break

            if config.cut_image.lower() == 'true':
                black_image = np.zeros_like(frame, dtype=np.uint8)
                black_image[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2] = 255
                frame = cv2.bitwise_and(frame, black_image)

            
            current_time = datetime.now().timestamp()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            if config.Quantization == 'intel':
                results = self.model(frame, classes=0, save=False, verbose = False , device=f'intel:{self.device.lower()}')

            elif config.Quantization == 'mac':
                results = self.model(frame ,classes = 0 , save = False , verbose = False)
            else:                
                results = self.model(frame, classes=0, device=f'{self.config.device}', save=False, verbose=False)
        
            roi_people_count = 0
            frame_height, frame_width = frame.shape[:2]
            
            image = results[0].plot()
            
            image = self.draw_roi(image)
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    if self.is_in_roi(x1, y1, x2, y2, frame_width, frame_height):
                        roi_people_count += 1
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        cv2.putText(image, 'IN ROI', (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.circle(image, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 5, (0, 0, 255), -1)
                    else:
                        cv2.circle(image, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 5, (0, 255, 0), -1)
            
            people_count = roi_people_count  
            now = datetime.now()
            time_str = now.strftime("%H:%M:%S")
            cv2.putText(image , f'ROI People: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Total Detected: {len(results[0].boxes) if results[0].boxes is not None else 0}', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(image, f'Mouse: x={self.mouse_x}, y={self.mouse_y}', 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            if now.timestamp() - self.last_save_time > self.config.save_interval:
                self.worksheet.append([time_str, people_count])
                self.workbook.save(self.excel_file)
                logging.info(f"엑셀 업데이트: {time_str} - {people_count} people")
                self.last_save_time = now.timestamp()
                cv2.imwrite(os.path.join(self.image_save_path, f'{count}.jpg'), image) if self.config.save_image else None
                self.total_people_count += people_count
                count += 1            
        
            cv2.imshow('w', image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: 
                self.worksheet.append(['인원수',self.total_people_count])
                self.worksheet.append(['FPS', fps])
                self.workbook.save(self.excel_file)
                break

        cap.release()
        cv2.destroyAllWindows()

    def _init_excel(self):
        try:
            self.workbook = openpyxl.Workbook()
            self.worksheet = self.workbook.active
            self.worksheet['A1'] = '시간'
            self.worksheet['B1'] = '인원수'
            self.workbook.save(self.excel_file)
            logging.info(f"Excel file initialized and reset: {self.excel_file}")

        except Exception as e:
            logging.error(f"Failed to initialize Excel file: {e}")


    
    def get_available_openvino_device(self):
        """ Intel  GPU TPU 확인 """
        core = ov.Core()
        available_devices = core.available_devices
        logging.info(f"사용 가능한 OpenVINO 장치: {available_devices}")

        for device in ['GPU', 'NPU', 'CPU']:
            if device in available_devices:
                logging.info(f"선택된 장치: {device}")
                return device
        
        return 'CPU'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="object detection.")
    parser.add_argument('--cam' , type=str , default=0 , help='내장 카메라 0번  외부연결카메라1번이상..')
    parser.add_argument('--save_path', type=str, default='', help='저장경로')
    parser.add_argument('--device', type=str, default='cpu',choices = ['cpu','cuda'] , help='추론장치')
    parser.add_argument('--version',type=str,choices = ['yolo11s','yolo11l','yolo11x'] , default = 'yolo11l',help='추론버전')
    parser.add_argument('--save_interval' , type=int , default=0.5 , help='엑셀 저장간격(s)')
    parser.add_argument('--save_image',type = str , default = False ,help='이미지 저장 여부')
    parser.add_argument('--Quantization' , type= str , choices=['intel','mac','None'] , default='mac' , help='가속지원 + 양자화 FP16')
    
    # x1우상단 y1좌상단 x2우하단 y2좌하단
    parser.add_argument('--roi_x1', type=int, default=100, help='ROI 영역 좌상단 X 좌표 (픽셀)')
    parser.add_argument('--roi_y1', type=int, default=200, help='ROI 영역 좌상단 Y 좌표 (픽셀)')
    parser.add_argument('--roi_x2', type=int, default=1600, help='ROI 영역 우하단 X 좌표 (픽셀)')
    parser.add_argument('--roi_y2', type=int, default=800, help='ROI 영역 우하단 Y 좌표 (픽셀)')


    # 이미지 자르기 true -> 모델입력전에 프레임에서 ROI 영역제외 흑백처리
    parser.add_argument('--cut_image', type=str, default='true', help='이미지 자르기 여부')

    config = parser.parse_args()
    for key, value in vars(config).items():
        logging.info(f"{key}: {value}")
    
    logging.info('='*30)
    run = main(config)
    run.run()
