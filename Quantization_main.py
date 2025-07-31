from ultralytics import YOLO
import cv2
import logging
import openpyxl
import argparse
from datetime import datetime
import os
import shutil
import openvino as ov
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class main():
    def __init__(self,config):
        self.config = config
        self.model = YOLO(f'{config.version}.pt')
        self.model.to(f'{config.device}')  
        logging.info(f"Model loaded: {config.version} on {config.device}")

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
            logging.info(f"Model type : {type(ov_model)}")
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

    def run(self ):
        cv2.namedWindow('w', cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(0)
            
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

            
            current_time = datetime.now().timestamp()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            if config.Quantization == 'intel':
                results = self.model(frame, classes=0, save=False, verbose = False , device=f'intel:{self.device.lower()}')

            elif config.Quantization == 'mac':
                results = self.model(frame ,classes = 0 , save = False , verbose = False)
            else:                
                results = self.model(frame, classes=0, device=f'{self.config.device}', save=False, verbose=False)
        
            people_count = len(results[0].boxes) if results[0].boxes is not None else 0
            now = datetime.now()
            time_str = now.strftime("%H:%M:%S")

            image = results[0].plot()
            cv2.putText(image , f'count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
    parser.add_argument('--save_path', type=str, default='', help='저장경로')
    parser.add_argument('--device', type=str, default='cpu',choices = ['cpu','cuda'] , help='추론장치')
    parser.add_argument('--version',type=str,choices = ['yolo11s', 'yolo11m','yolo11l','yolo11x'] , default = 'yolo11l',help='추론버전')
    parser.add_argument('--save_interval' , type=int , default=0.5 , help='엑셀 저장간격(s)')
    parser.add_argument('--save_image',type = str , default = True ,help='이미지 저장 여부')
    parser.add_argument('--Quantization' , type= str , choices=['intel','mac','None'] , default='mac' , help='가속지원 + 양자화 FP16')
    config = parser.parse_args()
    for key, value in vars(config).items():
        logging.info(f"{key}: {value}")
    
    logging.info('='*30)
    run = main(config)
    run.run()
