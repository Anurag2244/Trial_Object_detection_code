import cv2
import numpy as np
from ultralytics import YOLO

class ObjectChangeDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")     # Load YOLOv8 model (nano version for speed)
        
        self.prev_objects = {}    
        self.current_objects = {}
        
        self.frame_count = 0   
        self.prev_time = 0
        
    def detect_changes(self, frame):

        results = self.model(frame)# Run YOLO detection
        
        self.current_objects = {}       # Reset current objects
        
        for result in results:      # Extract detected objects
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = box
                obj_id = f"{x1:.0f}_{y1:.0f}_{x2:.0f}_{y2:.0f}"  # Simple ID (not robust)
                self.current_objects[obj_id] = (box, cls)
        
        missing_objs = set(self.prev_objects.keys()) - set(self.current_objects.keys())     # Check for missing objects
        for obj_id in missing_objs:
            cls = self.prev_objects[obj_id][1]
            print(f"[ALERT] Object {obj_id} (Class: {cls}) disappeared!")
        
        new_objs = set(self.current_objects.keys()) - set(self.prev_objects.keys())     # Check for new objects
        for obj_id in new_objs:
            cls = self.current_objects[obj_id][1]
            print(f"[ALERT] New object {obj_id} (Class: {cls}) appeared!")
        
        self.prev_objects = self.current_objects.copy()     # Update previous objects  
        
        annotated_frame = results[0].plot()     # Draw bounding boxes (optional)
        cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame

def main():
    detector = ObjectChangeDetector()
    cap = cv2.VideoCapture(0)  # Use webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = detector.detect_changes(frame)
        cv2.imshow("Object Change Detection", processed_frame)
    
    cap.release()

if __name__ == "__main__":
    main()
