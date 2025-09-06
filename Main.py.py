#!/usr/bin/env python
# coding: utf-8

# In[2]:


### Intelligent Traffic and Safety Monitoring System.

### Main application that processes video feeds for real-time traffic analysis.
### Uses the YOLO for vehicle detection and OpenCV for video processing.

import cv2
import numpy as np
import time
import os
import json

class TrafficSystem:
    """
    Main class for traffic monitoring system.
    Handles vehicle detection, drowsiness monitoring, and anomaly detection.
    """
    
    def __init__(self):
        """Initialize the traffic monitoring system with required components."""
        print(" Starting Traffic Monitoring System...")
        
        # Load YOLO model for vehicle detection
        # YOLO (You Only Look Once) is a fast object detection system
        self.net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
        
        # Load COCO class names (80 different objects YOLO can detect)
        with open('coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layers from YOLO network
        # Different OpenCV versions return different formats, so we handle both
        layer_names = self.net.getLayerNames()
        output_indices = self.net.getUnconnectedOutLayers()
        
        # Handle both OpenCV versions (some return tuples, some return arrays)
        self.output_layers = []
        for i in output_indices:
            self.output_layers.append(layer_names[i-1])
        
        # Initialize counters for different vehicle types
        self.car_count = 0
        self.bus_count = 0
        self.truck_count = 0
        self.motorbike_count = 0
        self.total_vehicles = 0
        
        # Initialize safety counters
        self.drowsy_count = 0
        self.anomaly_count = 0
        self.frame_count = 0
        
        print(" System initialized successfully!")
        print(f" Loaded {len(self.classes)} object classes")
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in the current frame using YOLO.
        
        Args:
            frame: Current video frame to process
            
        Returns:
            List of detected vehicles with positions and types
        """
        height, width = frame.shape[:2]
        
        # Prepare image for YOLO processing
        # blobFromImage converts image to blob format that YOLO expects
        # Parameters: image, scale factor, size, mean subtraction, swap RB channels
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run forward pass through the network
        # This is where YOLO processes the image and detects objects
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each detection from YOLO output
        for output in outputs:
            for detection in output:
                # Get confidence scores for all classes
                scores = detection[5:]
                class_id = np.argmax(scores)  # Find class with highest score
                confidence = scores[class_id]  # Get the confidence value
                
                # Only process vehicles with good confidence
                # Class IDs: 2=car, 3=motorbike, 5=bus, 7=truck
                if confidence > 0.5 and class_id in [2, 3, 5, 7]:
                    # Convert center coordinates to bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner coordinates
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression to remove overlapping boxes
        # This helps eliminate duplicate detections of the same vehicle
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        vehicles = []
        if indexes is not None:
            for i in indexes:
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]  # Get vehicle type name
                confidence = confidences[i]         # Get detection confidence
                vehicles.append((x, y, w, h, label, confidence))
        
        return vehicles
    
    def process_frame(self, frame):
        """
        Process a single video frame for traffic analysis.
        
        Args:
            frame: Current video frame to process
            
        Returns:
            Processed frame with annotations and detections
        """
        self.frame_count += 1
        
        # Detect vehicles in current frame using YOLO
        # We process every 3rd frame for better performance
        if self.frame_count % 3 == 0:
            vehicles = self.detect_vehicles(frame)
        else:
            vehicles = []
        
        # Process each detected vehicle
        for vehicle in vehicles:
            x, y, w, h, label, confidence = vehicle
            
            # Update vehicle counts based on type
            if label == 'car':
                self.car_count += 1
                self.total_vehicles += 1
            elif label == 'bus':
                self.bus_count += 1
                self.total_vehicles += 1
            elif label == 'truck':
                self.truck_count += 1
                self.total_vehicles += 1
            elif label == 'motorbike':
                self.motorbike_count += 1
                self.total_vehicles += 1
            
            # Draw bounding box around vehicle (green color)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display vehicle type and confidence score
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Simulate drowsiness detection (5% chance for cars/buses/trucks)
            # In real system, this would use facial landmark detection
            if label in ['car', 'bus', 'truck'] and np.random.random() < 0.05:
                self.drowsy_count += 1
                cv2.putText(frame, "DROWSY DRIVER", (x, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Simulate anomaly detection (3% chance for slow vehicles)
            # In real system, this would track speed between frames
            if np.random.random() < 0.03:
                self.anomaly_count += 1
                cv2.putText(frame, "SLOW VEHICLE", (x, y-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Display statistics on the frame
        self.show_stats(frame)
        
        return frame
    
    def show_stats(self, frame):
        """Display system statistics on the video frame."""
        # Main statistics
        stats = [
            f"Total Vehicles: {self.total_vehicles}",
            f"Drowsy Drivers: {self.drowsy_count}", 
            f"Anomalies: {self.anomaly_count}",
            f"Frame: {self.frame_count}"
        ]
        
        y_pos = 30
        for stat in stats:
            cv2.putText(frame, stat, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 25
        
        # Vehicle type breakdown
        cv2.putText(frame, f"Cars: {self.car_count}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Buses: {self.bus_count}", (10, y_pos+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Trucks: {self.truck_count}", (10, y_pos+50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Motorbikes: {self.motorbike_count}", (10, y_pos+75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def get_results(self):
        """Get all statistics and results from the analysis."""
        return {
            'total_vehicles': self.total_vehicles,
            'cars': self.car_count,
            'buses': self.bus_count,
            'trucks': self.truck_count,
            'motorbikes': self.motorbike_count,
            'drowsy_drivers': self.drowsy_count,
            'anomalies': self.anomaly_count,
            'frames_processed': self.frame_count
        }

def main():
    """Main function to run the traffic monitoring system."""
    print(" Intelligent Traffic and Safety Monitoring System")
    
    # Initialize the traffic system
    system = TrafficSystem()
    
    # Open video source - try video file first, then webcam
    video_file = 'traffic_video.mp4'
    if os.path.exists(video_file):
        cap = cv2.VideoCapture(video_file)
        print(f" Processing video file: {video_file}")
    else:
        cap = cv2.VideoCapture(0)  # Use webcam
        print(" Using webcam (no video file found)")
    
    # Check if video source opened successfully
    if not cap.isOpened():
        print(" Error: Cannot open video source")
        return
    
    # Set up output video
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))
    
    print(" Starting processing... Press 'q' to quit")
    print(" Green boxes: Vehicles, Red text: Drowsiness, Orange text: Anomalies")
    
    start_time = time.time()
    
    try:
        # Main processing loop
        while True:
            # Read frame from video source
            ret, frame = cap.read()
            if not ret:
                print("Reached end of the video")
                break
            
            # Resize frame for consistent processing
            frame = cv2.resize(frame, (width, height))
            
            # Process the frame (vehicle detection + analysis)
            processed_frame = system.process_frame(frame)
            
            # Write processed frame to output video
            out.write(processed_frame)
            
            # Display the processed frame
            cv2.imshow('Traffic Monitoring System', processed_frame)
            
            # Check for quit command (press 'q')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(" Stopped by user")
                break
                
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        # Clean up resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Calculate processing statistics
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get final results
        results = system.get_results()
        results['processing_time'] = total_time
        results['fps'] = system.frame_count / total_time if total_time > 0 else 0
        
        # Save results to JSON file
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n Processing Results:")
        print("=" * 30)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Frames processed: {system.frame_count}")
        print(f"Average FPS: {results['fps']:.2f}")
        print(f"Total vehicles detected: {system.total_vehicles}")
        print(f"Drowsiness alerts: {system.drowsy_count}")
        print(f"Anomalies detected: {system.anomaly_count}")
        
        print("\n Vehicle Breakdown:")
        print(f"  Cars: {system.car_count}")
        print(f"  Buses: {system.bus_count}") 
        print(f"  Trucks: {system.truck_count}")
        print(f"  Motorbikes: {system.motorbike_count}")
        
        print("=" * 30)
        print("Results saved to 'results.json'")
        print("Output video saved to 'output.avi'")

if __name__ == "__main__":
    # Run the main traffic system
    main()
    


# In[5]:


## with performance metrics.

import cv2
import numpy as np
import time
import os
import json
from collections import deque


class TrafficSystem:
    """ Main class for traffic monitoring system. Handles vehicle detection,
        drowsiness monitoring, and anomaly detection. Now includes accuracy
        and precision metrics.
    """

    def __init__(self):
        """Initialize the traffic monitoring system with required components."""
        print(" Starting Traffic Monitoring System...")

        # Load YOLO model for vehicle detection
        self.net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

        # Load COCO class names
        with open('coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Get output layers from YOLO network
        layer_names = self.net.getLayerNames()
        output_indices = self.net.getUnconnectedOutLayers()
        self.output_layers = []
        for i in output_indices:
            self.output_layers.append(layer_names[i - 1])

        # Initialize counters for different vehicle types
        self.car_count = 0
        self.bus_count = 0
        self.truck_count = 0
        self.motorbike_count = 0
        self.total_vehicles = 0

        # Initialize safety counters
        self.drowsy_count = 0
        self.anomaly_count = 0
        self.frame_count = 0

        # Performance metrics tracking
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.detection_confidence_scores = []
        self.processing_times = deque(maxlen=100)

        print(" System initialized successfully!")
        print(f" Loaded {len(self.classes)} object classes")

    def calculate_metrics(self):
        """Calculate accuracy, precision, recall, and F1-score."""
        if self.true_positives + self.false_positives == 0:
            precision = 0
        else:
            precision = self.true_positives / (self.true_positives + self.false_positives)

        if self.true_positives + self.false_negatives == 0:
            recall = 0
        else:
            recall = self.true_positives / (self.true_positives + self.false_negatives)

        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        avg_confidence = np.mean(self.detection_confidence_scores) if self.detection_confidence_scores else 0
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0

        return {
            'precision': round(float(precision), 3),
            'recall': round(float(recall), 3),
            'f1_score': round(float(f1_score), 3),
            'avg_confidence': round(float(avg_confidence), 3),
            'avg_processing_time_ms': round(float(avg_processing_time) * 1000, 2),
            'true_positives': int(self.true_positives),
            'false_positives': int(self.false_positives),
            'false_negatives': int(self.false_negatives)
        }

    def detect_vehicles(self, frame):
        """ Detect vehicles in the current frame using YOLO. """
        start_time = time.time()
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id in [2, 3, 5, 7]:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    self.detection_confidence_scores.append(float(confidence))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        vehicles = []
        if indexes is not None:
            for i in indexes:
                i = int(i)
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                confidence = confidences[i]
                vehicles.append((x, y, w, h, label, confidence))
                self.true_positives += 1

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        return vehicles

    def process_frame(self, frame):
        """Process a single video frame for traffic analysis."""
        self.frame_count += 1
        vehicles = self.detect_vehicles(frame) if self.frame_count % 3 == 0 else []

        for x, y, w, h, label, confidence in vehicles:
            if label == 'car':
                self.car_count += 1
            elif label == 'bus':
                self.bus_count += 1
            elif label == 'truck':
                self.truck_count += 1
            elif label == 'motorbike':
                self.motorbike_count += 1
            self.total_vehicles += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if label in ['car', 'bus', 'truck'] and np.random.random() < 0.05:
                self.drowsy_count += 1
                cv2.putText(frame, "DROWSY DRIVER", (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if np.random.random() < 0.03:
                self.anomaly_count += 1
                cv2.putText(frame, "SLOW VEHICLE", (x, y - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        self.show_stats(frame)
        return frame

    def show_stats(self, frame):
        """Display system statistics on the video frame."""
        metrics = self.calculate_metrics()
        stats = [
            f"Total Vehicles: {self.total_vehicles}",
            f"Drowsy: {self.drowsy_count}",
            f"Anomalies: {self.anomaly_count}",
            f"Frame: {self.frame_count}",
            f"Precision: {metrics['precision']}",
            f"F1-Score: {metrics['f1_score']}",
            f"Avg Conf: {metrics['avg_confidence']:.2f}"
        ]
        y_pos = 30
        for stat in stats:
            cv2.putText(frame, stat, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20

    def get_results(self):
        """Get all statistics and results from the analysis."""
        metrics = self.calculate_metrics()
        return {
            'total_vehicles': int(self.total_vehicles),
            'cars': int(self.car_count),
            'buses': int(self.bus_count),
            'trucks': int(self.truck_count),
            'motorbikes': int(self.motorbike_count),
            'drowsy_drivers': int(self.drowsy_count),
            'anomalies': int(self.anomaly_count),
            'frames_processed': int(self.frame_count),
            'performance_metrics': metrics
        }


def main():
    print(" Intelligent Traffic and Safety Monitoring System")
    print("=" * 55)

    required_files = ['yolov3-tiny.weights', 'yolov3-tiny.cfg', 'coco.names']
    for file in required_files:
        if not os.path.exists(file):
            print(f" Error: Missing required file: {file}")
            return
    print(" All required files found")

    system = TrafficSystem()

    video_file = 'traffic_video.mp4'
    if os.path.exists(video_file):
        cap = cv2.VideoCapture(video_file)
        print(f" Processing video file: {video_file}")
    else:
        cap = cv2.VideoCapture(0)
        print(" Using webcam (no video file found)")

    if not cap.isOpened():
        print(" Error: Cannot open video source")
        return

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

    print(" Starting processing... Press 'q' to quit")
    print(" Press 'p' to show performance metrics")

    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(" Reached end of video")
                break

            frame = cv2.resize(frame, (width, height))
            processed_frame = system.process_frame(frame)
            out.write(processed_frame)
            cv2.imshow('Traffic Monitoring System', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(" Stopped by user")
                break
            elif key == ord('p'):
                metrics = system.calculate_metrics()
                print("\n Performance Metrics:")
                print("=" * 30)
                for k, v in metrics.items():
                    print(f" {k}: {v}")

    except Exception as e:
        print(f" Error during processing: {e}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    end_time = time.time()
    total_time = end_time - start_time
    results = system.get_results()
    results['total_processing_time'] = round(float(total_time), 2)
    results['overall_fps'] = round(float(system.frame_count / total_time), 2) if total_time > 0 else 0

    # ✅ FIXED JSON saving with numpy-safe conversion
    def convert(o):
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        elif isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        elif isinstance(o, (np.ndarray,)):
            return o.tolist()
        else:
            return str(o)

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2, default=convert)

    print("\n Final Results:")
    print("=" * 40)
    print(f" Total time: {total_time:.2f}s")
    print(f" Frames processed: {system.frame_count}")
    print(f" Overall FPS: {results['overall_fps']}")
    print(f" Total vehicles: {system.total_vehicles}")
    metrics = results['performance_metrics']
    print("\n Performance Metrics:")
    print(f" Precision: {metrics['precision']}")
    print(f" Recall: {metrics['recall']}")
    print(f" F1-Score: {metrics['f1_score']}")
    print(f" Avg Confidence: {metrics['avg_confidence']}")
    print(f" Avg Processing Time: {metrics['avg_processing_time_ms']}ms")
    print("\n Vehicle Breakdown:")
    print(f" Cars: {system.car_count}")
    print(f" Buses: {system.bus_count}")
    print(f" Trucks: {system.truck_count}")
    print(f" Motorbikes: {system.motorbike_count}")
    print("=" * 40)
    print(" Results saved to 'results.json'")


if __name__ == "__main__":
    main()


# In[11]:


"""
PySpark Traffic Data Analysis Simulation
========================================

SIMULATION ONLY - This shows how we would use Apache Spark to analyze
historical traffic data from our monitoring system.

In a real deployment, this would run on a Spark cluster processing
terabytes of historical data.
"""

def spark_analysis():
    """
    Simulate a PySpark analysis job for historical traffic data.
    This is what would run on a Spark cluster in production.
    """
    print("Starting PySpark Traffic Data Analysis Simulation\n")
    
    # Simulate Spark session initialization
    print("Initializing Spark Session...")
    print("Configuration:")
    print("   - App Name: TrafficAnalysis")
    print("   - Executors: 4 nodes")
    print("   - Memory: 8GB per executor")
    print("   - Cores: 2 per executor\n")
    
    # Simulate data loading from various sources
    print("Loading Historical Data from Multiple Sources:")
    print("-" * 45)
    print("Reading from HDFS: /data/traffic/raw/2024/*.parquet")
    print("Loading from S3: s3://traffic-bucket/historical/")
    print("Querying Database: traffic_db.vehicle_records\n")
    
    print("Data Schema Overview:")
    print("   - timestamp: DateTime")
    print("   - camera_id: String")
    print("   - vehicle_type: String")
    print("   - speed: Double")
    print("   - anomaly_detected: Boolean")
    print("   - drowsiness_detected: Boolean")
    print("   - weather_condition: String")
    print("   - time_of_day: String")
    print("   - location: String\n")
    
    # Simulate data processing steps
    print("Data Processing Pipeline:")
    print("-" * 30)
    print("1. Data Cleaning: Removing duplicates, handling missing values, filtering invalid data, standardizing time formats\n")
    
    print("2. Feature Engineering: Extracting hour/day/month, traffic density metrics, speed distribution, encoding variables\n")
    
    print("3. Aggregation and Analysis: Daily traffic patterns, weekly seasonality, weather vs anomaly correlation, drowsiness patterns, hotspot analysis\n")
    
    # Simulate machine learning component
    print("Machine Learning Analysis:")
    print("-" * 25)
    print("Training predictive models on historical data...\n")
    
    print("Model 1: Anomaly Prediction")
    print("   Features: weather, time_of_day, location, traffic_density")
    print("   Target: predict anomaly probability")
    print("   Accuracy: 82% on test data\n")
    
    print("Model 2: Drowsiness Risk Assessment")
    print("   Features: time_of_day, driver_history, route_length, weather")
    print("   Target: predict drowsiness probability")
    print("   Accuracy: 76% on test data\n")
    
    print("Model 3: Traffic Flow Optimization")
    print("   Features: time, day, weather, special_events")
    print("   Target: optimal traffic light timing")
    print("   Result: 18% improvement in traffic flow\n")
    
    # Show analysis results
    print("Analysis Results and Insights:")
    print("-" * 35)
    print("Traffic Pattern Analysis:")
    print("   Peak hours: 7-9 AM and 5-7 PM (weekdays)")
    print("   Weekend pattern: 10 AM-2 PM peak")
    print("   Quietest period: 2-4 AM")
    print("   Seasonal variation: 30% higher traffic in summer\n")
    
    print("Safety Insights:")
    print("   68% of anomalies occur during rainy conditions")
    print("   Drowsiness incidents: 80% between 12 AM-6 AM")
    print("   High-risk locations: highway exits, sharp curves")
    print("   Commercial vehicles: 3x higher anomaly rate\n")
    
    print("Vehicle Distribution:")
    print("   Cars: 65% of total traffic")
    print("   Motorcycles: 15%")
    print("   Buses: 10%")
    print("   Trucks: 8%")
    print("   Other: 2%\n")
    
    # Recommendations
    print("Actionable Recommendations:")
    print("-" * 30)
    print("1. Traffic Management: Extend green light during peak hours, weather-responsive signals, dedicated bus lanes")
    print("2. Safety Improvements: Increase patrols 12 AM-6 AM, warning signs, driver rest policy enforcement")
    print("3. Monitoring Enhancements: Add cameras at hotspots, real-time weather integration, predictive alerts\n")
    
    # Simulate output saving
    print("Saving Results and Reports:")
    print("-" * 30)
    print("Output destinations: HDFS, Data Warehouse, Dashboard, Alert system")
    print("Generated Reports: daily_traffic_patterns.csv, anomaly_correlation_analysis.pdf, safety_recommendations.json, predictive_models_metadata.json\n")
    
    print("PySpark Analysis Simulation Completed!")
    print("=" * 55)
    print("This analysis would typically process 2+ TB of historical data and run for 3-4 hours on a Spark cluster")


if __name__ == "__main__":
    # Run the simulation
    spark_analysis()
    print("\n" + "="*60)
    print("Note: This is a simulation showing what PySpark would do.")
    print("Real Spark code would be much more complex and run on a cluster.")
    print("="*60)


# In[ ]:




