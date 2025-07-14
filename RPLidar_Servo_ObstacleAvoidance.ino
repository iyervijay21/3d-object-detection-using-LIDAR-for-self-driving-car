#include <RPLidar.h>
#include <Servo.h>

// Motor control pins
#define LEFT_MOTOR_PIN1 10
#define LEFT_MOTOR_PIN2 11
#define RIGHT_MOTOR_PIN1 12
#define RIGHT_MOTOR_PIN2 13
#define LIDAR_MOTOR 8
#define SERVO_PIN 9

// Constants
const float SAFE_DISTANCE = 50.0;  // Safe distance in cm
const float MIN_DISTANCE = 30.0;   // Minimum distance threshold
const int SCAN_INTERVAL = 200;     // Time between scans in ms

RPLidar lidar;
Servo steeringServo;

float minDistance = 10000.0;
float minAngle = 0.0;
bool obstacleDetected = false;

void setup() {
  Serial.begin(115200);
  lidar.begin(Serial);
  
  // Initialize motor control pins
  pinMode(LEFT_MOTOR_PIN1, OUTPUT);
  pinMode(LEFT_MOTOR_PIN2, OUTPUT);
  pinMode(RIGHT_MOTOR_PIN1, OUTPUT);
  pinMode(RIGHT_MOTOR_PIN2, OUTPUT);
  
  // Initialize LIDAR motor control
  pinMode(LIDAR_MOTOR, OUTPUT);
  
  // Attach servo
  steeringServo.attach(SERVO_PIN);
  steeringServo.write(90);  // Center position
  
  // Start LIDAR
  analogWrite(LIDAR_MOTOR, 255);
  delay(1000);
  lidar.startScan();
}

void loop() {
  if (IS_OK(lidar.waitPoint())) {
    processLidarData();
  } else {
    handleLidarError();
  }
}

void processLidarData() {
  float distance = lidar.getCurrentPoint().distance;
  float angle = lidar.getCurrentPoint().angle;
  bool isNewScan = lidar.getCurrentPoint().startBit;

  if (isNewScan) {
    handleObstacle();
    minDistance = 10000.0;
    obstacleDetected = false;
    delay(SCAN_INTERVAL);
  } 
  else if (distance > 0) {
    updateMinDistance(distance, angle);
  }
}

void updateMinDistance(float distance, float angle) {
  // Consider front 180 degrees (270° to 90° crossing 0°)
  if ((angle >= 270.0 || angle <= 90.0) && distance < minDistance) {
    minDistance = distance;
    minAngle = angle;
    obstacleDetected = true;
  }
}

void handleObstacle() {
  if (!obstacleDetected) {
    moveForward();
    steeringServo.write(90);  // Center steering
    return;
  }

  if (minDistance < MIN_DISTANCE) {
    stopMotors();
    avoidCollision();
  } 
  else if (minDistance < SAFE_DISTANCE) {
    navigateAroundObstacle();
  } 
  else {
    moveForward();
    steeringServo.write(90);  // Center steering
  }
}

void avoidCollision() {
  // Adjust angle to -180° to 180° range
  float adjustedAngle = (minAngle > 180) ? minAngle - 360 : minAngle;
  
  if (adjustedAngle < 0) {
    // Obstacle on left, turn right
    steeringServo.write(180);
  } else {
    // Obstacle on right, turn left
    steeringServo.write(0);
  }
  // Back up slightly
  moveBackward();
  delay(500);
  stopMotors();
}

void navigateAroundObstacle() {
  float adjustedAngle = (minAngle > 180) ? minAngle - 360 : minAngle;
  
  if (adjustedAngle < 0) {
    steeringServo.write(135);  // Moderate right turn
  } else {
    steeringServo.write(45);   // Moderate left turn
  }
  moveForward();
}

void handleLidarError() {
  analogWrite(LIDAR_MOTOR, 0);  // Stop LIDAR motor
  stopMotors();
  
  rplidar_response_device_info_t info;
  if (IS_OK(lidar.getDeviceInfo(info, 100))) {
    lidar.startScan();
    analogWrite(LIDAR_MOTOR, 255);
    delay(1000);
  }
}

// Motor control functions
void moveForward() {
  digitalWrite(LEFT_MOTOR_PIN1, HIGH);
  digitalWrite(LEFT_MOTOR_PIN2, LOW);
  digitalWrite(RIGHT_MOTOR_PIN1, HIGH);
  digitalWrite(RIGHT_MOTOR_PIN2, LOW);
}

void moveBackward() {
  digitalWrite(LEFT_MOTOR_PIN1, LOW);
  digitalWrite(LEFT_MOTOR_PIN2, HIGH);
  digitalWrite(RIGHT_MOTOR_PIN1, LOW);
  digitalWrite(RIGHT_MOTOR_PIN2, HIGH);
}

void stopMotors() {
  digitalWrite(LEFT_MOTOR_PIN1, LOW);
  digitalWrite(LEFT_MOTOR_PIN2, LOW);
  digitalWrite(RIGHT_MOTOR_PIN1, LOW);
  digitalWrite(RIGHT_MOTOR_PIN2, LOW);
}
