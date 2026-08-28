#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>

// [사용자 설정] 본인의 WiFi 정보로 변경하세요.
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

WebServer server(80);
const int servoPin = 13;
Servo myServo;

int currentPos = 0;

void setup() {
  Serial.begin(115200);

  // ESP32Servo 타이머 할당
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  
  myServo.setPeriodHertz(50);
  myServo.attach(servoPin, 500, 2400);

  // 초기 상태: 0도 (밸브 닫힘)
  myServo.write(0);
  currentPos = 0;

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // ---------------------------------------------------------
  // 1. /open → 0도에서 90도로 회전 (밸브 열기, 물 흐름)
  // ---------------------------------------------------------
  server.on("/open", []() {
    Serial.println("[수신] /open 명령 수신!");
    
    for (int pos = currentPos; pos <= 90; pos += 2) {
      myServo.write(pos);
      delay(15);
    }
    currentPos = 90;

    Serial.println("[모터 완료] 밸브 열림! (90도)");
    server.send(200, "text/plain", "Servo Opened: 90");
  });

  // ---------------------------------------------------------
  // 2. /stop → 90도에서 0도로 복귀 (밸브 잠금, 물 멈춤)
  // ---------------------------------------------------------
  server.on("/stop", []() {
    Serial.println("[수신] /stop 명령 수신!");

    for (int pos = currentPos; pos >= 0; pos -= 2) {
      myServo.write(pos);
      delay(15);
    }
    currentPos = 0;

    Serial.println("[모터 완료] 밸브 잠금! (0도)");
    server.send(200, "text/plain", "Servo Stopped: 0");
  });

  server.begin();
  Serial.println("HTTP Server Started. 대기 중...");
}

void loop() {
  server.handleClient();
}
