#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebServer.h>
#include <ESP32Servo.h>
#include <driver/i2s.h>

// ===============================================
// [사용자 설정] 반드시 본인의 환경에 맞게 수정하세요!
// ===============================================
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// 파이썬 코드를 실행할 PC의 로컬 IP 주소
const char* pc_ip = "192.168.0.206"; // <-- ★ 파이썬 PC의 IP로 변경 필수
const int udp_port = 5005;

// I2S 마이크 연결 핀 (INMP441 권장 핀 배열)
#define I2S_WS  15 // L/R Clock (Word Select)
#define I2S_SD  2  // Data Out (DOUT)
#define I2S_SCK 4  // Bit Clock (BCLK)

#define I2S_PORT I2S_NUM_0
// ===============================================

WebServer server(80);
const int servoPin = 13;
Servo myServo;

int currentPos = 0;
WiFiUDP udp;

void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 512,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_set_clk(I2S_PORT, 16000, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
}

void setup() {
  Serial.begin(115200);

  // 서보모터 설정
  ESP32PWM::allocateTimer(0);
  myServo.setPeriodHertz(50);
  myServo.attach(servoPin, 500, 2400);
  myServo.write(0);
  currentPos = 0;

  // 와이파이 연결
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  // I2S 마이크 초기화
  setupI2S();
  Serial.println("I2S Microphone Initialized.");

  // 모터 제어 라우트 설정
  server.on("/open", []() {
    Serial.println("[명령] /open 수신");
    for (int pos = currentPos; pos <= 90; pos += 2) {
      myServo.write(pos);
      delay(15);
    }
    currentPos = 90;
    server.send(200, "text/plain", "Servo Opened: 90");
  });

  server.on("/stop", []() {
    Serial.println("[명령] /stop 수신");
    for (int pos = currentPos; pos >= 0; pos -= 2) {
      myServo.write(pos);
      delay(15);
    }
    currentPos = 0;
    server.send(200, "text/plain", "Servo Stopped: 0");
  });

  server.on("/ping", []() {
    server.send(200, "text/plain", "pong");
  });

  server.begin();
  Serial.println("HTTP Server Started.");
}

void loop() {
  server.handleClient();

  // I2S 마이크에서 32비트 오디오 데이터 읽기
  size_t bytesIn = 0;
  // INMP441은 32비트 단위로 데이터를 보내므로 32비트 버퍼 사용
  // 256 samples * 4 bytes = 1024 bytes
  int32_t rawBuffer[256]; 
  
  esp_err_t result = i2s_read(I2S_PORT, &rawBuffer, sizeof(rawBuffer), &bytesIn, portMAX_DELAY);
  
  // 성공적으로 읽었으면 16비트로 가공하여 UDP로 PC에 전송
  if (result == ESP_OK && bytesIn > 0) {
    int samples = bytesIn / sizeof(int32_t);
    int16_t sBuffer[256];
    
    long sum = 0;
    for (int i = 0; i < samples; i++) {
      // INMP441은 24비트 데이터를 32비트 그릇의 상위 24비트(MSB)에 채워 보냅니다.
      // 16비트 오디오 데이터로 스케일링하기 위해 우측으로 16비트 산술 쉬프트(>> 16) 후 캐스팅합니다.
      sBuffer[i] = (int16_t)(rawBuffer[i] >> 16);
      sum += abs(sBuffer[i]);
    }
    
    // PC로 16비트 PCM 전송 (256샘플 = 512바이트)
    udp.beginPacket(pc_ip, udp_port);
    udp.write((uint8_t*)sBuffer, samples * sizeof(int16_t));
    udp.endPacket();

    // [테스트용] PC 연결 없이 시리얼 모니터로 마이크 작동 확인
    static unsigned long lastPrint = 0;
    if (millis() - lastPrint > 100) {
      Serial.print("Raw: ");
      Serial.print(rawBuffer[0]);
      Serial.print(" | Shifted: ");
      Serial.println(sBuffer[0]);
      lastPrint = millis();
    }
  }
}
