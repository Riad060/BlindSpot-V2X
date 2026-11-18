
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// ESP-NOW channel (must match Traffic Node)
#define ESPNOW_CHANNEL 1

// Built-in LED (common on ESP32 DevKit boards)
#define LED_PIN 2

bool hazardDetected = false;

// Called when a wireless message arrives
void onDataRecv(const esp_now_recv_info_t *info,
                const uint8_t *data,
                int len) {

  char message[32];

  if (len <= 0 || len >= sizeof(message)) {
    return;
  }

  memcpy(message, data, len);
  message[len] = '\0';

  Serial.print("Received: ");
  Serial.println(message);

  if (strcmp(message, "HAZARD") == 0) {
    hazardDetected = true;
    digitalWrite(LED_PIN, HIGH);
    Serial.println("WARNING: Hazard detected!");
  }
  else if (strcmp(message, "CLEAR") == 0) {
    hazardDetected = false;
    digitalWrite(LED_PIN, LOW);
    Serial.println("SAFE: No hazard detected");
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);

  Serial.print("Vehicle ESP32 MAC: ");
  uint8_t mac[6];
esp_wifi_get_mac(WIFI_IF_STA, mac);

Serial.printf("Vehicle ESP32 MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
              mac[0], mac[1], mac[2],
              mac[3], mac[4], mac[5]);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW initialization failed!");
    return;
  }

  esp_now_register_recv_cb(onDataRecv);

  Serial.println("BlindSpot V2X - Vehicle Node Ready");
}

void loop() {
  delay(100);
}
