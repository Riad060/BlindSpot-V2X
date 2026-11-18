
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// BlindSpot V2X - Traffic Node
// Raspberry Pi -> USB Serial -> Traffic ESP32
// Traffic ESP32 -> ESP-NOW -> Vehicle ESP32

#define ESPNOW_CHANNEL 1

// Vehicle ESP32 Wi-Fi MAC address
uint8_t vehicleMAC[] = {
  0x64, 0xB7, 0x08, 0x29, 0x2B, 0x7C
};

// ESP-NOW transmission callback
void onDataSent(const wifi_tx_info_t *info,
                esp_now_send_status_t status) {
  if (status == ESP_NOW_SEND_SUCCESS) {
    Serial.println("ESP-NOW: Transmission successful");
  } else {
    Serial.println("ESP-NOW: Transmission failed");
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  // Both ESP32 devices use channel 1
  esp_err_t channelResult = esp_wifi_set_channel(
    ESPNOW_CHANNEL,
    WIFI_SECOND_CHAN_NONE
  );

  if (channelResult != ESP_OK) {
    Serial.println("Failed to set Wi-Fi channel");
    return;
  }

  // Start ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW initialization failed");
    return;
  }

  esp_now_register_send_cb(onDataSent);

  // Register the Vehicle ESP32 as a peer
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, vehicleMAC, 6);
  peerInfo.channel = ESPNOW_CHANNEL;
  peerInfo.ifidx = WIFI_IF_STA;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add Vehicle Node");
    return;
  }

  Serial.println("BlindSpot V2X - Traffic Node Ready");
  Serial.println("Waiting for HAZARD or CLEAR...");
}

void loop() {
  if (Serial.available() > 0) {
    String message = Serial.readStringUntil('\n');
    message.trim();

    if (message == "HAZARD" || message == "CLEAR") {

      Serial.print("Pi message received: ");
      Serial.println(message);

      esp_err_t result = esp_now_send(
        vehicleMAC,
        (const uint8_t *)message.c_str(),
        message.length()
      );

      if (result == ESP_OK) {
        Serial.println("ESP-NOW: Message queued");
      } else {
        Serial.println("ESP-NOW: Send error");
      }
    }
    else if (message.length() > 0) {
      Serial.print("Unknown message: ");
      Serial.println(message);
    }
  }
}
