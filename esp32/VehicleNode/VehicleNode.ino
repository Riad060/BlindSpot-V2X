
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// ESP-NOW channel (must match Traffic Node)
#define ESPNOW_CHANNEL 1

// External LEDs
#define RED_LED 23
#define GREEN_LED 17  // Yellow LED for CLEAR

bool hazardDetected = false;
bool messageReceived = false;

// Runs automatically when an ESP-NOW message arrives
void onDataRecv(
    const esp_now_recv_info_t *info,
    const uint8_t *data,
    int len
) {
    // Convert received bytes into a String
    String message = "";

    for (int i = 0; i < len; i++) {
        if (data[i] == '\0') {
            break;
        }

        message += (char)data[i];
    }

    message.trim();

    Serial.print("Received: ");
    Serial.println(message);

    if (message == "HAZARD") {
        hazardDetected = true;
        messageReceived = true;

        digitalWrite(RED_LED, HIGH);
        digitalWrite(GREEN_LED, LOW);

        Serial.println("WARNING: Hazard detected!");
    }
    else if (message == "CLEAR") {
        hazardDetected = false;
        messageReceived = true;

        digitalWrite(RED_LED, LOW);
        digitalWrite(GREEN_LED, HIGH);

        Serial.println("SAFE: No hazard detected");
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    // Configure LED pins
    pinMode(RED_LED, OUTPUT);
    pinMode(GREEN_LED, OUTPUT);

    // Both LEDs OFF until a message arrives
    digitalWrite(RED_LED, LOW);
    digitalWrite(GREEN_LED, LOW);

    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    esp_wifi_set_channel(
        ESPNOW_CHANNEL,
        WIFI_SECOND_CHAN_NONE
    );

    Serial.print("Vehicle ESP32 MAC: ");
    Serial.println(WiFi.macAddress());

    // Initialise ESP-NOW
    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW initialization failed");
        return;
    }

    // Register receive callback
    esp_now_register_recv_cb(onDataRecv);

    Serial.println("BlindSpot V2X - Vehicle Node Ready");
    Serial.println("Waiting for HAZARD or CLEAR...");
}

void loop() {
    delay(100);
}
