
#include <Arduino.h>

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("BlindSpot V2X - Traffic Node Ready");
}

void loop() {
  if (Serial.available() > 0) {
    String message = Serial.readStringUntil('\n');
    message.trim();

    if (message == "HAZARD") {
      Serial.println("WARNING: Hazard detected!");
    }
    else if (message == "CLEAR") {
      Serial.println("SAFE: No hazard detected");
    }
    else if (message.length() > 0) {
      Serial.print("Unknown message: ");
      Serial.println(message);
    }
  }
}
