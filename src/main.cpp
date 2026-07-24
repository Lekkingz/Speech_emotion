/*
  ESP32 Speech Emotion Client

  - Connects to WiFi using WiFiManager (captive portal AP if no saved credentials)
  - Records audio from INMP441 (I2S) for a fixed duration
  - Encodes PCM into a WAV buffer
  - Sends WAV via HTTP POST to Flask server
  - Displays status and results on SSD1306
  - Plays an audio response file from SPIFFS via I2S to MAX98357A

  This file is a production-quality scaffold with robust error handling.
*/

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <WiFiManager.h>
#include <ArduinoJson.h>
#include <SPIFFS.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Wire.h>

extern "C" {
  #include "driver/i2s.h"
}

// Configurable build flags (set in platformio.ini)
#ifndef SERVER_URL
#define SERVER_URL "http://192.168.1.100:5000/predict"
#endif

#ifndef AP_SSID
#define AP_SSID "ESP32_SpeechEmotion"
#endif

#ifndef AP_PASSWORD
#define AP_PASSWORD ""
#endif

#ifndef WIFI_CONNECT_TIMEOUT_MS
#define WIFI_CONNECT_TIMEOUT_MS 30000
#endif

#ifndef WIFI_PORTAL_TIMEOUT_S
#define WIFI_PORTAL_TIMEOUT_S 180
#endif

#define I2S_NUM           (i2s_port_t)0
#define SAMPLE_RATE       16000
#define RECORD_SECONDS    2
#define I2S_BCK_PIN       26
#define I2S_WS_PIN        25
#define I2S_DATA_IN_PIN   33
#define I2S_DATA_OUT_PIN  27
#define STATUS_LED_PIN    2
#define STATUS_LED_ACTIVE_LOW false
#define RECORD_BUTTON_PIN  4
#define REPLAY_BUTTON_PIN  13
#define WIFI_RESET_BUTTON_PIN 14
#define PLAY_RESPONSE_AUDIO true
#define VOICE_AVG_THRESHOLD 250
#define VOICE_PEAK_THRESHOLD 1500

#ifndef OLED_SDA_PIN
#define OLED_SDA_PIN 21
#endif

#ifndef OLED_SCL_PIN
#define OLED_SCL_PIN 22
#endif

#ifndef OLED_I2C_ADDRESS
#define OLED_I2C_ADDRESS 0x3C
#endif

// OLED (0.96 inch display: 128x32)
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
String lastEmotion = "";

// Forward declarations
bool initSPIFFS();
bool connectToWiFi();
void initI2SRecorder();
void stopI2S();
bool recordWavToBuffer(std::vector<uint8_t> &outWav);
bool postWavAndHandleResponse(const std::vector<uint8_t> &wavData);
void playResponse(const char *emotion);
void playTone(uint16_t frequency, uint16_t durationMs);
void setStatusLed(bool on);
void blinkStatusLed(uint8_t times);


void setStatusLed(bool on) {
  digitalWrite(STATUS_LED_PIN, STATUS_LED_ACTIVE_LOW ? !on : on);
}

void blinkStatusLed(uint8_t times) {
  for (uint8_t i = 0; i < times; i++) {
    setStatusLed(true);
    delay(150);
    setStatusLed(false);
    delay(150);
  }
}

bool connectToWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(true);
  WiFi.disconnect(false);
  delay(100);

  String configuredSsid = WIFI_SSID;
  String configuredPassword = WIFI_PASSWORD;

  if (configuredSsid.length() > 0) {
    Serial.printf("Trying configured WiFi SSID: %s\n", configuredSsid.c_str());
    if (configuredPassword.length() > 0) {
      WiFi.begin(configuredSsid.c_str(), configuredPassword.c_str());
    } else {
      WiFi.begin(configuredSsid.c_str());
    }
  } else {
    Serial.println("No configured WiFi SSID found; starting WiFiManager portal.");
  }

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < WIFI_CONNECT_TIMEOUT_MS) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi connected: ");
    Serial.println(WiFi.localIP());
    return true;
  }

  Serial.println("WiFi connect timed out. Starting configuration portal...");
  WiFiManager wm;
  wm.setDebugOutput(true);
  wm.setConfigPortalTimeout(WIFI_PORTAL_TIMEOUT_S);
  wm.setConnectTimeout(20);
  wm.setHostname("esp32-speech-emotion");

  bool connected = wm.autoConnect(AP_SSID, AP_PASSWORD);
  if (connected) {
    Serial.print("WiFi connected through portal: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi still unavailable. Continuing in AP mode.");
  }

  return connected;
}

void setup() {
  Serial.begin(115200);
  delay(100);

  pinMode(STATUS_LED_PIN, OUTPUT);
  pinMode(RECORD_BUTTON_PIN, INPUT_PULLUP);
  pinMode(REPLAY_BUTTON_PIN, INPUT_PULLUP);
  pinMode(WIFI_RESET_BUTTON_PIN, INPUT_PULLUP);
  setStatusLed(false);
  blinkStatusLed(3);

  // Init display
  // 4-pin OLED wiring: GND -> GND, VCC -> 3.3V, SDA -> GPIO21, SCL -> GPIO22
  Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN);
  Wire.setClock(400000);
  delay(100);

  bool displayReady = false;
  uint8_t displayAddress = OLED_I2C_ADDRESS;
  if (display.begin(SSD1306_SWITCHCAPVCC, displayAddress)) {
    displayReady = true;
  } else {
    Serial.println("OLED init failed at 0x3C, trying 0x3D");
    if (display.begin(SSD1306_SWITCHCAPVCC, 0x3D)) {
      displayAddress = 0x3D;
      displayReady = true;
    }
  }

  if (displayReady) {
    Serial.printf("OLED initialized on address 0x%02X\n", displayAddress);
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("Speech Emotion System");
    display.display();
  } else {
    Serial.println("SSD1306 not found. Check wiring: GND/VCC/SDA/SCL and I2C address.");
  }

  // SPIFFS for playback files
  if (!initSPIFFS()) {
    Serial.println("SPIFFS init failed");
  }

  display.setCursor(0, 16);
  display.println("Starting WiFi...");
  display.display();

  bool wifiConnected = connectToWiFi();
  if (!wifiConnected) {
    Serial.println("WiFi not connected; AP mode is active.");
  }

  Serial.print("Connected, IP: ");
  Serial.println(WiFi.localIP());

  setStatusLed(wifiConnected);
  display.clearDisplay();
  display.setCursor(0, 0);
  if (wifiConnected) {
    display.println("WiFi Connected");
    display.println(WiFi.localIP().toString());
  } else {
    display.println("WiFi setup mode");
    display.println("Connect to AP");
    display.println(AP_SSID);
  }
  display.display();

  initI2SRecorder();
}

void loop() {
  static bool promptShown = false;

  if (!promptShown) {
    setStatusLed(true);
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("Ready");
    display.println("B1: Record");
    display.println("B2: Replay");
    display.println("B3 hold: WiFi reset");
    display.display();
    Serial.println("Ready. B1 GPIO4=record, B2 GPIO13=replay, hold B3 GPIO14=reset WiFi.");
    Serial.println("Serial commands: r=record, p=replay, w=reset WiFi.");
    promptShown = true;
  }

  bool recordTrigger = false;
  bool replayTrigger = false;
  bool wifiResetTrigger = false;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'r' || c == 'R') recordTrigger = true;
    if (c == 'p' || c == 'P') replayTrigger = true;
    if (c == 'w' || c == 'W') wifiResetTrigger = true;
  }

  if (digitalRead(WIFI_RESET_BUTTON_PIN) == LOW) {
    Serial.println("Hold WiFi reset button for 3 seconds...");
    unsigned long start = millis();
    while (digitalRead(WIFI_RESET_BUTTON_PIN) == LOW) {
      if (millis() - start >= 3000) {
        wifiResetTrigger = true;
        break;
      }
      delay(20);
    }
  }

  if (wifiResetTrigger) {
    Serial.println("Resetting WiFi settings...");
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("Resetting WiFi");
    display.display();
    WiFiManager wm;
    wm.resetSettings();
    delay(1000);
    ESP.restart();
  }

  replayTrigger = replayTrigger || (digitalRead(REPLAY_BUTTON_PIN) == LOW);
  if (replayTrigger) {
    while (digitalRead(REPLAY_BUTTON_PIN) == LOW) delay(10);
    if (lastEmotion.length() == 0) {
      Serial.println("No previous emotion to replay.");
    } else {
      Serial.print("Replaying response for: ");
      Serial.println(lastEmotion);
      playResponse(lastEmotion.c_str());
    }
    promptShown = false;
    delay(300);
    return;
  }

  recordTrigger = recordTrigger || (digitalRead(RECORD_BUTTON_PIN) == LOW);
  if (!recordTrigger) {
    delay(50);
    return;
  }

  delay(50);
  while (digitalRead(RECORD_BUTTON_PIN) == LOW) delay(10);

  Serial.println("Get ready. Recording starts in 3 seconds...");
  setStatusLed(true);
  for (int secondsLeft = 3; secondsLeft > 0; secondsLeft--) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("Get ready");
    display.printf("Speak in %d...\n", secondsLeft);
    display.display();
    Serial.printf("Speak in %d...\n", secondsLeft);
    delay(1000);
  }

  Serial.println("Recording started. Speak now...");
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Speak now");
  display.println("Recording...");
  display.display();

  std::vector<uint8_t> wav;
  if (!recordWavToBuffer(wav)) {
    Serial.println("Recording failed");
    setStatusLed(false);
    promptShown = false;
    delay(1000);
    return;
  }

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Processing...");
  display.display();

  if (!postWavAndHandleResponse(wav)) {
    Serial.println("POST failed");
  }

  setStatusLed(false);
  promptShown = false;
  delay(500);
}

bool initSPIFFS() {
  if (!SPIFFS.begin(true)) {
    return false;
  }
  return true;
}

void initI2SRecorder() {
  // Configure I2S for PDM/PCM microphone (INMP441 acts as I2S slave)
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 4,
    .dma_buf_len = 512,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_BCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_DATA_IN_PIN
  };

  i2s_driver_install(I2S_NUM, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM, &pin_config);
}

void stopI2S() {
  i2s_driver_uninstall(I2S_NUM);
}

bool recordWavToBuffer(std::vector<uint8_t> &outWav) {
  const size_t totalSamples = SAMPLE_RATE * RECORD_SECONDS;
  const uint32_t data_bytes = totalSamples * sizeof(int16_t);
  const uint32_t riff_chunk_size = 36 + data_bytes;

  outWav.clear();
  size_t requiredBytes = 44 + data_bytes;
  if (ESP.getFreeHeap() < requiredBytes + 20000) {
    Serial.printf("Not enough heap for WAV buffer. Need %u, free %u\n", requiredBytes, ESP.getFreeHeap());
    return false;
  }
  outWav.reserve(requiredBytes);

  Serial.printf("Free heap before recording: %u\n", ESP.getFreeHeap());

  outWav.insert(outWav.end(), {'R','I','F','F'});
  outWav.push_back((uint8_t)(riff_chunk_size & 0xFF));
  outWav.push_back((uint8_t)((riff_chunk_size >> 8) & 0xFF));
  outWav.push_back((uint8_t)((riff_chunk_size >> 16) & 0xFF));
  outWav.push_back((uint8_t)((riff_chunk_size >> 24) & 0xFF));
  outWav.insert(outWav.end(), {'W','A','V','E','f','m','t',' '});
  uint32_t subchunk1_size = 16;
  uint16_t audio_format = 1;
  uint16_t num_channels = 1;
  uint32_t byte_rate = SAMPLE_RATE * num_channels * 16 / 8;
  uint16_t block_align = num_channels * 16 / 8;
  uint16_t bits_per_sample = 16;

  outWav.push_back((uint8_t)(subchunk1_size & 0xFF));
  outWav.push_back((uint8_t)((subchunk1_size >> 8) & 0xFF));
  outWav.push_back((uint8_t)((subchunk1_size >> 16) & 0xFF));
  outWav.push_back((uint8_t)((subchunk1_size >> 24) & 0xFF));
  outWav.push_back((uint8_t)(audio_format & 0xFF));
  outWav.push_back((uint8_t)((audio_format >> 8) & 0xFF));
  outWav.push_back((uint8_t)(num_channels & 0xFF));
  outWav.push_back((uint8_t)((num_channels >> 8) & 0xFF));
  outWav.push_back((uint8_t)(SAMPLE_RATE & 0xFF));
  outWav.push_back((uint8_t)((SAMPLE_RATE >> 8) & 0xFF));
  outWav.push_back((uint8_t)((SAMPLE_RATE >> 16) & 0xFF));
  outWav.push_back((uint8_t)((SAMPLE_RATE >> 24) & 0xFF));
  outWav.push_back((uint8_t)(byte_rate & 0xFF));
  outWav.push_back((uint8_t)((byte_rate >> 8) & 0xFF));
  outWav.push_back((uint8_t)((byte_rate >> 16) & 0xFF));
  outWav.push_back((uint8_t)((byte_rate >> 24) & 0xFF));
  outWav.push_back((uint8_t)(block_align & 0xFF));
  outWav.push_back((uint8_t)((block_align >> 8) & 0xFF));
  outWav.push_back((uint8_t)(bits_per_sample & 0xFF));
  outWav.push_back((uint8_t)((bits_per_sample >> 8) & 0xFF));
  outWav.insert(outWav.end(), {'d','a','t','a'});
  outWav.push_back((uint8_t)(data_bytes & 0xFF));
  outWav.push_back((uint8_t)((data_bytes >> 8) & 0xFF));
  outWav.push_back((uint8_t)((data_bytes >> 16) & 0xFF));
  outWav.push_back((uint8_t)((data_bytes >> 24) & 0xFF));

  const size_t chunkSamples = 256;
  int32_t raw[chunkSamples];
  size_t samplesRead = 0;
  uint64_t absSum = 0;
  int16_t peakAbs = 0;

  while (samplesRead < totalSamples) {
    size_t wantedSamples = min(chunkSamples, totalSamples - samplesRead);
    size_t bytesRead = 0;
    esp_err_t res = i2s_read(I2S_NUM, raw, wantedSamples * sizeof(int32_t), &bytesRead, pdMS_TO_TICKS(2000));
    if (res != ESP_OK) {
      Serial.printf("i2s_read failed: %d\n", res);
      return false;
    }

    size_t gotSamples = bytesRead / sizeof(int32_t);
    if (gotSamples == 0) {
      Serial.println("i2s_read returned no samples");
      return false;
    }

    for (size_t i = 0; i < gotSamples; i++) {
      int32_t v = raw[i] >> 16;
      if (v > 32767) v = 32767;
      if (v < -32768) v = -32768;
      int16_t sample = (int16_t)v;
      int16_t absSample = sample < 0 ? -sample : sample;
      absSum += absSample;
      if (absSample > peakAbs) peakAbs = absSample;
      outWav.push_back((uint8_t)(sample & 0xFF));
      outWav.push_back((uint8_t)((sample >> 8) & 0xFF));
    }
    samplesRead += gotSamples;
  }

  uint32_t avgAbs = samplesRead > 0 ? absSum / samplesRead : 0;
  Serial.printf("WAV bytes: %u, avg amplitude: %u, peak: %d, free heap after recording: %u\n", outWav.size(), avgAbs, peakAbs, ESP.getFreeHeap());
  if (avgAbs < VOICE_AVG_THRESHOLD && peakAbs < VOICE_PEAK_THRESHOLD) {
    Serial.println("No speech detected; prediction skipped.");
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("No speech heard");
    display.println("Try again");
    display.display();
    return false;
  }
  return true;
}

bool postWavAndHandleResponse(const std::vector<uint8_t> &wavData) {
  if (WiFi.status() != WL_CONNECTED) return false;

  String url = SERVER_URL;
  bool useHttps = url.startsWith("https://");
  if (useHttps) {
    url.remove(0, 8);
  } else if (url.startsWith("http://")) {
    url.remove(0, 7);
  } else {
    Serial.println("SERVER_URL must start with http:// or https://");
    return false;
  }
  int slashIndex = url.indexOf('/');
  String hostPort = slashIndex >= 0 ? url.substring(0, slashIndex) : url;
  String requestPath = slashIndex >= 0 ? url.substring(slashIndex) : "/";
  int port = useHttps ? 443 : 80;
  int colonIndex = hostPort.indexOf(':');
  String host = hostPort;
  if (colonIndex >= 0) {
    host = hostPort.substring(0, colonIndex);
    port = hostPort.substring(colonIndex + 1).toInt();
  }

  WiFiClient tcpClient;
  WiFiClientSecure secureClient;
  WiFiClient *client = nullptr;

  if (useHttps) {
    secureClient.setInsecure();
    secureClient.setTimeout(8000);
    if (!secureClient.connect(host.c_str(), port)) {
      Serial.printf("HTTPS connection failed: %s:%d\n", host.c_str(), port);
      return false;
    }
    client = &secureClient;
  } else {
    tcpClient.setTimeout(8000);
    if (!tcpClient.connect(host.c_str(), port)) {
      Serial.printf("HTTP connection failed: %s:%d\n", host.c_str(), port);
      return false;
    }
    client = &tcpClient;
  }

  Serial.printf("Posting WAV %u bytes to %s via %s, free heap before POST: %u\n", wavData.size(), host.c_str(), useHttps ? "HTTPS" : "HTTP", ESP.getFreeHeap());

  client->printf("POST %s HTTP/1.1\r\n", requestPath.c_str());
  client->printf("Host: %s:%d\r\n", host.c_str(), port);
  client->print("Content-Type: audio/wav\r\n");
  client->printf("Content-Length: %u\r\n", wavData.size());
  client->print("Connection: close\r\n\r\n");

  const size_t chunkSize = 1024;
  for (size_t offset = 0; offset < wavData.size(); offset += chunkSize) {
    size_t toWrite = min(chunkSize, wavData.size() - offset);
    size_t written = client->write(wavData.data() + offset, toWrite);
    if (written != toWrite) {
      Serial.printf("Short write: %u/%u\n", written, toWrite);
      client->stop();
      return false;
    }
  }
  client->flush();

  unsigned long responseStart = millis();
  while (!client->available() && millis() - responseStart < 8000) {
    delay(10);
  }

  if (!client->available()) {
    Serial.println("No HTTP response from server");
    client->stop();
    return false;
  }

  String statusLine = client->readStringUntil('\n');
  statusLine.trim();
  Serial.print("HTTP status line: ");
  Serial.println(statusLine);
  int firstSpace = statusLine.indexOf(' ');
  int secondSpace = statusLine.indexOf(' ', firstSpace + 1);
  int code = firstSpace >= 0 ? statusLine.substring(firstSpace + 1, secondSpace).toInt() : -1;

  int contentLength = -1;
  bool chunked = false;
  while (client->available()) {
    String line = client->readStringUntil('\n');
    line.trim();
    if (line.length() == 0) break;
    Serial.print("Header: ");
    Serial.println(line);
    String lower = line;
    lower.toLowerCase();
    if (lower.startsWith("content-length:")) {
      contentLength = line.substring(15).toInt();
    }
    if (lower.indexOf("transfer-encoding: chunked") >= 0) {
      chunked = true;
    }
  }

  String response;
  unsigned long bodyStart = millis();
  while (millis() - bodyStart < 8000) {
    while (client->available()) {
      response += (char)client->read();
      bodyStart = millis();
      if (contentLength > 0 && response.length() >= contentLength) break;
    }
    if (contentLength > 0 && response.length() >= contentLength) break;
    if (!client->connected() && !client->available()) break;
    delay(10);
  }
  client->stop();

  if (chunked) {
    int openBrace = response.indexOf('{');
    int closeBrace = response.lastIndexOf('}');
    if (openBrace >= 0 && closeBrace > openBrace) {
      response = response.substring(openBrace, closeBrace + 1);
    }
  }

  response.trim();
  Serial.printf("HTTP code: %d, free heap after POST: %u\n", code, ESP.getFreeHeap());
  Serial.print("Response: ");
  Serial.println(response);

  if (code != 200) {
    Serial.printf("Server returned code: %d\n", code);
    return false;
  }

  JsonDocument doc;
  auto err = deserializeJson(doc, response);
  if (err) {
    Serial.println("Failed to parse JSON");
    return false;
  }

  const char *emotion = doc["emotion"] | "unknown";
  float confidence = doc["confidence"] | 0.0;
  lastEmotion = String(emotion);

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Emotion Detected:");
  display.println(emotion);
  display.printf("Confidence: %.2f\n", confidence);
  display.display();

  if (PLAY_RESPONSE_AUDIO) {
    playResponse(emotion);
  } else {
    Serial.println("Speaker playback disabled. Use OLED/serial result for now.");
  }
  return true;
}

void playTone(uint16_t frequency, uint16_t durationMs) {
  const uint16_t sampleRate = 16000;
  const uint16_t amplitude = 5000;
  const size_t frameCount = 128;
  int16_t frames[frameCount * 2];
  uint32_t totalFrames = ((uint32_t)sampleRate * durationMs) / 1000;
  uint32_t phase = 0;
  uint32_t phaseStep = ((uint32_t)frequency << 16) / sampleRate;

  while (totalFrames > 0) {
    size_t n = min((uint32_t)frameCount, totalFrames);
    for (size_t i = 0; i < n; i++) {
      int16_t sample = (phase & 0x8000) ? amplitude : -amplitude;
      frames[i * 2] = sample;
      frames[i * 2 + 1] = sample;
      phase += phaseStep;
    }
    size_t written = 0;
    esp_err_t writeResult = i2s_write(I2S_NUM, frames, n * 2 * sizeof(int16_t), &written, portMAX_DELAY);
    if (writeResult != ESP_OK) {
      Serial.printf("I2S beep write failed: %d\n", writeResult);
      break;
    }
    totalFrames -= n;
  }
}

void playResponse(const char *emotion) {
  stopI2S();
  delay(50);

  i2s_config_t i2s_config_out = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 4,
    .dma_buf_len = 256,
    .use_apll = false
  };

  i2s_pin_config_t pin_config_out = {
    .bck_io_num = I2S_BCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_DATA_OUT_PIN,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  esp_err_t installResult = i2s_driver_install(I2S_NUM, &i2s_config_out, 0, NULL);
  esp_err_t pinResult = i2s_set_pin(I2S_NUM, &pin_config_out);
  if (installResult != ESP_OK || pinResult != ESP_OK) {
    Serial.printf("I2S beep init failed: install=%d pin=%d\n", installResult, pinResult);
    i2s_driver_uninstall(I2S_NUM);
    initI2SRecorder();
    return;
  }

  String e = String(emotion);
  e.toLowerCase();
  Serial.print("Emotion beep: ");
  Serial.println(e);

  if (e == "neutral") {
    playTone(440, 180);
  } else if (e == "calm") {
    playTone(392, 140); delay(90); playTone(392, 140);
  } else if (e == "happy") {
    playTone(660, 120); delay(80); playTone(880, 180);
  } else if (e == "sad") {
    playTone(330, 250); delay(120); playTone(262, 300);
  } else if (e == "angry") {
    playTone(900, 80); delay(60); playTone(900, 80); delay(60); playTone(900, 160);
  } else if (e == "fearful") {
    playTone(760, 90); delay(70); playTone(520, 90); delay(70); playTone(760, 90);
  } else if (e == "disgust") {
    playTone(220, 120); delay(80); playTone(180, 220);
  } else if (e == "surprised") {
    playTone(520, 90); delay(60); playTone(700, 90); delay(60); playTone(950, 180);
  } else {
    playTone(500, 150);
  }

  delay(50);
  i2s_zero_dma_buffer(I2S_NUM);
  i2s_driver_uninstall(I2S_NUM);
  initI2SRecorder();
}



