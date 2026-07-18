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

#define I2S_NUM           (i2s_port_t)0
#define SAMPLE_RATE       22050
#define RECORD_SECONDS    4
#define I2S_BCK_PIN       26
#define I2S_WS_PIN        25
#define I2S_DATA_IN_PIN   33
#define I2S_DATA_OUT_PIN  27
#define STATUS_LED_PIN    2

// OLED
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Forward declarations
bool initSPIFFS();
void initI2SRecorder();
void stopI2S();
bool recordWavToBuffer(std::vector<uint8_t> &outWav);
bool postWavAndHandleResponse(const std::vector<uint8_t> &wavData);
void playResponse(const char *emotion);

void setup() {
  Serial.begin(115200);
  delay(100);

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  // Init display
  // Initialize I2C for 4-pin OLED (SDA=GPIO21, SCL=GPIO22)
  Wire.begin(21, 22);
  Wire.setClock(400000);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("SSD1306 allocation failed");
  } else {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("Speech Emotion System");
    display.display();
  }

  // SPIFFS for playback files
  if (!initSPIFFS()) {
    Serial.println("SPIFFS init failed");
  }

  // WiFiManager: creates AP portal if needed
  WiFiManager wm;
  wm.setDebugOutput(true);
  display.setCursor(0, 16);
  display.println("Starting WiFi...");
  display.display();

  if (!wm.autoConnect(AP_SSID, AP_PASSWORD)) {
    Serial.println("Failed to connect and hit timeout");
    // Stay in AP mode
  }

  Serial.print("Connected, IP: ");
  Serial.println(WiFi.localIP());

  digitalWrite(STATUS_LED_PIN, HIGH);
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("WiFi Connected");
  display.println(WiFi.localIP().toString());
  display.display();

  initI2SRecorder();
}

void loop() {
  // Main loop: record, send, display, play response
  digitalWrite(STATUS_LED_PIN, LOW);
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Listening...");
  display.display();

  std::vector<uint8_t> wav;
  if (!recordWavToBuffer(wav)) {
    Serial.println("Recording failed");
    delay(1000);
    return;
  }

  digitalWrite(STATUS_LED_PIN, HIGH);
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Processing...");
  display.display();

  if (!postWavAndHandleResponse(wav)) {
    Serial.println("POST failed");
  }

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
  // Record PCM data from I2S into a buffer and prepend WAV header
  size_t bytes_to_read = SAMPLE_RATE * RECORD_SECONDS * 4; // 32-bit samples
  std::vector<int32_t> samples(bytes_to_read / 4);

  size_t bytes_read = 0;
  size_t offset = 0;
  while (offset < samples.size()) {
    size_t to_read = min((size_t)1024, samples.size() - offset) * 4;
    size_t ret = 0;
    esp_err_t res = i2s_read(I2S_NUM, (void *)&samples[offset], to_read, &ret, pdMS_TO_TICKS(2000));
    if (res != ESP_OK) {
      Serial.printf("i2s_read failed: %d\n", res);
      return false;
    }
    offset += ret / 4;
  }

  // Convert 32-bit samples to 16-bit for WAV
  std::vector<int16_t> pcm16;
  pcm16.reserve(samples.size());
  for (size_t i = 0; i < samples.size(); i++) {
    int32_t v = samples[i] >> 16; // simple downshift
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    pcm16.push_back((int16_t)v);
  }

  // Build WAV header
  uint32_t data_bytes = pcm16.size() * sizeof(int16_t);
  uint32_t riff_chunk_size = 36 + data_bytes;
  outWav.clear();
  // RIFF header
  outWav.insert(outWav.end(), {'R','I','F','F'});
  outWav.push_back((uint8_t)(riff_chunk_size & 0xFF));
  outWav.push_back((uint8_t)((riff_chunk_size >> 8) & 0xFF));
  outWav.push_back((uint8_t)((riff_chunk_size >> 16) & 0xFF));
  outWav.push_back((uint8_t)((riff_chunk_size >> 24) & 0xFF));
  outWav.insert(outWav.end(), {'W','A','V','E','f','m','t',' '});
  uint32_t subchunk1_size = 16;
  uint16_t audio_format = 1; // PCM
  uint16_t num_channels = 1;
  uint32_t byte_rate = SAMPLE_RATE * num_channels * 16/8;
  uint16_t block_align = num_channels * 16/8;
  uint16_t bits_per_sample = 16;
  // subchunk1
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
  // data subchunk header
  outWav.insert(outWav.end(), {'d','a','t','a'});
  outWav.push_back((uint8_t)(data_bytes & 0xFF));
  outWav.push_back((uint8_t)((data_bytes >> 8) & 0xFF));
  outWav.push_back((uint8_t)((data_bytes >> 16) & 0xFF));
  outWav.push_back((uint8_t)((data_bytes >> 24) & 0xFF));

  // Append PCM data little-endian
  for (size_t i = 0; i < pcm16.size(); i++) {
    int16_t v = pcm16[i];
    outWav.push_back((uint8_t)(v & 0xFF));
    outWav.push_back((uint8_t)((v >> 8) & 0xFF));
  }

  return true;
}

bool postWavAndHandleResponse(const std::vector<uint8_t> &wavData) {
  if (WiFi.status() != WL_CONNECTED) return false;

  HTTPClient http;
  String url = SERVER_URL;
  http.begin(url);

  String boundary = "----ESP32Boundary";
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  // Build multipart body in memory (careful with memory usage)
  String head = "--" + boundary + "\r\n";
  head += "Content-Disposition: form-data; name=\"audio\"; filename=\"recording.wav\"\r\n";
  head += "Content-Type: audio/wav\r\n\r\n";

  String tail = "\r\n--" + boundary + "--\r\n";

  // HTTPClient sends the request when sendRequest() is called. Build the full
  // multipart body first instead of writing to its response stream afterward.
  std::vector<uint8_t> body;
  body.reserve(head.length() + wavData.size() + tail.length());
  body.insert(body.end(), head.begin(), head.end());
  body.insert(body.end(), wavData.begin(), wavData.end());
  body.insert(body.end(), tail.begin(), tail.end());

  int code = http.sendRequest("POST", body.data(), body.size());
  String response = code > 0 ? http.getString() : "";
  http.end();

  if (code != 200) {
    Serial.printf("Server returned code: %d\n", code);
    return false;
  }

  // Parse JSON
  JsonDocument doc;
  auto err = deserializeJson(doc, response);
  if (err) {
    Serial.println("Failed to parse JSON");
    return false;
  }

  const char *emotion = doc["emotion"] | "unknown";
  float confidence = doc["confidence"] | 0.0;

  // Update display
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Emotion Detected:");
  display.println(emotion);
  display.printf("Confidence: %.2f\n", confidence);
  display.display();

  playResponse(emotion);

  return true;
}

void playResponse(const char *emotion) {
  // Map emotion to a WAV filename in SPIFFS
  String fname = "/responses/";
  fname += emotion;
  fname += ".wav";

  if (!SPIFFS.exists(fname)) {
    Serial.printf("Response file not found: %s\n", fname.c_str());
    return;
  }

  File f = SPIFFS.open(fname, FILE_READ);
  if (!f) return;

  // Skip WAV header (assumes 44 bytes)
  f.seek(44);

  // Start I2S for playback (reuse pins for output)
  // Configure I2S for output
  i2s_config_t i2s_config_out = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 4,
    .dma_buf_len = 512,
    .use_apll = false
  };

  i2s_pin_config_t pin_config_out = {
    .bck_io_num = I2S_BCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_DATA_OUT_PIN,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  i2s_driver_install(I2S_NUM, &i2s_config_out, 0, NULL);
  i2s_set_pin(I2S_NUM, &pin_config_out);

  const size_t chunkSize = 512;
  uint8_t buf[chunkSize];
  while (f.available()) {
    size_t toRead = f.read(buf, chunkSize);
    size_t written = 0;
    i2s_write(I2S_NUM, buf, toRead, &written, portMAX_DELAY);
  }

  f.close();
  i2s_driver_uninstall(I2S_NUM);
}


