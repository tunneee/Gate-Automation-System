#include <Servo.h>
#include <SPI.h>
#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

Servo myservo; 
String inByte;
int pos;
int led = 13;
const int trig = 8; // chan trig
const int echo = 7; // chan echo



void setup() {

  lcd.begin(16, 2);
  lcd.print("Arduino!");
  myservo.attach(9);
  pinMode(led, OUTPUT);   
  Serial.begin(9600);
  pinMode(trig, OUTPUT);
  pinMode(echo, INPUT);
}

void distance() {
    unsigned long duration; // biến đo thời gian
    int distance;           // biến lưu khoảng cách
    
    /* Phát xung từ chân trig */
    digitalWrite(trig,0);   // tắt chân trig
    delayMicroseconds(2);
    digitalWrite(trig,1);   // phát xung từ chân trig
    delayMicroseconds(5);   // xung có độ dài 5 microSeconds
    digitalWrite(trig,0);   // tắt chân trig
    pinMode(led, OUTPUT);   
    /* Tính toán thời gian */
    // Đo độ rộng xung HIGH ở chân echo. 
    duration = pulseIn(echo,HIGH);  
    // Tính khoảng cách đến vật.
    distance = int(duration/2/29.412);
    lcd.setCursor(0, 1);
    lcd.print("Welcome back");
    /* In kết quả ra Serial Monitor */
    Serial.print(distance);
    Serial.println("cm");
    delay(100);
} 

void loop()
{    
  distance();
  if(Serial.available())  // if data available in serial port
    { 
    inByte = Serial.readStringUntil('\n'); // read data until newline
    pos = inByte.toInt();   // change datatype from string to integer 
    if (pos > 100) {
      digitalWrite(13,HIGH);
    }
    else {
      digitalWrite(13,LOW);
    }
    // if (pos > 180) {
    //   pos = 180;           
    // }       
    // if (pos < 90) {
    //   pos = 90;
    // }
    myservo.write(pos);     // move servo
    // Serial.print("Servo in position: ");  
    // Serial.println(inByte);
    }
}