
#include <LiquidCrystal.h>

int BUTTON = 11;
LiquidCrystal lcd(9, 2, 4, 5, 6, 7);


void setup() {
  lcd.begin(16, 2);

  pinMode(BUTTON, INPUT);
  Serial.begin(115200);
  lcd.print("COGS118A   SP17");

  lcd.setCursor(1, 1);
  lcd.print("Final  Project");
  delay(1500);
  lcd.clear();
  lcd.print("Record Condition:");
  lcd.setCursor(1, 1);

}

void loop() {
  if (digitalRead(BUTTON) == HIGH)
  {
    lcd.setCursor(2, 1);
    lcd.print("IS_RECORDING");
    // Serial.println("button is pressed");
  }
  else
  {
    lcd.setCursor(2, 1);
    lcd.print("NO_RECORDING");
  }
}
