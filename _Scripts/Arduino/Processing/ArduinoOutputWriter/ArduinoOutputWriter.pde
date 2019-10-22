import processing.serial.*;
import java.text.SimpleDateFormat;  
import java.util.Date; 
import java.lang.StringBuilder;

boolean USE_SUBJECT_NAME = false;
String SUBJECT_NAME = "Russell";
String TRIAL_NUMBER = "1";

Serial mySerial;
Date date;
PrintWriter output;
boolean isReady;
StringBuilder buffer;
int char_counter;

void setup() {
   date = new Date();  
   
   int last_idx = Serial.list().length - 1;
   mySerial = new Serial( this, Serial.list()[last_idx], 115200 );
   
   println("Serial Port is: " + mySerial.toString());
   isReady = false;
   
   buffer = new StringBuilder();
   char_counter = 0;
}

void draw() {
    if (mySerial.available() > 0 ) {
         String value = mySerial.readString();
         if ( value != null ) {
              print(value);
              if (!isReady){
                if (value.indexOf("Ready to go!") != -1) {
                  isReady = true;
                  char curr_char = CreateNewFile();
                  println("Now writing for the letter " + curr_char);
                }
              }
              else{
                buffer.append(value);
              }
         }
    }
}

void keyReleased(){
  
  // Upon completing writing a letter, flush the buffer to the outputfile, and Next
  if(key == 'n' || key == 'N'){
    
    if (char_counter > 26){
      println("Confirmed exiting.");
      exit();
    }
    
    SaveBufferData();
    
    if (char_counter == 26){
      output.flush();
      output.close();
      
      println("Completed collecting letters. Press R to restart collecting. Press N again to confirm exit.");
      char_counter += 1;
    }
    
    if (char_counter < 26){
      char curr_char = CreateNewFile();
      println("Now writing for the letter " + curr_char);
    }
  }
  
  if (key == 'd' || key == 'D'){
    buffer.append("#\n");
    println("Marked the previous letter motion as invalid using #");
  }
  
  if (key == 'p' || key == 'P'){
    buffer.setLength(0);
    char curr_char = (char)('a' + char_counter - 1);
    println("Purified the buffer, recollect motion data for letter " + curr_char);
  }
  
  if (key == 'r' || key == 'R'){
    output.flush();
    output.close();
    char_counter = 0;
    date = new Date();  
    char curr_char = CreateNewFile();
    println("Started recollecting data.");
    println("Now writing for the letter " + curr_char);
  }
    
}

char CreateNewFile(){
  char curr_char = (char)('a' + char_counter);
  SimpleDateFormat formatter = new SimpleDateFormat("MM_dd_HH_mm");  
  String filename = "Raw_" + curr_char + "_" + formatter.format(date) + ".csv";
  
  if (USE_SUBJECT_NAME){
    filename = "Raw_" + curr_char + "_" + SUBJECT_NAME + "_" + TRIAL_NUMBER + ".csv";
  }
  
  if (output != null){
    output.flush();
    output.close();
  }
  
  output = createWriter( filename );
  char_counter += 1;
  
  return curr_char;
}

void SaveBufferData(){
  output.print(buffer.toString());
  output.flush();
  buffer.setLength(0);
}
