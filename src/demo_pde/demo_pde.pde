import processing.serial.*;
import java.text.SimpleDateFormat;  
import java.util.Date; 
import java.lang.StringBuilder;

boolean USE_SUBJECT_NAME = true;
String FOLDER_NAME = "../demo_data";

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
                  println("\nNow please collect Calibration data. Press N when done.");
                }
              }
              else{
                buffer.append(value);
              }
         }
    }
}


// Key Board Control Explained:
// N: "Next" Collect the next data sequence (handwriting data for one letter)
// P: "Purify" You want to clear all data about this current letter before writing to file, e.g. because you suddenly notice that you should write letter 'b' but not 'a'
// S: "Stop" Stop execution and exit
void keyReleased(){
  
  // Next:
  if(key == 'n' || key == 'N'){
    
    SaveBufferToFile();
    println("\nSaved data to file. Please now write the " + char_counter + "th character. Press N to save.");
    
  }
  
  // Purify:
  if (key == 'p' || key == 'P'){
    buffer.setLength(0);
    println("\nPurified the buffer, You may now rewrite the letter.");
  }
  
  // Stop:
  if (key == 's' || key == 'S'){
    if (output != null){
      output.flush();
      output.close();
    }
    println("\nConfirmed exit. Unsaved data discarded.");
    exit();
  }
}

// Save current things in the buffer to the output.csv
boolean SaveBufferToFile(){
    
  String directoryName = FOLDER_NAME;
  String filename = "a.csv";
  
  if (char_counter == 0){
    filename = "calibration.csv";
  }
  
  File directory = new File(directoryName);
  if (! directory.exists()){
    directory.mkdir();
  }
  
  filename = directoryName + "/" + filename;
  
  output = createWriter( filename );
  output.print(buffer.toString());
  output.flush();
  output.close();
  buffer.setLength(0);
  
  char_counter += 1;
  
  return true; 
}

//void SaveBufferData(){
//  output.print(buffer.toString());
//  output.flush();
//  buffer.setLength(0);
//}
