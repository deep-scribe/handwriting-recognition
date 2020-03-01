import processing.serial.*;
import java.text.SimpleDateFormat;  
import java.util.Date; 
import java.lang.StringBuilder;
import java.util.Arrays;

String[] testwords = new String[]{"hello", "world"};

String[] words_mini_easy_2 = new String[]{
"exams", "focus", "sauce", "coax", "same", "sex", "awe", "fem", "axe", "age"
};

String[] words_mini_hard_2 = new String[]{
"kanji", "flick", "beach", "cabin", "hack", "lack", "kami", "kid" , "ban", "bad"
};

String[] words_20 = new String[]{
"exams", "focus", "sauce", "coax", "same", "sex", "awe", "fem", "axe", "age", "kanji", "flick", "beach", "cabin", "hack", "lack", "kami", "kid" , "ban", "bad"
};



/* ------------------------------ */
/* Custom Field Start */
String DIR_NAME = "words_data";

String FOLDER_NAME = "Chen";
String[] word_list = words_20.clone();

boolean start_from_first_word = true;
String word_to_start_at = "NO_WORD_DEFINED";
//boolean start_from_first_word = false;
//String word_to_start_at = "fem";

boolean addCalibration = true;

/* Custom Field End */
/* ------------------------------ */

Serial mySerial;
Date date;
PFont f;
PrintWriter output;
boolean isReady;
StringBuilder buffer;
int next_word_index;
int curr_word_index;
int INDEX_LIMIT = word_list.length;

void setup() {
   // window size
   size(600,200);
   surface.setResizable(true);
  
   int last_idx = Serial.list().length - 1;
   mySerial = new Serial( this, Serial.list()[last_idx], 115200 );
   
   println("Serial Port is: " + mySerial.toString());
   isReady = false;
   
   buffer = new StringBuilder();
   
   if (start_from_first_word && addCalibration){
     // use "-1" if you wish to start with calibration, else enter corresponding letter's number
     next_word_index = -1;
   }
   else {
     if (start_from_first_word){
       next_word_index = 0;
     }
     else{
       int word_index = Arrays.asList(word_list).indexOf(word_to_start_at);
       print("word_index is: " + word_index);
       if (word_index == -1){
         print("WARNING: No starting word is defined. Start with first word by default. Check start_from_first_word.");
         next_word_index = 0;
       }
       else{
         next_word_index = word_index;
       }
     }
   }
   
   // For displaying text:
   f = createFont("Arial", 16, true);
   background(255);
   textFont(f,20);
   fill(0);
   text("Welcome to Data Collection! Now loading hardware...\n", 10, 100);
   textFont(f,14);
   text("\nNote: If taking too long. Try restart.", 10, 110);
}

void draw() {
    delay(2000);
    if (mySerial.available() > 0 ) {
         String value = mySerial.readString();
         if ( value != null ) {
              print(value);
              if (!isReady){
                if (value.indexOf("Ready to go!") != -1) {
                  isReady = true;
                  String filename = CreateNewFile();
                  println("Now word index is: " + curr_word_index + ". Writing for the file:  " + filename);
                }
              }
              else{
                buffer.append(value);
              }
         }
    }
}


// Key Board Control Explained:
// N: "Next" Collect the next word, or exit if all words are collected.
// D: "Delete" You just wrote a letter and released the arduino button, but you want to mark the previous data sequence as invalid.
// P: "Purify" You want to clear all data about this current letter before writing to file, e.g. because you suddenly notice that you should write letter 'b' but not 'a'
// R: "Recollect" You want to re-start from the very beginning, and overwrite everything you've collected so far in that folder.
void keyReleased(){
  
  // Upon completing writing a letter, flush the buffer to the outputfile, and Next
  if(key == 'n' || key == 'N'){
    
    if (next_word_index > INDEX_LIMIT){
      background(255);
      fill(0);
      textFont(f,20);
      String message = "Exit Confirmed.";
      text(message, 10, 100);
      println("Confirmed exiting.");
      exit();
    }
    
    SaveBufferData();
    
    if (next_word_index == INDEX_LIMIT){
      output.flush();
      output.close();
      
      println("Completed collecting data. Press R to restart collecting. Press N again to confirm exit.");
      next_word_index += 1;
      
      background(255);
      fill(0);
      textFont(f,20);
      String message = "All words have been collected.";
      text(message, 10, 100);
      textFont(f,14);
      text("\nPress N again to confirm exit. Or press R to restart. ", 10, 110);
    }
    
    if (next_word_index < INDEX_LIMIT){
      String filename = CreateNewFile();
      println("Now word index is: " + curr_word_index + ". Writing for the file:  " + filename);
    }
  }
  
  if (key == 'd' || key == 'D'){
    buffer.append("#\n");
    println("Marked the previous data sequence as invalid by using # as the current row.");
  }
  
  if (key == 'p' || key == 'P'){
    buffer.setLength(0);
    println("Purified the buffer, recollect motion data for word " + word_list[curr_word_index]);
  }
  
  if (key == 'r' || key == 'R'){
    output.flush();
    output.close();
    next_word_index = 0;
    String filename = CreateNewFile();
    println("Started recollecting data.");
    println("Now word index is: " + curr_word_index + ". Writing for the file:  " + filename);
  }
}

String CreateNewFile(){
  // look at current word
  curr_word_index = next_word_index;
  
  String filename = ".csv";
  String curr_word = "NO_WORD_ERROR";
  String directoryName = DIR_NAME + "/" + FOLDER_NAME;
  
  if (curr_word_index == -1){
    filename = "calibration" + filename;
    curr_word = "Calibration data.";
  }
  else{
    curr_word = word_list[curr_word_index];
    filename = curr_word + filename;
  }
  
  File directory = new File(directoryName);
  if (! directory.exists()){
    directory.mkdir();
  }
  
  String full_path = directoryName + "/" + filename;
  
  if (output != null){
    output.flush();
    output.close();
  }
  
  output = createWriter( full_path );
  next_word_index += 1;
  
  // for displaying text.
  background(255);
  fill(0);
  textFont(f,20);
  String message = "Now Writing : " + curr_word; 
  text(message, 10, 100);
  textFont(f,14);
  message = "\n\nPress N for next word. Press P to clean current buffer.\nSaved at: " + full_path;
  text(message, 10, 110);
  
  return filename;
}

void SaveBufferData(){
  output.print(buffer.toString());
  output.flush();
  buffer.setLength(0);
}
