import processing.serial.*;
import java.text.SimpleDateFormat;  
import java.util.Date; 
import java.lang.StringBuilder;

//boolean USE_SUBJECT_NAME = true;

String[] mini_easy_2 = new String[]{
"exams", "focus", "sauce", "coax", "same", "sex", "awe", "fem", "axe", "age"
};

String[] mini_hard_2 = new String[]{
"kanji", "flick", "beach", "cabin", "hack", "lack", "kami", "kid" , "ban", "bad"
};


/* ------------------------------ */
/* Custom Field Start */
String DIR_NAME = "Words";
String SUBJECT_NAME = "words_mini_easy_2";

String[] word_list = mini_easy_2.clone();

boolean addCalibration = true;
int word_index_to_start_at = 0;

/* Custom Field End */
/* ------------------------------ */

Serial mySerial;
Date date;
PrintWriter output;
boolean isReady;
StringBuilder buffer;
int next_word_index;
int curr_word_index;
int INDEX_LIMIT = word_list.length;

void setup() {
   date = new Date();  
   
   int last_idx = Serial.list().length - 1;
   mySerial = new Serial( this, Serial.list()[last_idx], 115200 );
   
   println("Serial Port is: " + mySerial.toString());
   isReady = false;
   
   buffer = new StringBuilder();
   
   if (addCalibration){
     // use "-1" if you wish to start with calibration, else enter corresponding letter's number
     next_word_index = -1;
   }
   else {
     next_word_index = word_index_to_start_at;
   }
}

void draw() {
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
      println("Confirmed exiting.");
      exit();
    }
    
    SaveBufferData();
    
    if (next_word_index == INDEX_LIMIT){
      output.flush();
      output.close();
      
      println("Completed collecting data. Press R to restart collecting. Press N again to confirm exit.");
      next_word_index += 1;
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
  String directoryName = DIR_NAME + "/" + SUBJECT_NAME;
  
  if (curr_word_index == -1){
    filename = "calibration" + filename;
  }
  else{
    curr_word = word_list[curr_word_index];
    filename = curr_word + filename;
  }
  
  File directory = new File(directoryName);
  if (! directory.exists()){
    directory.mkdir();
  }
  
  filename = directoryName + "/" + filename;
  
  if (output != null){
    output.flush();
    output.close();
  }
  
  output = createWriter( filename );
  next_word_index += 1;
  
  return filename;
}

void SaveBufferData(){
  output.print(buffer.toString());
  output.flush();
  buffer.setLength(0);
}
