import processing.serial.*;
Serial mySerial;
PrintWriter output;
import java.text.SimpleDateFormat;  
import java.util.Date;  

void setup() {
   SimpleDateFormat formatter = new SimpleDateFormat("MM_dd_HH_mm");  
   Date date = new Date();  
   String filename = "Output_" + formatter.format(date) + ".txt";  
   
   int last_idx = Serial.list().length - 1;
   mySerial = new Serial( this, Serial.list()[last_idx], 115200 );
   output = createWriter( filename );
   println(mySerial);
}
void draw() {
    if (mySerial.available() > 0 ) {
         String value = mySerial.readStringUntil('\n');
         if ( value != null ) {
              println(value);
              output.println( value );
         }
    }
}

void keyPressed() {
    output.flush();  // Writes the remaining data to the file
    output.close();  // Finishes the file
    exit();  // Stops the program
}
