import processing.serial.*;
Serial mySerial;
PrintWriter output;
import java.text.SimpleDateFormat;  
import java.util.Date;  
boolean isReady;

void setup() {
   SimpleDateFormat formatter = new SimpleDateFormat("MM_dd_HH_mm");  
   Date date = new Date();  
   String filename = "Output_" + formatter.format(date) + ".txt";  
   
   int last_idx = Serial.list().length - 1;
   mySerial = new Serial( this, Serial.list()[last_idx], 115200 );
   output = createWriter( filename );
   println("Serial Port is: " + mySerial.toString());
   isReady = false;
}
void draw() {
    if (mySerial.available() > 0 ) {
         String value = mySerial.readString();
         if ( value != null ) {
              print(value);
              if (!isReady){
                if (value.indexOf("Ready to go!") != -1) isReady = true;
              }
              else{
                output.print( value );
                output.flush();
              }
         }
    }
}
