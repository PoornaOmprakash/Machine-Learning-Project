import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;


public class PickBestParameterADTree {
	
	static class ADTreeStats
	{
		int B;
		int E;
	    double avg;
	    double max;
	    
	    ADTreeStats()
	    {
	    	B=0;
	    	E=0;
	    	avg=0;
	    	max=0;
	    }
	}
public static void main(String[] argv) throws IOException {
	 
		FileInputStream fr = new FileInputStream(argv[0]);
	    BufferedReader br = new BufferedReader(new InputStreamReader(fr)); 
		Scanner s=new Scanner(new File(argv[0]));
		double mean=0.0;
		double max=0.0;
		String str=null;
		String str1=null;
		int index;
		List<Double> means = new ArrayList<Double>();
		while((str=br.readLine())!=null)   //Create a hashmap with the user ID as a key (Corresponding to the row of the matrix)
		{
		   str1=str;
		   index=str.indexOf(",");
		   //movieID=Integer.parseInt(str.substring(0,posMovieID));
		   str1=str.substring(index+1);
	//	   System.out.println(str1);
	/*	   index = str1.indexOf(",");
		   str1 = str1.substring(index+1);
		   index = str1.indexOf(",");
		   str1 = str1.substring(index+1);
		   index = str1.indexOf(",");
		   str1 = str1.substring(index+1);
//		   System.out.println(str1);
		   index = str1.indexOf(","); */
		   str1 = str1.substring(0,index);
//		   System.out.println(str1);
		   mean = Double.parseDouble(str1);
		   means.add(mean);
		}
	    Collections.sort(means);
	    System.out.println(means.get(0));
	}	
}
