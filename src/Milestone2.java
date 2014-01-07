import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Dagging;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.rules.JRip;
import weka.core.Instances;

public class Milestone2 {

//	/**
//	 * Method which reads in .arff files from a directory, loads a serialized model for each data set from the same directory
//	 * and computes the relevant performance metrics.
//	 * @param dirPath The path to the directory containing the .arff files.
//	 */
//	public static void loadModels(String dirPath) {
//		try {
//			// Get the name of each data set
//			File dir = new File(dirPath);
//			File [] trainFiles = dir.listFiles(new FilenameFilter() {
//				@Override
//				public boolean accept(File dir, String name) {
//					return name.contains("_train");
//				}
//			});
//
//			ArrayList<String> names = new ArrayList<String>();
//			for (int i = 0; i < trainFiles.length; i++)
//				names.add(((trainFiles[i]).getName()).split("_")[0]);
//			Collections.sort(names);
//
//			double avgE = 0.0;
//			double maxE = 0.0;
//
//			BufferedWriter writer = new BufferedWriter(new FileWriter("serialiedresults.txt"));
//
//			for (String name : names) {
//				String trainName = dirPath + "/" + name + "_train.arff";
//				String testName = dirPath + "/" + name + "_test.arff";
//				BufferedReader reader = new BufferedReader(
//						new FileReader(trainName));
//				Instances trainData = new Instances(reader);
//				reader.close();
//				reader = new BufferedReader(new FileReader(testName));
//				Instances testData = new Instances(reader);
//				reader.close();
//				trainData.setClassIndex(trainData.numAttributes() - 1);
//				testData.setClassIndex(testData.numAttributes() - 1);
//				
//				// Train NB classifier (baseline)
//				NaiveBayes nb = new NaiveBayes();
//				nb.buildClassifier(trainData);
//
//				// Make predictions on test set for NB
//				Evaluation nbEval = new Evaluation(trainData);
//				nbEval.evaluateModel(nb, testData);
//				double nbE = nbEval.errorRate();
//
//				ObjectInputStream ois = new ObjectInputStream(
//						new FileInputStream(dirPath + "/" + name + ".model"));
//				Classifier cls = (Classifier) ois.readObject();
//				ois.close();
//				
//				Evaluation clsEval = new Evaluation(trainData);
//				clsEval.evaluateModel(cls, testData);
//				double clsE = clsEval.errorRate();
//
//				double relError = clsE / nbE;
//				avgE += relError;
//				if (relError > maxE)
//					maxE = relError;
//				writer.write(relError + "\n");;
//			}
//
//			avgE /= (double) names.size();
//			writer.write(avgE + "\n");
//			writer.write(maxE + "");
//			writer.close();
//		}
//		catch (Exception e) {
//			e.printStackTrace();
//		}
//	}

	/**
	 * Method which reads in .arff files from a directory, trains the models and
	 * computes the relevant performance metrics.
	 * @param dirPath The path to the directory containing the .arff files.
	 */
	public static void trainModels(String dirPath) {
		try {
			// Get the name of each data set
			File dir = new File(dirPath);
			File [] trainFiles = dir.listFiles(new FilenameFilter() {
				@Override
				public boolean accept(File dir, String name) {
					return name.contains("_train");
				}
			});

			ArrayList<String> names = new ArrayList<String>();
			for (int i = 0; i < trainFiles.length; i++)
				names.add(((trainFiles[i]).getName()).split("_")[0]);
			Collections.sort(names);

			double avgE = 0.0;
			double maxE = 0.0;

			BufferedWriter writer = new BufferedWriter(new FileWriter("result0.txt"));

			for (String name : names) {
				String trainName = dirPath + "/" + name + "_train.arff";
				String testName = dirPath + "/" + name + "_test.arff";
				BufferedReader reader = new BufferedReader(
						new FileReader(trainName));
				Instances trainData = new Instances(reader);
				reader.close();
				reader = new BufferedReader(new FileReader(testName));
				Instances testData = new Instances(reader);
				reader.close();
				trainData.setClassIndex(trainData.numAttributes() - 1);
				testData.setClassIndex(testData.numAttributes() - 1);

				// Train NB classifier (baseline)
				NaiveBayes nb = new NaiveBayes();
				nb.buildClassifier(trainData);

				// Make predictions on test set for NB
				Evaluation nbEval = new Evaluation(trainData);
				nbEval.evaluateModel(nb, testData);
				double nbE = nbEval.errorRate();

				// Train ADTree classifier
				MultiClassClassifier multi = new MultiClassClassifier();
				String[] options = new String[2];
				options[0] = "-W";
				options[1] = "weka.classifiers.trees.ADTree";
				multi.setOptions(options);
				multi.buildClassifier(trainData);

				// Make predictions on test set for ADTree
				Evaluation adEval = new Evaluation(trainData);
				adEval.evaluateModel(multi, testData);
				double adE = adEval.errorRate();
               
				// Train JRip classifier
				JRip jrip = new JRip();
				jrip.buildClassifier(trainData);

				// Make predictions on test set for JRip
				Evaluation jripEval = new Evaluation(trainData);
				jripEval.evaluateModel(jrip, testData);
				double jripE = jripEval.errorRate();
				
				// Train Dagging classifier
				Dagging dag = new Dagging();
				dag.buildClassifier(trainData);

				// Make predictions on test set for 
				Evaluation dagEval = new Evaluation(trainData);
				dagEval.evaluateModel(dag, testData);
				double dagE = dagEval.errorRate();
				
//				System.out.println("Statistics for " + name + " data set:");
//				System.out.println("nbE: " + nbE + "\tadE: " + adE + "\tjripE: " + jripE + "\tdagE: " + dagE);

				ArrayList<Double> errs = new ArrayList<Double>();
				errs.add(adE); errs.add(jripE); errs.add(dagE);
				Collections.sort(errs);
				double minE = errs.get(0);

				// System.out.println("Selected minimum error: " + minE + "\n");

				double relError = minE / nbE;
				avgE += relError;
				if (relError > maxE)
					maxE = relError;
				writer.write(relError + "\n");

				// Write out the model
				
				String serializedModel = dirPath + "/" + name + "0.model";
				ObjectOutputStream oos = new ObjectOutputStream(
						new FileOutputStream(serializedModel));
				if((adE <= dagE) && (adE <= jripE))
					oos.writeObject(multi);
				else if((jripE <= dagE) && (jripE <= adE))
					oos.writeObject(jrip);
				else
					oos.writeObject(dag);
				oos.flush();
				oos.close();
			}

			avgE /= (double) names.size();
			writer.write(avgE + "\n");
			writer.write(maxE + "");
			writer.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Main method.
	 * @param args Program arguments. Should only contain the path to the directory containing the .arff files.
	 */
	public static void main(String[] args) {
		Milestone2.trainModels(args[0]);
//		ClassifierTrainer.loadModels(args[0]);
	}
}
