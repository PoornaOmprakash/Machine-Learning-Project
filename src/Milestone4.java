import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Collections;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Dagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.meta.Vote;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.FT;
import weka.classifiers.trees.LMT;
import weka.core.Instances;

public class Milestone4 {

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

			BufferedWriter writer = new BufferedWriter(new FileWriter("result.txt"));

			for (String name : names) {
				// Collect info on training/test data
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
				Evaluation nbEval = new Evaluation(trainData);
				nbEval.evaluateModel(nb, testData);
				double nbE = nbEval.errorRate();
				
				// Train FT classifier
				FT ft = new FT();
				String[] ftOptions = new String[9];
				ftOptions[0] = "-B";
				ftOptions[1] = "-P";
				ftOptions[2] = "-I";
				ftOptions[3] = "5";
				ftOptions[4] = "-F";
				ftOptions[5] = "0";
				ftOptions[6] = "-M";
				ftOptions[7] = "10";
				ftOptions[8] = "-A";
				ft.setOptions(ftOptions);
				ft.buildClassifier(trainData);
				
				// Train LB classifier
				LogitBoost lb = new LogitBoost();
				String[] lbOptions = new String[10];
				lbOptions[0] = "-I";
				lbOptions[1] = "100";
				lbOptions[2] = "-P";
				lbOptions[3] = "100";
				lbOptions[4] = "-F";
				lbOptions[5] = "0";
				lbOptions[6] = "-R";
				lbOptions[7] = "1";
				lbOptions[8] = "-H";
				lbOptions[9] = "1.5";
				lb.setOptions(lbOptions);
				lb.buildClassifier(trainData);
				
				// Train LMT classifier (parameters will vary on data set)
				LMT lmt = new LMT();
				String[] lmtOptions = new String[6];
				lmtOptions[0] = "-I";
				lmtOptions[2] = "-M";
				lmtOptions[4] = "-W";
				if (name.equals("anneal")) {
					lmtOptions[1] = "-1";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.0";
				}
				if (name.equals("audiology")) {
					lmtOptions[1] = "15";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.1";
				}
				if (name.equals("autos")) {
					lmtOptions[1] = "30";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.0";
				}
				if (name.equals("balance-scale")) {
					lmtOptions[1] = "30";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.2";
				}
				if (name.equals("breast-cancer")) {
					lmtOptions[1] = "-1";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.2";
				}
				if (name.equals("colic")) {
					lmtOptions[1] = "15";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.0";
				}
				if (name.equals("credit-a")) {
					lmtOptions[1] = "-1";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.2";
				}
				if (name.equals("diabetes")) {
					lmtOptions[1] = "15";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.2";
				}
				if (name.equals("glass")) {
					lmtOptions[1] = "-1";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.0";
				}
				if (name.equals("heart-c")) {
					lmtOptions[1] = "-1";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.0";
				}
				if (name.equals("hepatitis")) {
					lmtOptions[1] = "-1";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.2";
				}
				if (name.equals("hypothyroid")) {
					lmtOptions[1] = "-4";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.0";
				}
				else
				{
					lmtOptions[1] = "-1";
					lmtOptions[3] = "15";
					lmtOptions[5] = "0.0";
				}
				lmt.setOptions(lmtOptions);
				lmt.buildClassifier(trainData);
				
				// Build vote classifier
				Vote vote = new Vote();
				if(name.equals("splice"))
				{
					Classifier[] classifiers = new Classifier[2];
					classifiers[0] = ft;
					classifiers[1] = lb;
					vote.setClassifiers(classifiers);
				}
				
				else
				{
				    Classifier[] classifiers = new Classifier[3];
				    classifiers[0] = ft;
				    classifiers[1] = lmt;
				    classifiers[2] = lb;
				    vote.setClassifiers(classifiers);
				}
				
				String preds = "";
				Instances instances = new Instances(testData);
				for (int i = 0; i < instances.numInstances(); i++) {
					double pred = vote.classifyInstance(instances.instance(i));
					if (i != 0)
						preds += "\n";
					preds += (pred + "");
				}
//				Evaluation voteEval = new Evaluation(trainData);
//				voteEval.evaluateModel(vote, testData);
//				double eVote = voteEval.errorRate();
//				
//				double relError = eVote / nbE;
//				avgE += relError;
//				if (relError > maxE)
//					maxE = relError;
//				writer.write(relError + "\n");
				
				// Write out the predictions
				System.out.println(name);
				String modelName = dirPath + "/" + name + "1.predict";
				BufferedWriter predWriter = new BufferedWriter(new FileWriter(modelName));
				predWriter.write(preds);
				predWriter.close();
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
	 * Method which reads in .arff files from a directory and determines
	 * an optimal set of parameters for the ADTree classifier.
	 * @param dirPath The path to the directory containing the .arff files.
	 */
	public static void determineADTreeParameters(String dirPath) {
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

			BufferedWriter writer = new BufferedWriter(new FileWriter("adtree_result.txt"));
			for (int i = 0; i <= 50; i++) {
				for (int j = 0; j <= 3; j++) {
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

						// Train ADTree classifier
						MultiClassClassifier multi = new MultiClassClassifier();
						String[] options = new String[7];
						options[0] = "-W";
						options[1] = "weka.classifiers.trees.ADTree";
						options[2] = "--";
						options[3] = "-B";
						options[4] = i + "";
						options[5] = "-E";
						options[6] = j + "";
						multi.setOptions(options);
						multi.buildClassifier(trainData);

						// Make predictions on test set for ADTree
						Evaluation adEval = new Evaluation(trainData);
						adEval.evaluateModel(multi, testData);
						double adE = adEval.errorRate();
						avgE += adE;
						if (adE > maxE)
							maxE = adE;					
					}
					avgE /= (double) names.size();
					writer.write("B=" + i +",E=" + j + "," + avgE + "," + maxE + "\n");
				}
			}
			writer.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Method which reads in .arff files from a directory and determines
	 * an optimal set of parameters for the JRip classifier.
	 * @param dirPath The path to the directory containing the .arff files.
	 */
	public static void determineJRipParameters(String dirPath) {
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

			BufferedWriter writer = new BufferedWriter(new FileWriter("jrip_result.txt"));
			boolean[] bools = {true, false};
			for (int i = 1; i <= 10; i++) {
				for (int j = 0; j <= 10; j++) {
					for (int k = 0; k <= 10; k++) {
						for (int n = 0; n < bools.length; n++) {
							boolean curBool = bools[n];
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

								// Train JRip
								JRip jrip = new JRip();
								String[] options;
								if (curBool)
									options = new String[7];
								else
									options = new String[6];
								options[0] = "-F";
								options[1] = i + "";
								options[2] = "-N";
								options[3] = j + "";
								options[4] = "-O";
								options[5] = k + "";
								if (curBool)
									options[6] = "-P";
								jrip.setOptions(options);
								jrip.buildClassifier(trainData);

								// Make predictions on test set for ADTree
								Evaluation adEval = new Evaluation(trainData);
								adEval.evaluateModel(jrip, testData);
								double adE = adEval.errorRate();
								avgE += adE;
								if (adE > maxE)
									maxE = adE;	
							}
							avgE /= (double) names.size();
							writer.write("F=" + i + ",N=" + j + ",O=" + k + ",P=" + curBool + "," + avgE + "," + maxE + "\n");
						}
					}
				}
			}
			writer.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Method which reads in .arff files from a directory and determines
	 * an optimal set of parameters for the Dagging meta-classifier.
	 * @param dirPath The path to the directory containing the .arff files.
	 */
	public static void determineDaggingParameters(String dirPath) {
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

			BufferedWriter writer = new BufferedWriter(new FileWriter("dagging_result.txt"));
			for (int i = 1; i <= 20; i++) {
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

						// Train ADTree classifier
						Dagging dagging = new Dagging();
						String[] options = new String[2];
						options[0] = "-F";
						options[1] = i + "";
						dagging.setOptions(options);
						dagging.buildClassifier(trainData);

						// Make predictions on test set for ADTree
						Evaluation adEval = new Evaluation(trainData);
						adEval.evaluateModel(dagging, testData);
						double adE = adEval.errorRate();
						avgE += adE;
						if (adE > maxE)
							maxE = adE;					
					}
					avgE /= (double) names.size();
					writer.write("F=" + i + "," + avgE + "," + maxE + "\n");
			}
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
		//ClassifierTrainer.determineADTreeParameters(args[0]);
		//ClassifierTrainer.determineJRipParameters(args[0]);
//		ClassifierTrainer.determineDaggingParameters(args[0]);
		Milestone4.trainModels(args[0]);
	}
}
