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
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.Dagging;
import weka.classifiers.meta.END;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.FT;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class Milestone5_2 {

	/**
	 * Method which reads in .arff files from a directory, trains the models and
	 * computes the relevant performance metrics (milestone 4 version).
	 * @param dirPath The path to the directory containing the .arff files.
	 */
	public static void milestone4Ensemble(String dirPath) {
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
				/*String[] ftOptions = new String[9];
				ftOptions[0] = "-B";
				ftOptions[1] = "-P";
				ftOptions[2] = "-I";
				ftOptions[3] = "5";
				ftOptions[4] = "-F";
				ftOptions[5] = "0";
				ftOptions[6] = "-M";
				ftOptions[7] = "10";
				ftOptions[8] = "-A";
				ft.setOptions(ftOptions); */
				ft.buildClassifier(trainData);

				// Train LB classifier
				LogitBoost lb = new LogitBoost();
				/*String[] lbOptions = new String[10];
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
				lb.setOptions(lbOptions);*/
				lb.buildClassifier(trainData);

				// Train LMT classifier (parameters will vary on data set)
				LMT lmt = new LMT();
				/*String[] lmtOptions = new String[6];
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
				lmt.setOptions(lmtOptions);*/
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
				String modelName = dirPath + "/" + name + "-LB.predict";
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
	 * Method which reads in .arff files from a directory, trains the models and
	 * computes the relevant performance metrics (milestone 5 version).
	 * @param dirPath The path to the directory containing the .arff files.
	 */
	public static void milestone5Ensemble(String dirPath) {
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

			double bestAvgE = 0.0;
			double bestMaxE = 0.0;

			BufferedWriter writer = new BufferedWriter(new FileWriter("result.txt"));

			for (String name : names) {
				// Collect info on training/test data
				String trainName = dirPath + "/" + name + "_train.arff";
				String testName = dirPath + "/" + name + "_test.arff";
				BufferedReader reader = new BufferedReader(
						new FileReader(trainName));
				final Instances trainData = new Instances(reader);
				reader.close();
				reader = new BufferedReader(new FileReader(testName));
				final Instances testData = new Instances(reader);
				reader.close();
				trainData.setClassIndex(trainData.numAttributes() - 1);
				testData.setClassIndex(testData.numAttributes() - 1);

				int NUM_CLASSIFIERS = 8;
				final Classifier[] classifiers = new Classifier[NUM_CLASSIFIERS];
				final String[] predList = new String[NUM_CLASSIFIERS];
				final double[] errors = new double[NUM_CLASSIFIERS];

				// Train NB classifier (baseline)
				NaiveBayes nb = new NaiveBayes();
				nb.buildClassifier(trainData);
				Evaluation nbEval = new Evaluation(trainData);
				nbEval.evaluateModel(nb, testData);
				final double nbE = nbEval.errorRate();

				// Prepare LB classifier for training
				LogitBoost lb = new LogitBoost();
				classifiers[7] = lb;

				// Prepare Dagging classifier for training
				Dagging dagging = new Dagging();
				classifiers[2] = dagging;

				// Prepare AdaBoost classifier for training
				AdaBoostM1 ada = new AdaBoostM1();
				classifiers[3] = ada;

				// Prepare LR classifier for training
				Logistic lr = new Logistic();
				classifiers[4] = lr;

				// Prepare random forest classifier for training
				RandomForest rf = new RandomForest();
				classifiers[5] = rf;

				// Prepare bagging classifier for training
				Bagging bag = new Bagging();
				classifiers[6] = bag;

				// Prepare END classifier
				END end = new END();
				classifiers[0] = end;

				// Prepare BayesNet classifier
				BayesNet bn = new BayesNet();
				classifiers[1] = bn;

				// Train classifiers in parallel
				Thread[] threads = new Thread[NUM_CLASSIFIERS];
				for (int i = 0; i < NUM_CLASSIFIERS / 2; i++) {
					final int k = i;
					Thread t = new Thread() {
						public void run() {
							try {
								// Find optimal parameters for the classifier
								Classifier c = classifiers[k];
								if (!(c instanceof BayesNet)) {
									CVParameterSelection pSelect = new CVParameterSelection();
									String[] pSelectOptions = new String[2];
									pSelectOptions[0] = "-W";
									pSelectOptions[1] = ((c.getClass()).toString()).split(" ")[1];
									pSelect.setOptions(pSelectOptions);
									pSelect.buildClassifier(trainData);
									c.setOptions(pSelect.getBestClassifierOptions());
								}

								// Build the classifier								
								c.buildClassifier(trainData);
								String preds = "";
								int total = testData.numInstances();
								for (int j = 0; j < total; j++) {
									double pred = c.classifyInstance(testData.instance(j));;
									if (j != 0)
										preds += "\n";
									preds += (pred + "");
								}
								predList[k] = preds;
								Evaluation eval = new Evaluation(trainData);
								eval.evaluateModel(c, testData);
								double eRate = eval.errorRate();
								errors[k] = eRate / nbE;
							}
							catch (Exception e) {
								e.printStackTrace();
							}
						}

					};
					t.start();
					threads[k] = t;
				}

				for (int i = 0; i < NUM_CLASSIFIERS / 2 ; i++)
					threads[i].join();

				for (int i = NUM_CLASSIFIERS / 2; i < NUM_CLASSIFIERS; i++) {
					final int k = i;
					Thread t = new Thread() {
						public void run() {
							try {
								// Find optimal parameters for the classifier
								Classifier c = classifiers[k];
								CVParameterSelection pSelect = new CVParameterSelection();
								String[] pSelectOptions = new String[2];
								pSelectOptions[0] = "-W";
								pSelectOptions[1] = ((c.getClass()).toString()).split(" ")[1];
								pSelect.setOptions(pSelectOptions);
								pSelect.buildClassifier(trainData);
								c.setOptions(pSelect.getBestClassifierOptions());

								// Build the classifier
								c.buildClassifier(trainData);
								String preds = "";
								int total = testData.numInstances();
								for (int j = 0; j < total; j++) {
									double pred = c.classifyInstance(testData.instance(j));;
									if (j != 0)
										preds += "\n";
									preds += (pred + "");
								}
								predList[k] = preds;
								Evaluation eval = new Evaluation(trainData);
								eval.evaluateModel(c, testData);
								double eRate = eval.errorRate();
								errors[k] = eRate / nbE;
							}
							catch (Exception e) {
								e.printStackTrace();
							}
						}

					};
					t.start();
					threads[k] = t;
				}

				for (int i = NUM_CLASSIFIERS / 2; i < NUM_CLASSIFIERS ; i++)
					threads[i].join();

				// Get result (best average error)
				double bestBase = Double.MAX_VALUE;
				String preds = "";
				for (int j = 0; j < errors.length; j++) {
					double cur = errors[j];
					if (cur < bestBase) {
						bestBase = cur;
						preds = predList[j];
					}
				};
				bestAvgE += bestBase;
				if (bestBase > bestMaxE)
					bestMaxE = bestBase;
				writer.write(bestBase + "\n");

				// Write out the predictions
				System.out.println("Writing predictions for " + name);
				String modelName = dirPath + "/" + name + "-L5.predict";
				BufferedWriter predWriter = new BufferedWriter(new FileWriter(modelName));
				predWriter.write(preds);
				predWriter.close();
			}

			bestAvgE /= (double) names.size();
			writer.write(bestAvgE + "\n");
			writer.write(bestMaxE + "");
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
	public static void main (String[] args) {
		Milestone5.milestone5Ensemble(args[0]);
		Milestone5.milestone4Ensemble(args[0]);
	}
}
