	import java.io.BufferedReader;
	import java.io.BufferedWriter;
	import java.io.File;
	import java.io.FileOutputStream;
	import java.io.FileReader;
	import java.io.FileWriter;
	import java.io.FilenameFilter;
	import java.io.ObjectOutputStream;
	import java.util.ArrayList;
	import java.util.Arrays;
	import java.util.Collections;

	import weka.classifiers.Evaluation;
	import weka.classifiers.bayes.NaiveBayes;
	import weka.classifiers.meta.Dagging;
	import weka.classifiers.meta.MultiClassClassifier;
	import weka.classifiers.rules.JRip;
	import weka.core.Instances;

	public class ClassifierTrainer3 {

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


					ArrayList<Double> errs = new ArrayList<Double>();
					errs.add(adE); errs.add(jripE); errs.add(dagE);
					Collections.sort(errs);
					double minE = errs.get(0);

					double relError = minE / nbE;
					avgE += relError;
					if (relError > maxE)
						maxE = relError;
					writer.write(relError + "\n");

					// Write out the model
					String serializedModel = dirPath + "/" + name + ".model";
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
			  //ClassifierTrainer3.determineADTreeParameters(args[0]);
			 // ClassifierTrainer3.determineJRipParameters(args[0]);
			 ClassifierTrainer3.determineDaggingParameters(args[0]);
		}
	}
	
