Programming Assignment 2 - Christopher Thai

Overview
	This is a python project
	Each part can be specified to be run or not in the command line. Each part outputs a csv and other data to the output folder

Dependencies
	This code uses several outside libraries, namely Numpy, Pandas, Scipy, Keras, and SKLearn. The required modules are listed in requirements.txt

Basic Usage
	Basic usage for generating classification results of machine learning methods is:
	main.py -p3 -p5 -p6a -p6b -p6c -p6d
		Most of the machine learning methods require PCA, which is done in part 5, so -p5 is required with any machine learning method
		-p3 cleans outliers, and will affect the machine learning methods, but isn't a requirement

	Basic usage for parts 1, 2, and 4:
	main.py -p1 -lf iris_data_for_cleansing.csv
		The default is to use updated_iris.csv, so for part 1 override that with -lf <file name>
	main.py -p2
	main.py -p4

Command Line Args
	-lf <file name>
		Specifies which file to load as a csv. File must be in the input folder as a csv
		Defaults to iris_updated.csv, so don't need to specify this usually
	-p1
		Runs Part 1: Data Cleansing, this only does something on iris_data_for_cleansing.csv, so use -p1 with -lf iris_data_for_cleansing.csv
		Affects Machine Learning Methods
	-p2
		Runs Part 2: Feature Generation
		Does not affect Machine Learning Methods
	-p3
		Runs Part 3: Feature Preprocessing, cleans outliers
		Affects Machine Learning Methods
	-p4
		Runs Part 4: Feature Ranking
		Does not affect Machine Learning Methods
	-p5
		Runs Part 5: Dimensionality Reduction
		Requirement for Machine Learning Methods
	-p6a
		Runs Part 6a: Expectation Maximization
		Does not need to be run with other Machine Learning Methods
	-p6b
		Runs Part 6b: Fisher's LDA Classification
		Does not need to be run with other Machine Learning Methods
	-p6c
		Runs Part 6c: Neural Network
		Does not need to be run with other Machine Learning Methods
	-p6d
		Runs Part 6d: Support Vector Machine
		Does not need to be run with other Machine Learning Methods

Other Libraries Used:
	Numpy: Matrix Manipulation
	Pandas: Data Storage and Manipulation
	Scipy (part 6a): Multivariate Normal Distribution
	Keras (part 6c): Neural Network
	SKLearn (part 6d): Support Vector Machine

Files:
	main.py
	ReadMe.txt
	requirements.txt
	input/
		updated_iris.csv
		iris_data_for_cleansing.csv
	machine_learning/
		support_vector_machine.py
		fisher_linear_discriminant.py
		feed_forward_neural_network.py
		expectation_maximization.py
	output/
		final_for_submission/
			same as output folder
		part1_data_cleansing_data.csv
		part2_feature_generation_algorithm.json
		part2_feature_generation_data.csv
		part3_feature_preprocessing_data.csv
		part3_feature_preprocessing_outlier_algorithm.json
		part3_feature_preprocessing_outlier_data.csv
		part4_feature_ranking_algorithm.json
		part4_feature_ranking_data.csv
		part5_dimensionality_reduction_algorithm.json
		part5_dimensionality_reduction_data.csv
		part6a_expectation_maximization_data.csv
		part6b_fisher_linear_discriminant_data.csv
		part6c_feed_forward_neural_network_data.csv
		part6d_support_vector_machine_data.csv
	pdf_details/
		Programming Assignment Guideline(4).pdf
		PA2(4).pdf
	preprocessing/
		load_data.py
		feature_ranking.py
		feature_preprocessing.py
		feature_generation.py
		dimensionality_reduction.py
		data_cleansing.py
	utils/
		parse_args.py