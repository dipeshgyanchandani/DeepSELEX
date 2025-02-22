# Introduction
DeepSELEX, originally developed to infer DNA-binding preferences of transcription factors using HT-SELEX data, leverages deep neural networks to analyze high-throughput sequencing data. An architectural model is tailored specifically for the theophylline dataset, aiming to enhance predictive capabilities. Initial results demonstrate improved performance over traditional methods, underscoring the potential of generative AI in advancing biomedical research and personalized medicine. This adaptation highlights the versatility of DeepSELEX and its potential application to diverse health-related datasets, paving the way for more accurate and efficient biomedical discoveries. Further research will focus on refining this model and extending its applicability to other critical areas in health science.

# About DeepSELEX
Inference of DNA-binding preferences from HT-SELEX data using deep neural networks

# Flags
The flags for the command line interface:
1. - `learning_file_list`: A list of HT-SELEX files. Should be written as follows:
  -lfl demo_data/ALX4_TGTGTC20NGA_W_0.fastq demo_data/ALX4_TGTGTC20NGA_W_1.fastq demo_data/ALX4_TGTGTC20NGA_W_2.fastq demo_data/ALX4_TGTGTC20NGA_W_3.fastq demo_data/ALX4_TGTGTC20NGA_W_4.fastq
2. - `primary_selex_sequence`: the sequence which is the HT-SELEX experiment primary sequence.
   If the selex file is of the form: ALX4_TGTGTC20NGA_W_0.fastq, the primary sequence is: TGTGTC20NGA
   this sequence should be supplied in the cmd.  Should be written as follows:
   -pss TGTGTC20NGA
3. - `prediction_file`: Prediction data file.
   Should be written as follows:
   -pf demo_data/Alx4_1744.1_deBruijn.txt or any other predicted file
4. - `output_file_location`: The output file name and location.
   Should be written as follows:
   -ofl results.csv
5. - `saved_model_location`: If supply, saves the model in the supplied address.
   Should be written as follows:
   -sml output_model.h5
6. - `loaded_model_location`: Loads the model from the supplied address
   Should be written as follows:
   -lml loaded_model_name.h5

# Examples
1. Training command line example:\
   ```python deep_selex.py -lfl demo_data/ALX4_TGTGTC20NGA_W_0.fastq demo_data/ALX4_TGTGTC20NGA_W_1.fastq demo_data/ALX4_TGTGTC20NGA_W_2.fastq demo_data/ALX4_TGTGTC20NGA_W_3.fastq demo_data/ALX4_TGTGTC20NGA_W_4.fastq -pss TGTGTC20NGA -sml output_model.h5```
2.  Training and predicting command line example:\
   ```python deep_selex.py -lfl demo_data/ALX4_TGTGTC20NGA_W_0.fastq demo_data/ALX4_TGTGTC20NGA_W_1.fastq demo_data/ALX4_TGTGTC20NGA_W_2.fastq demo_data/ALX4_TGTGTC20NGA_W_3.fastq demo_data/ALX4_TGTGTC20NGA_W_4.fastq -pss TGTGTC20NGA -pf demo_data/Alx4_1744.1_deBruijn.txt -ofl results.csv -sml output_model.h5```
3. Using pre-trained model command line example:\
   ```python deep_selex.py -lml test_model.h5 -pf demo_data/Alx4_1744.1_deBruijn.txt -ofl results.csv```


# Requirements:

Linux based operating system (the trained models at the "models" directory can sometime have errors under other operating systems)

python interpreter > = 3.6

python software packages:\
   xlrd >= 1.2.0\
	numpy >= 1.17.5\
	pandas >= 0.25.3\
	tensorflow version == 1.14.0\
	keras version == 2.3.1\

	

