This repository holds the inference code that is associated with the research paper titled "Deep learning, data ramping, and uncertainty estimation for detecting artifacts in large, imbalanced databases of MRI images" by Pizarro et al., published in Medical Image Analysis in 2023.

The computational model leveraged in this code has been transitioned to the ONNX format for enhanced interoperability and lightweight operations, making it feasible to execute on a CPU within a few seconds. Please find the installation and execution instructions below:

```bash
# Set up the software environment for CPU
$ conda env create -n artifact -f ./artifact_env.yml
$ conda activate artifact

# Verify the correct installation of the software
$ pytest  test_infer_utils.py

# Here's how you can run the inference script. It can take multiple inputs and the output is structured for easy parsing.
$ infer_onnx.py -i my_scan_1.mnc my_scan_2.mnc
my_scan_1.mnc,clean,0.0
my_scan_2.mnc,clean,0.0
```

For a greater degree of customization, the inference script comes equipped with several options:

```bash
infer_onnx.py --help
usage: infer_onnx.py [-h] (-i image_paths [image_paths ...] | -f inputs_file)
                     [-c CSV_FILEPATH] [-m MC_RUNS] [-q QUEUE_MAX_ITEMS]
                     [-s SEED] [-t THRESHOLD] [-v] [-r] [-d {gpu,cpu}]

Image processing script

optional arguments:
  -h, --help            show this help message and exit
  -i image_paths [image_paths ...], --image_paths image_paths [image_paths ...]
                        List of image paths to process
  -f inputs_file, --inputs_file inputs_file
                        Path to a file containing a list of image paths
  -c CSV_FILEPATH, --csv_filepath CSV_FILEPATH
                        Optional path to a CSV file
  -m MC_RUNS, --mc_runs MC_RUNS
                        Number of Monte Carlo runs
  -q QUEUE_MAX_ITEMS, --queue_max_items QUEUE_MAX_ITEMS
                        Number of items in the queue
  -s SEED, --seed SEED  Seed for random generation
  -t THRESHOLD, --threshold THRESHOLD
                        Score threshold for detecting artifact
  -v, --verbose         Verbose output if set
  -r, --no-reorient     No reorient input images
  -d {gpu,cpu}, --device {gpu,cpu}
                        Device to use for inference
```