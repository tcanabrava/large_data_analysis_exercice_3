#!/bin/bash
#SBATCH -J ex3_test_prob3
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --time=8:00:00
#SBATCH --output=/mnt/aiongpfs/users/jcastroribeiro/Semester2/BigDataAnalytics/Exercise_3/large_data_analysis_exercice_3/logs/test_prob3_hpc_%j.out
#SBATCH --error=/mnt/aiongpfs/users/jcastroribeiro/Semester2/BigDataAnalytics/Exercise_3/large_data_analysis_exercice_3/logs/test_prob3_hpc_%j.err

set -euo pipefail

REPO=/mnt/aiongpfs/users/jcastroribeiro/Semester2/BigDataAnalytics/Exercise_3/large_data_analysis_exercice_3
mkdir -p "$REPO/logs"
cd "$REPO"

module load env/development/2024a
module load tools/EasyBuild/5.2.0
module use $HOME/.local/easybuild/modules/all
module load devel/Spark/4.0.1-foss-2024a-Java-21
module load devel/Maven/3.9.7

spark-submit problem_3_gearbox.py \
  --data-path "/home/users/jcastroribeiro/Semester2/BigDataAnalytics/Exercise_3/large_data_analysis_exercice_3/Data/gearbox/*.csv" \
  --k-min 2 --k-max 12 \
  --top-outliers 25 \


