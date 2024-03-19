# change directory 
cd "$(dirname "$0")" 

# activate virtual environment
source ../../env/bin/activate

# run all analysis
echo -e "[INFO:] RUNNING PCA ..."
python analysis/run_pca.py

echo -e "[INFO:] COMPUTING DISTANCES ..."
python analysis/run_distance.py 

echo -e "[INFO:] PLOTTING DISTANCES ..."
python analysis/plot_distance.py

echo -e "[INFO:] EXTRACTING MEDIAN DISTANCES ..."
python analysis/median_distance.py