# change directory 
cd "$(dirname "$0")" 

echo -e "[INFO:] RUNNING PCA ..."
python run_pca.py

echo -e "[INFO:] COMPUTING DISTANCES ..."
python run_distance.py 

echo -e "[INFO:] PLOTTING DISTANCES ..."
python plot_distance.py

echo -e "[INFO:] EXTRACTING MEDIAN DISTANCES ..."
python median_distance.py