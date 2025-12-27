# 대규모 데이터를 이용해 모델을 연합학습하는 방법 연구
## Large-Scale-Data-FL (Year 2)

## Setup and Run Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AdibaOrz/Large-Scale-Data-FL.git
   cd Large-Scale-Data-FL
   ```
2. **Set up the conda environment from the .yml file**:
   ```bash
   conda env create -f medai_fl.yml
   conda activate fb-torch-conda
   ```
3. **Run the example shell script:** 
   ```bash
   bash cifar10_fl/run_y2.sh
   ```