#include <cuda_runtime.h>
#include "utils.cuh"
#include "naive_rmsnorm.cuh"
#include "optim_rms.cuh"


int main(){
  
  for(int i = 0 ; i < 5 ; i++){
    int M = 1024 * (i + 1);
    int N = 1024 * (i + 1);
    timeIt(launchNaiveRms , M , N, "NaiveRmsNorm");
    timeIt(launchOptimRms , M , N , "OptimRmsNorm");
  }

}