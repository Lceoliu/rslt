if [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
elif ls -d /usr/local/cuda-*/bin >/dev/null 2>&1; then
  CUDA_BIN=$(ls -d /usr/local/cuda-*/bin | sort -V | tail -1)
  export PATH="$CUDA_BIN:$PATH"
  export LD_LIBRARY_PATH="${CUDA_BIN%/bin}/lib64:$LD_LIBRARY_PATH"
fi