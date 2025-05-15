naive_gemm:
	nvcc -lineinfo -o naive_gemm gemm.cu 

pinned_gemm:
	nvcc -lineinfo -o pinned_gemm pinned_gemm.cu

nvtx_gemm:
	nvcc -lineinfo -o nvtx_gemm nvtx_gemm.cu -lnvToolsExt

tiled_gemm: 
	nvcc -lineinfo -o tiled_gemm tiled_gemm.cu -lnvToolsExt

all: naive_gemm pinned_gemm tiled_gemm nvtx_gemm
	make naive_gemm
	make pinned_gemm
	make tiled_gemm
	make nvtx_gemm
	@echo "All targets built successfully."

clean:
	rm -f naive_gemm pinned_gemm tiled_gemm nvtx_gemm