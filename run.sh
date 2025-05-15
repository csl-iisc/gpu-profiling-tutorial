./naive_gemm 32768

nsys profile --trace=nvtx,cuda,osrt --stats=true --force-overwrite true \
    -o naive_gemm_report ./naive_gemm 32768

nsys stats naive_gemm_report.nsys-rep

nsys profile --trace=nvtx,cuda,osrt --stats=true --force-overwrite true \
    -o pinned_gemm_report ./pinned_gemm 32768

nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --force-overwrite true \
    -o nvtx_gemm_report ./nvtx_gemm 32768

nsys stats pinned_gemm_report.nsys-rep
nsys stats nvtx_gemm_report.nsys-rep 

ncu --target-processes all --set full \
    -o pinned_gemm_report ./pinned_gemm 32768

ncu --target-processes all --set full \
    -o tiled_gemm_report ./tiled_gemm 32768