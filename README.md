# EVM_color
   复现了欧拉运动放大的matlab的原码，BGR转NTSC，构建高斯金字塔，理想带通滤波，这三步中每一步结果都与matlab运行的结果几乎一致(可以精确到小数点后几位)，在最后一步重构时，需要将滤波放大的结果resize到
原始图像大小，这一步可能与matlab不太一样，因为opencv 没有像matlab一样的三次样条插值，但不会影响总体的结果。在未来的不久，也将会实现欧拉运动放大的python 版本
