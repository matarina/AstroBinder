"""已验证的各步骤计算脚本(从 pipeline_code/ 原样复制,逻辑不改)。

每个脚本都是独立可执行入口,各 step 通过 subprocess 调用它们,
脚本内部硬编码了本机所需的环境路径(Rosetta / RFdiffusion / ProteinMPNN / AF3)。
"""
