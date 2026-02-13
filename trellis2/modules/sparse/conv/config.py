SPCONV_ALGO = 'auto'                                # 'auto', 'implicit_gemm', 'native'
FLEX_GEMM_ALGO = 'masked_implicit_gemm_splitk'      # 'explicit_gemm', 'implicit_gemm', 'implicit_gemm_splitk', 'masked_implicit_gemm', 'masked_implicit_gemm_splitk'
FLEX_GEMM_HASHMAP_RATIO = 2.0                       # Ratio of hashmap size to input size
FLEX_GEMM_SPLITK_SM_MULTIPLIER = 32                # Upper bound for SPLITK autotuning: MAX_NUM_BLOCKS = multiplier * num_SMs (default in FlexGEMM is 32)
