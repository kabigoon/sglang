#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
