#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
