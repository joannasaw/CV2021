#!/bin/sh
cd CODE
python pipeline_test.py --model resnet_lstm
python pipeline_test.py --model resnet_fpm_lstm
python pipeline_test.py --model resnet_lstm_attention
python pipeline_test.py --model resnet_fpm_lstm_attention
python pipeline_test.py --model resnet_fpm_blstm_attention
