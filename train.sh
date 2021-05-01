#!/bin/sh
cd CODE
python pipeline_test.py --model vgg16_lstm
python pipeline_test.py --model vgg16_fpm_lstm
python pipeline_test.py --model vgg16_lstm_attention
python pipeline_test.py --model vgg16_fpm_lstm_attention
python pipeline_test.py --model vgg16_fpm_blstm_attention
