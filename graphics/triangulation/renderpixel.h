#include "teg_cuda_runtime.h"
__device__ generic_array<float,3> renderpixelbody_block_2(float tc0_14_1,float tc1_15_1,float tc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float x_17_1,float y_18_1,float ty3_9_1,float tx3_6_1){
	generic_array<float,3> _t_4;
	float _t_5;
	float _t_6;
	float _t_7;
	float _t_8;
	float _t_9;
	float _t_10;
	float _t_11;
	float _t_12;
	float _t_13;
	float _t_14;
	float _t_15;
	float _t_16;
	bool _t_17;
	float _t_18;
	float _t_19;
	float _t_20;
	float _t_21;
	float _t_22;
	float _t_23;
	float _t_24;
	float _t_25;
	float _t_26;
	float _t_27;
	float _t_28;
	float _t_29;
	float _t_30;
	bool _t_31;
	float _t_32;
	float _t_33;
	float _t_34;
	float _t_35;
	float _t_36;
	float _t_37;
	float _t_38;
	float _t_39;
	float _t_40;
	float _t_41;
	float _t_42;
	float _t_43;
	float _t_44;
	float _t_45;
	bool _t_46;
	float _t_47;
	float _t_48;

	generic_array<float,3> _t_3;

	_t_4[0] =tc0_14_1;
_t_4[1] =tc1_15_1;
_t_4[2] =tc2_16_1;
	_t_5 = tx1_4_1 * ty2_8_1;
	_t_6 = tx2_5_1 * ty1_7_1;
	_t_7 = _t_6 * -1.0f;
	_t_8 = _t_5 + _t_7;
	_t_9 = -1.0f * ty2_8_1;
	_t_10 = ty1_7_1 + _t_9;
	_t_11 = _t_10 * x_17_1;
	_t_12 = _t_8 + _t_11;
	_t_13 = -1.0f * tx1_4_1;
	_t_14 = tx2_5_1 + _t_13;
	_t_15 = _t_14 * y_18_1;
	_t_16 = _t_12 + _t_15;
	_t_17 = _t_16 < 0.0f;
	if(_t_17)
		{
		
			_t_18 = 1.0f;
		
		}
else
		{
		
			_t_18 = 0.0f;
		
		}

	_t_19 = tx2_5_1 * ty3_9_1;
	_t_20 = tx3_6_1 * ty2_8_1;
	_t_21 = _t_20 * -1.0f;
	_t_22 = _t_19 + _t_21;
	_t_23 = -1.0f * ty3_9_1;
	_t_24 = ty2_8_1 + _t_23;
	_t_25 = _t_24 * x_17_1;
	_t_26 = _t_22 + _t_25;
	_t_27 = -1.0f * tx2_5_1;
	_t_28 = tx3_6_1 + _t_27;
	_t_29 = _t_28 * y_18_1;
	_t_30 = _t_26 + _t_29;
	_t_31 = _t_30 < 0.0f;
	if(_t_31)
		{
		
			_t_32 = 1.0f;
		
		}
else
		{
		
			_t_32 = 0.0f;
		
		}

	_t_33 = _t_18 * _t_32;
	_t_34 = tx3_6_1 * ty1_7_1;
	_t_35 = tx1_4_1 * ty3_9_1;
	_t_36 = _t_35 * -1.0f;
	_t_37 = _t_34 + _t_36;
	_t_38 = -1.0f * ty1_7_1;
	_t_39 = ty3_9_1 + _t_38;
	_t_40 = _t_39 * x_17_1;
	_t_41 = _t_37 + _t_40;
	_t_42 = -1.0f * tx3_6_1;
	_t_43 = tx1_4_1 + _t_42;
	_t_44 = _t_43 * y_18_1;
	_t_45 = _t_41 + _t_44;
	_t_46 = _t_45 < 0.0f;
	if(_t_46)
		{
		
			_t_47 = 1.0f;
		
		}
else
		{
		
			_t_47 = 0.0f;
		
		}

	_t_48 = _t_33 * _t_47;
	for(int __iter__ = 0; __iter__ < 3; __iter__++)   _t_3[__iter__] = _t_4[__iter__] * _t_48;

	return _t_3;
}
__device__ generic_array<float,3> renderpixelintegrator_2(float px0_10_1,float ty2_8_1,float tc0_14_1,float ty1_7_1,float tx3_6_1,float ty3_9_1,float px1_11_1,float tx2_5_1,float tc1_15_1,float tc2_16_1,float y_18_1,float tx1_4_1){
    float x_17_1;
    generic_array<float,3> __output__ = 0;
    float __step__ = ((float)(px1_11_1 - px0_10_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        x_17_1 = px0_10_1 + __step__ * (i + (float)(0.5));
        generic_array<float,3> _t_3;
		_t_3 = renderpixelbody_block_2(tc0_14_1,tc1_15_1,tc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,x_17_1,y_18_1,ty3_9_1,tx3_6_1);;
        __output__ = __output__ + _t_3 * __step__;
    }
    return __output__;
}
__device__ generic_array<float,3> renderpixelbody_block_1(float px1_11_1,float px0_10_1,float tc0_14_1,float tc1_15_1,float tc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float y_18_1,float ty3_9_1,float tx3_6_1){

	generic_array<float,3> _t_1;

	_t_1 = renderpixelintegrator_2(px0_10_1,ty2_8_1,tc0_14_1,ty1_7_1,tx3_6_1,ty3_9_1,px1_11_1,tx2_5_1,tc1_15_1,tc2_16_1,y_18_1,tx1_4_1);

	return _t_1;
}
__device__ generic_array<float,3> renderpixelintegrator_1(float px0_10_1,float tc0_14_1,float ty2_8_1,float ty1_7_1,float tx3_6_1,float px1_11_1,float ty3_9_1,float py0_12_1,float tx2_5_1,float tc1_15_1,float py1_13_1,float tc2_16_1,float tx1_4_1){
    float y_18_1;
    generic_array<float,3> __output__ = 0;
    float __step__ = ((float)(py1_13_1 - py0_12_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y_18_1 = py0_12_1 + __step__ * (i + (float)(0.5));
        generic_array<float,3> _t_1;
		_t_1 = renderpixelbody_block_1(px1_11_1,px0_10_1,tc0_14_1,tc1_15_1,tc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,y_18_1,ty3_9_1,tx3_6_1);;
        __output__ = __output__ + _t_1 * __step__;
    }
    return __output__;
}
__device__ generic_array<float,3> renderpixel(float tx1_4_1,float ty1_7_1,float tx2_5_1,float ty2_8_1,float tx3_6_1,float ty3_9_1,float px0_10_1,float px1_11_1,float py0_12_1,float py1_13_1,float tc0_14_1,float tc1_15_1,float tc2_16_1){

	generic_array<float,3> _t_2;

	_t_2 = renderpixelintegrator_1(px0_10_1,tc0_14_1,ty2_8_1,ty1_7_1,tx3_6_1,px1_11_1,ty3_9_1,py0_12_1,tx2_5_1,tc1_15_1,py1_13_1,tc2_16_1,tx1_4_1);

	return _t_2;
}
