#include "teg_cuda_runtime.h"
__device__ float tegpixelbody_block_1(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float px0_10_1,float y_21_1,float ty3_9_1,float tx3_6_1){
	float _t_599;
	float _t_600;
	float _t_601;
	float _t_602;
	float _t_603;
	float _t_604;
	float _t_605;
	float _t_606;
	float _t_607;
	float _t_608;
	float _t_609;
	float _t_610;
	float _t_611;
	float _t_612;
	float _t_613;
	float _t_614;
	float _t_615;
	float _t_616;
	float _t_617;
	float _t_618;
	float _t_619;
	float _t_620;
	float _t_621;
	bool _t_622;
	float _t_623;
	float _t_624;
	float _t_625;
	float _t_626;
	float _t_627;
	float _t_628;
	float _t_629;
	float _t_630;
	float _t_631;
	float _t_632;
	float _t_633;
	float _t_634;
	float _t_635;
	bool _t_636;
	float _t_637;
	float _t_638;
	float _t_639;
	float _t_640;
	float _t_641;
	float _t_642;
	float _t_643;
	float _t_644;
	float _t_645;
	float _t_646;
	float _t_647;
	float _t_648;
	float _t_649;
	float _t_650;
	bool _t_651;
	float _t_652;
	float _t_653;
	float _t_654;

	float _t_1;

	_t_599 = -1.0f * pc0_14_1;
	_t_600 = tc0_17_1 + _t_599;
	_t_601 = _t_600 * _t_600;
	_t_602 = -1.0f * pc1_15_1;
	_t_603 = tc1_18_1 + _t_602;
	_t_604 = _t_603 * _t_603;
	_t_605 = _t_601 + _t_604;
	_t_606 = -1.0f * pc2_16_1;
	_t_607 = tc2_19_1 + _t_606;
	_t_608 = _t_607 * _t_607;
	_t_609 = _t_605 + _t_608;
	_t_610 = tx1_4_1 * ty2_8_1;
	_t_611 = tx2_5_1 * ty1_7_1;
	_t_612 = _t_611 * -1.0f;
	_t_613 = _t_610 + _t_612;
	_t_614 = -1.0f * ty2_8_1;
	_t_615 = ty1_7_1 + _t_614;
	_t_616 = _t_615 * px0_10_1;
	_t_617 = _t_613 + _t_616;
	_t_618 = -1.0f * tx1_4_1;
	_t_619 = tx2_5_1 + _t_618;
	_t_620 = _t_619 * y_21_1;
	_t_621 = _t_617 + _t_620;
	_t_622 = _t_621 < 0.0f;
	if(_t_622)
		{
		
			_t_623 = 1.0f;
		
		}
else
		{
		
			_t_623 = 0.0f;
		
		}

	_t_624 = tx2_5_1 * ty3_9_1;
	_t_625 = tx3_6_1 * ty2_8_1;
	_t_626 = _t_625 * -1.0f;
	_t_627 = _t_624 + _t_626;
	_t_628 = -1.0f * ty3_9_1;
	_t_629 = ty2_8_1 + _t_628;
	_t_630 = _t_629 * px0_10_1;
	_t_631 = _t_627 + _t_630;
	_t_632 = -1.0f * tx2_5_1;
	_t_633 = tx3_6_1 + _t_632;
	_t_634 = _t_633 * y_21_1;
	_t_635 = _t_631 + _t_634;
	_t_636 = _t_635 < 0.0f;
	if(_t_636)
		{
		
			_t_637 = 1.0f;
		
		}
else
		{
		
			_t_637 = 0.0f;
		
		}

	_t_638 = _t_623 * _t_637;
	_t_639 = tx3_6_1 * ty1_7_1;
	_t_640 = tx1_4_1 * ty3_9_1;
	_t_641 = _t_640 * -1.0f;
	_t_642 = _t_639 + _t_641;
	_t_643 = -1.0f * ty1_7_1;
	_t_644 = ty3_9_1 + _t_643;
	_t_645 = _t_644 * px0_10_1;
	_t_646 = _t_642 + _t_645;
	_t_647 = -1.0f * tx3_6_1;
	_t_648 = tx1_4_1 + _t_647;
	_t_649 = _t_648 * y_21_1;
	_t_650 = _t_646 + _t_649;
	_t_651 = _t_650 < 0.0f;
	if(_t_651)
		{
		
			_t_652 = 1.0f;
		
		}
else
		{
		
			_t_652 = 0.0f;
		
		}

	_t_653 = _t_638 * _t_652;
	_t_654 = _t_609 * _t_653;
	_t_1 = _t_654 * -1.0f;

	return _t_1;
}
__device__ float tegpixelintegrator_1(float tx1_4_1,float tx3_6_1,float pc1_15_1,float pc2_16_1,float tx2_5_1,float ty3_9_1,float tc0_17_1,float py1_13_1,float py0_12_1,float ty1_7_1,float tc2_19_1,float tc1_18_1,float px0_10_1,float ty2_8_1,float pc0_14_1){
    float y_21_1;
    float __output__ = 0;
    float __step__ = ((float)(py1_13_1 - py0_12_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y_21_1 = py0_12_1 + __step__ * (i + (float)(0.5));
        float _t_1;
		_t_1 = tegpixelbody_block_1(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,px0_10_1,y_21_1,ty3_9_1,tx3_6_1);;
        __output__ = __output__ + _t_1 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_2(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float px1_11_1,float y_21_2,float ty3_9_1,float tx3_6_1){
	float _t_655;
	float _t_656;
	float _t_657;
	float _t_658;
	float _t_659;
	float _t_660;
	float _t_661;
	float _t_662;
	float _t_663;
	float _t_664;
	float _t_665;
	float _t_666;
	float _t_667;
	float _t_668;
	float _t_669;
	float _t_670;
	float _t_671;
	float _t_672;
	float _t_673;
	float _t_674;
	float _t_675;
	float _t_676;
	float _t_677;
	bool _t_678;
	float _t_679;
	float _t_680;
	float _t_681;
	float _t_682;
	float _t_683;
	float _t_684;
	float _t_685;
	float _t_686;
	float _t_687;
	float _t_688;
	float _t_689;
	float _t_690;
	float _t_691;
	bool _t_692;
	float _t_693;
	float _t_694;
	float _t_695;
	float _t_696;
	float _t_697;
	float _t_698;
	float _t_699;
	float _t_700;
	float _t_701;
	float _t_702;
	float _t_703;
	float _t_704;
	float _t_705;
	float _t_706;
	bool _t_707;
	float _t_708;
	float _t_709;

	float _t_3;

	_t_655 = -1.0f * pc0_14_1;
	_t_656 = tc0_17_1 + _t_655;
	_t_657 = _t_656 * _t_656;
	_t_658 = -1.0f * pc1_15_1;
	_t_659 = tc1_18_1 + _t_658;
	_t_660 = _t_659 * _t_659;
	_t_661 = _t_657 + _t_660;
	_t_662 = -1.0f * pc2_16_1;
	_t_663 = tc2_19_1 + _t_662;
	_t_664 = _t_663 * _t_663;
	_t_665 = _t_661 + _t_664;
	_t_666 = tx1_4_1 * ty2_8_1;
	_t_667 = tx2_5_1 * ty1_7_1;
	_t_668 = _t_667 * -1.0f;
	_t_669 = _t_666 + _t_668;
	_t_670 = -1.0f * ty2_8_1;
	_t_671 = ty1_7_1 + _t_670;
	_t_672 = _t_671 * px1_11_1;
	_t_673 = _t_669 + _t_672;
	_t_674 = -1.0f * tx1_4_1;
	_t_675 = tx2_5_1 + _t_674;
	_t_676 = _t_675 * y_21_2;
	_t_677 = _t_673 + _t_676;
	_t_678 = _t_677 < 0.0f;
	if(_t_678)
		{
		
			_t_679 = 1.0f;
		
		}
else
		{
		
			_t_679 = 0.0f;
		
		}

	_t_680 = tx2_5_1 * ty3_9_1;
	_t_681 = tx3_6_1 * ty2_8_1;
	_t_682 = _t_681 * -1.0f;
	_t_683 = _t_680 + _t_682;
	_t_684 = -1.0f * ty3_9_1;
	_t_685 = ty2_8_1 + _t_684;
	_t_686 = _t_685 * px1_11_1;
	_t_687 = _t_683 + _t_686;
	_t_688 = -1.0f * tx2_5_1;
	_t_689 = tx3_6_1 + _t_688;
	_t_690 = _t_689 * y_21_2;
	_t_691 = _t_687 + _t_690;
	_t_692 = _t_691 < 0.0f;
	if(_t_692)
		{
		
			_t_693 = 1.0f;
		
		}
else
		{
		
			_t_693 = 0.0f;
		
		}

	_t_694 = _t_679 * _t_693;
	_t_695 = tx3_6_1 * ty1_7_1;
	_t_696 = tx1_4_1 * ty3_9_1;
	_t_697 = _t_696 * -1.0f;
	_t_698 = _t_695 + _t_697;
	_t_699 = -1.0f * ty1_7_1;
	_t_700 = ty3_9_1 + _t_699;
	_t_701 = _t_700 * px1_11_1;
	_t_702 = _t_698 + _t_701;
	_t_703 = -1.0f * tx3_6_1;
	_t_704 = tx1_4_1 + _t_703;
	_t_705 = _t_704 * y_21_2;
	_t_706 = _t_702 + _t_705;
	_t_707 = _t_706 < 0.0f;
	if(_t_707)
		{
		
			_t_708 = 1.0f;
		
		}
else
		{
		
			_t_708 = 0.0f;
		
		}

	_t_709 = _t_694 * _t_708;
	_t_3 = _t_665 * _t_709;

	return _t_3;
}
__device__ float tegpixelintegrator_2(float tx1_4_1,float tx3_6_1,float pc1_15_1,float pc2_16_1,float tx2_5_1,float ty3_9_1,float tc0_17_1,float px1_11_1,float py1_13_1,float py0_12_1,float ty1_7_1,float tc2_19_1,float tc1_18_1,float ty2_8_1,float pc0_14_1){
    float y_21_2;
    float __output__ = 0;
    float __step__ = ((float)(py1_13_1 - py0_12_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y_21_2 = py0_12_1 + __step__ * (i + (float)(0.5));
        float _t_3;
		_t_3 = tegpixelbody_block_2(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,px1_11_1,y_21_2,ty3_9_1,tx3_6_1);;
        __output__ = __output__ + _t_3 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_3(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float x_20_1,float py0_12_1,float ty3_9_1,float tx3_6_1){
	float _t_710;
	float _t_711;
	float _t_712;
	float _t_713;
	float _t_714;
	float _t_715;
	float _t_716;
	float _t_717;
	float _t_718;
	float _t_719;
	float _t_720;
	float _t_721;
	float _t_722;
	float _t_723;
	float _t_724;
	float _t_725;
	float _t_726;
	float _t_727;
	float _t_728;
	float _t_729;
	float _t_730;
	float _t_731;
	float _t_732;
	bool _t_733;
	float _t_734;
	float _t_735;
	float _t_736;
	float _t_737;
	float _t_738;
	float _t_739;
	float _t_740;
	float _t_741;
	float _t_742;
	float _t_743;
	float _t_744;
	float _t_745;
	float _t_746;
	bool _t_747;
	float _t_748;
	float _t_749;
	float _t_750;
	float _t_751;
	float _t_752;
	float _t_753;
	float _t_754;
	float _t_755;
	float _t_756;
	float _t_757;
	float _t_758;
	float _t_759;
	float _t_760;
	float _t_761;
	bool _t_762;
	float _t_763;
	float _t_764;

	float _t_5;

	_t_710 = -1.0f * pc0_14_1;
	_t_711 = tc0_17_1 + _t_710;
	_t_712 = _t_711 * _t_711;
	_t_713 = -1.0f * pc1_15_1;
	_t_714 = tc1_18_1 + _t_713;
	_t_715 = _t_714 * _t_714;
	_t_716 = _t_712 + _t_715;
	_t_717 = -1.0f * pc2_16_1;
	_t_718 = tc2_19_1 + _t_717;
	_t_719 = _t_718 * _t_718;
	_t_720 = _t_716 + _t_719;
	_t_721 = tx1_4_1 * ty2_8_1;
	_t_722 = tx2_5_1 * ty1_7_1;
	_t_723 = _t_722 * -1.0f;
	_t_724 = _t_721 + _t_723;
	_t_725 = -1.0f * ty2_8_1;
	_t_726 = ty1_7_1 + _t_725;
	_t_727 = _t_726 * x_20_1;
	_t_728 = _t_724 + _t_727;
	_t_729 = -1.0f * tx1_4_1;
	_t_730 = tx2_5_1 + _t_729;
	_t_731 = _t_730 * py0_12_1;
	_t_732 = _t_728 + _t_731;
	_t_733 = _t_732 < 0.0f;
	if(_t_733)
		{
		
			_t_734 = 1.0f;
		
		}
else
		{
		
			_t_734 = 0.0f;
		
		}

	_t_735 = tx2_5_1 * ty3_9_1;
	_t_736 = tx3_6_1 * ty2_8_1;
	_t_737 = _t_736 * -1.0f;
	_t_738 = _t_735 + _t_737;
	_t_739 = -1.0f * ty3_9_1;
	_t_740 = ty2_8_1 + _t_739;
	_t_741 = _t_740 * x_20_1;
	_t_742 = _t_738 + _t_741;
	_t_743 = -1.0f * tx2_5_1;
	_t_744 = tx3_6_1 + _t_743;
	_t_745 = _t_744 * py0_12_1;
	_t_746 = _t_742 + _t_745;
	_t_747 = _t_746 < 0.0f;
	if(_t_747)
		{
		
			_t_748 = 1.0f;
		
		}
else
		{
		
			_t_748 = 0.0f;
		
		}

	_t_749 = _t_734 * _t_748;
	_t_750 = tx3_6_1 * ty1_7_1;
	_t_751 = tx1_4_1 * ty3_9_1;
	_t_752 = _t_751 * -1.0f;
	_t_753 = _t_750 + _t_752;
	_t_754 = -1.0f * ty1_7_1;
	_t_755 = ty3_9_1 + _t_754;
	_t_756 = _t_755 * x_20_1;
	_t_757 = _t_753 + _t_756;
	_t_758 = -1.0f * tx3_6_1;
	_t_759 = tx1_4_1 + _t_758;
	_t_760 = _t_759 * py0_12_1;
	_t_761 = _t_757 + _t_760;
	_t_762 = _t_761 < 0.0f;
	if(_t_762)
		{
		
			_t_763 = 1.0f;
		
		}
else
		{
		
			_t_763 = 0.0f;
		
		}

	_t_764 = _t_749 * _t_763;
	_t_5 = _t_720 * _t_764;

	return _t_5;
}
__device__ float tegpixelintegrator_3(float tx1_4_1,float tx3_6_1,float pc1_15_1,float pc2_16_1,float tx2_5_1,float ty3_9_1,float tc0_17_1,float px1_11_1,float py0_12_1,float ty1_7_1,float tc2_19_1,float tc1_18_1,float px0_10_1,float ty2_8_1,float pc0_14_1){
    float x_20_1;
    float __output__ = 0;
    float __step__ = ((float)(px1_11_1 - px0_10_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        x_20_1 = px0_10_1 + __step__ * (i + (float)(0.5));
        float _t_5;
		_t_5 = tegpixelbody_block_3(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,x_20_1,py0_12_1,ty3_9_1,tx3_6_1);;
        __output__ = __output__ + _t_5 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_4(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float x_20_2,float py1_13_1,float ty3_9_1,float tx3_6_1){
	float _t_765;
	float _t_766;
	float _t_767;
	float _t_768;
	float _t_769;
	float _t_770;
	float _t_771;
	float _t_772;
	float _t_773;
	float _t_774;
	float _t_775;
	float _t_776;
	float _t_777;
	float _t_778;
	float _t_779;
	float _t_780;
	float _t_781;
	float _t_782;
	float _t_783;
	float _t_784;
	float _t_785;
	float _t_786;
	float _t_787;
	bool _t_788;
	float _t_789;
	float _t_790;
	float _t_791;
	float _t_792;
	float _t_793;
	float _t_794;
	float _t_795;
	float _t_796;
	float _t_797;
	float _t_798;
	float _t_799;
	float _t_800;
	float _t_801;
	bool _t_802;
	float _t_803;
	float _t_804;
	float _t_805;
	float _t_806;
	float _t_807;
	float _t_808;
	float _t_809;
	float _t_810;
	float _t_811;
	float _t_812;
	float _t_813;
	float _t_814;
	float _t_815;
	float _t_816;
	bool _t_817;
	float _t_818;
	float _t_819;

	float _t_8;

	_t_765 = -1.0f * pc0_14_1;
	_t_766 = tc0_17_1 + _t_765;
	_t_767 = _t_766 * _t_766;
	_t_768 = -1.0f * pc1_15_1;
	_t_769 = tc1_18_1 + _t_768;
	_t_770 = _t_769 * _t_769;
	_t_771 = _t_767 + _t_770;
	_t_772 = -1.0f * pc2_16_1;
	_t_773 = tc2_19_1 + _t_772;
	_t_774 = _t_773 * _t_773;
	_t_775 = _t_771 + _t_774;
	_t_776 = tx1_4_1 * ty2_8_1;
	_t_777 = tx2_5_1 * ty1_7_1;
	_t_778 = _t_777 * -1.0f;
	_t_779 = _t_776 + _t_778;
	_t_780 = -1.0f * ty2_8_1;
	_t_781 = ty1_7_1 + _t_780;
	_t_782 = _t_781 * x_20_2;
	_t_783 = _t_779 + _t_782;
	_t_784 = -1.0f * tx1_4_1;
	_t_785 = tx2_5_1 + _t_784;
	_t_786 = _t_785 * py1_13_1;
	_t_787 = _t_783 + _t_786;
	_t_788 = _t_787 < 0.0f;
	if(_t_788)
		{
		
			_t_789 = 1.0f;
		
		}
else
		{
		
			_t_789 = 0.0f;
		
		}

	_t_790 = tx2_5_1 * ty3_9_1;
	_t_791 = tx3_6_1 * ty2_8_1;
	_t_792 = _t_791 * -1.0f;
	_t_793 = _t_790 + _t_792;
	_t_794 = -1.0f * ty3_9_1;
	_t_795 = ty2_8_1 + _t_794;
	_t_796 = _t_795 * x_20_2;
	_t_797 = _t_793 + _t_796;
	_t_798 = -1.0f * tx2_5_1;
	_t_799 = tx3_6_1 + _t_798;
	_t_800 = _t_799 * py1_13_1;
	_t_801 = _t_797 + _t_800;
	_t_802 = _t_801 < 0.0f;
	if(_t_802)
		{
		
			_t_803 = 1.0f;
		
		}
else
		{
		
			_t_803 = 0.0f;
		
		}

	_t_804 = _t_789 * _t_803;
	_t_805 = tx3_6_1 * ty1_7_1;
	_t_806 = tx1_4_1 * ty3_9_1;
	_t_807 = _t_806 * -1.0f;
	_t_808 = _t_805 + _t_807;
	_t_809 = -1.0f * ty1_7_1;
	_t_810 = ty3_9_1 + _t_809;
	_t_811 = _t_810 * x_20_2;
	_t_812 = _t_808 + _t_811;
	_t_813 = -1.0f * tx3_6_1;
	_t_814 = tx1_4_1 + _t_813;
	_t_815 = _t_814 * py1_13_1;
	_t_816 = _t_812 + _t_815;
	_t_817 = _t_816 < 0.0f;
	if(_t_817)
		{
		
			_t_818 = 1.0f;
		
		}
else
		{
		
			_t_818 = 0.0f;
		
		}

	_t_819 = _t_804 * _t_818;
	_t_8 = _t_775 * _t_819;

	return _t_8;
}
__device__ float tegpixelintegrator_4(float tx1_4_1,float tx3_6_1,float pc1_15_1,float pc2_16_1,float tx2_5_1,float py1_13_1,float tc0_17_1,float ty3_9_1,float px1_11_1,float ty1_7_1,float tc2_19_1,float tc1_18_1,float px0_10_1,float ty2_8_1,float pc0_14_1){
    float x_20_2;
    float __output__ = 0;
    float __step__ = ((float)(px1_11_1 - px0_10_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        x_20_2 = px0_10_1 + __step__ * (i + (float)(0.5));
        float _t_8;
		_t_8 = tegpixelbody_block_4(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,x_20_2,py1_13_1,ty3_9_1,tx3_6_1);;
        __output__ = __output__ + _t_8 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_11(float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float x_20_3,float y_21_3,float ty3_9_1,float tx3_6_1,float tc0_17_1,float pc0_14_1){
	float _t_821;
	float _t_822;
	float _t_823;
	float _t_824;
	float _t_825;
	float _t_826;
	float _t_827;
	float _t_828;
	float _t_829;
	float _t_830;
	float _t_831;
	float _t_832;
	bool _t_833;
	float _t_834;
	float _t_835;
	float _t_836;
	float _t_837;
	float _t_838;
	float _t_839;
	float _t_840;
	float _t_841;
	float _t_842;
	float _t_843;
	float _t_844;
	float _t_845;
	float _t_846;
	bool _t_847;
	float _t_848;
	float _t_849;
	float _t_850;
	float _t_851;
	float _t_852;
	float _t_853;
	float _t_854;
	float _t_855;
	float _t_856;
	float _t_857;
	float _t_858;
	float _t_859;
	float _t_860;
	float _t_861;
	bool _t_862;
	float _t_863;
	float _t_864;
	float _t_865;
	float _t_866;
	float _t_867;
	float _t_868;

	float _t_820;

	_t_821 = tx1_4_1 * ty2_8_1;
	_t_822 = tx2_5_1 * ty1_7_1;
	_t_823 = _t_822 * -1.0f;
	_t_824 = _t_821 + _t_823;
	_t_825 = -1.0f * ty2_8_1;
	_t_826 = ty1_7_1 + _t_825;
	_t_827 = _t_826 * x_20_3;
	_t_828 = _t_824 + _t_827;
	_t_829 = -1.0f * tx1_4_1;
	_t_830 = tx2_5_1 + _t_829;
	_t_831 = _t_830 * y_21_3;
	_t_832 = _t_828 + _t_831;
	_t_833 = _t_832 < 0.0f;
	if(_t_833)
		{
		
			_t_834 = 1.0f;
		
		}
else
		{
		
			_t_834 = 0.0f;
		
		}

	_t_835 = tx2_5_1 * ty3_9_1;
	_t_836 = tx3_6_1 * ty2_8_1;
	_t_837 = _t_836 * -1.0f;
	_t_838 = _t_835 + _t_837;
	_t_839 = -1.0f * ty3_9_1;
	_t_840 = ty2_8_1 + _t_839;
	_t_841 = _t_840 * x_20_3;
	_t_842 = _t_838 + _t_841;
	_t_843 = -1.0f * tx2_5_1;
	_t_844 = tx3_6_1 + _t_843;
	_t_845 = _t_844 * y_21_3;
	_t_846 = _t_842 + _t_845;
	_t_847 = _t_846 < 0.0f;
	if(_t_847)
		{
		
			_t_848 = 1.0f;
		
		}
else
		{
		
			_t_848 = 0.0f;
		
		}

	_t_849 = _t_834 * _t_848;
	_t_850 = tx3_6_1 * ty1_7_1;
	_t_851 = tx1_4_1 * ty3_9_1;
	_t_852 = _t_851 * -1.0f;
	_t_853 = _t_850 + _t_852;
	_t_854 = -1.0f * ty1_7_1;
	_t_855 = ty3_9_1 + _t_854;
	_t_856 = _t_855 * x_20_3;
	_t_857 = _t_853 + _t_856;
	_t_858 = -1.0f * tx3_6_1;
	_t_859 = tx1_4_1 + _t_858;
	_t_860 = _t_859 * y_21_3;
	_t_861 = _t_857 + _t_860;
	_t_862 = _t_861 < 0.0f;
	if(_t_862)
		{
		
			_t_863 = 1.0f;
		
		}
else
		{
		
			_t_863 = 0.0f;
		
		}

	_t_864 = _t_849 * _t_863;
	_t_865 = _t_864 * 2.0f;
	_t_866 = -1.0f * pc0_14_1;
	_t_867 = tc0_17_1 + _t_866;
	_t_868 = _t_865 * _t_867;
	_t_820 = _t_868 * -1.0f;

	return _t_820;
}
__device__ float tegpixelintegrator_11(float tx1_4_1,float tx3_6_1,float tx2_5_1,float ty3_9_1,float tc0_17_1,float px1_11_1,float pc0_14_1,float ty2_8_1,float px0_10_1,float y_21_3,float ty1_7_1){
    float x_20_3;
    float __output__ = 0;
    float __step__ = ((float)(px1_11_1 - px0_10_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        x_20_3 = px0_10_1 + __step__ * (i + (float)(0.5));
        float _t_820;
		_t_820 = tegpixelbody_block_11(tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,x_20_3,y_21_3,ty3_9_1,tx3_6_1,tc0_17_1,pc0_14_1);;
        __output__ = __output__ + _t_820 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_5(float px1_11_1,float px0_10_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float y_21_3,float ty3_9_1,float tx3_6_1,float tc0_17_1,float pc0_14_1){

	float _t_10;

	_t_10 = tegpixelintegrator_11(tx1_4_1,tx3_6_1,tx2_5_1,ty3_9_1,tc0_17_1,px1_11_1,pc0_14_1,ty2_8_1,px0_10_1,y_21_3,ty1_7_1);

	return _t_10;
}
__device__ float tegpixelintegrator_5(float tx1_4_1,float tx3_6_1,float tx2_5_1,float ty3_9_1,float py1_13_1,float px1_11_1,float tc0_17_1,float py0_12_1,float pc0_14_1,float px0_10_1,float ty2_8_1,float ty1_7_1){
    float y_21_3;
    float __output__ = 0;
    float __step__ = ((float)(py1_13_1 - py0_12_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y_21_3 = py0_12_1 + __step__ * (i + (float)(0.5));
        float _t_10;
		_t_10 = tegpixelbody_block_5(px1_11_1,px0_10_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,y_21_3,ty3_9_1,tx3_6_1,tc0_17_1,pc0_14_1);;
        __output__ = __output__ + _t_10 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_12(float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float x_20_4,float y_21_4,float ty3_9_1,float tx3_6_1,float tc1_18_1,float pc1_15_1){
	float _t_870;
	float _t_871;
	float _t_872;
	float _t_873;
	float _t_874;
	float _t_875;
	float _t_876;
	float _t_877;
	float _t_878;
	float _t_879;
	float _t_880;
	float _t_881;
	bool _t_882;
	float _t_883;
	float _t_884;
	float _t_885;
	float _t_886;
	float _t_887;
	float _t_888;
	float _t_889;
	float _t_890;
	float _t_891;
	float _t_892;
	float _t_893;
	float _t_894;
	float _t_895;
	bool _t_896;
	float _t_897;
	float _t_898;
	float _t_899;
	float _t_900;
	float _t_901;
	float _t_902;
	float _t_903;
	float _t_904;
	float _t_905;
	float _t_906;
	float _t_907;
	float _t_908;
	float _t_909;
	float _t_910;
	bool _t_911;
	float _t_912;
	float _t_913;
	float _t_914;
	float _t_915;
	float _t_916;
	float _t_917;

	float _t_869;

	_t_870 = tx1_4_1 * ty2_8_1;
	_t_871 = tx2_5_1 * ty1_7_1;
	_t_872 = _t_871 * -1.0f;
	_t_873 = _t_870 + _t_872;
	_t_874 = -1.0f * ty2_8_1;
	_t_875 = ty1_7_1 + _t_874;
	_t_876 = _t_875 * x_20_4;
	_t_877 = _t_873 + _t_876;
	_t_878 = -1.0f * tx1_4_1;
	_t_879 = tx2_5_1 + _t_878;
	_t_880 = _t_879 * y_21_4;
	_t_881 = _t_877 + _t_880;
	_t_882 = _t_881 < 0.0f;
	if(_t_882)
		{
		
			_t_883 = 1.0f;
		
		}
else
		{
		
			_t_883 = 0.0f;
		
		}

	_t_884 = tx2_5_1 * ty3_9_1;
	_t_885 = tx3_6_1 * ty2_8_1;
	_t_886 = _t_885 * -1.0f;
	_t_887 = _t_884 + _t_886;
	_t_888 = -1.0f * ty3_9_1;
	_t_889 = ty2_8_1 + _t_888;
	_t_890 = _t_889 * x_20_4;
	_t_891 = _t_887 + _t_890;
	_t_892 = -1.0f * tx2_5_1;
	_t_893 = tx3_6_1 + _t_892;
	_t_894 = _t_893 * y_21_4;
	_t_895 = _t_891 + _t_894;
	_t_896 = _t_895 < 0.0f;
	if(_t_896)
		{
		
			_t_897 = 1.0f;
		
		}
else
		{
		
			_t_897 = 0.0f;
		
		}

	_t_898 = _t_883 * _t_897;
	_t_899 = tx3_6_1 * ty1_7_1;
	_t_900 = tx1_4_1 * ty3_9_1;
	_t_901 = _t_900 * -1.0f;
	_t_902 = _t_899 + _t_901;
	_t_903 = -1.0f * ty1_7_1;
	_t_904 = ty3_9_1 + _t_903;
	_t_905 = _t_904 * x_20_4;
	_t_906 = _t_902 + _t_905;
	_t_907 = -1.0f * tx3_6_1;
	_t_908 = tx1_4_1 + _t_907;
	_t_909 = _t_908 * y_21_4;
	_t_910 = _t_906 + _t_909;
	_t_911 = _t_910 < 0.0f;
	if(_t_911)
		{
		
			_t_912 = 1.0f;
		
		}
else
		{
		
			_t_912 = 0.0f;
		
		}

	_t_913 = _t_898 * _t_912;
	_t_914 = _t_913 * 2.0f;
	_t_915 = -1.0f * pc1_15_1;
	_t_916 = tc1_18_1 + _t_915;
	_t_917 = _t_914 * _t_916;
	_t_869 = _t_917 * -1.0f;

	return _t_869;
}
__device__ float tegpixelintegrator_12(float tx1_4_1,float tx3_6_1,float y_21_4,float tx2_5_1,float ty3_9_1,float pc1_15_1,float px1_11_1,float tc1_18_1,float px0_10_1,float ty2_8_1,float ty1_7_1){
    float x_20_4;
    float __output__ = 0;
    float __step__ = ((float)(px1_11_1 - px0_10_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        x_20_4 = px0_10_1 + __step__ * (i + (float)(0.5));
        float _t_869;
		_t_869 = tegpixelbody_block_12(tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,x_20_4,y_21_4,ty3_9_1,tx3_6_1,tc1_18_1,pc1_15_1);;
        __output__ = __output__ + _t_869 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_6(float px1_11_1,float px0_10_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float y_21_4,float ty3_9_1,float tx3_6_1,float tc1_18_1,float pc1_15_1){

	float _t_12;

	_t_12 = tegpixelintegrator_12(tx1_4_1,tx3_6_1,y_21_4,tx2_5_1,ty3_9_1,pc1_15_1,px1_11_1,tc1_18_1,px0_10_1,ty2_8_1,ty1_7_1);

	return _t_12;
}
__device__ float tegpixelintegrator_6(float tx1_4_1,float tx3_6_1,float tx2_5_1,float ty3_9_1,float pc1_15_1,float py1_13_1,float px1_11_1,float py0_12_1,float tc1_18_1,float px0_10_1,float ty2_8_1,float ty1_7_1){
    float y_21_4;
    float __output__ = 0;
    float __step__ = ((float)(py1_13_1 - py0_12_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y_21_4 = py0_12_1 + __step__ * (i + (float)(0.5));
        float _t_12;
		_t_12 = tegpixelbody_block_6(px1_11_1,px0_10_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,y_21_4,ty3_9_1,tx3_6_1,tc1_18_1,pc1_15_1);;
        __output__ = __output__ + _t_12 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_13(float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float x_20_5,float y_21_5,float ty3_9_1,float tx3_6_1,float tc2_19_1,float pc2_16_1){
	float _t_919;
	float _t_920;
	float _t_921;
	float _t_922;
	float _t_923;
	float _t_924;
	float _t_925;
	float _t_926;
	float _t_927;
	float _t_928;
	float _t_929;
	float _t_930;
	bool _t_931;
	float _t_932;
	float _t_933;
	float _t_934;
	float _t_935;
	float _t_936;
	float _t_937;
	float _t_938;
	float _t_939;
	float _t_940;
	float _t_941;
	float _t_942;
	float _t_943;
	float _t_944;
	bool _t_945;
	float _t_946;
	float _t_947;
	float _t_948;
	float _t_949;
	float _t_950;
	float _t_951;
	float _t_952;
	float _t_953;
	float _t_954;
	float _t_955;
	float _t_956;
	float _t_957;
	float _t_958;
	float _t_959;
	bool _t_960;
	float _t_961;
	float _t_962;
	float _t_963;
	float _t_964;
	float _t_965;
	float _t_966;

	float _t_918;

	_t_919 = tx1_4_1 * ty2_8_1;
	_t_920 = tx2_5_1 * ty1_7_1;
	_t_921 = _t_920 * -1.0f;
	_t_922 = _t_919 + _t_921;
	_t_923 = -1.0f * ty2_8_1;
	_t_924 = ty1_7_1 + _t_923;
	_t_925 = _t_924 * x_20_5;
	_t_926 = _t_922 + _t_925;
	_t_927 = -1.0f * tx1_4_1;
	_t_928 = tx2_5_1 + _t_927;
	_t_929 = _t_928 * y_21_5;
	_t_930 = _t_926 + _t_929;
	_t_931 = _t_930 < 0.0f;
	if(_t_931)
		{
		
			_t_932 = 1.0f;
		
		}
else
		{
		
			_t_932 = 0.0f;
		
		}

	_t_933 = tx2_5_1 * ty3_9_1;
	_t_934 = tx3_6_1 * ty2_8_1;
	_t_935 = _t_934 * -1.0f;
	_t_936 = _t_933 + _t_935;
	_t_937 = -1.0f * ty3_9_1;
	_t_938 = ty2_8_1 + _t_937;
	_t_939 = _t_938 * x_20_5;
	_t_940 = _t_936 + _t_939;
	_t_941 = -1.0f * tx2_5_1;
	_t_942 = tx3_6_1 + _t_941;
	_t_943 = _t_942 * y_21_5;
	_t_944 = _t_940 + _t_943;
	_t_945 = _t_944 < 0.0f;
	if(_t_945)
		{
		
			_t_946 = 1.0f;
		
		}
else
		{
		
			_t_946 = 0.0f;
		
		}

	_t_947 = _t_932 * _t_946;
	_t_948 = tx3_6_1 * ty1_7_1;
	_t_949 = tx1_4_1 * ty3_9_1;
	_t_950 = _t_949 * -1.0f;
	_t_951 = _t_948 + _t_950;
	_t_952 = -1.0f * ty1_7_1;
	_t_953 = ty3_9_1 + _t_952;
	_t_954 = _t_953 * x_20_5;
	_t_955 = _t_951 + _t_954;
	_t_956 = -1.0f * tx3_6_1;
	_t_957 = tx1_4_1 + _t_956;
	_t_958 = _t_957 * y_21_5;
	_t_959 = _t_955 + _t_958;
	_t_960 = _t_959 < 0.0f;
	if(_t_960)
		{
		
			_t_961 = 1.0f;
		
		}
else
		{
		
			_t_961 = 0.0f;
		
		}

	_t_962 = _t_947 * _t_961;
	_t_963 = _t_962 * 2.0f;
	_t_964 = -1.0f * pc2_16_1;
	_t_965 = tc2_19_1 + _t_964;
	_t_966 = _t_963 * _t_965;
	_t_918 = _t_966 * -1.0f;

	return _t_918;
}
__device__ float tegpixelintegrator_13(float tx1_4_1,float tx3_6_1,float tx2_5_1,float ty3_9_1,float pc2_16_1,float px1_11_1,float tc2_19_1,float y_21_5,float px0_10_1,float ty2_8_1,float ty1_7_1){
    float x_20_5;
    float __output__ = 0;
    float __step__ = ((float)(px1_11_1 - px0_10_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        x_20_5 = px0_10_1 + __step__ * (i + (float)(0.5));
        float _t_918;
		_t_918 = tegpixelbody_block_13(tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,x_20_5,y_21_5,ty3_9_1,tx3_6_1,tc2_19_1,pc2_16_1);;
        __output__ = __output__ + _t_918 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_7(float px1_11_1,float px0_10_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float y_21_5,float ty3_9_1,float tx3_6_1,float tc2_19_1,float pc2_16_1){

	float _t_14;

	_t_14 = tegpixelintegrator_13(tx1_4_1,tx3_6_1,tx2_5_1,ty3_9_1,pc2_16_1,px1_11_1,tc2_19_1,y_21_5,px0_10_1,ty2_8_1,ty1_7_1);

	return _t_14;
}
__device__ float tegpixelintegrator_7(float tx1_4_1,float tx3_6_1,float tx2_5_1,float ty3_9_1,float pc2_16_1,float py1_13_1,float px1_11_1,float py0_12_1,float tc2_19_1,float px0_10_1,float ty2_8_1,float ty1_7_1){
    float y_21_5;
    float __output__ = 0;
    float __step__ = ((float)(py1_13_1 - py0_12_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y_21_5 = py0_12_1 + __step__ * (i + (float)(0.5));
        float _t_14;
		_t_14 = tegpixelbody_block_7(px1_11_1,px0_10_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,y_21_5,ty3_9_1,tx3_6_1,tc2_19_1,pc2_16_1);;
        __output__ = __output__ + _t_14 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_14(float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float x_20_6,float y_21_6,float ty3_9_1,float tx3_6_1,float tc0_17_1,float pc0_14_1){
	float _t_968;
	float _t_969;
	float _t_970;
	float _t_971;
	float _t_972;
	float _t_973;
	float _t_974;
	float _t_975;
	float _t_976;
	float _t_977;
	float _t_978;
	float _t_979;
	bool _t_980;
	float _t_981;
	float _t_982;
	float _t_983;
	float _t_984;
	float _t_985;
	float _t_986;
	float _t_987;
	float _t_988;
	float _t_989;
	float _t_990;
	float _t_991;
	float _t_992;
	float _t_993;
	bool _t_994;
	float _t_995;
	float _t_996;
	float _t_997;
	float _t_998;
	float _t_999;
	float _t_1000;
	float _t_1001;
	float _t_1002;
	float _t_1003;
	float _t_1004;
	float _t_1005;
	float _t_1006;
	float _t_1007;
	float _t_1008;
	bool _t_1009;
	float _t_1010;
	float _t_1011;
	float _t_1012;
	float _t_1013;
	float _t_1014;

	float _t_967;

	_t_968 = tx1_4_1 * ty2_8_1;
	_t_969 = tx2_5_1 * ty1_7_1;
	_t_970 = _t_969 * -1.0f;
	_t_971 = _t_968 + _t_970;
	_t_972 = -1.0f * ty2_8_1;
	_t_973 = ty1_7_1 + _t_972;
	_t_974 = _t_973 * x_20_6;
	_t_975 = _t_971 + _t_974;
	_t_976 = -1.0f * tx1_4_1;
	_t_977 = tx2_5_1 + _t_976;
	_t_978 = _t_977 * y_21_6;
	_t_979 = _t_975 + _t_978;
	_t_980 = _t_979 < 0.0f;
	if(_t_980)
		{
		
			_t_981 = 1.0f;
		
		}
else
		{
		
			_t_981 = 0.0f;
		
		}

	_t_982 = tx2_5_1 * ty3_9_1;
	_t_983 = tx3_6_1 * ty2_8_1;
	_t_984 = _t_983 * -1.0f;
	_t_985 = _t_982 + _t_984;
	_t_986 = -1.0f * ty3_9_1;
	_t_987 = ty2_8_1 + _t_986;
	_t_988 = _t_987 * x_20_6;
	_t_989 = _t_985 + _t_988;
	_t_990 = -1.0f * tx2_5_1;
	_t_991 = tx3_6_1 + _t_990;
	_t_992 = _t_991 * y_21_6;
	_t_993 = _t_989 + _t_992;
	_t_994 = _t_993 < 0.0f;
	if(_t_994)
		{
		
			_t_995 = 1.0f;
		
		}
else
		{
		
			_t_995 = 0.0f;
		
		}

	_t_996 = _t_981 * _t_995;
	_t_997 = tx3_6_1 * ty1_7_1;
	_t_998 = tx1_4_1 * ty3_9_1;
	_t_999 = _t_998 * -1.0f;
	_t_1000 = _t_997 + _t_999;
	_t_1001 = -1.0f * ty1_7_1;
	_t_1002 = ty3_9_1 + _t_1001;
	_t_1003 = _t_1002 * x_20_6;
	_t_1004 = _t_1000 + _t_1003;
	_t_1005 = -1.0f * tx3_6_1;
	_t_1006 = tx1_4_1 + _t_1005;
	_t_1007 = _t_1006 * y_21_6;
	_t_1008 = _t_1004 + _t_1007;
	_t_1009 = _t_1008 < 0.0f;
	if(_t_1009)
		{
		
			_t_1010 = 1.0f;
		
		}
else
		{
		
			_t_1010 = 0.0f;
		
		}

	_t_1011 = _t_996 * _t_1010;
	_t_1012 = _t_1011 * 2.0f;
	_t_1013 = -1.0f * pc0_14_1;
	_t_1014 = tc0_17_1 + _t_1013;
	_t_967 = _t_1012 * _t_1014;

	return _t_967;
}
__device__ float tegpixelintegrator_14(float tx1_4_1,float tx3_6_1,float y_21_6,float tx2_5_1,float ty3_9_1,float tc0_17_1,float px1_11_1,float pc0_14_1,float px0_10_1,float ty2_8_1,float ty1_7_1){
    float x_20_6;
    float __output__ = 0;
    float __step__ = ((float)(px1_11_1 - px0_10_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        x_20_6 = px0_10_1 + __step__ * (i + (float)(0.5));
        float _t_967;
		_t_967 = tegpixelbody_block_14(tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,x_20_6,y_21_6,ty3_9_1,tx3_6_1,tc0_17_1,pc0_14_1);;
        __output__ = __output__ + _t_967 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_8(float px1_11_1,float px0_10_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float y_21_6,float ty3_9_1,float tx3_6_1,float tc0_17_1,float pc0_14_1){

	float _t_16;

	_t_16 = tegpixelintegrator_14(tx1_4_1,tx3_6_1,y_21_6,tx2_5_1,ty3_9_1,tc0_17_1,px1_11_1,pc0_14_1,px0_10_1,ty2_8_1,ty1_7_1);

	return _t_16;
}
__device__ float tegpixelintegrator_8(float tx1_4_1,float tx3_6_1,float tx2_5_1,float ty3_9_1,float py1_13_1,float px1_11_1,float tc0_17_1,float py0_12_1,float pc0_14_1,float px0_10_1,float ty2_8_1,float ty1_7_1){
    float y_21_6;
    float __output__ = 0;
    float __step__ = ((float)(py1_13_1 - py0_12_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y_21_6 = py0_12_1 + __step__ * (i + (float)(0.5));
        float _t_16;
		_t_16 = tegpixelbody_block_8(px1_11_1,px0_10_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,y_21_6,ty3_9_1,tx3_6_1,tc0_17_1,pc0_14_1);;
        __output__ = __output__ + _t_16 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_15(float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float x_20_7,float y_21_7,float ty3_9_1,float tx3_6_1,float tc1_18_1,float pc1_15_1){
	float _t_1016;
	float _t_1017;
	float _t_1018;
	float _t_1019;
	float _t_1020;
	float _t_1021;
	float _t_1022;
	float _t_1023;
	float _t_1024;
	float _t_1025;
	float _t_1026;
	float _t_1027;
	bool _t_1028;
	float _t_1029;
	float _t_1030;
	float _t_1031;
	float _t_1032;
	float _t_1033;
	float _t_1034;
	float _t_1035;
	float _t_1036;
	float _t_1037;
	float _t_1038;
	float _t_1039;
	float _t_1040;
	float _t_1041;
	bool _t_1042;
	float _t_1043;
	float _t_1044;
	float _t_1045;
	float _t_1046;
	float _t_1047;
	float _t_1048;
	float _t_1049;
	float _t_1050;
	float _t_1051;
	float _t_1052;
	float _t_1053;
	float _t_1054;
	float _t_1055;
	float _t_1056;
	bool _t_1057;
	float _t_1058;
	float _t_1059;
	float _t_1060;
	float _t_1061;
	float _t_1062;

	float _t_1015;

	_t_1016 = tx1_4_1 * ty2_8_1;
	_t_1017 = tx2_5_1 * ty1_7_1;
	_t_1018 = _t_1017 * -1.0f;
	_t_1019 = _t_1016 + _t_1018;
	_t_1020 = -1.0f * ty2_8_1;
	_t_1021 = ty1_7_1 + _t_1020;
	_t_1022 = _t_1021 * x_20_7;
	_t_1023 = _t_1019 + _t_1022;
	_t_1024 = -1.0f * tx1_4_1;
	_t_1025 = tx2_5_1 + _t_1024;
	_t_1026 = _t_1025 * y_21_7;
	_t_1027 = _t_1023 + _t_1026;
	_t_1028 = _t_1027 < 0.0f;
	if(_t_1028)
		{
		
			_t_1029 = 1.0f;
		
		}
else
		{
		
			_t_1029 = 0.0f;
		
		}

	_t_1030 = tx2_5_1 * ty3_9_1;
	_t_1031 = tx3_6_1 * ty2_8_1;
	_t_1032 = _t_1031 * -1.0f;
	_t_1033 = _t_1030 + _t_1032;
	_t_1034 = -1.0f * ty3_9_1;
	_t_1035 = ty2_8_1 + _t_1034;
	_t_1036 = _t_1035 * x_20_7;
	_t_1037 = _t_1033 + _t_1036;
	_t_1038 = -1.0f * tx2_5_1;
	_t_1039 = tx3_6_1 + _t_1038;
	_t_1040 = _t_1039 * y_21_7;
	_t_1041 = _t_1037 + _t_1040;
	_t_1042 = _t_1041 < 0.0f;
	if(_t_1042)
		{
		
			_t_1043 = 1.0f;
		
		}
else
		{
		
			_t_1043 = 0.0f;
		
		}

	_t_1044 = _t_1029 * _t_1043;
	_t_1045 = tx3_6_1 * ty1_7_1;
	_t_1046 = tx1_4_1 * ty3_9_1;
	_t_1047 = _t_1046 * -1.0f;
	_t_1048 = _t_1045 + _t_1047;
	_t_1049 = -1.0f * ty1_7_1;
	_t_1050 = ty3_9_1 + _t_1049;
	_t_1051 = _t_1050 * x_20_7;
	_t_1052 = _t_1048 + _t_1051;
	_t_1053 = -1.0f * tx3_6_1;
	_t_1054 = tx1_4_1 + _t_1053;
	_t_1055 = _t_1054 * y_21_7;
	_t_1056 = _t_1052 + _t_1055;
	_t_1057 = _t_1056 < 0.0f;
	if(_t_1057)
		{
		
			_t_1058 = 1.0f;
		
		}
else
		{
		
			_t_1058 = 0.0f;
		
		}

	_t_1059 = _t_1044 * _t_1058;
	_t_1060 = _t_1059 * 2.0f;
	_t_1061 = -1.0f * pc1_15_1;
	_t_1062 = tc1_18_1 + _t_1061;
	_t_1015 = _t_1060 * _t_1062;

	return _t_1015;
}
__device__ float tegpixelintegrator_15(float tx1_4_1,float y_21_7,float tx3_6_1,float tx2_5_1,float ty3_9_1,float pc1_15_1,float px1_11_1,float tc1_18_1,float px0_10_1,float ty2_8_1,float ty1_7_1){
    float x_20_7;
    float __output__ = 0;
    float __step__ = ((float)(px1_11_1 - px0_10_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        x_20_7 = px0_10_1 + __step__ * (i + (float)(0.5));
        float _t_1015;
		_t_1015 = tegpixelbody_block_15(tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,x_20_7,y_21_7,ty3_9_1,tx3_6_1,tc1_18_1,pc1_15_1);;
        __output__ = __output__ + _t_1015 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_9(float px1_11_1,float px0_10_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float y_21_7,float ty3_9_1,float tx3_6_1,float tc1_18_1,float pc1_15_1){

	float _t_18;

	_t_18 = tegpixelintegrator_15(tx1_4_1,y_21_7,tx3_6_1,tx2_5_1,ty3_9_1,pc1_15_1,px1_11_1,tc1_18_1,px0_10_1,ty2_8_1,ty1_7_1);

	return _t_18;
}
__device__ float tegpixelintegrator_9(float tx1_4_1,float tx3_6_1,float tx2_5_1,float ty3_9_1,float pc1_15_1,float py1_13_1,float px1_11_1,float py0_12_1,float tc1_18_1,float px0_10_1,float ty2_8_1,float ty1_7_1){
    float y_21_7;
    float __output__ = 0;
    float __step__ = ((float)(py1_13_1 - py0_12_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y_21_7 = py0_12_1 + __step__ * (i + (float)(0.5));
        float _t_18;
		_t_18 = tegpixelbody_block_9(px1_11_1,px0_10_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,y_21_7,ty3_9_1,tx3_6_1,tc1_18_1,pc1_15_1);;
        __output__ = __output__ + _t_18 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_16(float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float x_20_8,float y_21_8,float ty3_9_1,float tx3_6_1,float tc2_19_1,float pc2_16_1){
	float _t_1064;
	float _t_1065;
	float _t_1066;
	float _t_1067;
	float _t_1068;
	float _t_1069;
	float _t_1070;
	float _t_1071;
	float _t_1072;
	float _t_1073;
	float _t_1074;
	float _t_1075;
	bool _t_1076;
	float _t_1077;
	float _t_1078;
	float _t_1079;
	float _t_1080;
	float _t_1081;
	float _t_1082;
	float _t_1083;
	float _t_1084;
	float _t_1085;
	float _t_1086;
	float _t_1087;
	float _t_1088;
	float _t_1089;
	bool _t_1090;
	float _t_1091;
	float _t_1092;
	float _t_1093;
	float _t_1094;
	float _t_1095;
	float _t_1096;
	float _t_1097;
	float _t_1098;
	float _t_1099;
	float _t_1100;
	float _t_1101;
	float _t_1102;
	float _t_1103;
	float _t_1104;
	bool _t_1105;
	float _t_1106;
	float _t_1107;
	float _t_1108;
	float _t_1109;
	float _t_1110;

	float _t_1063;

	_t_1064 = tx1_4_1 * ty2_8_1;
	_t_1065 = tx2_5_1 * ty1_7_1;
	_t_1066 = _t_1065 * -1.0f;
	_t_1067 = _t_1064 + _t_1066;
	_t_1068 = -1.0f * ty2_8_1;
	_t_1069 = ty1_7_1 + _t_1068;
	_t_1070 = _t_1069 * x_20_8;
	_t_1071 = _t_1067 + _t_1070;
	_t_1072 = -1.0f * tx1_4_1;
	_t_1073 = tx2_5_1 + _t_1072;
	_t_1074 = _t_1073 * y_21_8;
	_t_1075 = _t_1071 + _t_1074;
	_t_1076 = _t_1075 < 0.0f;
	if(_t_1076)
		{
		
			_t_1077 = 1.0f;
		
		}
else
		{
		
			_t_1077 = 0.0f;
		
		}

	_t_1078 = tx2_5_1 * ty3_9_1;
	_t_1079 = tx3_6_1 * ty2_8_1;
	_t_1080 = _t_1079 * -1.0f;
	_t_1081 = _t_1078 + _t_1080;
	_t_1082 = -1.0f * ty3_9_1;
	_t_1083 = ty2_8_1 + _t_1082;
	_t_1084 = _t_1083 * x_20_8;
	_t_1085 = _t_1081 + _t_1084;
	_t_1086 = -1.0f * tx2_5_1;
	_t_1087 = tx3_6_1 + _t_1086;
	_t_1088 = _t_1087 * y_21_8;
	_t_1089 = _t_1085 + _t_1088;
	_t_1090 = _t_1089 < 0.0f;
	if(_t_1090)
		{
		
			_t_1091 = 1.0f;
		
		}
else
		{
		
			_t_1091 = 0.0f;
		
		}

	_t_1092 = _t_1077 * _t_1091;
	_t_1093 = tx3_6_1 * ty1_7_1;
	_t_1094 = tx1_4_1 * ty3_9_1;
	_t_1095 = _t_1094 * -1.0f;
	_t_1096 = _t_1093 + _t_1095;
	_t_1097 = -1.0f * ty1_7_1;
	_t_1098 = ty3_9_1 + _t_1097;
	_t_1099 = _t_1098 * x_20_8;
	_t_1100 = _t_1096 + _t_1099;
	_t_1101 = -1.0f * tx3_6_1;
	_t_1102 = tx1_4_1 + _t_1101;
	_t_1103 = _t_1102 * y_21_8;
	_t_1104 = _t_1100 + _t_1103;
	_t_1105 = _t_1104 < 0.0f;
	if(_t_1105)
		{
		
			_t_1106 = 1.0f;
		
		}
else
		{
		
			_t_1106 = 0.0f;
		
		}

	_t_1107 = _t_1092 * _t_1106;
	_t_1108 = _t_1107 * 2.0f;
	_t_1109 = -1.0f * pc2_16_1;
	_t_1110 = tc2_19_1 + _t_1109;
	_t_1063 = _t_1108 * _t_1110;

	return _t_1063;
}
__device__ float tegpixelintegrator_16(float tx1_4_1,float tx3_6_1,float tx2_5_1,float ty3_9_1,float pc2_16_1,float px1_11_1,float tc2_19_1,float ty2_8_1,float px0_10_1,float y_21_8,float ty1_7_1){
    float x_20_8;
    float __output__ = 0;
    float __step__ = ((float)(px1_11_1 - px0_10_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        x_20_8 = px0_10_1 + __step__ * (i + (float)(0.5));
        float _t_1063;
		_t_1063 = tegpixelbody_block_16(tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,x_20_8,y_21_8,ty3_9_1,tx3_6_1,tc2_19_1,pc2_16_1);;
        __output__ = __output__ + _t_1063 * __step__;
    }
    return __output__;
}
__device__ float tegpixelbody_block_10(float px1_11_1,float px0_10_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float y_21_8,float ty3_9_1,float tx3_6_1,float tc2_19_1,float pc2_16_1){

	float _t_20;

	_t_20 = tegpixelintegrator_16(tx1_4_1,tx3_6_1,tx2_5_1,ty3_9_1,pc2_16_1,px1_11_1,tc2_19_1,ty2_8_1,px0_10_1,y_21_8,ty1_7_1);

	return _t_20;
}
__device__ float tegpixelintegrator_10(float tx1_4_1,float tx3_6_1,float tx2_5_1,float ty3_9_1,float pc2_16_1,float py1_13_1,float px1_11_1,float py0_12_1,float tc2_19_1,float px0_10_1,float ty2_8_1,float ty1_7_1){
    float y_21_8;
    float __output__ = 0;
    float __step__ = ((float)(py1_13_1 - py0_12_1)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y_21_8 = py0_12_1 + __step__ * (i + (float)(0.5));
        float _t_20;
		_t_20 = tegpixelbody_block_10(px1_11_1,px0_10_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,y_21_8,ty3_9_1,tx3_6_1,tc2_19_1,pc2_16_1);;
        __output__ = __output__ + _t_20 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_20(float py0_12_1,float _t_1566,float py1_13_1,float px0_10_1,float _t_1513,float px1_11_1,float ty1_7_1,float ty2_8_1,float tx2_5_1,float tx1_4_1,float _t_119,float y__2573_1,float _t_1486,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	bool _t_1567;
	bool _t_1568;
	bool _t_1569;
	bool _t_1570;
	bool _t_1571;
	bool _t_1572;
	bool _t_1573;
	float _t_1903;
	float _t_1904;
	float _t_1905;
	float _t_1906;
	float _t_1907;
	float _t_1908;
	float _t_1909;
	float _t_1910;
	float _t_1911;
	float _t_1912;
	float _t_1913;
	float _t_1914;
	float _t_1915;
	float _t_1916;
	float _t_1917;
	float _t_1918;
	float _t_1919;
	float _t_1920;
	float _t_1921;
	float _t_1922;
	float _t_1923;
	float _t_1924;
	float _t_1925;
	float _t_1926;
	bool _t_1927;
	float _t_1928;
	float _t_1929;
	float _t_1930;
	float _t_1931;
	float _t_1932;
	float _t_1933;
	float _t_1934;
	float _t_1935;
	float _t_1936;
	float _t_1937;
	float _t_1938;
	float _t_1939;
	float _t_1940;
	float _t_1941;
	bool _t_1942;
	float _t_1943;
	float _t_1944;
	float _t_1945;
	float _t_1946;
	float _t_1947;
	float _t_1948;
	float _t_1949;
	float _t_1950;
	float _t_1951;
	float _t_1952;
	float _t_1953;
	float _t_1954;
	float _t_1955;
	float _t_1956;
	float _t_1957;
	float _t_1958;
	float _t_1959;
	float _t_1960;
	float _t_1961;
	float _t_1962;
	float _t_1963;
	float _t_1964;
	float _t_1965;
	float _t_1966;
	float _t_1967;
	float _t_1968;
	float _t_1969;
	bool _t_1970;
	float _t_1971;
	float _t_1972;
	float _t_1973;
	float _t_1974;
	float _t_1975;
	float _t_1976;
	float _t_1977;
	float _t_1978;
	float _t_1979;
	float _t_1980;
	float _t_1981;
	float _t_1982;
	float _t_1983;
	float _t_1984;
	bool _t_1985;
	float _t_1986;
	float _t_1987;
	float _t_1988;
	float _t_1989;

	float _t_1487;

	_t_1567 = py0_12_1 < _t_1566;
	_t_1568 = _t_1566 < py1_13_1;
	_t_1569 = _t_1567 && _t_1568;
	_t_1570 = px0_10_1 < _t_1513;
	_t_1571 = _t_1513 < px1_11_1;
	_t_1572 = _t_1570 && _t_1571;
	_t_1573 = _t_1569 && _t_1572;
	if(_t_1573)
		{
			float _t_1574;
			float _t_1575;
			float _t_1576;
			bool _t_1577;
			float _t_1580;
			float _t_1584;
			float _t_1585;
			float _t_1586;
			float _t_1587;
			float _t_1588;
			bool _t_1589;
			float _t_1592;
			float _t_1596;
			float _t_1597;
			bool _t_1598;
			float _t_1599;
			float _t_1600;
			float _t_1601;
			float _t_1602;
			float _t_1603;
			bool _t_1604;
			float _t_1607;
			float _t_1611;
			float _t_1612;
			float _t_1613;
			float _t_1614;
			bool _t_1615;
			float _t_1618;
			float _t_1622;
			float _t_1623;
			float _t_1624;
			float _t_1625;
			float _t_1626;
			bool _t_1627;
			float _t_1630;
			float _t_1634;
			float _t_1635;
			float _t_1636;
			float _t_1637;
			float _t_1638;
			float _t_1639;
			float _t_1640;
			float _t_1641;
			float _t_1642;
			bool _t_1643;
			float _t_1646;
			float _t_1650;
			float _t_1651;
			float _t_1652;
			float _t_1653;
			bool _t_1654;
			float _t_1657;
			float _t_1661;
			float _t_1662;
			float _t_1663;
			float _t_1664;
			float _t_1665;
			bool _t_1666;
			float _t_1669;
			float _t_1673;
			float _t_1674;
			float _t_1675;
			float _t_1676;
			float _t_1677;
			float _t_1678;
			bool _t_1679;
			float _t_1680;
			float _t_1681;
			float _t_1682;
			bool _t_1683;
			float _t_1684;
			float _t_1685;
			float _t_1686;
			bool _t_1687;
			float _t_1690;
			float _t_1694;
			float _t_1695;
			float _t_1696;
			float _t_1697;
			float _t_1698;
			bool _t_1699;
			float _t_1702;
			float _t_1706;
			float _t_1707;
			bool _t_1708;
			float _t_1709;
			float _t_1710;
			float _t_1711;
			float _t_1712;
			float _t_1713;
			bool _t_1714;
			float _t_1717;
			float _t_1721;
			float _t_1722;
			float _t_1723;
			float _t_1724;
			bool _t_1725;
			float _t_1728;
			float _t_1732;
			float _t_1733;
			float _t_1734;
			float _t_1735;
			float _t_1736;
			bool _t_1737;
			float _t_1740;
			float _t_1744;
			float _t_1745;
			float _t_1746;
			float _t_1747;
			float _t_1748;
			float _t_1749;
			float _t_1750;
			float _t_1751;
			float _t_1752;
			bool _t_1753;
			float _t_1756;
			float _t_1760;
			float _t_1761;
			float _t_1762;
			float _t_1763;
			bool _t_1764;
			float _t_1767;
			float _t_1771;
			float _t_1772;
			float _t_1773;
			float _t_1774;
			float _t_1775;
			bool _t_1776;
			float _t_1779;
			float _t_1783;
			float _t_1784;
			float _t_1785;
			float _t_1786;
			float _t_1787;
			float _t_1788;
			bool _t_1789;
			float _t_1790;
			float _t_1791;
			float _t_1792;
			bool _t_1793;
			bool _t_1794;
			float _t_1795;
			float _t_1796;
			float _t_1797;
			bool _t_1798;
			float _t_1801;
			float _t_1805;
			float _t_1806;
			float _t_1807;
			float _t_1808;
			bool _t_1809;
			float _t_1812;
			float _t_1816;
			bool _t_1817;
			float _t_1818;
			float _t_1819;
			float _t_1820;
			float _t_1821;
			float _t_1822;
			bool _t_1823;
			float _t_1826;
			float _t_1830;
			float _t_1831;
			float _t_1832;
			float _t_1833;
			bool _t_1834;
			float _t_1837;
			float _t_1841;
			bool _t_1842;
			float _t_1843;
			float _t_1844;
			float _t_1845;
			bool _t_1846;
			float _t_1847;
			float _t_1848;
			float _t_1849;
			bool _t_1850;
			float _t_1853;
			float _t_1857;
			float _t_1858;
			float _t_1859;
			float _t_1860;
			bool _t_1861;
			float _t_1864;
			float _t_1868;
			bool _t_1869;
			float _t_1870;
			float _t_1871;
			float _t_1872;
			float _t_1873;
			float _t_1874;
			bool _t_1875;
			float _t_1878;
			float _t_1882;
			float _t_1883;
			float _t_1884;
			float _t_1885;
			bool _t_1886;
			float _t_1889;
			float _t_1893;
			bool _t_1894;
			float _t_1895;
			float _t_1896;
			float _t_1897;
			bool _t_1898;
			bool _t_1899;
			bool _t_1900;
			float _t_1901;
			float _t_1902;
		
			_t_1574 = -1.0f * ty2_8_1;
			_t_1575 = ty1_7_1 + _t_1574;
			_t_1576 = -1.0f * _t_1575;
			_t_1577 = _t_1576 < 0.0f;
			if(_t_1577)
				{
					float _t_1578;
					float _t_1579;
				
					_t_1578 = -1.0f * tx1_4_1;
					_t_1579 = tx2_5_1 + _t_1578;
					_t_1580 = _t_1579;
				
				}
		else
				{
					float _t_1581;
					float _t_1582;
					float _t_1583;
				
					_t_1581 = -1.0f * tx1_4_1;
					_t_1582 = tx2_5_1 + _t_1581;
					_t_1583 = -1.0f * _t_1582;
					_t_1580 = _t_1583;
				
				}
		
			_t_1584 = _t_1580 * _t_119;
			_t_1585 = _t_1584 * -1.0f;
			_t_1586 = -1.0f * ty2_8_1;
			_t_1587 = ty1_7_1 + _t_1586;
			_t_1588 = -1.0f * _t_1587;
			_t_1589 = _t_1588 < 0.0f;
			if(_t_1589)
				{
					float _t_1590;
					float _t_1591;
				
					_t_1590 = -1.0f * tx1_4_1;
					_t_1591 = tx2_5_1 + _t_1590;
					_t_1592 = _t_1591;
				
				}
		else
				{
					float _t_1593;
					float _t_1594;
					float _t_1595;
				
					_t_1593 = -1.0f * tx1_4_1;
					_t_1594 = tx2_5_1 + _t_1593;
					_t_1595 = -1.0f * _t_1594;
					_t_1592 = _t_1595;
				
				}
		
			_t_1596 = _t_1592 * _t_119;
			_t_1597 = _t_1596 * -1.0f;
			_t_1598 = 0.0f < _t_1597;
			if(_t_1598)
				{
				
					_t_1599 = px0_10_1;
				
				}
		else
				{
				
					_t_1599 = px1_11_1;
				
				}
		
			_t_1600 = _t_1585 * _t_1599;
			_t_1601 = -1.0f * ty2_8_1;
			_t_1602 = ty1_7_1 + _t_1601;
			_t_1603 = -1.0f * _t_1602;
			_t_1604 = _t_1603 < 0.0f;
			if(_t_1604)
				{
					float _t_1605;
					float _t_1606;
				
					_t_1605 = -1.0f * tx1_4_1;
					_t_1606 = tx2_5_1 + _t_1605;
					_t_1607 = _t_1606;
				
				}
		else
				{
					float _t_1608;
					float _t_1609;
					float _t_1610;
				
					_t_1608 = -1.0f * tx1_4_1;
					_t_1609 = tx2_5_1 + _t_1608;
					_t_1610 = -1.0f * _t_1609;
					_t_1607 = _t_1610;
				
				}
		
			_t_1611 = _t_1607 * _t_119;
			_t_1612 = -1.0f * ty2_8_1;
			_t_1613 = ty1_7_1 + _t_1612;
			_t_1614 = -1.0f * _t_1613;
			_t_1615 = _t_1614 < 0.0f;
			if(_t_1615)
				{
					float _t_1616;
					float _t_1617;
				
					_t_1616 = -1.0f * tx1_4_1;
					_t_1617 = tx2_5_1 + _t_1616;
					_t_1618 = _t_1617;
				
				}
		else
				{
					float _t_1619;
					float _t_1620;
					float _t_1621;
				
					_t_1619 = -1.0f * tx1_4_1;
					_t_1620 = tx2_5_1 + _t_1619;
					_t_1621 = -1.0f * _t_1620;
					_t_1618 = _t_1621;
				
				}
		
			_t_1622 = _t_1618 * _t_119;
			_t_1623 = _t_1611 * _t_1622;
			_t_1624 = -1.0f * ty2_8_1;
			_t_1625 = ty1_7_1 + _t_1624;
			_t_1626 = -1.0f * _t_1625;
			_t_1627 = _t_1626 < 0.0f;
			if(_t_1627)
				{
					float _t_1628;
					float _t_1629;
				
					_t_1628 = -1.0f * ty2_8_1;
					_t_1629 = ty1_7_1 + _t_1628;
					_t_1630 = _t_1629;
				
				}
		else
				{
					float _t_1631;
					float _t_1632;
					float _t_1633;
				
					_t_1631 = -1.0f * ty2_8_1;
					_t_1632 = ty1_7_1 + _t_1631;
					_t_1633 = -1.0f * _t_1632;
					_t_1630 = _t_1633;
				
				}
		
			_t_1634 = _t_1630 * _t_119;
			_t_1635 = 1.0f + _t_1634;
			_t_1636 = 1.0f / _t_1635;
			_t_1637 = _t_1623 * _t_1636;
			_t_1638 = _t_1637 * -1.0f;
			_t_1639 = 1.0f + _t_1638;
			_t_1640 = -1.0f * ty2_8_1;
			_t_1641 = ty1_7_1 + _t_1640;
			_t_1642 = -1.0f * _t_1641;
			_t_1643 = _t_1642 < 0.0f;
			if(_t_1643)
				{
					float _t_1644;
					float _t_1645;
				
					_t_1644 = -1.0f * tx1_4_1;
					_t_1645 = tx2_5_1 + _t_1644;
					_t_1646 = _t_1645;
				
				}
		else
				{
					float _t_1647;
					float _t_1648;
					float _t_1649;
				
					_t_1647 = -1.0f * tx1_4_1;
					_t_1648 = tx2_5_1 + _t_1647;
					_t_1649 = -1.0f * _t_1648;
					_t_1646 = _t_1649;
				
				}
		
			_t_1650 = _t_1646 * _t_119;
			_t_1651 = -1.0f * ty2_8_1;
			_t_1652 = ty1_7_1 + _t_1651;
			_t_1653 = -1.0f * _t_1652;
			_t_1654 = _t_1653 < 0.0f;
			if(_t_1654)
				{
					float _t_1655;
					float _t_1656;
				
					_t_1655 = -1.0f * tx1_4_1;
					_t_1656 = tx2_5_1 + _t_1655;
					_t_1657 = _t_1656;
				
				}
		else
				{
					float _t_1658;
					float _t_1659;
					float _t_1660;
				
					_t_1658 = -1.0f * tx1_4_1;
					_t_1659 = tx2_5_1 + _t_1658;
					_t_1660 = -1.0f * _t_1659;
					_t_1657 = _t_1660;
				
				}
		
			_t_1661 = _t_1657 * _t_119;
			_t_1662 = _t_1650 * _t_1661;
			_t_1663 = -1.0f * ty2_8_1;
			_t_1664 = ty1_7_1 + _t_1663;
			_t_1665 = -1.0f * _t_1664;
			_t_1666 = _t_1665 < 0.0f;
			if(_t_1666)
				{
					float _t_1667;
					float _t_1668;
				
					_t_1667 = -1.0f * ty2_8_1;
					_t_1668 = ty1_7_1 + _t_1667;
					_t_1669 = _t_1668;
				
				}
		else
				{
					float _t_1670;
					float _t_1671;
					float _t_1672;
				
					_t_1670 = -1.0f * ty2_8_1;
					_t_1671 = ty1_7_1 + _t_1670;
					_t_1672 = -1.0f * _t_1671;
					_t_1669 = _t_1672;
				
				}
		
			_t_1673 = _t_1669 * _t_119;
			_t_1674 = 1.0f + _t_1673;
			_t_1675 = 1.0f / _t_1674;
			_t_1676 = _t_1662 * _t_1675;
			_t_1677 = _t_1676 * -1.0f;
			_t_1678 = 1.0f + _t_1677;
			_t_1679 = 0.0f < _t_1678;
			if(_t_1679)
				{
				
					_t_1680 = py0_12_1;
				
				}
		else
				{
				
					_t_1680 = py1_13_1;
				
				}
		
			_t_1681 = _t_1639 * _t_1680;
			_t_1682 = _t_1600 + _t_1681;
			_t_1683 = _t_1682 < y__2573_1;
			_t_1684 = -1.0f * ty2_8_1;
			_t_1685 = ty1_7_1 + _t_1684;
			_t_1686 = -1.0f * _t_1685;
			_t_1687 = _t_1686 < 0.0f;
			if(_t_1687)
				{
					float _t_1688;
					float _t_1689;
				
					_t_1688 = -1.0f * tx1_4_1;
					_t_1689 = tx2_5_1 + _t_1688;
					_t_1690 = _t_1689;
				
				}
		else
				{
					float _t_1691;
					float _t_1692;
					float _t_1693;
				
					_t_1691 = -1.0f * tx1_4_1;
					_t_1692 = tx2_5_1 + _t_1691;
					_t_1693 = -1.0f * _t_1692;
					_t_1690 = _t_1693;
				
				}
		
			_t_1694 = _t_1690 * _t_119;
			_t_1695 = _t_1694 * -1.0f;
			_t_1696 = -1.0f * ty2_8_1;
			_t_1697 = ty1_7_1 + _t_1696;
			_t_1698 = -1.0f * _t_1697;
			_t_1699 = _t_1698 < 0.0f;
			if(_t_1699)
				{
					float _t_1700;
					float _t_1701;
				
					_t_1700 = -1.0f * tx1_4_1;
					_t_1701 = tx2_5_1 + _t_1700;
					_t_1702 = _t_1701;
				
				}
		else
				{
					float _t_1703;
					float _t_1704;
					float _t_1705;
				
					_t_1703 = -1.0f * tx1_4_1;
					_t_1704 = tx2_5_1 + _t_1703;
					_t_1705 = -1.0f * _t_1704;
					_t_1702 = _t_1705;
				
				}
		
			_t_1706 = _t_1702 * _t_119;
			_t_1707 = _t_1706 * -1.0f;
			_t_1708 = 0.0f < _t_1707;
			if(_t_1708)
				{
				
					_t_1709 = px1_11_1;
				
				}
		else
				{
				
					_t_1709 = px0_10_1;
				
				}
		
			_t_1710 = _t_1695 * _t_1709;
			_t_1711 = -1.0f * ty2_8_1;
			_t_1712 = ty1_7_1 + _t_1711;
			_t_1713 = -1.0f * _t_1712;
			_t_1714 = _t_1713 < 0.0f;
			if(_t_1714)
				{
					float _t_1715;
					float _t_1716;
				
					_t_1715 = -1.0f * tx1_4_1;
					_t_1716 = tx2_5_1 + _t_1715;
					_t_1717 = _t_1716;
				
				}
		else
				{
					float _t_1718;
					float _t_1719;
					float _t_1720;
				
					_t_1718 = -1.0f * tx1_4_1;
					_t_1719 = tx2_5_1 + _t_1718;
					_t_1720 = -1.0f * _t_1719;
					_t_1717 = _t_1720;
				
				}
		
			_t_1721 = _t_1717 * _t_119;
			_t_1722 = -1.0f * ty2_8_1;
			_t_1723 = ty1_7_1 + _t_1722;
			_t_1724 = -1.0f * _t_1723;
			_t_1725 = _t_1724 < 0.0f;
			if(_t_1725)
				{
					float _t_1726;
					float _t_1727;
				
					_t_1726 = -1.0f * tx1_4_1;
					_t_1727 = tx2_5_1 + _t_1726;
					_t_1728 = _t_1727;
				
				}
		else
				{
					float _t_1729;
					float _t_1730;
					float _t_1731;
				
					_t_1729 = -1.0f * tx1_4_1;
					_t_1730 = tx2_5_1 + _t_1729;
					_t_1731 = -1.0f * _t_1730;
					_t_1728 = _t_1731;
				
				}
		
			_t_1732 = _t_1728 * _t_119;
			_t_1733 = _t_1721 * _t_1732;
			_t_1734 = -1.0f * ty2_8_1;
			_t_1735 = ty1_7_1 + _t_1734;
			_t_1736 = -1.0f * _t_1735;
			_t_1737 = _t_1736 < 0.0f;
			if(_t_1737)
				{
					float _t_1738;
					float _t_1739;
				
					_t_1738 = -1.0f * ty2_8_1;
					_t_1739 = ty1_7_1 + _t_1738;
					_t_1740 = _t_1739;
				
				}
		else
				{
					float _t_1741;
					float _t_1742;
					float _t_1743;
				
					_t_1741 = -1.0f * ty2_8_1;
					_t_1742 = ty1_7_1 + _t_1741;
					_t_1743 = -1.0f * _t_1742;
					_t_1740 = _t_1743;
				
				}
		
			_t_1744 = _t_1740 * _t_119;
			_t_1745 = 1.0f + _t_1744;
			_t_1746 = 1.0f / _t_1745;
			_t_1747 = _t_1733 * _t_1746;
			_t_1748 = _t_1747 * -1.0f;
			_t_1749 = 1.0f + _t_1748;
			_t_1750 = -1.0f * ty2_8_1;
			_t_1751 = ty1_7_1 + _t_1750;
			_t_1752 = -1.0f * _t_1751;
			_t_1753 = _t_1752 < 0.0f;
			if(_t_1753)
				{
					float _t_1754;
					float _t_1755;
				
					_t_1754 = -1.0f * tx1_4_1;
					_t_1755 = tx2_5_1 + _t_1754;
					_t_1756 = _t_1755;
				
				}
		else
				{
					float _t_1757;
					float _t_1758;
					float _t_1759;
				
					_t_1757 = -1.0f * tx1_4_1;
					_t_1758 = tx2_5_1 + _t_1757;
					_t_1759 = -1.0f * _t_1758;
					_t_1756 = _t_1759;
				
				}
		
			_t_1760 = _t_1756 * _t_119;
			_t_1761 = -1.0f * ty2_8_1;
			_t_1762 = ty1_7_1 + _t_1761;
			_t_1763 = -1.0f * _t_1762;
			_t_1764 = _t_1763 < 0.0f;
			if(_t_1764)
				{
					float _t_1765;
					float _t_1766;
				
					_t_1765 = -1.0f * tx1_4_1;
					_t_1766 = tx2_5_1 + _t_1765;
					_t_1767 = _t_1766;
				
				}
		else
				{
					float _t_1768;
					float _t_1769;
					float _t_1770;
				
					_t_1768 = -1.0f * tx1_4_1;
					_t_1769 = tx2_5_1 + _t_1768;
					_t_1770 = -1.0f * _t_1769;
					_t_1767 = _t_1770;
				
				}
		
			_t_1771 = _t_1767 * _t_119;
			_t_1772 = _t_1760 * _t_1771;
			_t_1773 = -1.0f * ty2_8_1;
			_t_1774 = ty1_7_1 + _t_1773;
			_t_1775 = -1.0f * _t_1774;
			_t_1776 = _t_1775 < 0.0f;
			if(_t_1776)
				{
					float _t_1777;
					float _t_1778;
				
					_t_1777 = -1.0f * ty2_8_1;
					_t_1778 = ty1_7_1 + _t_1777;
					_t_1779 = _t_1778;
				
				}
		else
				{
					float _t_1780;
					float _t_1781;
					float _t_1782;
				
					_t_1780 = -1.0f * ty2_8_1;
					_t_1781 = ty1_7_1 + _t_1780;
					_t_1782 = -1.0f * _t_1781;
					_t_1779 = _t_1782;
				
				}
		
			_t_1783 = _t_1779 * _t_119;
			_t_1784 = 1.0f + _t_1783;
			_t_1785 = 1.0f / _t_1784;
			_t_1786 = _t_1772 * _t_1785;
			_t_1787 = _t_1786 * -1.0f;
			_t_1788 = 1.0f + _t_1787;
			_t_1789 = 0.0f < _t_1788;
			if(_t_1789)
				{
				
					_t_1790 = py1_13_1;
				
				}
		else
				{
				
					_t_1790 = py0_12_1;
				
				}
		
			_t_1791 = _t_1749 * _t_1790;
			_t_1792 = _t_1710 + _t_1791;
			_t_1793 = y__2573_1 < _t_1792;
			_t_1794 = _t_1683 && _t_1793;
			_t_1795 = -1.0f * ty2_8_1;
			_t_1796 = ty1_7_1 + _t_1795;
			_t_1797 = -1.0f * _t_1796;
			_t_1798 = _t_1797 < 0.0f;
			if(_t_1798)
				{
					float _t_1799;
					float _t_1800;
				
					_t_1799 = -1.0f * ty2_8_1;
					_t_1800 = ty1_7_1 + _t_1799;
					_t_1801 = _t_1800;
				
				}
		else
				{
					float _t_1802;
					float _t_1803;
					float _t_1804;
				
					_t_1802 = -1.0f * ty2_8_1;
					_t_1803 = ty1_7_1 + _t_1802;
					_t_1804 = -1.0f * _t_1803;
					_t_1801 = _t_1804;
				
				}
		
			_t_1805 = _t_1801 * _t_119;
			_t_1806 = -1.0f * ty2_8_1;
			_t_1807 = ty1_7_1 + _t_1806;
			_t_1808 = -1.0f * _t_1807;
			_t_1809 = _t_1808 < 0.0f;
			if(_t_1809)
				{
					float _t_1810;
					float _t_1811;
				
					_t_1810 = -1.0f * ty2_8_1;
					_t_1811 = ty1_7_1 + _t_1810;
					_t_1812 = _t_1811;
				
				}
		else
				{
					float _t_1813;
					float _t_1814;
					float _t_1815;
				
					_t_1813 = -1.0f * ty2_8_1;
					_t_1814 = ty1_7_1 + _t_1813;
					_t_1815 = -1.0f * _t_1814;
					_t_1812 = _t_1815;
				
				}
		
			_t_1816 = _t_1812 * _t_119;
			_t_1817 = 0.0f < _t_1816;
			if(_t_1817)
				{
				
					_t_1818 = px0_10_1;
				
				}
		else
				{
				
					_t_1818 = px1_11_1;
				
				}
		
			_t_1819 = _t_1805 * _t_1818;
			_t_1820 = -1.0f * ty2_8_1;
			_t_1821 = ty1_7_1 + _t_1820;
			_t_1822 = -1.0f * _t_1821;
			_t_1823 = _t_1822 < 0.0f;
			if(_t_1823)
				{
					float _t_1824;
					float _t_1825;
				
					_t_1824 = -1.0f * tx1_4_1;
					_t_1825 = tx2_5_1 + _t_1824;
					_t_1826 = _t_1825;
				
				}
		else
				{
					float _t_1827;
					float _t_1828;
					float _t_1829;
				
					_t_1827 = -1.0f * tx1_4_1;
					_t_1828 = tx2_5_1 + _t_1827;
					_t_1829 = -1.0f * _t_1828;
					_t_1826 = _t_1829;
				
				}
		
			_t_1830 = _t_1826 * _t_119;
			_t_1831 = -1.0f * ty2_8_1;
			_t_1832 = ty1_7_1 + _t_1831;
			_t_1833 = -1.0f * _t_1832;
			_t_1834 = _t_1833 < 0.0f;
			if(_t_1834)
				{
					float _t_1835;
					float _t_1836;
				
					_t_1835 = -1.0f * tx1_4_1;
					_t_1836 = tx2_5_1 + _t_1835;
					_t_1837 = _t_1836;
				
				}
		else
				{
					float _t_1838;
					float _t_1839;
					float _t_1840;
				
					_t_1838 = -1.0f * tx1_4_1;
					_t_1839 = tx2_5_1 + _t_1838;
					_t_1840 = -1.0f * _t_1839;
					_t_1837 = _t_1840;
				
				}
		
			_t_1841 = _t_1837 * _t_119;
			_t_1842 = 0.0f < _t_1841;
			if(_t_1842)
				{
				
					_t_1843 = py0_12_1;
				
				}
		else
				{
				
					_t_1843 = py1_13_1;
				
				}
		
			_t_1844 = _t_1830 * _t_1843;
			_t_1845 = _t_1819 + _t_1844;
			_t_1846 = _t_1845 < _t_1486;
			_t_1847 = -1.0f * ty2_8_1;
			_t_1848 = ty1_7_1 + _t_1847;
			_t_1849 = -1.0f * _t_1848;
			_t_1850 = _t_1849 < 0.0f;
			if(_t_1850)
				{
					float _t_1851;
					float _t_1852;
				
					_t_1851 = -1.0f * ty2_8_1;
					_t_1852 = ty1_7_1 + _t_1851;
					_t_1853 = _t_1852;
				
				}
		else
				{
					float _t_1854;
					float _t_1855;
					float _t_1856;
				
					_t_1854 = -1.0f * ty2_8_1;
					_t_1855 = ty1_7_1 + _t_1854;
					_t_1856 = -1.0f * _t_1855;
					_t_1853 = _t_1856;
				
				}
		
			_t_1857 = _t_1853 * _t_119;
			_t_1858 = -1.0f * ty2_8_1;
			_t_1859 = ty1_7_1 + _t_1858;
			_t_1860 = -1.0f * _t_1859;
			_t_1861 = _t_1860 < 0.0f;
			if(_t_1861)
				{
					float _t_1862;
					float _t_1863;
				
					_t_1862 = -1.0f * ty2_8_1;
					_t_1863 = ty1_7_1 + _t_1862;
					_t_1864 = _t_1863;
				
				}
		else
				{
					float _t_1865;
					float _t_1866;
					float _t_1867;
				
					_t_1865 = -1.0f * ty2_8_1;
					_t_1866 = ty1_7_1 + _t_1865;
					_t_1867 = -1.0f * _t_1866;
					_t_1864 = _t_1867;
				
				}
		
			_t_1868 = _t_1864 * _t_119;
			_t_1869 = 0.0f < _t_1868;
			if(_t_1869)
				{
				
					_t_1870 = px1_11_1;
				
				}
		else
				{
				
					_t_1870 = px0_10_1;
				
				}
		
			_t_1871 = _t_1857 * _t_1870;
			_t_1872 = -1.0f * ty2_8_1;
			_t_1873 = ty1_7_1 + _t_1872;
			_t_1874 = -1.0f * _t_1873;
			_t_1875 = _t_1874 < 0.0f;
			if(_t_1875)
				{
					float _t_1876;
					float _t_1877;
				
					_t_1876 = -1.0f * tx1_4_1;
					_t_1877 = tx2_5_1 + _t_1876;
					_t_1878 = _t_1877;
				
				}
		else
				{
					float _t_1879;
					float _t_1880;
					float _t_1881;
				
					_t_1879 = -1.0f * tx1_4_1;
					_t_1880 = tx2_5_1 + _t_1879;
					_t_1881 = -1.0f * _t_1880;
					_t_1878 = _t_1881;
				
				}
		
			_t_1882 = _t_1878 * _t_119;
			_t_1883 = -1.0f * ty2_8_1;
			_t_1884 = ty1_7_1 + _t_1883;
			_t_1885 = -1.0f * _t_1884;
			_t_1886 = _t_1885 < 0.0f;
			if(_t_1886)
				{
					float _t_1887;
					float _t_1888;
				
					_t_1887 = -1.0f * tx1_4_1;
					_t_1888 = tx2_5_1 + _t_1887;
					_t_1889 = _t_1888;
				
				}
		else
				{
					float _t_1890;
					float _t_1891;
					float _t_1892;
				
					_t_1890 = -1.0f * tx1_4_1;
					_t_1891 = tx2_5_1 + _t_1890;
					_t_1892 = -1.0f * _t_1891;
					_t_1889 = _t_1892;
				
				}
		
			_t_1893 = _t_1889 * _t_119;
			_t_1894 = 0.0f < _t_1893;
			if(_t_1894)
				{
				
					_t_1895 = py1_13_1;
				
				}
		else
				{
				
					_t_1895 = py0_12_1;
				
				}
		
			_t_1896 = _t_1882 * _t_1895;
			_t_1897 = _t_1871 + _t_1896;
			_t_1898 = _t_1486 < _t_1897;
			_t_1899 = _t_1846 && _t_1898;
			_t_1900 = _t_1794 && _t_1899;
			if(_t_1900)
				{
				
					_t_1901 = 1.0f;
				
				}
		else
				{
				
					_t_1901 = 0.0f;
				
				}
		
			_t_1902 = _t_1901 * _t_119;
			_t_1903 = _t_1902;
		
		}
else
		{
		
			_t_1903 = 0.0f;
		
		}

	_t_1904 = -1.0f * pc0_14_1;
	_t_1905 = tc0_17_1 + _t_1904;
	_t_1906 = _t_1905 * _t_1905;
	_t_1907 = -1.0f * pc1_15_1;
	_t_1908 = tc1_18_1 + _t_1907;
	_t_1909 = _t_1908 * _t_1908;
	_t_1910 = _t_1906 + _t_1909;
	_t_1911 = -1.0f * pc2_16_1;
	_t_1912 = tc2_19_1 + _t_1911;
	_t_1913 = _t_1912 * _t_1912;
	_t_1914 = _t_1910 + _t_1913;
	_t_1915 = tx3_6_1 * ty1_7_1;
	_t_1916 = tx1_4_1 * ty3_9_1;
	_t_1917 = _t_1916 * -1.0f;
	_t_1918 = _t_1915 + _t_1917;
	_t_1919 = -1.0f * ty1_7_1;
	_t_1920 = ty3_9_1 + _t_1919;
	_t_1921 = _t_1920 * _t_1513;
	_t_1922 = _t_1918 + _t_1921;
	_t_1923 = -1.0f * tx3_6_1;
	_t_1924 = tx1_4_1 + _t_1923;
	_t_1925 = _t_1924 * _t_1566;
	_t_1926 = _t_1922 + _t_1925;
	_t_1927 = _t_1926 < 0.0f;
	if(_t_1927)
		{
		
			_t_1928 = 1.0f;
		
		}
else
		{
		
			_t_1928 = 0.0f;
		
		}

	_t_1929 = _t_1914 * _t_1928;
	_t_1930 = tx2_5_1 * ty3_9_1;
	_t_1931 = tx3_6_1 * ty2_8_1;
	_t_1932 = _t_1931 * -1.0f;
	_t_1933 = _t_1930 + _t_1932;
	_t_1934 = -1.0f * ty3_9_1;
	_t_1935 = ty2_8_1 + _t_1934;
	_t_1936 = _t_1935 * _t_1513;
	_t_1937 = _t_1933 + _t_1936;
	_t_1938 = -1.0f * tx2_5_1;
	_t_1939 = tx3_6_1 + _t_1938;
	_t_1940 = _t_1939 * _t_1566;
	_t_1941 = _t_1937 + _t_1940;
	_t_1942 = _t_1941 < 0.0f;
	if(_t_1942)
		{
		
			_t_1943 = 1.0f;
		
		}
else
		{
		
			_t_1943 = 0.0f;
		
		}

	_t_1944 = _t_1929 * _t_1943;
	_t_1945 = _t_1944 * ty2_8_1;
	_t_1946 = _t_1945 * -1.0f;
	_t_1947 = -1.0f * pc0_14_1;
	_t_1948 = tc0_17_1 + _t_1947;
	_t_1949 = _t_1948 * _t_1948;
	_t_1950 = -1.0f * pc1_15_1;
	_t_1951 = tc1_18_1 + _t_1950;
	_t_1952 = _t_1951 * _t_1951;
	_t_1953 = _t_1949 + _t_1952;
	_t_1954 = -1.0f * pc2_16_1;
	_t_1955 = tc2_19_1 + _t_1954;
	_t_1956 = _t_1955 * _t_1955;
	_t_1957 = _t_1953 + _t_1956;
	_t_1958 = tx3_6_1 * ty1_7_1;
	_t_1959 = tx1_4_1 * ty3_9_1;
	_t_1960 = _t_1959 * -1.0f;
	_t_1961 = _t_1958 + _t_1960;
	_t_1962 = -1.0f * ty1_7_1;
	_t_1963 = ty3_9_1 + _t_1962;
	_t_1964 = _t_1963 * _t_1513;
	_t_1965 = _t_1961 + _t_1964;
	_t_1966 = -1.0f * tx3_6_1;
	_t_1967 = tx1_4_1 + _t_1966;
	_t_1968 = _t_1967 * _t_1566;
	_t_1969 = _t_1965 + _t_1968;
	_t_1970 = _t_1969 < 0.0f;
	if(_t_1970)
		{
		
			_t_1971 = 1.0f;
		
		}
else
		{
		
			_t_1971 = 0.0f;
		
		}

	_t_1972 = _t_1957 * _t_1971;
	_t_1973 = tx2_5_1 * ty3_9_1;
	_t_1974 = tx3_6_1 * ty2_8_1;
	_t_1975 = _t_1974 * -1.0f;
	_t_1976 = _t_1973 + _t_1975;
	_t_1977 = -1.0f * ty3_9_1;
	_t_1978 = ty2_8_1 + _t_1977;
	_t_1979 = _t_1978 * _t_1513;
	_t_1980 = _t_1976 + _t_1979;
	_t_1981 = -1.0f * tx2_5_1;
	_t_1982 = tx3_6_1 + _t_1981;
	_t_1983 = _t_1982 * _t_1566;
	_t_1984 = _t_1980 + _t_1983;
	_t_1985 = _t_1984 < 0.0f;
	if(_t_1985)
		{
		
			_t_1986 = 1.0f;
		
		}
else
		{
		
			_t_1986 = 0.0f;
		
		}

	_t_1987 = _t_1972 * _t_1986;
	_t_1988 = _t_1987 * _t_1566;
	_t_1989 = _t_1946 + _t_1988;
	_t_1487 = _t_1903 * _t_1989;

	return _t_1487;
}
__device__ float tegpixellet_block_19(float ty1_7_1,float ty2_8_1,float _t_119,float _t_1486,float tx2_5_1,float tx1_4_1,float y__2573_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	float _t_1488;
	float _t_1489;
	float _t_1490;
	bool _t_1491;
	float _t_1494;
	float _t_1498;
	float _t_1499;
	float _t_1500;
	float _t_1501;
	float _t_1502;
	bool _t_1503;
	float _t_1506;
	float _t_1510;
	float _t_1511;
	float _t_1512;
	float _t_1513;
	float _t_1514;
	float _t_1515;
	float _t_1516;
	bool _t_1517;
	float _t_1520;
	float _t_1524;
	float _t_1525;
	float _t_1526;
	float _t_1527;
	bool _t_1528;
	float _t_1531;
	float _t_1535;
	float _t_1536;
	float _t_1537;
	float _t_1538;
	float _t_1539;
	bool _t_1540;
	float _t_1543;
	float _t_1547;
	float _t_1548;
	float _t_1549;
	float _t_1550;
	float _t_1551;
	float _t_1552;
	float _t_1553;
	float _t_1554;
	float _t_1555;
	float _t_1556;
	bool _t_1557;
	float _t_1560;
	float _t_1564;
	float _t_1565;
	float _t_1566;

	float _t_1487;

	_t_1488 = -1.0f * ty2_8_1;
	_t_1489 = ty1_7_1 + _t_1488;
	_t_1490 = -1.0f * _t_1489;
	_t_1491 = _t_1490 < 0.0f;
	if(_t_1491)
		{
			float _t_1492;
			float _t_1493;
		
			_t_1492 = -1.0f * ty2_8_1;
			_t_1493 = ty1_7_1 + _t_1492;
			_t_1494 = _t_1493;
		
		}
else
		{
			float _t_1495;
			float _t_1496;
			float _t_1497;
		
			_t_1495 = -1.0f * ty2_8_1;
			_t_1496 = ty1_7_1 + _t_1495;
			_t_1497 = -1.0f * _t_1496;
			_t_1494 = _t_1497;
		
		}

	_t_1498 = _t_1494 * _t_119;
	_t_1499 = _t_1498 * _t_1486;
	_t_1500 = -1.0f * ty2_8_1;
	_t_1501 = ty1_7_1 + _t_1500;
	_t_1502 = -1.0f * _t_1501;
	_t_1503 = _t_1502 < 0.0f;
	if(_t_1503)
		{
			float _t_1504;
			float _t_1505;
		
			_t_1504 = -1.0f * tx1_4_1;
			_t_1505 = tx2_5_1 + _t_1504;
			_t_1506 = _t_1505;
		
		}
else
		{
			float _t_1507;
			float _t_1508;
			float _t_1509;
		
			_t_1507 = -1.0f * tx1_4_1;
			_t_1508 = tx2_5_1 + _t_1507;
			_t_1509 = -1.0f * _t_1508;
			_t_1506 = _t_1509;
		
		}

	_t_1510 = _t_1506 * _t_119;
	_t_1511 = _t_1510 * -1.0f;
	_t_1512 = _t_1511 * y__2573_1;
	_t_1513 = _t_1499 + _t_1512;
	_t_1514 = -1.0f * ty2_8_1;
	_t_1515 = ty1_7_1 + _t_1514;
	_t_1516 = -1.0f * _t_1515;
	_t_1517 = _t_1516 < 0.0f;
	if(_t_1517)
		{
			float _t_1518;
			float _t_1519;
		
			_t_1518 = -1.0f * tx1_4_1;
			_t_1519 = tx2_5_1 + _t_1518;
			_t_1520 = _t_1519;
		
		}
else
		{
			float _t_1521;
			float _t_1522;
			float _t_1523;
		
			_t_1521 = -1.0f * tx1_4_1;
			_t_1522 = tx2_5_1 + _t_1521;
			_t_1523 = -1.0f * _t_1522;
			_t_1520 = _t_1523;
		
		}

	_t_1524 = _t_1520 * _t_119;
	_t_1525 = -1.0f * ty2_8_1;
	_t_1526 = ty1_7_1 + _t_1525;
	_t_1527 = -1.0f * _t_1526;
	_t_1528 = _t_1527 < 0.0f;
	if(_t_1528)
		{
			float _t_1529;
			float _t_1530;
		
			_t_1529 = -1.0f * tx1_4_1;
			_t_1530 = tx2_5_1 + _t_1529;
			_t_1531 = _t_1530;
		
		}
else
		{
			float _t_1532;
			float _t_1533;
			float _t_1534;
		
			_t_1532 = -1.0f * tx1_4_1;
			_t_1533 = tx2_5_1 + _t_1532;
			_t_1534 = -1.0f * _t_1533;
			_t_1531 = _t_1534;
		
		}

	_t_1535 = _t_1531 * _t_119;
	_t_1536 = _t_1524 * _t_1535;
	_t_1537 = -1.0f * ty2_8_1;
	_t_1538 = ty1_7_1 + _t_1537;
	_t_1539 = -1.0f * _t_1538;
	_t_1540 = _t_1539 < 0.0f;
	if(_t_1540)
		{
			float _t_1541;
			float _t_1542;
		
			_t_1541 = -1.0f * ty2_8_1;
			_t_1542 = ty1_7_1 + _t_1541;
			_t_1543 = _t_1542;
		
		}
else
		{
			float _t_1544;
			float _t_1545;
			float _t_1546;
		
			_t_1544 = -1.0f * ty2_8_1;
			_t_1545 = ty1_7_1 + _t_1544;
			_t_1546 = -1.0f * _t_1545;
			_t_1543 = _t_1546;
		
		}

	_t_1547 = _t_1543 * _t_119;
	_t_1548 = 1.0f + _t_1547;
	_t_1549 = 1.0f / _t_1548;
	_t_1550 = _t_1536 * _t_1549;
	_t_1551 = _t_1550 * -1.0f;
	_t_1552 = 1.0f + _t_1551;
	_t_1553 = _t_1552 * y__2573_1;
	_t_1554 = -1.0f * ty2_8_1;
	_t_1555 = ty1_7_1 + _t_1554;
	_t_1556 = -1.0f * _t_1555;
	_t_1557 = _t_1556 < 0.0f;
	if(_t_1557)
		{
			float _t_1558;
			float _t_1559;
		
			_t_1558 = -1.0f * tx1_4_1;
			_t_1559 = tx2_5_1 + _t_1558;
			_t_1560 = _t_1559;
		
		}
else
		{
			float _t_1561;
			float _t_1562;
			float _t_1563;
		
			_t_1561 = -1.0f * tx1_4_1;
			_t_1562 = tx2_5_1 + _t_1561;
			_t_1563 = -1.0f * _t_1562;
			_t_1560 = _t_1563;
		
		}

	_t_1564 = _t_1560 * _t_119;
	_t_1565 = _t_1564 * _t_1486;
	_t_1566 = _t_1553 + _t_1565;
	_t_1487 = tegpixellet_block_20(py0_12_1,_t_1566,py1_13_1,px0_10_1,_t_1513,px1_11_1,ty1_7_1,ty2_8_1,tx2_5_1,tx1_4_1,_t_119,y__2573_1,_t_1486,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);

	return _t_1487;
}
__device__ float tegpixelbody_block_17(float ty1_7_1,float ty2_8_1,float _t_119,float px0_10_1,float px1_11_1,float tx2_5_1,float tx1_4_1,float py0_12_1,float py1_13_1,float y__2573_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	float _t_1330;
	float _t_1331;
	float _t_1332;
	bool _t_1333;
	float _t_1336;
	float _t_1340;
	float _t_1341;
	float _t_1342;
	float _t_1343;
	bool _t_1344;
	float _t_1347;
	float _t_1351;
	bool _t_1352;
	float _t_1353;
	float _t_1354;
	float _t_1355;
	float _t_1356;
	float _t_1357;
	bool _t_1358;
	float _t_1361;
	float _t_1365;
	float _t_1366;
	float _t_1367;
	float _t_1368;
	bool _t_1369;
	float _t_1372;
	float _t_1376;
	bool _t_1377;
	float _t_1378;
	float _t_1379;
	float _t_1380;
	float _t_1381;
	float _t_1382;
	float _t_1383;
	bool _t_1384;
	float _t_1389;
	float _t_1395;
	float _t_1396;
	float _t_1397;
	float _t_1398;
	bool _t_1399;
	float _t_1400;
	float _t_1401;
	float _t_1402;
	bool _t_1403;
	float _t_1406;
	float _t_1410;
	float _t_1411;
	float _t_1412;
	float _t_1413;
	bool _t_1414;
	float _t_1417;
	float _t_1421;
	bool _t_1422;
	float _t_1423;
	float _t_1424;
	float _t_1425;
	float _t_1426;
	float _t_1427;
	bool _t_1428;
	float _t_1431;
	float _t_1435;
	float _t_1436;
	float _t_1437;
	float _t_1438;
	bool _t_1439;
	float _t_1442;
	float _t_1446;
	bool _t_1447;
	float _t_1448;
	float _t_1449;
	float _t_1450;
	float _t_1451;
	float _t_1452;
	float _t_1453;
	bool _t_1454;
	float _t_1459;
	float _t_1465;
	float _t_1466;
	float _t_1467;
	float _t_1468;
	bool _t_1469;
	bool _t_1470;

	float _t_1329;

	_t_1330 = -1.0f * ty2_8_1;
	_t_1331 = ty1_7_1 + _t_1330;
	_t_1332 = -1.0f * _t_1331;
	_t_1333 = _t_1332 < 0.0f;
	if(_t_1333)
		{
			float _t_1334;
			float _t_1335;
		
			_t_1334 = -1.0f * ty2_8_1;
			_t_1335 = ty1_7_1 + _t_1334;
			_t_1336 = _t_1335;
		
		}
else
		{
			float _t_1337;
			float _t_1338;
			float _t_1339;
		
			_t_1337 = -1.0f * ty2_8_1;
			_t_1338 = ty1_7_1 + _t_1337;
			_t_1339 = -1.0f * _t_1338;
			_t_1336 = _t_1339;
		
		}

	_t_1340 = _t_1336 * _t_119;
	_t_1341 = -1.0f * ty2_8_1;
	_t_1342 = ty1_7_1 + _t_1341;
	_t_1343 = -1.0f * _t_1342;
	_t_1344 = _t_1343 < 0.0f;
	if(_t_1344)
		{
			float _t_1345;
			float _t_1346;
		
			_t_1345 = -1.0f * ty2_8_1;
			_t_1346 = ty1_7_1 + _t_1345;
			_t_1347 = _t_1346;
		
		}
else
		{
			float _t_1348;
			float _t_1349;
			float _t_1350;
		
			_t_1348 = -1.0f * ty2_8_1;
			_t_1349 = ty1_7_1 + _t_1348;
			_t_1350 = -1.0f * _t_1349;
			_t_1347 = _t_1350;
		
		}

	_t_1351 = _t_1347 * _t_119;
	_t_1352 = 0.0f < _t_1351;
	if(_t_1352)
		{
		
			_t_1353 = px0_10_1;
		
		}
else
		{
		
			_t_1353 = px1_11_1;
		
		}

	_t_1354 = _t_1340 * _t_1353;
	_t_1355 = -1.0f * ty2_8_1;
	_t_1356 = ty1_7_1 + _t_1355;
	_t_1357 = -1.0f * _t_1356;
	_t_1358 = _t_1357 < 0.0f;
	if(_t_1358)
		{
			float _t_1359;
			float _t_1360;
		
			_t_1359 = -1.0f * tx1_4_1;
			_t_1360 = tx2_5_1 + _t_1359;
			_t_1361 = _t_1360;
		
		}
else
		{
			float _t_1362;
			float _t_1363;
			float _t_1364;
		
			_t_1362 = -1.0f * tx1_4_1;
			_t_1363 = tx2_5_1 + _t_1362;
			_t_1364 = -1.0f * _t_1363;
			_t_1361 = _t_1364;
		
		}

	_t_1365 = _t_1361 * _t_119;
	_t_1366 = -1.0f * ty2_8_1;
	_t_1367 = ty1_7_1 + _t_1366;
	_t_1368 = -1.0f * _t_1367;
	_t_1369 = _t_1368 < 0.0f;
	if(_t_1369)
		{
			float _t_1370;
			float _t_1371;
		
			_t_1370 = -1.0f * tx1_4_1;
			_t_1371 = tx2_5_1 + _t_1370;
			_t_1372 = _t_1371;
		
		}
else
		{
			float _t_1373;
			float _t_1374;
			float _t_1375;
		
			_t_1373 = -1.0f * tx1_4_1;
			_t_1374 = tx2_5_1 + _t_1373;
			_t_1375 = -1.0f * _t_1374;
			_t_1372 = _t_1375;
		
		}

	_t_1376 = _t_1372 * _t_119;
	_t_1377 = 0.0f < _t_1376;
	if(_t_1377)
		{
		
			_t_1378 = py0_12_1;
		
		}
else
		{
		
			_t_1378 = py1_13_1;
		
		}

	_t_1379 = _t_1365 * _t_1378;
	_t_1380 = _t_1354 + _t_1379;
	_t_1381 = -1.0f * ty2_8_1;
	_t_1382 = ty1_7_1 + _t_1381;
	_t_1383 = -1.0f * _t_1382;
	_t_1384 = _t_1383 < 0.0f;
	if(_t_1384)
		{
			float _t_1385;
			float _t_1386;
			float _t_1387;
			float _t_1388;
		
			_t_1385 = tx1_4_1 * ty2_8_1;
			_t_1386 = tx2_5_1 * ty1_7_1;
			_t_1387 = _t_1386 * -1.0f;
			_t_1388 = _t_1385 + _t_1387;
			_t_1389 = _t_1388;
		
		}
else
		{
			float _t_1390;
			float _t_1391;
			float _t_1392;
			float _t_1393;
			float _t_1394;
		
			_t_1390 = tx1_4_1 * ty2_8_1;
			_t_1391 = tx2_5_1 * ty1_7_1;
			_t_1392 = _t_1391 * -1.0f;
			_t_1393 = _t_1390 + _t_1392;
			_t_1394 = -1.0f * _t_1393;
			_t_1389 = _t_1394;
		
		}

	_t_1395 = -1.0f * _t_1389;
	_t_1396 = _t_1395 * _t_119;
	_t_1397 = _t_1396 * -1.0f;
	_t_1398 = _t_1380 + _t_1397;
	_t_1399 = _t_1398 < 0.0f;
	_t_1400 = -1.0f * ty2_8_1;
	_t_1401 = ty1_7_1 + _t_1400;
	_t_1402 = -1.0f * _t_1401;
	_t_1403 = _t_1402 < 0.0f;
	if(_t_1403)
		{
			float _t_1404;
			float _t_1405;
		
			_t_1404 = -1.0f * ty2_8_1;
			_t_1405 = ty1_7_1 + _t_1404;
			_t_1406 = _t_1405;
		
		}
else
		{
			float _t_1407;
			float _t_1408;
			float _t_1409;
		
			_t_1407 = -1.0f * ty2_8_1;
			_t_1408 = ty1_7_1 + _t_1407;
			_t_1409 = -1.0f * _t_1408;
			_t_1406 = _t_1409;
		
		}

	_t_1410 = _t_1406 * _t_119;
	_t_1411 = -1.0f * ty2_8_1;
	_t_1412 = ty1_7_1 + _t_1411;
	_t_1413 = -1.0f * _t_1412;
	_t_1414 = _t_1413 < 0.0f;
	if(_t_1414)
		{
			float _t_1415;
			float _t_1416;
		
			_t_1415 = -1.0f * ty2_8_1;
			_t_1416 = ty1_7_1 + _t_1415;
			_t_1417 = _t_1416;
		
		}
else
		{
			float _t_1418;
			float _t_1419;
			float _t_1420;
		
			_t_1418 = -1.0f * ty2_8_1;
			_t_1419 = ty1_7_1 + _t_1418;
			_t_1420 = -1.0f * _t_1419;
			_t_1417 = _t_1420;
		
		}

	_t_1421 = _t_1417 * _t_119;
	_t_1422 = 0.0f < _t_1421;
	if(_t_1422)
		{
		
			_t_1423 = px1_11_1;
		
		}
else
		{
		
			_t_1423 = px0_10_1;
		
		}

	_t_1424 = _t_1410 * _t_1423;
	_t_1425 = -1.0f * ty2_8_1;
	_t_1426 = ty1_7_1 + _t_1425;
	_t_1427 = -1.0f * _t_1426;
	_t_1428 = _t_1427 < 0.0f;
	if(_t_1428)
		{
			float _t_1429;
			float _t_1430;
		
			_t_1429 = -1.0f * tx1_4_1;
			_t_1430 = tx2_5_1 + _t_1429;
			_t_1431 = _t_1430;
		
		}
else
		{
			float _t_1432;
			float _t_1433;
			float _t_1434;
		
			_t_1432 = -1.0f * tx1_4_1;
			_t_1433 = tx2_5_1 + _t_1432;
			_t_1434 = -1.0f * _t_1433;
			_t_1431 = _t_1434;
		
		}

	_t_1435 = _t_1431 * _t_119;
	_t_1436 = -1.0f * ty2_8_1;
	_t_1437 = ty1_7_1 + _t_1436;
	_t_1438 = -1.0f * _t_1437;
	_t_1439 = _t_1438 < 0.0f;
	if(_t_1439)
		{
			float _t_1440;
			float _t_1441;
		
			_t_1440 = -1.0f * tx1_4_1;
			_t_1441 = tx2_5_1 + _t_1440;
			_t_1442 = _t_1441;
		
		}
else
		{
			float _t_1443;
			float _t_1444;
			float _t_1445;
		
			_t_1443 = -1.0f * tx1_4_1;
			_t_1444 = tx2_5_1 + _t_1443;
			_t_1445 = -1.0f * _t_1444;
			_t_1442 = _t_1445;
		
		}

	_t_1446 = _t_1442 * _t_119;
	_t_1447 = 0.0f < _t_1446;
	if(_t_1447)
		{
		
			_t_1448 = py1_13_1;
		
		}
else
		{
		
			_t_1448 = py0_12_1;
		
		}

	_t_1449 = _t_1435 * _t_1448;
	_t_1450 = _t_1424 + _t_1449;
	_t_1451 = -1.0f * ty2_8_1;
	_t_1452 = ty1_7_1 + _t_1451;
	_t_1453 = -1.0f * _t_1452;
	_t_1454 = _t_1453 < 0.0f;
	if(_t_1454)
		{
			float _t_1455;
			float _t_1456;
			float _t_1457;
			float _t_1458;
		
			_t_1455 = tx1_4_1 * ty2_8_1;
			_t_1456 = tx2_5_1 * ty1_7_1;
			_t_1457 = _t_1456 * -1.0f;
			_t_1458 = _t_1455 + _t_1457;
			_t_1459 = _t_1458;
		
		}
else
		{
			float _t_1460;
			float _t_1461;
			float _t_1462;
			float _t_1463;
			float _t_1464;
		
			_t_1460 = tx1_4_1 * ty2_8_1;
			_t_1461 = tx2_5_1 * ty1_7_1;
			_t_1462 = _t_1461 * -1.0f;
			_t_1463 = _t_1460 + _t_1462;
			_t_1464 = -1.0f * _t_1463;
			_t_1459 = _t_1464;
		
		}

	_t_1465 = -1.0f * _t_1459;
	_t_1466 = _t_1465 * _t_119;
	_t_1467 = _t_1466 * -1.0f;
	_t_1468 = _t_1450 + _t_1467;
	_t_1469 = 0.0f < _t_1468;
	_t_1470 = _t_1399 && _t_1469;
	if(_t_1470)
		{
			float _t_1471;
			float _t_1472;
			float _t_1473;
			bool _t_1474;
			float _t_1479;
			float _t_1485;
			float _t_1486;
			float _t_1487;
		
			_t_1471 = -1.0f * ty2_8_1;
			_t_1472 = ty1_7_1 + _t_1471;
			_t_1473 = -1.0f * _t_1472;
			_t_1474 = _t_1473 < 0.0f;
			if(_t_1474)
				{
					float _t_1475;
					float _t_1476;
					float _t_1477;
					float _t_1478;
				
					_t_1475 = tx1_4_1 * ty2_8_1;
					_t_1476 = tx2_5_1 * ty1_7_1;
					_t_1477 = _t_1476 * -1.0f;
					_t_1478 = _t_1475 + _t_1477;
					_t_1479 = _t_1478;
				
				}
		else
				{
					float _t_1480;
					float _t_1481;
					float _t_1482;
					float _t_1483;
					float _t_1484;
				
					_t_1480 = tx1_4_1 * ty2_8_1;
					_t_1481 = tx2_5_1 * ty1_7_1;
					_t_1482 = _t_1481 * -1.0f;
					_t_1483 = _t_1480 + _t_1482;
					_t_1484 = -1.0f * _t_1483;
					_t_1479 = _t_1484;
				
				}
		
			_t_1485 = -1.0f * _t_1479;
			_t_1486 = _t_1485 * _t_119;
			_t_1487 = tegpixellet_block_19(ty1_7_1,ty2_8_1,_t_119,_t_1486,tx2_5_1,tx1_4_1,y__2573_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);
			_t_1329 = _t_1487;
		
		}
else
		{
		
			_t_1329 = 0.0f;
		
		}


	return _t_1329;
}
__device__ float tegpixelintegrator_17(float pc1_15_1,float ty3_9_1,float tc2_19_1,float _t_1219,float ty2_8_1,float pc0_14_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float tx2_5_1,float py1_13_1,float pc2_16_1,float px1_11_1,float _t_119,float tc0_17_1,float py0_12_1,float _t_1328,float tc1_18_1,float px0_10_1){
    float y__2573_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_1328 - _t_1219)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__2573_1 = _t_1219 + __step__ * (i + (float)(0.5));
        float _t_1329;
		_t_1329 = tegpixelbody_block_17(ty1_7_1,ty2_8_1,_t_119,px0_10_1,px1_11_1,tx2_5_1,tx1_4_1,py0_12_1,py1_13_1,y__2573_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);;
        __output__ = __output__ + _t_1329 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_1(float ty1_7_1,float ty2_8_1,float tx2_5_1,float tx1_4_1,float _t_119,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	float _t_1111;
	float _t_1112;
	float _t_1113;
	bool _t_1114;
	float _t_1117;
	float _t_1121;
	float _t_1122;
	float _t_1123;
	float _t_1124;
	float _t_1125;
	bool _t_1126;
	float _t_1129;
	float _t_1133;
	float _t_1134;
	bool _t_1135;
	float _t_1136;
	float _t_1137;
	float _t_1138;
	float _t_1139;
	float _t_1140;
	bool _t_1141;
	float _t_1144;
	float _t_1148;
	float _t_1149;
	float _t_1150;
	float _t_1151;
	bool _t_1152;
	float _t_1155;
	float _t_1159;
	float _t_1160;
	float _t_1161;
	float _t_1162;
	float _t_1163;
	bool _t_1164;
	float _t_1167;
	float _t_1171;
	float _t_1172;
	float _t_1173;
	float _t_1174;
	float _t_1175;
	float _t_1176;
	float _t_1177;
	float _t_1178;
	float _t_1179;
	bool _t_1180;
	float _t_1183;
	float _t_1187;
	float _t_1188;
	float _t_1189;
	float _t_1190;
	bool _t_1191;
	float _t_1194;
	float _t_1198;
	float _t_1199;
	float _t_1200;
	float _t_1201;
	float _t_1202;
	bool _t_1203;
	float _t_1206;
	float _t_1210;
	float _t_1211;
	float _t_1212;
	float _t_1213;
	float _t_1214;
	float _t_1215;
	bool _t_1216;
	float _t_1217;
	float _t_1218;
	float _t_1219;
	float _t_1220;
	float _t_1221;
	float _t_1222;
	bool _t_1223;
	float _t_1226;
	float _t_1230;
	float _t_1231;
	float _t_1232;
	float _t_1233;
	float _t_1234;
	bool _t_1235;
	float _t_1238;
	float _t_1242;
	float _t_1243;
	bool _t_1244;
	float _t_1245;
	float _t_1246;
	float _t_1247;
	float _t_1248;
	float _t_1249;
	bool _t_1250;
	float _t_1253;
	float _t_1257;
	float _t_1258;
	float _t_1259;
	float _t_1260;
	bool _t_1261;
	float _t_1264;
	float _t_1268;
	float _t_1269;
	float _t_1270;
	float _t_1271;
	float _t_1272;
	bool _t_1273;
	float _t_1276;
	float _t_1280;
	float _t_1281;
	float _t_1282;
	float _t_1283;
	float _t_1284;
	float _t_1285;
	float _t_1286;
	float _t_1287;
	float _t_1288;
	bool _t_1289;
	float _t_1292;
	float _t_1296;
	float _t_1297;
	float _t_1298;
	float _t_1299;
	bool _t_1300;
	float _t_1303;
	float _t_1307;
	float _t_1308;
	float _t_1309;
	float _t_1310;
	float _t_1311;
	bool _t_1312;
	float _t_1315;
	float _t_1319;
	float _t_1320;
	float _t_1321;
	float _t_1322;
	float _t_1323;
	float _t_1324;
	bool _t_1325;
	float _t_1326;
	float _t_1327;
	float _t_1328;

	float _t_120;

	_t_1111 = -1.0f * ty2_8_1;
	_t_1112 = ty1_7_1 + _t_1111;
	_t_1113 = -1.0f * _t_1112;
	_t_1114 = _t_1113 < 0.0f;
	if(_t_1114)
		{
			float _t_1115;
			float _t_1116;
		
			_t_1115 = -1.0f * tx1_4_1;
			_t_1116 = tx2_5_1 + _t_1115;
			_t_1117 = _t_1116;
		
		}
else
		{
			float _t_1118;
			float _t_1119;
			float _t_1120;
		
			_t_1118 = -1.0f * tx1_4_1;
			_t_1119 = tx2_5_1 + _t_1118;
			_t_1120 = -1.0f * _t_1119;
			_t_1117 = _t_1120;
		
		}

	_t_1121 = _t_1117 * _t_119;
	_t_1122 = _t_1121 * -1.0f;
	_t_1123 = -1.0f * ty2_8_1;
	_t_1124 = ty1_7_1 + _t_1123;
	_t_1125 = -1.0f * _t_1124;
	_t_1126 = _t_1125 < 0.0f;
	if(_t_1126)
		{
			float _t_1127;
			float _t_1128;
		
			_t_1127 = -1.0f * tx1_4_1;
			_t_1128 = tx2_5_1 + _t_1127;
			_t_1129 = _t_1128;
		
		}
else
		{
			float _t_1130;
			float _t_1131;
			float _t_1132;
		
			_t_1130 = -1.0f * tx1_4_1;
			_t_1131 = tx2_5_1 + _t_1130;
			_t_1132 = -1.0f * _t_1131;
			_t_1129 = _t_1132;
		
		}

	_t_1133 = _t_1129 * _t_119;
	_t_1134 = _t_1133 * -1.0f;
	_t_1135 = 0.0f < _t_1134;
	if(_t_1135)
		{
		
			_t_1136 = px0_10_1;
		
		}
else
		{
		
			_t_1136 = px1_11_1;
		
		}

	_t_1137 = _t_1122 * _t_1136;
	_t_1138 = -1.0f * ty2_8_1;
	_t_1139 = ty1_7_1 + _t_1138;
	_t_1140 = -1.0f * _t_1139;
	_t_1141 = _t_1140 < 0.0f;
	if(_t_1141)
		{
			float _t_1142;
			float _t_1143;
		
			_t_1142 = -1.0f * tx1_4_1;
			_t_1143 = tx2_5_1 + _t_1142;
			_t_1144 = _t_1143;
		
		}
else
		{
			float _t_1145;
			float _t_1146;
			float _t_1147;
		
			_t_1145 = -1.0f * tx1_4_1;
			_t_1146 = tx2_5_1 + _t_1145;
			_t_1147 = -1.0f * _t_1146;
			_t_1144 = _t_1147;
		
		}

	_t_1148 = _t_1144 * _t_119;
	_t_1149 = -1.0f * ty2_8_1;
	_t_1150 = ty1_7_1 + _t_1149;
	_t_1151 = -1.0f * _t_1150;
	_t_1152 = _t_1151 < 0.0f;
	if(_t_1152)
		{
			float _t_1153;
			float _t_1154;
		
			_t_1153 = -1.0f * tx1_4_1;
			_t_1154 = tx2_5_1 + _t_1153;
			_t_1155 = _t_1154;
		
		}
else
		{
			float _t_1156;
			float _t_1157;
			float _t_1158;
		
			_t_1156 = -1.0f * tx1_4_1;
			_t_1157 = tx2_5_1 + _t_1156;
			_t_1158 = -1.0f * _t_1157;
			_t_1155 = _t_1158;
		
		}

	_t_1159 = _t_1155 * _t_119;
	_t_1160 = _t_1148 * _t_1159;
	_t_1161 = -1.0f * ty2_8_1;
	_t_1162 = ty1_7_1 + _t_1161;
	_t_1163 = -1.0f * _t_1162;
	_t_1164 = _t_1163 < 0.0f;
	if(_t_1164)
		{
			float _t_1165;
			float _t_1166;
		
			_t_1165 = -1.0f * ty2_8_1;
			_t_1166 = ty1_7_1 + _t_1165;
			_t_1167 = _t_1166;
		
		}
else
		{
			float _t_1168;
			float _t_1169;
			float _t_1170;
		
			_t_1168 = -1.0f * ty2_8_1;
			_t_1169 = ty1_7_1 + _t_1168;
			_t_1170 = -1.0f * _t_1169;
			_t_1167 = _t_1170;
		
		}

	_t_1171 = _t_1167 * _t_119;
	_t_1172 = 1.0f + _t_1171;
	_t_1173 = 1.0f / _t_1172;
	_t_1174 = _t_1160 * _t_1173;
	_t_1175 = _t_1174 * -1.0f;
	_t_1176 = 1.0f + _t_1175;
	_t_1177 = -1.0f * ty2_8_1;
	_t_1178 = ty1_7_1 + _t_1177;
	_t_1179 = -1.0f * _t_1178;
	_t_1180 = _t_1179 < 0.0f;
	if(_t_1180)
		{
			float _t_1181;
			float _t_1182;
		
			_t_1181 = -1.0f * tx1_4_1;
			_t_1182 = tx2_5_1 + _t_1181;
			_t_1183 = _t_1182;
		
		}
else
		{
			float _t_1184;
			float _t_1185;
			float _t_1186;
		
			_t_1184 = -1.0f * tx1_4_1;
			_t_1185 = tx2_5_1 + _t_1184;
			_t_1186 = -1.0f * _t_1185;
			_t_1183 = _t_1186;
		
		}

	_t_1187 = _t_1183 * _t_119;
	_t_1188 = -1.0f * ty2_8_1;
	_t_1189 = ty1_7_1 + _t_1188;
	_t_1190 = -1.0f * _t_1189;
	_t_1191 = _t_1190 < 0.0f;
	if(_t_1191)
		{
			float _t_1192;
			float _t_1193;
		
			_t_1192 = -1.0f * tx1_4_1;
			_t_1193 = tx2_5_1 + _t_1192;
			_t_1194 = _t_1193;
		
		}
else
		{
			float _t_1195;
			float _t_1196;
			float _t_1197;
		
			_t_1195 = -1.0f * tx1_4_1;
			_t_1196 = tx2_5_1 + _t_1195;
			_t_1197 = -1.0f * _t_1196;
			_t_1194 = _t_1197;
		
		}

	_t_1198 = _t_1194 * _t_119;
	_t_1199 = _t_1187 * _t_1198;
	_t_1200 = -1.0f * ty2_8_1;
	_t_1201 = ty1_7_1 + _t_1200;
	_t_1202 = -1.0f * _t_1201;
	_t_1203 = _t_1202 < 0.0f;
	if(_t_1203)
		{
			float _t_1204;
			float _t_1205;
		
			_t_1204 = -1.0f * ty2_8_1;
			_t_1205 = ty1_7_1 + _t_1204;
			_t_1206 = _t_1205;
		
		}
else
		{
			float _t_1207;
			float _t_1208;
			float _t_1209;
		
			_t_1207 = -1.0f * ty2_8_1;
			_t_1208 = ty1_7_1 + _t_1207;
			_t_1209 = -1.0f * _t_1208;
			_t_1206 = _t_1209;
		
		}

	_t_1210 = _t_1206 * _t_119;
	_t_1211 = 1.0f + _t_1210;
	_t_1212 = 1.0f / _t_1211;
	_t_1213 = _t_1199 * _t_1212;
	_t_1214 = _t_1213 * -1.0f;
	_t_1215 = 1.0f + _t_1214;
	_t_1216 = 0.0f < _t_1215;
	if(_t_1216)
		{
		
			_t_1217 = py0_12_1;
		
		}
else
		{
		
			_t_1217 = py1_13_1;
		
		}

	_t_1218 = _t_1176 * _t_1217;
	_t_1219 = _t_1137 + _t_1218;
	_t_1220 = -1.0f * ty2_8_1;
	_t_1221 = ty1_7_1 + _t_1220;
	_t_1222 = -1.0f * _t_1221;
	_t_1223 = _t_1222 < 0.0f;
	if(_t_1223)
		{
			float _t_1224;
			float _t_1225;
		
			_t_1224 = -1.0f * tx1_4_1;
			_t_1225 = tx2_5_1 + _t_1224;
			_t_1226 = _t_1225;
		
		}
else
		{
			float _t_1227;
			float _t_1228;
			float _t_1229;
		
			_t_1227 = -1.0f * tx1_4_1;
			_t_1228 = tx2_5_1 + _t_1227;
			_t_1229 = -1.0f * _t_1228;
			_t_1226 = _t_1229;
		
		}

	_t_1230 = _t_1226 * _t_119;
	_t_1231 = _t_1230 * -1.0f;
	_t_1232 = -1.0f * ty2_8_1;
	_t_1233 = ty1_7_1 + _t_1232;
	_t_1234 = -1.0f * _t_1233;
	_t_1235 = _t_1234 < 0.0f;
	if(_t_1235)
		{
			float _t_1236;
			float _t_1237;
		
			_t_1236 = -1.0f * tx1_4_1;
			_t_1237 = tx2_5_1 + _t_1236;
			_t_1238 = _t_1237;
		
		}
else
		{
			float _t_1239;
			float _t_1240;
			float _t_1241;
		
			_t_1239 = -1.0f * tx1_4_1;
			_t_1240 = tx2_5_1 + _t_1239;
			_t_1241 = -1.0f * _t_1240;
			_t_1238 = _t_1241;
		
		}

	_t_1242 = _t_1238 * _t_119;
	_t_1243 = _t_1242 * -1.0f;
	_t_1244 = 0.0f < _t_1243;
	if(_t_1244)
		{
		
			_t_1245 = px1_11_1;
		
		}
else
		{
		
			_t_1245 = px0_10_1;
		
		}

	_t_1246 = _t_1231 * _t_1245;
	_t_1247 = -1.0f * ty2_8_1;
	_t_1248 = ty1_7_1 + _t_1247;
	_t_1249 = -1.0f * _t_1248;
	_t_1250 = _t_1249 < 0.0f;
	if(_t_1250)
		{
			float _t_1251;
			float _t_1252;
		
			_t_1251 = -1.0f * tx1_4_1;
			_t_1252 = tx2_5_1 + _t_1251;
			_t_1253 = _t_1252;
		
		}
else
		{
			float _t_1254;
			float _t_1255;
			float _t_1256;
		
			_t_1254 = -1.0f * tx1_4_1;
			_t_1255 = tx2_5_1 + _t_1254;
			_t_1256 = -1.0f * _t_1255;
			_t_1253 = _t_1256;
		
		}

	_t_1257 = _t_1253 * _t_119;
	_t_1258 = -1.0f * ty2_8_1;
	_t_1259 = ty1_7_1 + _t_1258;
	_t_1260 = -1.0f * _t_1259;
	_t_1261 = _t_1260 < 0.0f;
	if(_t_1261)
		{
			float _t_1262;
			float _t_1263;
		
			_t_1262 = -1.0f * tx1_4_1;
			_t_1263 = tx2_5_1 + _t_1262;
			_t_1264 = _t_1263;
		
		}
else
		{
			float _t_1265;
			float _t_1266;
			float _t_1267;
		
			_t_1265 = -1.0f * tx1_4_1;
			_t_1266 = tx2_5_1 + _t_1265;
			_t_1267 = -1.0f * _t_1266;
			_t_1264 = _t_1267;
		
		}

	_t_1268 = _t_1264 * _t_119;
	_t_1269 = _t_1257 * _t_1268;
	_t_1270 = -1.0f * ty2_8_1;
	_t_1271 = ty1_7_1 + _t_1270;
	_t_1272 = -1.0f * _t_1271;
	_t_1273 = _t_1272 < 0.0f;
	if(_t_1273)
		{
			float _t_1274;
			float _t_1275;
		
			_t_1274 = -1.0f * ty2_8_1;
			_t_1275 = ty1_7_1 + _t_1274;
			_t_1276 = _t_1275;
		
		}
else
		{
			float _t_1277;
			float _t_1278;
			float _t_1279;
		
			_t_1277 = -1.0f * ty2_8_1;
			_t_1278 = ty1_7_1 + _t_1277;
			_t_1279 = -1.0f * _t_1278;
			_t_1276 = _t_1279;
		
		}

	_t_1280 = _t_1276 * _t_119;
	_t_1281 = 1.0f + _t_1280;
	_t_1282 = 1.0f / _t_1281;
	_t_1283 = _t_1269 * _t_1282;
	_t_1284 = _t_1283 * -1.0f;
	_t_1285 = 1.0f + _t_1284;
	_t_1286 = -1.0f * ty2_8_1;
	_t_1287 = ty1_7_1 + _t_1286;
	_t_1288 = -1.0f * _t_1287;
	_t_1289 = _t_1288 < 0.0f;
	if(_t_1289)
		{
			float _t_1290;
			float _t_1291;
		
			_t_1290 = -1.0f * tx1_4_1;
			_t_1291 = tx2_5_1 + _t_1290;
			_t_1292 = _t_1291;
		
		}
else
		{
			float _t_1293;
			float _t_1294;
			float _t_1295;
		
			_t_1293 = -1.0f * tx1_4_1;
			_t_1294 = tx2_5_1 + _t_1293;
			_t_1295 = -1.0f * _t_1294;
			_t_1292 = _t_1295;
		
		}

	_t_1296 = _t_1292 * _t_119;
	_t_1297 = -1.0f * ty2_8_1;
	_t_1298 = ty1_7_1 + _t_1297;
	_t_1299 = -1.0f * _t_1298;
	_t_1300 = _t_1299 < 0.0f;
	if(_t_1300)
		{
			float _t_1301;
			float _t_1302;
		
			_t_1301 = -1.0f * tx1_4_1;
			_t_1302 = tx2_5_1 + _t_1301;
			_t_1303 = _t_1302;
		
		}
else
		{
			float _t_1304;
			float _t_1305;
			float _t_1306;
		
			_t_1304 = -1.0f * tx1_4_1;
			_t_1305 = tx2_5_1 + _t_1304;
			_t_1306 = -1.0f * _t_1305;
			_t_1303 = _t_1306;
		
		}

	_t_1307 = _t_1303 * _t_119;
	_t_1308 = _t_1296 * _t_1307;
	_t_1309 = -1.0f * ty2_8_1;
	_t_1310 = ty1_7_1 + _t_1309;
	_t_1311 = -1.0f * _t_1310;
	_t_1312 = _t_1311 < 0.0f;
	if(_t_1312)
		{
			float _t_1313;
			float _t_1314;
		
			_t_1313 = -1.0f * ty2_8_1;
			_t_1314 = ty1_7_1 + _t_1313;
			_t_1315 = _t_1314;
		
		}
else
		{
			float _t_1316;
			float _t_1317;
			float _t_1318;
		
			_t_1316 = -1.0f * ty2_8_1;
			_t_1317 = ty1_7_1 + _t_1316;
			_t_1318 = -1.0f * _t_1317;
			_t_1315 = _t_1318;
		
		}

	_t_1319 = _t_1315 * _t_119;
	_t_1320 = 1.0f + _t_1319;
	_t_1321 = 1.0f / _t_1320;
	_t_1322 = _t_1308 * _t_1321;
	_t_1323 = _t_1322 * -1.0f;
	_t_1324 = 1.0f + _t_1323;
	_t_1325 = 0.0f < _t_1324;
	if(_t_1325)
		{
		
			_t_1326 = py1_13_1;
		
		}
else
		{
		
			_t_1326 = py0_12_1;
		
		}

	_t_1327 = _t_1285 * _t_1326;
	_t_1328 = _t_1246 + _t_1327;
	_t_120 = tegpixelintegrator_17(pc1_15_1,ty3_9_1,tc2_19_1,_t_1219,ty2_8_1,pc0_14_1,ty1_7_1,tx1_4_1,tx3_6_1,tx2_5_1,py1_13_1,pc2_16_1,px1_11_1,_t_119,tc0_17_1,py0_12_1,_t_1328,tc1_18_1,px0_10_1);

	return _t_120;
}
__device__ float tegpixellet_block_22(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float _t_2392,float _t_2445,float ty3_9_1,float tx3_6_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float _t_147,float y__2647_1,float _t_2365){
	float _t_2446;
	float _t_2447;
	float _t_2448;
	float _t_2449;
	float _t_2450;
	float _t_2451;
	float _t_2452;
	float _t_2453;
	float _t_2454;
	float _t_2455;
	float _t_2456;
	float _t_2457;
	float _t_2458;
	float _t_2459;
	float _t_2460;
	float _t_2461;
	float _t_2462;
	float _t_2463;
	float _t_2464;
	float _t_2465;
	float _t_2466;
	float _t_2467;
	float _t_2468;
	bool _t_2469;
	float _t_2470;
	float _t_2471;
	float _t_2472;
	float _t_2473;
	float _t_2474;
	float _t_2475;
	float _t_2476;
	float _t_2477;
	float _t_2478;
	float _t_2479;
	float _t_2480;
	float _t_2481;
	float _t_2482;
	bool _t_2483;
	float _t_2484;
	float _t_2485;
	float _t_2486;
	float _t_2487;
	bool _t_2488;
	bool _t_2489;
	bool _t_2490;
	bool _t_2491;
	bool _t_2492;
	bool _t_2493;
	bool _t_2494;
	float _t_2824;

	float _t_2366;

	_t_2446 = -1.0f * pc0_14_1;
	_t_2447 = tc0_17_1 + _t_2446;
	_t_2448 = _t_2447 * _t_2447;
	_t_2449 = -1.0f * pc1_15_1;
	_t_2450 = tc1_18_1 + _t_2449;
	_t_2451 = _t_2450 * _t_2450;
	_t_2452 = _t_2448 + _t_2451;
	_t_2453 = -1.0f * pc2_16_1;
	_t_2454 = tc2_19_1 + _t_2453;
	_t_2455 = _t_2454 * _t_2454;
	_t_2456 = _t_2452 + _t_2455;
	_t_2457 = tx1_4_1 * ty2_8_1;
	_t_2458 = tx2_5_1 * ty1_7_1;
	_t_2459 = _t_2458 * -1.0f;
	_t_2460 = _t_2457 + _t_2459;
	_t_2461 = -1.0f * ty2_8_1;
	_t_2462 = ty1_7_1 + _t_2461;
	_t_2463 = _t_2462 * _t_2392;
	_t_2464 = _t_2460 + _t_2463;
	_t_2465 = -1.0f * tx1_4_1;
	_t_2466 = tx2_5_1 + _t_2465;
	_t_2467 = _t_2466 * _t_2445;
	_t_2468 = _t_2464 + _t_2467;
	_t_2469 = _t_2468 < 0.0f;
	if(_t_2469)
		{
		
			_t_2470 = 1.0f;
		
		}
else
		{
		
			_t_2470 = 0.0f;
		
		}

	_t_2471 = tx2_5_1 * ty3_9_1;
	_t_2472 = tx3_6_1 * ty2_8_1;
	_t_2473 = _t_2472 * -1.0f;
	_t_2474 = _t_2471 + _t_2473;
	_t_2475 = -1.0f * ty3_9_1;
	_t_2476 = ty2_8_1 + _t_2475;
	_t_2477 = _t_2476 * _t_2392;
	_t_2478 = _t_2474 + _t_2477;
	_t_2479 = -1.0f * tx2_5_1;
	_t_2480 = tx3_6_1 + _t_2479;
	_t_2481 = _t_2480 * _t_2445;
	_t_2482 = _t_2478 + _t_2481;
	_t_2483 = _t_2482 < 0.0f;
	if(_t_2483)
		{
		
			_t_2484 = 1.0f;
		
		}
else
		{
		
			_t_2484 = 0.0f;
		
		}

	_t_2485 = _t_2470 * _t_2484;
	_t_2486 = _t_2456 * _t_2485;
	_t_2487 = _t_2486 * ty3_9_1;
	_t_2488 = py0_12_1 < _t_2445;
	_t_2489 = _t_2445 < py1_13_1;
	_t_2490 = _t_2488 && _t_2489;
	_t_2491 = px0_10_1 < _t_2392;
	_t_2492 = _t_2392 < px1_11_1;
	_t_2493 = _t_2491 && _t_2492;
	_t_2494 = _t_2490 && _t_2493;
	if(_t_2494)
		{
			float _t_2495;
			float _t_2496;
			float _t_2497;
			bool _t_2498;
			float _t_2501;
			float _t_2505;
			float _t_2506;
			float _t_2507;
			float _t_2508;
			float _t_2509;
			bool _t_2510;
			float _t_2513;
			float _t_2517;
			float _t_2518;
			bool _t_2519;
			float _t_2520;
			float _t_2521;
			float _t_2522;
			float _t_2523;
			float _t_2524;
			bool _t_2525;
			float _t_2528;
			float _t_2532;
			float _t_2533;
			float _t_2534;
			float _t_2535;
			bool _t_2536;
			float _t_2539;
			float _t_2543;
			float _t_2544;
			float _t_2545;
			float _t_2546;
			float _t_2547;
			bool _t_2548;
			float _t_2551;
			float _t_2555;
			float _t_2556;
			float _t_2557;
			float _t_2558;
			float _t_2559;
			float _t_2560;
			float _t_2561;
			float _t_2562;
			float _t_2563;
			bool _t_2564;
			float _t_2567;
			float _t_2571;
			float _t_2572;
			float _t_2573;
			float _t_2574;
			bool _t_2575;
			float _t_2578;
			float _t_2582;
			float _t_2583;
			float _t_2584;
			float _t_2585;
			float _t_2586;
			bool _t_2587;
			float _t_2590;
			float _t_2594;
			float _t_2595;
			float _t_2596;
			float _t_2597;
			float _t_2598;
			float _t_2599;
			bool _t_2600;
			float _t_2601;
			float _t_2602;
			float _t_2603;
			bool _t_2604;
			float _t_2605;
			float _t_2606;
			float _t_2607;
			bool _t_2608;
			float _t_2611;
			float _t_2615;
			float _t_2616;
			float _t_2617;
			float _t_2618;
			float _t_2619;
			bool _t_2620;
			float _t_2623;
			float _t_2627;
			float _t_2628;
			bool _t_2629;
			float _t_2630;
			float _t_2631;
			float _t_2632;
			float _t_2633;
			float _t_2634;
			bool _t_2635;
			float _t_2638;
			float _t_2642;
			float _t_2643;
			float _t_2644;
			float _t_2645;
			bool _t_2646;
			float _t_2649;
			float _t_2653;
			float _t_2654;
			float _t_2655;
			float _t_2656;
			float _t_2657;
			bool _t_2658;
			float _t_2661;
			float _t_2665;
			float _t_2666;
			float _t_2667;
			float _t_2668;
			float _t_2669;
			float _t_2670;
			float _t_2671;
			float _t_2672;
			float _t_2673;
			bool _t_2674;
			float _t_2677;
			float _t_2681;
			float _t_2682;
			float _t_2683;
			float _t_2684;
			bool _t_2685;
			float _t_2688;
			float _t_2692;
			float _t_2693;
			float _t_2694;
			float _t_2695;
			float _t_2696;
			bool _t_2697;
			float _t_2700;
			float _t_2704;
			float _t_2705;
			float _t_2706;
			float _t_2707;
			float _t_2708;
			float _t_2709;
			bool _t_2710;
			float _t_2711;
			float _t_2712;
			float _t_2713;
			bool _t_2714;
			bool _t_2715;
			float _t_2716;
			float _t_2717;
			float _t_2718;
			bool _t_2719;
			float _t_2722;
			float _t_2726;
			float _t_2727;
			float _t_2728;
			float _t_2729;
			bool _t_2730;
			float _t_2733;
			float _t_2737;
			bool _t_2738;
			float _t_2739;
			float _t_2740;
			float _t_2741;
			float _t_2742;
			float _t_2743;
			bool _t_2744;
			float _t_2747;
			float _t_2751;
			float _t_2752;
			float _t_2753;
			float _t_2754;
			bool _t_2755;
			float _t_2758;
			float _t_2762;
			bool _t_2763;
			float _t_2764;
			float _t_2765;
			float _t_2766;
			bool _t_2767;
			float _t_2768;
			float _t_2769;
			float _t_2770;
			bool _t_2771;
			float _t_2774;
			float _t_2778;
			float _t_2779;
			float _t_2780;
			float _t_2781;
			bool _t_2782;
			float _t_2785;
			float _t_2789;
			bool _t_2790;
			float _t_2791;
			float _t_2792;
			float _t_2793;
			float _t_2794;
			float _t_2795;
			bool _t_2796;
			float _t_2799;
			float _t_2803;
			float _t_2804;
			float _t_2805;
			float _t_2806;
			bool _t_2807;
			float _t_2810;
			float _t_2814;
			bool _t_2815;
			float _t_2816;
			float _t_2817;
			float _t_2818;
			bool _t_2819;
			bool _t_2820;
			bool _t_2821;
			float _t_2822;
			float _t_2823;
		
			_t_2495 = -1.0f * ty1_7_1;
			_t_2496 = ty3_9_1 + _t_2495;
			_t_2497 = -1.0f * _t_2496;
			_t_2498 = _t_2497 < 0.0f;
			if(_t_2498)
				{
					float _t_2499;
					float _t_2500;
				
					_t_2499 = -1.0f * tx3_6_1;
					_t_2500 = tx1_4_1 + _t_2499;
					_t_2501 = _t_2500;
				
				}
		else
				{
					float _t_2502;
					float _t_2503;
					float _t_2504;
				
					_t_2502 = -1.0f * tx3_6_1;
					_t_2503 = tx1_4_1 + _t_2502;
					_t_2504 = -1.0f * _t_2503;
					_t_2501 = _t_2504;
				
				}
		
			_t_2505 = _t_2501 * _t_147;
			_t_2506 = _t_2505 * -1.0f;
			_t_2507 = -1.0f * ty1_7_1;
			_t_2508 = ty3_9_1 + _t_2507;
			_t_2509 = -1.0f * _t_2508;
			_t_2510 = _t_2509 < 0.0f;
			if(_t_2510)
				{
					float _t_2511;
					float _t_2512;
				
					_t_2511 = -1.0f * tx3_6_1;
					_t_2512 = tx1_4_1 + _t_2511;
					_t_2513 = _t_2512;
				
				}
		else
				{
					float _t_2514;
					float _t_2515;
					float _t_2516;
				
					_t_2514 = -1.0f * tx3_6_1;
					_t_2515 = tx1_4_1 + _t_2514;
					_t_2516 = -1.0f * _t_2515;
					_t_2513 = _t_2516;
				
				}
		
			_t_2517 = _t_2513 * _t_147;
			_t_2518 = _t_2517 * -1.0f;
			_t_2519 = 0.0f < _t_2518;
			if(_t_2519)
				{
				
					_t_2520 = px0_10_1;
				
				}
		else
				{
				
					_t_2520 = px1_11_1;
				
				}
		
			_t_2521 = _t_2506 * _t_2520;
			_t_2522 = -1.0f * ty1_7_1;
			_t_2523 = ty3_9_1 + _t_2522;
			_t_2524 = -1.0f * _t_2523;
			_t_2525 = _t_2524 < 0.0f;
			if(_t_2525)
				{
					float _t_2526;
					float _t_2527;
				
					_t_2526 = -1.0f * tx3_6_1;
					_t_2527 = tx1_4_1 + _t_2526;
					_t_2528 = _t_2527;
				
				}
		else
				{
					float _t_2529;
					float _t_2530;
					float _t_2531;
				
					_t_2529 = -1.0f * tx3_6_1;
					_t_2530 = tx1_4_1 + _t_2529;
					_t_2531 = -1.0f * _t_2530;
					_t_2528 = _t_2531;
				
				}
		
			_t_2532 = _t_2528 * _t_147;
			_t_2533 = -1.0f * ty1_7_1;
			_t_2534 = ty3_9_1 + _t_2533;
			_t_2535 = -1.0f * _t_2534;
			_t_2536 = _t_2535 < 0.0f;
			if(_t_2536)
				{
					float _t_2537;
					float _t_2538;
				
					_t_2537 = -1.0f * tx3_6_1;
					_t_2538 = tx1_4_1 + _t_2537;
					_t_2539 = _t_2538;
				
				}
		else
				{
					float _t_2540;
					float _t_2541;
					float _t_2542;
				
					_t_2540 = -1.0f * tx3_6_1;
					_t_2541 = tx1_4_1 + _t_2540;
					_t_2542 = -1.0f * _t_2541;
					_t_2539 = _t_2542;
				
				}
		
			_t_2543 = _t_2539 * _t_147;
			_t_2544 = _t_2532 * _t_2543;
			_t_2545 = -1.0f * ty1_7_1;
			_t_2546 = ty3_9_1 + _t_2545;
			_t_2547 = -1.0f * _t_2546;
			_t_2548 = _t_2547 < 0.0f;
			if(_t_2548)
				{
					float _t_2549;
					float _t_2550;
				
					_t_2549 = -1.0f * ty1_7_1;
					_t_2550 = ty3_9_1 + _t_2549;
					_t_2551 = _t_2550;
				
				}
		else
				{
					float _t_2552;
					float _t_2553;
					float _t_2554;
				
					_t_2552 = -1.0f * ty1_7_1;
					_t_2553 = ty3_9_1 + _t_2552;
					_t_2554 = -1.0f * _t_2553;
					_t_2551 = _t_2554;
				
				}
		
			_t_2555 = _t_2551 * _t_147;
			_t_2556 = 1.0f + _t_2555;
			_t_2557 = 1.0f / _t_2556;
			_t_2558 = _t_2544 * _t_2557;
			_t_2559 = _t_2558 * -1.0f;
			_t_2560 = 1.0f + _t_2559;
			_t_2561 = -1.0f * ty1_7_1;
			_t_2562 = ty3_9_1 + _t_2561;
			_t_2563 = -1.0f * _t_2562;
			_t_2564 = _t_2563 < 0.0f;
			if(_t_2564)
				{
					float _t_2565;
					float _t_2566;
				
					_t_2565 = -1.0f * tx3_6_1;
					_t_2566 = tx1_4_1 + _t_2565;
					_t_2567 = _t_2566;
				
				}
		else
				{
					float _t_2568;
					float _t_2569;
					float _t_2570;
				
					_t_2568 = -1.0f * tx3_6_1;
					_t_2569 = tx1_4_1 + _t_2568;
					_t_2570 = -1.0f * _t_2569;
					_t_2567 = _t_2570;
				
				}
		
			_t_2571 = _t_2567 * _t_147;
			_t_2572 = -1.0f * ty1_7_1;
			_t_2573 = ty3_9_1 + _t_2572;
			_t_2574 = -1.0f * _t_2573;
			_t_2575 = _t_2574 < 0.0f;
			if(_t_2575)
				{
					float _t_2576;
					float _t_2577;
				
					_t_2576 = -1.0f * tx3_6_1;
					_t_2577 = tx1_4_1 + _t_2576;
					_t_2578 = _t_2577;
				
				}
		else
				{
					float _t_2579;
					float _t_2580;
					float _t_2581;
				
					_t_2579 = -1.0f * tx3_6_1;
					_t_2580 = tx1_4_1 + _t_2579;
					_t_2581 = -1.0f * _t_2580;
					_t_2578 = _t_2581;
				
				}
		
			_t_2582 = _t_2578 * _t_147;
			_t_2583 = _t_2571 * _t_2582;
			_t_2584 = -1.0f * ty1_7_1;
			_t_2585 = ty3_9_1 + _t_2584;
			_t_2586 = -1.0f * _t_2585;
			_t_2587 = _t_2586 < 0.0f;
			if(_t_2587)
				{
					float _t_2588;
					float _t_2589;
				
					_t_2588 = -1.0f * ty1_7_1;
					_t_2589 = ty3_9_1 + _t_2588;
					_t_2590 = _t_2589;
				
				}
		else
				{
					float _t_2591;
					float _t_2592;
					float _t_2593;
				
					_t_2591 = -1.0f * ty1_7_1;
					_t_2592 = ty3_9_1 + _t_2591;
					_t_2593 = -1.0f * _t_2592;
					_t_2590 = _t_2593;
				
				}
		
			_t_2594 = _t_2590 * _t_147;
			_t_2595 = 1.0f + _t_2594;
			_t_2596 = 1.0f / _t_2595;
			_t_2597 = _t_2583 * _t_2596;
			_t_2598 = _t_2597 * -1.0f;
			_t_2599 = 1.0f + _t_2598;
			_t_2600 = 0.0f < _t_2599;
			if(_t_2600)
				{
				
					_t_2601 = py0_12_1;
				
				}
		else
				{
				
					_t_2601 = py1_13_1;
				
				}
		
			_t_2602 = _t_2560 * _t_2601;
			_t_2603 = _t_2521 + _t_2602;
			_t_2604 = _t_2603 < y__2647_1;
			_t_2605 = -1.0f * ty1_7_1;
			_t_2606 = ty3_9_1 + _t_2605;
			_t_2607 = -1.0f * _t_2606;
			_t_2608 = _t_2607 < 0.0f;
			if(_t_2608)
				{
					float _t_2609;
					float _t_2610;
				
					_t_2609 = -1.0f * tx3_6_1;
					_t_2610 = tx1_4_1 + _t_2609;
					_t_2611 = _t_2610;
				
				}
		else
				{
					float _t_2612;
					float _t_2613;
					float _t_2614;
				
					_t_2612 = -1.0f * tx3_6_1;
					_t_2613 = tx1_4_1 + _t_2612;
					_t_2614 = -1.0f * _t_2613;
					_t_2611 = _t_2614;
				
				}
		
			_t_2615 = _t_2611 * _t_147;
			_t_2616 = _t_2615 * -1.0f;
			_t_2617 = -1.0f * ty1_7_1;
			_t_2618 = ty3_9_1 + _t_2617;
			_t_2619 = -1.0f * _t_2618;
			_t_2620 = _t_2619 < 0.0f;
			if(_t_2620)
				{
					float _t_2621;
					float _t_2622;
				
					_t_2621 = -1.0f * tx3_6_1;
					_t_2622 = tx1_4_1 + _t_2621;
					_t_2623 = _t_2622;
				
				}
		else
				{
					float _t_2624;
					float _t_2625;
					float _t_2626;
				
					_t_2624 = -1.0f * tx3_6_1;
					_t_2625 = tx1_4_1 + _t_2624;
					_t_2626 = -1.0f * _t_2625;
					_t_2623 = _t_2626;
				
				}
		
			_t_2627 = _t_2623 * _t_147;
			_t_2628 = _t_2627 * -1.0f;
			_t_2629 = 0.0f < _t_2628;
			if(_t_2629)
				{
				
					_t_2630 = px1_11_1;
				
				}
		else
				{
				
					_t_2630 = px0_10_1;
				
				}
		
			_t_2631 = _t_2616 * _t_2630;
			_t_2632 = -1.0f * ty1_7_1;
			_t_2633 = ty3_9_1 + _t_2632;
			_t_2634 = -1.0f * _t_2633;
			_t_2635 = _t_2634 < 0.0f;
			if(_t_2635)
				{
					float _t_2636;
					float _t_2637;
				
					_t_2636 = -1.0f * tx3_6_1;
					_t_2637 = tx1_4_1 + _t_2636;
					_t_2638 = _t_2637;
				
				}
		else
				{
					float _t_2639;
					float _t_2640;
					float _t_2641;
				
					_t_2639 = -1.0f * tx3_6_1;
					_t_2640 = tx1_4_1 + _t_2639;
					_t_2641 = -1.0f * _t_2640;
					_t_2638 = _t_2641;
				
				}
		
			_t_2642 = _t_2638 * _t_147;
			_t_2643 = -1.0f * ty1_7_1;
			_t_2644 = ty3_9_1 + _t_2643;
			_t_2645 = -1.0f * _t_2644;
			_t_2646 = _t_2645 < 0.0f;
			if(_t_2646)
				{
					float _t_2647;
					float _t_2648;
				
					_t_2647 = -1.0f * tx3_6_1;
					_t_2648 = tx1_4_1 + _t_2647;
					_t_2649 = _t_2648;
				
				}
		else
				{
					float _t_2650;
					float _t_2651;
					float _t_2652;
				
					_t_2650 = -1.0f * tx3_6_1;
					_t_2651 = tx1_4_1 + _t_2650;
					_t_2652 = -1.0f * _t_2651;
					_t_2649 = _t_2652;
				
				}
		
			_t_2653 = _t_2649 * _t_147;
			_t_2654 = _t_2642 * _t_2653;
			_t_2655 = -1.0f * ty1_7_1;
			_t_2656 = ty3_9_1 + _t_2655;
			_t_2657 = -1.0f * _t_2656;
			_t_2658 = _t_2657 < 0.0f;
			if(_t_2658)
				{
					float _t_2659;
					float _t_2660;
				
					_t_2659 = -1.0f * ty1_7_1;
					_t_2660 = ty3_9_1 + _t_2659;
					_t_2661 = _t_2660;
				
				}
		else
				{
					float _t_2662;
					float _t_2663;
					float _t_2664;
				
					_t_2662 = -1.0f * ty1_7_1;
					_t_2663 = ty3_9_1 + _t_2662;
					_t_2664 = -1.0f * _t_2663;
					_t_2661 = _t_2664;
				
				}
		
			_t_2665 = _t_2661 * _t_147;
			_t_2666 = 1.0f + _t_2665;
			_t_2667 = 1.0f / _t_2666;
			_t_2668 = _t_2654 * _t_2667;
			_t_2669 = _t_2668 * -1.0f;
			_t_2670 = 1.0f + _t_2669;
			_t_2671 = -1.0f * ty1_7_1;
			_t_2672 = ty3_9_1 + _t_2671;
			_t_2673 = -1.0f * _t_2672;
			_t_2674 = _t_2673 < 0.0f;
			if(_t_2674)
				{
					float _t_2675;
					float _t_2676;
				
					_t_2675 = -1.0f * tx3_6_1;
					_t_2676 = tx1_4_1 + _t_2675;
					_t_2677 = _t_2676;
				
				}
		else
				{
					float _t_2678;
					float _t_2679;
					float _t_2680;
				
					_t_2678 = -1.0f * tx3_6_1;
					_t_2679 = tx1_4_1 + _t_2678;
					_t_2680 = -1.0f * _t_2679;
					_t_2677 = _t_2680;
				
				}
		
			_t_2681 = _t_2677 * _t_147;
			_t_2682 = -1.0f * ty1_7_1;
			_t_2683 = ty3_9_1 + _t_2682;
			_t_2684 = -1.0f * _t_2683;
			_t_2685 = _t_2684 < 0.0f;
			if(_t_2685)
				{
					float _t_2686;
					float _t_2687;
				
					_t_2686 = -1.0f * tx3_6_1;
					_t_2687 = tx1_4_1 + _t_2686;
					_t_2688 = _t_2687;
				
				}
		else
				{
					float _t_2689;
					float _t_2690;
					float _t_2691;
				
					_t_2689 = -1.0f * tx3_6_1;
					_t_2690 = tx1_4_1 + _t_2689;
					_t_2691 = -1.0f * _t_2690;
					_t_2688 = _t_2691;
				
				}
		
			_t_2692 = _t_2688 * _t_147;
			_t_2693 = _t_2681 * _t_2692;
			_t_2694 = -1.0f * ty1_7_1;
			_t_2695 = ty3_9_1 + _t_2694;
			_t_2696 = -1.0f * _t_2695;
			_t_2697 = _t_2696 < 0.0f;
			if(_t_2697)
				{
					float _t_2698;
					float _t_2699;
				
					_t_2698 = -1.0f * ty1_7_1;
					_t_2699 = ty3_9_1 + _t_2698;
					_t_2700 = _t_2699;
				
				}
		else
				{
					float _t_2701;
					float _t_2702;
					float _t_2703;
				
					_t_2701 = -1.0f * ty1_7_1;
					_t_2702 = ty3_9_1 + _t_2701;
					_t_2703 = -1.0f * _t_2702;
					_t_2700 = _t_2703;
				
				}
		
			_t_2704 = _t_2700 * _t_147;
			_t_2705 = 1.0f + _t_2704;
			_t_2706 = 1.0f / _t_2705;
			_t_2707 = _t_2693 * _t_2706;
			_t_2708 = _t_2707 * -1.0f;
			_t_2709 = 1.0f + _t_2708;
			_t_2710 = 0.0f < _t_2709;
			if(_t_2710)
				{
				
					_t_2711 = py1_13_1;
				
				}
		else
				{
				
					_t_2711 = py0_12_1;
				
				}
		
			_t_2712 = _t_2670 * _t_2711;
			_t_2713 = _t_2631 + _t_2712;
			_t_2714 = y__2647_1 < _t_2713;
			_t_2715 = _t_2604 && _t_2714;
			_t_2716 = -1.0f * ty1_7_1;
			_t_2717 = ty3_9_1 + _t_2716;
			_t_2718 = -1.0f * _t_2717;
			_t_2719 = _t_2718 < 0.0f;
			if(_t_2719)
				{
					float _t_2720;
					float _t_2721;
				
					_t_2720 = -1.0f * ty1_7_1;
					_t_2721 = ty3_9_1 + _t_2720;
					_t_2722 = _t_2721;
				
				}
		else
				{
					float _t_2723;
					float _t_2724;
					float _t_2725;
				
					_t_2723 = -1.0f * ty1_7_1;
					_t_2724 = ty3_9_1 + _t_2723;
					_t_2725 = -1.0f * _t_2724;
					_t_2722 = _t_2725;
				
				}
		
			_t_2726 = _t_2722 * _t_147;
			_t_2727 = -1.0f * ty1_7_1;
			_t_2728 = ty3_9_1 + _t_2727;
			_t_2729 = -1.0f * _t_2728;
			_t_2730 = _t_2729 < 0.0f;
			if(_t_2730)
				{
					float _t_2731;
					float _t_2732;
				
					_t_2731 = -1.0f * ty1_7_1;
					_t_2732 = ty3_9_1 + _t_2731;
					_t_2733 = _t_2732;
				
				}
		else
				{
					float _t_2734;
					float _t_2735;
					float _t_2736;
				
					_t_2734 = -1.0f * ty1_7_1;
					_t_2735 = ty3_9_1 + _t_2734;
					_t_2736 = -1.0f * _t_2735;
					_t_2733 = _t_2736;
				
				}
		
			_t_2737 = _t_2733 * _t_147;
			_t_2738 = 0.0f < _t_2737;
			if(_t_2738)
				{
				
					_t_2739 = px0_10_1;
				
				}
		else
				{
				
					_t_2739 = px1_11_1;
				
				}
		
			_t_2740 = _t_2726 * _t_2739;
			_t_2741 = -1.0f * ty1_7_1;
			_t_2742 = ty3_9_1 + _t_2741;
			_t_2743 = -1.0f * _t_2742;
			_t_2744 = _t_2743 < 0.0f;
			if(_t_2744)
				{
					float _t_2745;
					float _t_2746;
				
					_t_2745 = -1.0f * tx3_6_1;
					_t_2746 = tx1_4_1 + _t_2745;
					_t_2747 = _t_2746;
				
				}
		else
				{
					float _t_2748;
					float _t_2749;
					float _t_2750;
				
					_t_2748 = -1.0f * tx3_6_1;
					_t_2749 = tx1_4_1 + _t_2748;
					_t_2750 = -1.0f * _t_2749;
					_t_2747 = _t_2750;
				
				}
		
			_t_2751 = _t_2747 * _t_147;
			_t_2752 = -1.0f * ty1_7_1;
			_t_2753 = ty3_9_1 + _t_2752;
			_t_2754 = -1.0f * _t_2753;
			_t_2755 = _t_2754 < 0.0f;
			if(_t_2755)
				{
					float _t_2756;
					float _t_2757;
				
					_t_2756 = -1.0f * tx3_6_1;
					_t_2757 = tx1_4_1 + _t_2756;
					_t_2758 = _t_2757;
				
				}
		else
				{
					float _t_2759;
					float _t_2760;
					float _t_2761;
				
					_t_2759 = -1.0f * tx3_6_1;
					_t_2760 = tx1_4_1 + _t_2759;
					_t_2761 = -1.0f * _t_2760;
					_t_2758 = _t_2761;
				
				}
		
			_t_2762 = _t_2758 * _t_147;
			_t_2763 = 0.0f < _t_2762;
			if(_t_2763)
				{
				
					_t_2764 = py0_12_1;
				
				}
		else
				{
				
					_t_2764 = py1_13_1;
				
				}
		
			_t_2765 = _t_2751 * _t_2764;
			_t_2766 = _t_2740 + _t_2765;
			_t_2767 = _t_2766 < _t_2365;
			_t_2768 = -1.0f * ty1_7_1;
			_t_2769 = ty3_9_1 + _t_2768;
			_t_2770 = -1.0f * _t_2769;
			_t_2771 = _t_2770 < 0.0f;
			if(_t_2771)
				{
					float _t_2772;
					float _t_2773;
				
					_t_2772 = -1.0f * ty1_7_1;
					_t_2773 = ty3_9_1 + _t_2772;
					_t_2774 = _t_2773;
				
				}
		else
				{
					float _t_2775;
					float _t_2776;
					float _t_2777;
				
					_t_2775 = -1.0f * ty1_7_1;
					_t_2776 = ty3_9_1 + _t_2775;
					_t_2777 = -1.0f * _t_2776;
					_t_2774 = _t_2777;
				
				}
		
			_t_2778 = _t_2774 * _t_147;
			_t_2779 = -1.0f * ty1_7_1;
			_t_2780 = ty3_9_1 + _t_2779;
			_t_2781 = -1.0f * _t_2780;
			_t_2782 = _t_2781 < 0.0f;
			if(_t_2782)
				{
					float _t_2783;
					float _t_2784;
				
					_t_2783 = -1.0f * ty1_7_1;
					_t_2784 = ty3_9_1 + _t_2783;
					_t_2785 = _t_2784;
				
				}
		else
				{
					float _t_2786;
					float _t_2787;
					float _t_2788;
				
					_t_2786 = -1.0f * ty1_7_1;
					_t_2787 = ty3_9_1 + _t_2786;
					_t_2788 = -1.0f * _t_2787;
					_t_2785 = _t_2788;
				
				}
		
			_t_2789 = _t_2785 * _t_147;
			_t_2790 = 0.0f < _t_2789;
			if(_t_2790)
				{
				
					_t_2791 = px1_11_1;
				
				}
		else
				{
				
					_t_2791 = px0_10_1;
				
				}
		
			_t_2792 = _t_2778 * _t_2791;
			_t_2793 = -1.0f * ty1_7_1;
			_t_2794 = ty3_9_1 + _t_2793;
			_t_2795 = -1.0f * _t_2794;
			_t_2796 = _t_2795 < 0.0f;
			if(_t_2796)
				{
					float _t_2797;
					float _t_2798;
				
					_t_2797 = -1.0f * tx3_6_1;
					_t_2798 = tx1_4_1 + _t_2797;
					_t_2799 = _t_2798;
				
				}
		else
				{
					float _t_2800;
					float _t_2801;
					float _t_2802;
				
					_t_2800 = -1.0f * tx3_6_1;
					_t_2801 = tx1_4_1 + _t_2800;
					_t_2802 = -1.0f * _t_2801;
					_t_2799 = _t_2802;
				
				}
		
			_t_2803 = _t_2799 * _t_147;
			_t_2804 = -1.0f * ty1_7_1;
			_t_2805 = ty3_9_1 + _t_2804;
			_t_2806 = -1.0f * _t_2805;
			_t_2807 = _t_2806 < 0.0f;
			if(_t_2807)
				{
					float _t_2808;
					float _t_2809;
				
					_t_2808 = -1.0f * tx3_6_1;
					_t_2809 = tx1_4_1 + _t_2808;
					_t_2810 = _t_2809;
				
				}
		else
				{
					float _t_2811;
					float _t_2812;
					float _t_2813;
				
					_t_2811 = -1.0f * tx3_6_1;
					_t_2812 = tx1_4_1 + _t_2811;
					_t_2813 = -1.0f * _t_2812;
					_t_2810 = _t_2813;
				
				}
		
			_t_2814 = _t_2810 * _t_147;
			_t_2815 = 0.0f < _t_2814;
			if(_t_2815)
				{
				
					_t_2816 = py1_13_1;
				
				}
		else
				{
				
					_t_2816 = py0_12_1;
				
				}
		
			_t_2817 = _t_2803 * _t_2816;
			_t_2818 = _t_2792 + _t_2817;
			_t_2819 = _t_2365 < _t_2818;
			_t_2820 = _t_2767 && _t_2819;
			_t_2821 = _t_2715 && _t_2820;
			if(_t_2821)
				{
				
					_t_2822 = 1.0f;
				
				}
		else
				{
				
					_t_2822 = 0.0f;
				
				}
		
			_t_2823 = _t_2822 * _t_147;
			_t_2824 = _t_2823;
		
		}
else
		{
		
			_t_2824 = 0.0f;
		
		}

	_t_2366 = _t_2487 * _t_2824;

	return _t_2366;
}
__device__ float tegpixellet_block_21(float ty3_9_1,float ty1_7_1,float _t_147,float _t_2365,float tx1_4_1,float tx3_6_1,float y__2647_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1){
	float _t_2367;
	float _t_2368;
	float _t_2369;
	bool _t_2370;
	float _t_2373;
	float _t_2377;
	float _t_2378;
	float _t_2379;
	float _t_2380;
	float _t_2381;
	bool _t_2382;
	float _t_2385;
	float _t_2389;
	float _t_2390;
	float _t_2391;
	float _t_2392;
	float _t_2393;
	float _t_2394;
	float _t_2395;
	bool _t_2396;
	float _t_2399;
	float _t_2403;
	float _t_2404;
	float _t_2405;
	float _t_2406;
	bool _t_2407;
	float _t_2410;
	float _t_2414;
	float _t_2415;
	float _t_2416;
	float _t_2417;
	float _t_2418;
	bool _t_2419;
	float _t_2422;
	float _t_2426;
	float _t_2427;
	float _t_2428;
	float _t_2429;
	float _t_2430;
	float _t_2431;
	float _t_2432;
	float _t_2433;
	float _t_2434;
	float _t_2435;
	bool _t_2436;
	float _t_2439;
	float _t_2443;
	float _t_2444;
	float _t_2445;

	float _t_2366;

	_t_2367 = -1.0f * ty1_7_1;
	_t_2368 = ty3_9_1 + _t_2367;
	_t_2369 = -1.0f * _t_2368;
	_t_2370 = _t_2369 < 0.0f;
	if(_t_2370)
		{
			float _t_2371;
			float _t_2372;
		
			_t_2371 = -1.0f * ty1_7_1;
			_t_2372 = ty3_9_1 + _t_2371;
			_t_2373 = _t_2372;
		
		}
else
		{
			float _t_2374;
			float _t_2375;
			float _t_2376;
		
			_t_2374 = -1.0f * ty1_7_1;
			_t_2375 = ty3_9_1 + _t_2374;
			_t_2376 = -1.0f * _t_2375;
			_t_2373 = _t_2376;
		
		}

	_t_2377 = _t_2373 * _t_147;
	_t_2378 = _t_2377 * _t_2365;
	_t_2379 = -1.0f * ty1_7_1;
	_t_2380 = ty3_9_1 + _t_2379;
	_t_2381 = -1.0f * _t_2380;
	_t_2382 = _t_2381 < 0.0f;
	if(_t_2382)
		{
			float _t_2383;
			float _t_2384;
		
			_t_2383 = -1.0f * tx3_6_1;
			_t_2384 = tx1_4_1 + _t_2383;
			_t_2385 = _t_2384;
		
		}
else
		{
			float _t_2386;
			float _t_2387;
			float _t_2388;
		
			_t_2386 = -1.0f * tx3_6_1;
			_t_2387 = tx1_4_1 + _t_2386;
			_t_2388 = -1.0f * _t_2387;
			_t_2385 = _t_2388;
		
		}

	_t_2389 = _t_2385 * _t_147;
	_t_2390 = _t_2389 * -1.0f;
	_t_2391 = _t_2390 * y__2647_1;
	_t_2392 = _t_2378 + _t_2391;
	_t_2393 = -1.0f * ty1_7_1;
	_t_2394 = ty3_9_1 + _t_2393;
	_t_2395 = -1.0f * _t_2394;
	_t_2396 = _t_2395 < 0.0f;
	if(_t_2396)
		{
			float _t_2397;
			float _t_2398;
		
			_t_2397 = -1.0f * tx3_6_1;
			_t_2398 = tx1_4_1 + _t_2397;
			_t_2399 = _t_2398;
		
		}
else
		{
			float _t_2400;
			float _t_2401;
			float _t_2402;
		
			_t_2400 = -1.0f * tx3_6_1;
			_t_2401 = tx1_4_1 + _t_2400;
			_t_2402 = -1.0f * _t_2401;
			_t_2399 = _t_2402;
		
		}

	_t_2403 = _t_2399 * _t_147;
	_t_2404 = -1.0f * ty1_7_1;
	_t_2405 = ty3_9_1 + _t_2404;
	_t_2406 = -1.0f * _t_2405;
	_t_2407 = _t_2406 < 0.0f;
	if(_t_2407)
		{
			float _t_2408;
			float _t_2409;
		
			_t_2408 = -1.0f * tx3_6_1;
			_t_2409 = tx1_4_1 + _t_2408;
			_t_2410 = _t_2409;
		
		}
else
		{
			float _t_2411;
			float _t_2412;
			float _t_2413;
		
			_t_2411 = -1.0f * tx3_6_1;
			_t_2412 = tx1_4_1 + _t_2411;
			_t_2413 = -1.0f * _t_2412;
			_t_2410 = _t_2413;
		
		}

	_t_2414 = _t_2410 * _t_147;
	_t_2415 = _t_2403 * _t_2414;
	_t_2416 = -1.0f * ty1_7_1;
	_t_2417 = ty3_9_1 + _t_2416;
	_t_2418 = -1.0f * _t_2417;
	_t_2419 = _t_2418 < 0.0f;
	if(_t_2419)
		{
			float _t_2420;
			float _t_2421;
		
			_t_2420 = -1.0f * ty1_7_1;
			_t_2421 = ty3_9_1 + _t_2420;
			_t_2422 = _t_2421;
		
		}
else
		{
			float _t_2423;
			float _t_2424;
			float _t_2425;
		
			_t_2423 = -1.0f * ty1_7_1;
			_t_2424 = ty3_9_1 + _t_2423;
			_t_2425 = -1.0f * _t_2424;
			_t_2422 = _t_2425;
		
		}

	_t_2426 = _t_2422 * _t_147;
	_t_2427 = 1.0f + _t_2426;
	_t_2428 = 1.0f / _t_2427;
	_t_2429 = _t_2415 * _t_2428;
	_t_2430 = _t_2429 * -1.0f;
	_t_2431 = 1.0f + _t_2430;
	_t_2432 = _t_2431 * y__2647_1;
	_t_2433 = -1.0f * ty1_7_1;
	_t_2434 = ty3_9_1 + _t_2433;
	_t_2435 = -1.0f * _t_2434;
	_t_2436 = _t_2435 < 0.0f;
	if(_t_2436)
		{
			float _t_2437;
			float _t_2438;
		
			_t_2437 = -1.0f * tx3_6_1;
			_t_2438 = tx1_4_1 + _t_2437;
			_t_2439 = _t_2438;
		
		}
else
		{
			float _t_2440;
			float _t_2441;
			float _t_2442;
		
			_t_2440 = -1.0f * tx3_6_1;
			_t_2441 = tx1_4_1 + _t_2440;
			_t_2442 = -1.0f * _t_2441;
			_t_2439 = _t_2442;
		
		}

	_t_2443 = _t_2439 * _t_147;
	_t_2444 = _t_2443 * _t_2365;
	_t_2445 = _t_2432 + _t_2444;
	_t_2366 = tegpixellet_block_22(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,_t_2392,_t_2445,ty3_9_1,tx3_6_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,_t_147,y__2647_1,_t_2365);

	return _t_2366;
}
__device__ float tegpixelbody_block_18(float ty3_9_1,float ty1_7_1,float _t_147,float px0_10_1,float px1_11_1,float tx1_4_1,float tx3_6_1,float py0_12_1,float py1_13_1,float y__2647_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_2209;
	float _t_2210;
	float _t_2211;
	bool _t_2212;
	float _t_2215;
	float _t_2219;
	float _t_2220;
	float _t_2221;
	float _t_2222;
	bool _t_2223;
	float _t_2226;
	float _t_2230;
	bool _t_2231;
	float _t_2232;
	float _t_2233;
	float _t_2234;
	float _t_2235;
	float _t_2236;
	bool _t_2237;
	float _t_2240;
	float _t_2244;
	float _t_2245;
	float _t_2246;
	float _t_2247;
	bool _t_2248;
	float _t_2251;
	float _t_2255;
	bool _t_2256;
	float _t_2257;
	float _t_2258;
	float _t_2259;
	float _t_2260;
	float _t_2261;
	float _t_2262;
	bool _t_2263;
	float _t_2268;
	float _t_2274;
	float _t_2275;
	float _t_2276;
	float _t_2277;
	bool _t_2278;
	float _t_2279;
	float _t_2280;
	float _t_2281;
	bool _t_2282;
	float _t_2285;
	float _t_2289;
	float _t_2290;
	float _t_2291;
	float _t_2292;
	bool _t_2293;
	float _t_2296;
	float _t_2300;
	bool _t_2301;
	float _t_2302;
	float _t_2303;
	float _t_2304;
	float _t_2305;
	float _t_2306;
	bool _t_2307;
	float _t_2310;
	float _t_2314;
	float _t_2315;
	float _t_2316;
	float _t_2317;
	bool _t_2318;
	float _t_2321;
	float _t_2325;
	bool _t_2326;
	float _t_2327;
	float _t_2328;
	float _t_2329;
	float _t_2330;
	float _t_2331;
	float _t_2332;
	bool _t_2333;
	float _t_2338;
	float _t_2344;
	float _t_2345;
	float _t_2346;
	float _t_2347;
	bool _t_2348;
	bool _t_2349;

	float _t_2208;

	_t_2209 = -1.0f * ty1_7_1;
	_t_2210 = ty3_9_1 + _t_2209;
	_t_2211 = -1.0f * _t_2210;
	_t_2212 = _t_2211 < 0.0f;
	if(_t_2212)
		{
			float _t_2213;
			float _t_2214;
		
			_t_2213 = -1.0f * ty1_7_1;
			_t_2214 = ty3_9_1 + _t_2213;
			_t_2215 = _t_2214;
		
		}
else
		{
			float _t_2216;
			float _t_2217;
			float _t_2218;
		
			_t_2216 = -1.0f * ty1_7_1;
			_t_2217 = ty3_9_1 + _t_2216;
			_t_2218 = -1.0f * _t_2217;
			_t_2215 = _t_2218;
		
		}

	_t_2219 = _t_2215 * _t_147;
	_t_2220 = -1.0f * ty1_7_1;
	_t_2221 = ty3_9_1 + _t_2220;
	_t_2222 = -1.0f * _t_2221;
	_t_2223 = _t_2222 < 0.0f;
	if(_t_2223)
		{
			float _t_2224;
			float _t_2225;
		
			_t_2224 = -1.0f * ty1_7_1;
			_t_2225 = ty3_9_1 + _t_2224;
			_t_2226 = _t_2225;
		
		}
else
		{
			float _t_2227;
			float _t_2228;
			float _t_2229;
		
			_t_2227 = -1.0f * ty1_7_1;
			_t_2228 = ty3_9_1 + _t_2227;
			_t_2229 = -1.0f * _t_2228;
			_t_2226 = _t_2229;
		
		}

	_t_2230 = _t_2226 * _t_147;
	_t_2231 = 0.0f < _t_2230;
	if(_t_2231)
		{
		
			_t_2232 = px0_10_1;
		
		}
else
		{
		
			_t_2232 = px1_11_1;
		
		}

	_t_2233 = _t_2219 * _t_2232;
	_t_2234 = -1.0f * ty1_7_1;
	_t_2235 = ty3_9_1 + _t_2234;
	_t_2236 = -1.0f * _t_2235;
	_t_2237 = _t_2236 < 0.0f;
	if(_t_2237)
		{
			float _t_2238;
			float _t_2239;
		
			_t_2238 = -1.0f * tx3_6_1;
			_t_2239 = tx1_4_1 + _t_2238;
			_t_2240 = _t_2239;
		
		}
else
		{
			float _t_2241;
			float _t_2242;
			float _t_2243;
		
			_t_2241 = -1.0f * tx3_6_1;
			_t_2242 = tx1_4_1 + _t_2241;
			_t_2243 = -1.0f * _t_2242;
			_t_2240 = _t_2243;
		
		}

	_t_2244 = _t_2240 * _t_147;
	_t_2245 = -1.0f * ty1_7_1;
	_t_2246 = ty3_9_1 + _t_2245;
	_t_2247 = -1.0f * _t_2246;
	_t_2248 = _t_2247 < 0.0f;
	if(_t_2248)
		{
			float _t_2249;
			float _t_2250;
		
			_t_2249 = -1.0f * tx3_6_1;
			_t_2250 = tx1_4_1 + _t_2249;
			_t_2251 = _t_2250;
		
		}
else
		{
			float _t_2252;
			float _t_2253;
			float _t_2254;
		
			_t_2252 = -1.0f * tx3_6_1;
			_t_2253 = tx1_4_1 + _t_2252;
			_t_2254 = -1.0f * _t_2253;
			_t_2251 = _t_2254;
		
		}

	_t_2255 = _t_2251 * _t_147;
	_t_2256 = 0.0f < _t_2255;
	if(_t_2256)
		{
		
			_t_2257 = py0_12_1;
		
		}
else
		{
		
			_t_2257 = py1_13_1;
		
		}

	_t_2258 = _t_2244 * _t_2257;
	_t_2259 = _t_2233 + _t_2258;
	_t_2260 = -1.0f * ty1_7_1;
	_t_2261 = ty3_9_1 + _t_2260;
	_t_2262 = -1.0f * _t_2261;
	_t_2263 = _t_2262 < 0.0f;
	if(_t_2263)
		{
			float _t_2264;
			float _t_2265;
			float _t_2266;
			float _t_2267;
		
			_t_2264 = tx3_6_1 * ty1_7_1;
			_t_2265 = tx1_4_1 * ty3_9_1;
			_t_2266 = _t_2265 * -1.0f;
			_t_2267 = _t_2264 + _t_2266;
			_t_2268 = _t_2267;
		
		}
else
		{
			float _t_2269;
			float _t_2270;
			float _t_2271;
			float _t_2272;
			float _t_2273;
		
			_t_2269 = tx3_6_1 * ty1_7_1;
			_t_2270 = tx1_4_1 * ty3_9_1;
			_t_2271 = _t_2270 * -1.0f;
			_t_2272 = _t_2269 + _t_2271;
			_t_2273 = -1.0f * _t_2272;
			_t_2268 = _t_2273;
		
		}

	_t_2274 = -1.0f * _t_2268;
	_t_2275 = _t_2274 * _t_147;
	_t_2276 = _t_2275 * -1.0f;
	_t_2277 = _t_2259 + _t_2276;
	_t_2278 = _t_2277 < 0.0f;
	_t_2279 = -1.0f * ty1_7_1;
	_t_2280 = ty3_9_1 + _t_2279;
	_t_2281 = -1.0f * _t_2280;
	_t_2282 = _t_2281 < 0.0f;
	if(_t_2282)
		{
			float _t_2283;
			float _t_2284;
		
			_t_2283 = -1.0f * ty1_7_1;
			_t_2284 = ty3_9_1 + _t_2283;
			_t_2285 = _t_2284;
		
		}
else
		{
			float _t_2286;
			float _t_2287;
			float _t_2288;
		
			_t_2286 = -1.0f * ty1_7_1;
			_t_2287 = ty3_9_1 + _t_2286;
			_t_2288 = -1.0f * _t_2287;
			_t_2285 = _t_2288;
		
		}

	_t_2289 = _t_2285 * _t_147;
	_t_2290 = -1.0f * ty1_7_1;
	_t_2291 = ty3_9_1 + _t_2290;
	_t_2292 = -1.0f * _t_2291;
	_t_2293 = _t_2292 < 0.0f;
	if(_t_2293)
		{
			float _t_2294;
			float _t_2295;
		
			_t_2294 = -1.0f * ty1_7_1;
			_t_2295 = ty3_9_1 + _t_2294;
			_t_2296 = _t_2295;
		
		}
else
		{
			float _t_2297;
			float _t_2298;
			float _t_2299;
		
			_t_2297 = -1.0f * ty1_7_1;
			_t_2298 = ty3_9_1 + _t_2297;
			_t_2299 = -1.0f * _t_2298;
			_t_2296 = _t_2299;
		
		}

	_t_2300 = _t_2296 * _t_147;
	_t_2301 = 0.0f < _t_2300;
	if(_t_2301)
		{
		
			_t_2302 = px1_11_1;
		
		}
else
		{
		
			_t_2302 = px0_10_1;
		
		}

	_t_2303 = _t_2289 * _t_2302;
	_t_2304 = -1.0f * ty1_7_1;
	_t_2305 = ty3_9_1 + _t_2304;
	_t_2306 = -1.0f * _t_2305;
	_t_2307 = _t_2306 < 0.0f;
	if(_t_2307)
		{
			float _t_2308;
			float _t_2309;
		
			_t_2308 = -1.0f * tx3_6_1;
			_t_2309 = tx1_4_1 + _t_2308;
			_t_2310 = _t_2309;
		
		}
else
		{
			float _t_2311;
			float _t_2312;
			float _t_2313;
		
			_t_2311 = -1.0f * tx3_6_1;
			_t_2312 = tx1_4_1 + _t_2311;
			_t_2313 = -1.0f * _t_2312;
			_t_2310 = _t_2313;
		
		}

	_t_2314 = _t_2310 * _t_147;
	_t_2315 = -1.0f * ty1_7_1;
	_t_2316 = ty3_9_1 + _t_2315;
	_t_2317 = -1.0f * _t_2316;
	_t_2318 = _t_2317 < 0.0f;
	if(_t_2318)
		{
			float _t_2319;
			float _t_2320;
		
			_t_2319 = -1.0f * tx3_6_1;
			_t_2320 = tx1_4_1 + _t_2319;
			_t_2321 = _t_2320;
		
		}
else
		{
			float _t_2322;
			float _t_2323;
			float _t_2324;
		
			_t_2322 = -1.0f * tx3_6_1;
			_t_2323 = tx1_4_1 + _t_2322;
			_t_2324 = -1.0f * _t_2323;
			_t_2321 = _t_2324;
		
		}

	_t_2325 = _t_2321 * _t_147;
	_t_2326 = 0.0f < _t_2325;
	if(_t_2326)
		{
		
			_t_2327 = py1_13_1;
		
		}
else
		{
		
			_t_2327 = py0_12_1;
		
		}

	_t_2328 = _t_2314 * _t_2327;
	_t_2329 = _t_2303 + _t_2328;
	_t_2330 = -1.0f * ty1_7_1;
	_t_2331 = ty3_9_1 + _t_2330;
	_t_2332 = -1.0f * _t_2331;
	_t_2333 = _t_2332 < 0.0f;
	if(_t_2333)
		{
			float _t_2334;
			float _t_2335;
			float _t_2336;
			float _t_2337;
		
			_t_2334 = tx3_6_1 * ty1_7_1;
			_t_2335 = tx1_4_1 * ty3_9_1;
			_t_2336 = _t_2335 * -1.0f;
			_t_2337 = _t_2334 + _t_2336;
			_t_2338 = _t_2337;
		
		}
else
		{
			float _t_2339;
			float _t_2340;
			float _t_2341;
			float _t_2342;
			float _t_2343;
		
			_t_2339 = tx3_6_1 * ty1_7_1;
			_t_2340 = tx1_4_1 * ty3_9_1;
			_t_2341 = _t_2340 * -1.0f;
			_t_2342 = _t_2339 + _t_2341;
			_t_2343 = -1.0f * _t_2342;
			_t_2338 = _t_2343;
		
		}

	_t_2344 = -1.0f * _t_2338;
	_t_2345 = _t_2344 * _t_147;
	_t_2346 = _t_2345 * -1.0f;
	_t_2347 = _t_2329 + _t_2346;
	_t_2348 = 0.0f < _t_2347;
	_t_2349 = _t_2278 && _t_2348;
	if(_t_2349)
		{
			float _t_2350;
			float _t_2351;
			float _t_2352;
			bool _t_2353;
			float _t_2358;
			float _t_2364;
			float _t_2365;
			float _t_2366;
		
			_t_2350 = -1.0f * ty1_7_1;
			_t_2351 = ty3_9_1 + _t_2350;
			_t_2352 = -1.0f * _t_2351;
			_t_2353 = _t_2352 < 0.0f;
			if(_t_2353)
				{
					float _t_2354;
					float _t_2355;
					float _t_2356;
					float _t_2357;
				
					_t_2354 = tx3_6_1 * ty1_7_1;
					_t_2355 = tx1_4_1 * ty3_9_1;
					_t_2356 = _t_2355 * -1.0f;
					_t_2357 = _t_2354 + _t_2356;
					_t_2358 = _t_2357;
				
				}
		else
				{
					float _t_2359;
					float _t_2360;
					float _t_2361;
					float _t_2362;
					float _t_2363;
				
					_t_2359 = tx3_6_1 * ty1_7_1;
					_t_2360 = tx1_4_1 * ty3_9_1;
					_t_2361 = _t_2360 * -1.0f;
					_t_2362 = _t_2359 + _t_2361;
					_t_2363 = -1.0f * _t_2362;
					_t_2358 = _t_2363;
				
				}
		
			_t_2364 = -1.0f * _t_2358;
			_t_2365 = _t_2364 * _t_147;
			_t_2366 = tegpixellet_block_21(ty3_9_1,ty1_7_1,_t_147,_t_2365,tx1_4_1,tx3_6_1,y__2647_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1);
			_t_2208 = _t_2366;
		
		}
else
		{
		
			_t_2208 = 0.0f;
		
		}


	return _t_2208;
}
__device__ float tegpixelintegrator_18(float ty3_9_1,float pc1_15_1,float _t_2207,float tc2_19_1,float _t_147,float ty2_8_1,float pc0_14_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float py1_13_1,float pc2_16_1,float tx2_5_1,float px1_11_1,float tc0_17_1,float _t_2098,float py0_12_1,float tc1_18_1,float px0_10_1){
    float y__2647_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_2207 - _t_2098)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__2647_1 = _t_2098 + __step__ * (i + (float)(0.5));
        float _t_2208;
		_t_2208 = tegpixelbody_block_18(ty3_9_1,ty1_7_1,_t_147,px0_10_1,px1_11_1,tx1_4_1,tx3_6_1,py0_12_1,py1_13_1,y__2647_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);;
        __output__ = __output__ + _t_2208 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_2(float ty3_9_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float _t_147,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_1990;
	float _t_1991;
	float _t_1992;
	bool _t_1993;
	float _t_1996;
	float _t_2000;
	float _t_2001;
	float _t_2002;
	float _t_2003;
	float _t_2004;
	bool _t_2005;
	float _t_2008;
	float _t_2012;
	float _t_2013;
	bool _t_2014;
	float _t_2015;
	float _t_2016;
	float _t_2017;
	float _t_2018;
	float _t_2019;
	bool _t_2020;
	float _t_2023;
	float _t_2027;
	float _t_2028;
	float _t_2029;
	float _t_2030;
	bool _t_2031;
	float _t_2034;
	float _t_2038;
	float _t_2039;
	float _t_2040;
	float _t_2041;
	float _t_2042;
	bool _t_2043;
	float _t_2046;
	float _t_2050;
	float _t_2051;
	float _t_2052;
	float _t_2053;
	float _t_2054;
	float _t_2055;
	float _t_2056;
	float _t_2057;
	float _t_2058;
	bool _t_2059;
	float _t_2062;
	float _t_2066;
	float _t_2067;
	float _t_2068;
	float _t_2069;
	bool _t_2070;
	float _t_2073;
	float _t_2077;
	float _t_2078;
	float _t_2079;
	float _t_2080;
	float _t_2081;
	bool _t_2082;
	float _t_2085;
	float _t_2089;
	float _t_2090;
	float _t_2091;
	float _t_2092;
	float _t_2093;
	float _t_2094;
	bool _t_2095;
	float _t_2096;
	float _t_2097;
	float _t_2098;
	float _t_2099;
	float _t_2100;
	float _t_2101;
	bool _t_2102;
	float _t_2105;
	float _t_2109;
	float _t_2110;
	float _t_2111;
	float _t_2112;
	float _t_2113;
	bool _t_2114;
	float _t_2117;
	float _t_2121;
	float _t_2122;
	bool _t_2123;
	float _t_2124;
	float _t_2125;
	float _t_2126;
	float _t_2127;
	float _t_2128;
	bool _t_2129;
	float _t_2132;
	float _t_2136;
	float _t_2137;
	float _t_2138;
	float _t_2139;
	bool _t_2140;
	float _t_2143;
	float _t_2147;
	float _t_2148;
	float _t_2149;
	float _t_2150;
	float _t_2151;
	bool _t_2152;
	float _t_2155;
	float _t_2159;
	float _t_2160;
	float _t_2161;
	float _t_2162;
	float _t_2163;
	float _t_2164;
	float _t_2165;
	float _t_2166;
	float _t_2167;
	bool _t_2168;
	float _t_2171;
	float _t_2175;
	float _t_2176;
	float _t_2177;
	float _t_2178;
	bool _t_2179;
	float _t_2182;
	float _t_2186;
	float _t_2187;
	float _t_2188;
	float _t_2189;
	float _t_2190;
	bool _t_2191;
	float _t_2194;
	float _t_2198;
	float _t_2199;
	float _t_2200;
	float _t_2201;
	float _t_2202;
	float _t_2203;
	bool _t_2204;
	float _t_2205;
	float _t_2206;
	float _t_2207;

	float _t_148;

	_t_1990 = -1.0f * ty1_7_1;
	_t_1991 = ty3_9_1 + _t_1990;
	_t_1992 = -1.0f * _t_1991;
	_t_1993 = _t_1992 < 0.0f;
	if(_t_1993)
		{
			float _t_1994;
			float _t_1995;
		
			_t_1994 = -1.0f * tx3_6_1;
			_t_1995 = tx1_4_1 + _t_1994;
			_t_1996 = _t_1995;
		
		}
else
		{
			float _t_1997;
			float _t_1998;
			float _t_1999;
		
			_t_1997 = -1.0f * tx3_6_1;
			_t_1998 = tx1_4_1 + _t_1997;
			_t_1999 = -1.0f * _t_1998;
			_t_1996 = _t_1999;
		
		}

	_t_2000 = _t_1996 * _t_147;
	_t_2001 = _t_2000 * -1.0f;
	_t_2002 = -1.0f * ty1_7_1;
	_t_2003 = ty3_9_1 + _t_2002;
	_t_2004 = -1.0f * _t_2003;
	_t_2005 = _t_2004 < 0.0f;
	if(_t_2005)
		{
			float _t_2006;
			float _t_2007;
		
			_t_2006 = -1.0f * tx3_6_1;
			_t_2007 = tx1_4_1 + _t_2006;
			_t_2008 = _t_2007;
		
		}
else
		{
			float _t_2009;
			float _t_2010;
			float _t_2011;
		
			_t_2009 = -1.0f * tx3_6_1;
			_t_2010 = tx1_4_1 + _t_2009;
			_t_2011 = -1.0f * _t_2010;
			_t_2008 = _t_2011;
		
		}

	_t_2012 = _t_2008 * _t_147;
	_t_2013 = _t_2012 * -1.0f;
	_t_2014 = 0.0f < _t_2013;
	if(_t_2014)
		{
		
			_t_2015 = px0_10_1;
		
		}
else
		{
		
			_t_2015 = px1_11_1;
		
		}

	_t_2016 = _t_2001 * _t_2015;
	_t_2017 = -1.0f * ty1_7_1;
	_t_2018 = ty3_9_1 + _t_2017;
	_t_2019 = -1.0f * _t_2018;
	_t_2020 = _t_2019 < 0.0f;
	if(_t_2020)
		{
			float _t_2021;
			float _t_2022;
		
			_t_2021 = -1.0f * tx3_6_1;
			_t_2022 = tx1_4_1 + _t_2021;
			_t_2023 = _t_2022;
		
		}
else
		{
			float _t_2024;
			float _t_2025;
			float _t_2026;
		
			_t_2024 = -1.0f * tx3_6_1;
			_t_2025 = tx1_4_1 + _t_2024;
			_t_2026 = -1.0f * _t_2025;
			_t_2023 = _t_2026;
		
		}

	_t_2027 = _t_2023 * _t_147;
	_t_2028 = -1.0f * ty1_7_1;
	_t_2029 = ty3_9_1 + _t_2028;
	_t_2030 = -1.0f * _t_2029;
	_t_2031 = _t_2030 < 0.0f;
	if(_t_2031)
		{
			float _t_2032;
			float _t_2033;
		
			_t_2032 = -1.0f * tx3_6_1;
			_t_2033 = tx1_4_1 + _t_2032;
			_t_2034 = _t_2033;
		
		}
else
		{
			float _t_2035;
			float _t_2036;
			float _t_2037;
		
			_t_2035 = -1.0f * tx3_6_1;
			_t_2036 = tx1_4_1 + _t_2035;
			_t_2037 = -1.0f * _t_2036;
			_t_2034 = _t_2037;
		
		}

	_t_2038 = _t_2034 * _t_147;
	_t_2039 = _t_2027 * _t_2038;
	_t_2040 = -1.0f * ty1_7_1;
	_t_2041 = ty3_9_1 + _t_2040;
	_t_2042 = -1.0f * _t_2041;
	_t_2043 = _t_2042 < 0.0f;
	if(_t_2043)
		{
			float _t_2044;
			float _t_2045;
		
			_t_2044 = -1.0f * ty1_7_1;
			_t_2045 = ty3_9_1 + _t_2044;
			_t_2046 = _t_2045;
		
		}
else
		{
			float _t_2047;
			float _t_2048;
			float _t_2049;
		
			_t_2047 = -1.0f * ty1_7_1;
			_t_2048 = ty3_9_1 + _t_2047;
			_t_2049 = -1.0f * _t_2048;
			_t_2046 = _t_2049;
		
		}

	_t_2050 = _t_2046 * _t_147;
	_t_2051 = 1.0f + _t_2050;
	_t_2052 = 1.0f / _t_2051;
	_t_2053 = _t_2039 * _t_2052;
	_t_2054 = _t_2053 * -1.0f;
	_t_2055 = 1.0f + _t_2054;
	_t_2056 = -1.0f * ty1_7_1;
	_t_2057 = ty3_9_1 + _t_2056;
	_t_2058 = -1.0f * _t_2057;
	_t_2059 = _t_2058 < 0.0f;
	if(_t_2059)
		{
			float _t_2060;
			float _t_2061;
		
			_t_2060 = -1.0f * tx3_6_1;
			_t_2061 = tx1_4_1 + _t_2060;
			_t_2062 = _t_2061;
		
		}
else
		{
			float _t_2063;
			float _t_2064;
			float _t_2065;
		
			_t_2063 = -1.0f * tx3_6_1;
			_t_2064 = tx1_4_1 + _t_2063;
			_t_2065 = -1.0f * _t_2064;
			_t_2062 = _t_2065;
		
		}

	_t_2066 = _t_2062 * _t_147;
	_t_2067 = -1.0f * ty1_7_1;
	_t_2068 = ty3_9_1 + _t_2067;
	_t_2069 = -1.0f * _t_2068;
	_t_2070 = _t_2069 < 0.0f;
	if(_t_2070)
		{
			float _t_2071;
			float _t_2072;
		
			_t_2071 = -1.0f * tx3_6_1;
			_t_2072 = tx1_4_1 + _t_2071;
			_t_2073 = _t_2072;
		
		}
else
		{
			float _t_2074;
			float _t_2075;
			float _t_2076;
		
			_t_2074 = -1.0f * tx3_6_1;
			_t_2075 = tx1_4_1 + _t_2074;
			_t_2076 = -1.0f * _t_2075;
			_t_2073 = _t_2076;
		
		}

	_t_2077 = _t_2073 * _t_147;
	_t_2078 = _t_2066 * _t_2077;
	_t_2079 = -1.0f * ty1_7_1;
	_t_2080 = ty3_9_1 + _t_2079;
	_t_2081 = -1.0f * _t_2080;
	_t_2082 = _t_2081 < 0.0f;
	if(_t_2082)
		{
			float _t_2083;
			float _t_2084;
		
			_t_2083 = -1.0f * ty1_7_1;
			_t_2084 = ty3_9_1 + _t_2083;
			_t_2085 = _t_2084;
		
		}
else
		{
			float _t_2086;
			float _t_2087;
			float _t_2088;
		
			_t_2086 = -1.0f * ty1_7_1;
			_t_2087 = ty3_9_1 + _t_2086;
			_t_2088 = -1.0f * _t_2087;
			_t_2085 = _t_2088;
		
		}

	_t_2089 = _t_2085 * _t_147;
	_t_2090 = 1.0f + _t_2089;
	_t_2091 = 1.0f / _t_2090;
	_t_2092 = _t_2078 * _t_2091;
	_t_2093 = _t_2092 * -1.0f;
	_t_2094 = 1.0f + _t_2093;
	_t_2095 = 0.0f < _t_2094;
	if(_t_2095)
		{
		
			_t_2096 = py0_12_1;
		
		}
else
		{
		
			_t_2096 = py1_13_1;
		
		}

	_t_2097 = _t_2055 * _t_2096;
	_t_2098 = _t_2016 + _t_2097;
	_t_2099 = -1.0f * ty1_7_1;
	_t_2100 = ty3_9_1 + _t_2099;
	_t_2101 = -1.0f * _t_2100;
	_t_2102 = _t_2101 < 0.0f;
	if(_t_2102)
		{
			float _t_2103;
			float _t_2104;
		
			_t_2103 = -1.0f * tx3_6_1;
			_t_2104 = tx1_4_1 + _t_2103;
			_t_2105 = _t_2104;
		
		}
else
		{
			float _t_2106;
			float _t_2107;
			float _t_2108;
		
			_t_2106 = -1.0f * tx3_6_1;
			_t_2107 = tx1_4_1 + _t_2106;
			_t_2108 = -1.0f * _t_2107;
			_t_2105 = _t_2108;
		
		}

	_t_2109 = _t_2105 * _t_147;
	_t_2110 = _t_2109 * -1.0f;
	_t_2111 = -1.0f * ty1_7_1;
	_t_2112 = ty3_9_1 + _t_2111;
	_t_2113 = -1.0f * _t_2112;
	_t_2114 = _t_2113 < 0.0f;
	if(_t_2114)
		{
			float _t_2115;
			float _t_2116;
		
			_t_2115 = -1.0f * tx3_6_1;
			_t_2116 = tx1_4_1 + _t_2115;
			_t_2117 = _t_2116;
		
		}
else
		{
			float _t_2118;
			float _t_2119;
			float _t_2120;
		
			_t_2118 = -1.0f * tx3_6_1;
			_t_2119 = tx1_4_1 + _t_2118;
			_t_2120 = -1.0f * _t_2119;
			_t_2117 = _t_2120;
		
		}

	_t_2121 = _t_2117 * _t_147;
	_t_2122 = _t_2121 * -1.0f;
	_t_2123 = 0.0f < _t_2122;
	if(_t_2123)
		{
		
			_t_2124 = px1_11_1;
		
		}
else
		{
		
			_t_2124 = px0_10_1;
		
		}

	_t_2125 = _t_2110 * _t_2124;
	_t_2126 = -1.0f * ty1_7_1;
	_t_2127 = ty3_9_1 + _t_2126;
	_t_2128 = -1.0f * _t_2127;
	_t_2129 = _t_2128 < 0.0f;
	if(_t_2129)
		{
			float _t_2130;
			float _t_2131;
		
			_t_2130 = -1.0f * tx3_6_1;
			_t_2131 = tx1_4_1 + _t_2130;
			_t_2132 = _t_2131;
		
		}
else
		{
			float _t_2133;
			float _t_2134;
			float _t_2135;
		
			_t_2133 = -1.0f * tx3_6_1;
			_t_2134 = tx1_4_1 + _t_2133;
			_t_2135 = -1.0f * _t_2134;
			_t_2132 = _t_2135;
		
		}

	_t_2136 = _t_2132 * _t_147;
	_t_2137 = -1.0f * ty1_7_1;
	_t_2138 = ty3_9_1 + _t_2137;
	_t_2139 = -1.0f * _t_2138;
	_t_2140 = _t_2139 < 0.0f;
	if(_t_2140)
		{
			float _t_2141;
			float _t_2142;
		
			_t_2141 = -1.0f * tx3_6_1;
			_t_2142 = tx1_4_1 + _t_2141;
			_t_2143 = _t_2142;
		
		}
else
		{
			float _t_2144;
			float _t_2145;
			float _t_2146;
		
			_t_2144 = -1.0f * tx3_6_1;
			_t_2145 = tx1_4_1 + _t_2144;
			_t_2146 = -1.0f * _t_2145;
			_t_2143 = _t_2146;
		
		}

	_t_2147 = _t_2143 * _t_147;
	_t_2148 = _t_2136 * _t_2147;
	_t_2149 = -1.0f * ty1_7_1;
	_t_2150 = ty3_9_1 + _t_2149;
	_t_2151 = -1.0f * _t_2150;
	_t_2152 = _t_2151 < 0.0f;
	if(_t_2152)
		{
			float _t_2153;
			float _t_2154;
		
			_t_2153 = -1.0f * ty1_7_1;
			_t_2154 = ty3_9_1 + _t_2153;
			_t_2155 = _t_2154;
		
		}
else
		{
			float _t_2156;
			float _t_2157;
			float _t_2158;
		
			_t_2156 = -1.0f * ty1_7_1;
			_t_2157 = ty3_9_1 + _t_2156;
			_t_2158 = -1.0f * _t_2157;
			_t_2155 = _t_2158;
		
		}

	_t_2159 = _t_2155 * _t_147;
	_t_2160 = 1.0f + _t_2159;
	_t_2161 = 1.0f / _t_2160;
	_t_2162 = _t_2148 * _t_2161;
	_t_2163 = _t_2162 * -1.0f;
	_t_2164 = 1.0f + _t_2163;
	_t_2165 = -1.0f * ty1_7_1;
	_t_2166 = ty3_9_1 + _t_2165;
	_t_2167 = -1.0f * _t_2166;
	_t_2168 = _t_2167 < 0.0f;
	if(_t_2168)
		{
			float _t_2169;
			float _t_2170;
		
			_t_2169 = -1.0f * tx3_6_1;
			_t_2170 = tx1_4_1 + _t_2169;
			_t_2171 = _t_2170;
		
		}
else
		{
			float _t_2172;
			float _t_2173;
			float _t_2174;
		
			_t_2172 = -1.0f * tx3_6_1;
			_t_2173 = tx1_4_1 + _t_2172;
			_t_2174 = -1.0f * _t_2173;
			_t_2171 = _t_2174;
		
		}

	_t_2175 = _t_2171 * _t_147;
	_t_2176 = -1.0f * ty1_7_1;
	_t_2177 = ty3_9_1 + _t_2176;
	_t_2178 = -1.0f * _t_2177;
	_t_2179 = _t_2178 < 0.0f;
	if(_t_2179)
		{
			float _t_2180;
			float _t_2181;
		
			_t_2180 = -1.0f * tx3_6_1;
			_t_2181 = tx1_4_1 + _t_2180;
			_t_2182 = _t_2181;
		
		}
else
		{
			float _t_2183;
			float _t_2184;
			float _t_2185;
		
			_t_2183 = -1.0f * tx3_6_1;
			_t_2184 = tx1_4_1 + _t_2183;
			_t_2185 = -1.0f * _t_2184;
			_t_2182 = _t_2185;
		
		}

	_t_2186 = _t_2182 * _t_147;
	_t_2187 = _t_2175 * _t_2186;
	_t_2188 = -1.0f * ty1_7_1;
	_t_2189 = ty3_9_1 + _t_2188;
	_t_2190 = -1.0f * _t_2189;
	_t_2191 = _t_2190 < 0.0f;
	if(_t_2191)
		{
			float _t_2192;
			float _t_2193;
		
			_t_2192 = -1.0f * ty1_7_1;
			_t_2193 = ty3_9_1 + _t_2192;
			_t_2194 = _t_2193;
		
		}
else
		{
			float _t_2195;
			float _t_2196;
			float _t_2197;
		
			_t_2195 = -1.0f * ty1_7_1;
			_t_2196 = ty3_9_1 + _t_2195;
			_t_2197 = -1.0f * _t_2196;
			_t_2194 = _t_2197;
		
		}

	_t_2198 = _t_2194 * _t_147;
	_t_2199 = 1.0f + _t_2198;
	_t_2200 = 1.0f / _t_2199;
	_t_2201 = _t_2187 * _t_2200;
	_t_2202 = _t_2201 * -1.0f;
	_t_2203 = 1.0f + _t_2202;
	_t_2204 = 0.0f < _t_2203;
	if(_t_2204)
		{
		
			_t_2205 = py1_13_1;
		
		}
else
		{
		
			_t_2205 = py0_12_1;
		
		}

	_t_2206 = _t_2164 * _t_2205;
	_t_2207 = _t_2125 + _t_2206;
	_t_148 = tegpixelintegrator_18(ty3_9_1,pc1_15_1,_t_2207,tc2_19_1,_t_147,ty2_8_1,pc0_14_1,ty1_7_1,tx1_4_1,tx3_6_1,py1_13_1,pc2_16_1,tx2_5_1,px1_11_1,tc0_17_1,_t_2098,py0_12_1,tc1_18_1,px0_10_1);

	return _t_148;
}
__device__ float tegpixellet_block_24(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float _t_3227,float _t_3280,float ty3_9_1,float tx3_6_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float _t_175,float y__2721_1,float _t_3200){
	float _t_3281;
	float _t_3282;
	float _t_3283;
	float _t_3284;
	float _t_3285;
	float _t_3286;
	float _t_3287;
	float _t_3288;
	float _t_3289;
	float _t_3290;
	float _t_3291;
	float _t_3292;
	float _t_3293;
	float _t_3294;
	float _t_3295;
	float _t_3296;
	float _t_3297;
	float _t_3298;
	float _t_3299;
	float _t_3300;
	float _t_3301;
	float _t_3302;
	float _t_3303;
	bool _t_3304;
	float _t_3305;
	float _t_3306;
	float _t_3307;
	float _t_3308;
	float _t_3309;
	float _t_3310;
	float _t_3311;
	float _t_3312;
	float _t_3313;
	float _t_3314;
	float _t_3315;
	float _t_3316;
	float _t_3317;
	bool _t_3318;
	float _t_3319;
	float _t_3320;
	float _t_3321;
	float _t_3322;
	float _t_3323;
	bool _t_3324;
	bool _t_3325;
	bool _t_3326;
	bool _t_3327;
	bool _t_3328;
	bool _t_3329;
	bool _t_3330;
	float _t_3660;

	float _t_3201;

	_t_3281 = -1.0f * pc0_14_1;
	_t_3282 = tc0_17_1 + _t_3281;
	_t_3283 = _t_3282 * _t_3282;
	_t_3284 = -1.0f * pc1_15_1;
	_t_3285 = tc1_18_1 + _t_3284;
	_t_3286 = _t_3285 * _t_3285;
	_t_3287 = _t_3283 + _t_3286;
	_t_3288 = -1.0f * pc2_16_1;
	_t_3289 = tc2_19_1 + _t_3288;
	_t_3290 = _t_3289 * _t_3289;
	_t_3291 = _t_3287 + _t_3290;
	_t_3292 = tx1_4_1 * ty2_8_1;
	_t_3293 = tx2_5_1 * ty1_7_1;
	_t_3294 = _t_3293 * -1.0f;
	_t_3295 = _t_3292 + _t_3294;
	_t_3296 = -1.0f * ty2_8_1;
	_t_3297 = ty1_7_1 + _t_3296;
	_t_3298 = _t_3297 * _t_3227;
	_t_3299 = _t_3295 + _t_3298;
	_t_3300 = -1.0f * tx1_4_1;
	_t_3301 = tx2_5_1 + _t_3300;
	_t_3302 = _t_3301 * _t_3280;
	_t_3303 = _t_3299 + _t_3302;
	_t_3304 = _t_3303 < 0.0f;
	if(_t_3304)
		{
		
			_t_3305 = 1.0f;
		
		}
else
		{
		
			_t_3305 = 0.0f;
		
		}

	_t_3306 = tx2_5_1 * ty3_9_1;
	_t_3307 = tx3_6_1 * ty2_8_1;
	_t_3308 = _t_3307 * -1.0f;
	_t_3309 = _t_3306 + _t_3308;
	_t_3310 = -1.0f * ty3_9_1;
	_t_3311 = ty2_8_1 + _t_3310;
	_t_3312 = _t_3311 * _t_3227;
	_t_3313 = _t_3309 + _t_3312;
	_t_3314 = -1.0f * tx2_5_1;
	_t_3315 = tx3_6_1 + _t_3314;
	_t_3316 = _t_3315 * _t_3280;
	_t_3317 = _t_3313 + _t_3316;
	_t_3318 = _t_3317 < 0.0f;
	if(_t_3318)
		{
		
			_t_3319 = 1.0f;
		
		}
else
		{
		
			_t_3319 = 0.0f;
		
		}

	_t_3320 = _t_3305 * _t_3319;
	_t_3321 = _t_3291 * _t_3320;
	_t_3322 = _t_3321 * _t_3280;
	_t_3323 = _t_3322 * -1.0f;
	_t_3324 = py0_12_1 < _t_3280;
	_t_3325 = _t_3280 < py1_13_1;
	_t_3326 = _t_3324 && _t_3325;
	_t_3327 = px0_10_1 < _t_3227;
	_t_3328 = _t_3227 < px1_11_1;
	_t_3329 = _t_3327 && _t_3328;
	_t_3330 = _t_3326 && _t_3329;
	if(_t_3330)
		{
			float _t_3331;
			float _t_3332;
			float _t_3333;
			bool _t_3334;
			float _t_3337;
			float _t_3341;
			float _t_3342;
			float _t_3343;
			float _t_3344;
			float _t_3345;
			bool _t_3346;
			float _t_3349;
			float _t_3353;
			float _t_3354;
			bool _t_3355;
			float _t_3356;
			float _t_3357;
			float _t_3358;
			float _t_3359;
			float _t_3360;
			bool _t_3361;
			float _t_3364;
			float _t_3368;
			float _t_3369;
			float _t_3370;
			float _t_3371;
			bool _t_3372;
			float _t_3375;
			float _t_3379;
			float _t_3380;
			float _t_3381;
			float _t_3382;
			float _t_3383;
			bool _t_3384;
			float _t_3387;
			float _t_3391;
			float _t_3392;
			float _t_3393;
			float _t_3394;
			float _t_3395;
			float _t_3396;
			float _t_3397;
			float _t_3398;
			float _t_3399;
			bool _t_3400;
			float _t_3403;
			float _t_3407;
			float _t_3408;
			float _t_3409;
			float _t_3410;
			bool _t_3411;
			float _t_3414;
			float _t_3418;
			float _t_3419;
			float _t_3420;
			float _t_3421;
			float _t_3422;
			bool _t_3423;
			float _t_3426;
			float _t_3430;
			float _t_3431;
			float _t_3432;
			float _t_3433;
			float _t_3434;
			float _t_3435;
			bool _t_3436;
			float _t_3437;
			float _t_3438;
			float _t_3439;
			bool _t_3440;
			float _t_3441;
			float _t_3442;
			float _t_3443;
			bool _t_3444;
			float _t_3447;
			float _t_3451;
			float _t_3452;
			float _t_3453;
			float _t_3454;
			float _t_3455;
			bool _t_3456;
			float _t_3459;
			float _t_3463;
			float _t_3464;
			bool _t_3465;
			float _t_3466;
			float _t_3467;
			float _t_3468;
			float _t_3469;
			float _t_3470;
			bool _t_3471;
			float _t_3474;
			float _t_3478;
			float _t_3479;
			float _t_3480;
			float _t_3481;
			bool _t_3482;
			float _t_3485;
			float _t_3489;
			float _t_3490;
			float _t_3491;
			float _t_3492;
			float _t_3493;
			bool _t_3494;
			float _t_3497;
			float _t_3501;
			float _t_3502;
			float _t_3503;
			float _t_3504;
			float _t_3505;
			float _t_3506;
			float _t_3507;
			float _t_3508;
			float _t_3509;
			bool _t_3510;
			float _t_3513;
			float _t_3517;
			float _t_3518;
			float _t_3519;
			float _t_3520;
			bool _t_3521;
			float _t_3524;
			float _t_3528;
			float _t_3529;
			float _t_3530;
			float _t_3531;
			float _t_3532;
			bool _t_3533;
			float _t_3536;
			float _t_3540;
			float _t_3541;
			float _t_3542;
			float _t_3543;
			float _t_3544;
			float _t_3545;
			bool _t_3546;
			float _t_3547;
			float _t_3548;
			float _t_3549;
			bool _t_3550;
			bool _t_3551;
			float _t_3552;
			float _t_3553;
			float _t_3554;
			bool _t_3555;
			float _t_3558;
			float _t_3562;
			float _t_3563;
			float _t_3564;
			float _t_3565;
			bool _t_3566;
			float _t_3569;
			float _t_3573;
			bool _t_3574;
			float _t_3575;
			float _t_3576;
			float _t_3577;
			float _t_3578;
			float _t_3579;
			bool _t_3580;
			float _t_3583;
			float _t_3587;
			float _t_3588;
			float _t_3589;
			float _t_3590;
			bool _t_3591;
			float _t_3594;
			float _t_3598;
			bool _t_3599;
			float _t_3600;
			float _t_3601;
			float _t_3602;
			bool _t_3603;
			float _t_3604;
			float _t_3605;
			float _t_3606;
			bool _t_3607;
			float _t_3610;
			float _t_3614;
			float _t_3615;
			float _t_3616;
			float _t_3617;
			bool _t_3618;
			float _t_3621;
			float _t_3625;
			bool _t_3626;
			float _t_3627;
			float _t_3628;
			float _t_3629;
			float _t_3630;
			float _t_3631;
			bool _t_3632;
			float _t_3635;
			float _t_3639;
			float _t_3640;
			float _t_3641;
			float _t_3642;
			bool _t_3643;
			float _t_3646;
			float _t_3650;
			bool _t_3651;
			float _t_3652;
			float _t_3653;
			float _t_3654;
			bool _t_3655;
			bool _t_3656;
			bool _t_3657;
			float _t_3658;
			float _t_3659;
		
			_t_3331 = -1.0f * ty1_7_1;
			_t_3332 = ty3_9_1 + _t_3331;
			_t_3333 = -1.0f * _t_3332;
			_t_3334 = _t_3333 < 0.0f;
			if(_t_3334)
				{
					float _t_3335;
					float _t_3336;
				
					_t_3335 = -1.0f * tx3_6_1;
					_t_3336 = tx1_4_1 + _t_3335;
					_t_3337 = _t_3336;
				
				}
		else
				{
					float _t_3338;
					float _t_3339;
					float _t_3340;
				
					_t_3338 = -1.0f * tx3_6_1;
					_t_3339 = tx1_4_1 + _t_3338;
					_t_3340 = -1.0f * _t_3339;
					_t_3337 = _t_3340;
				
				}
		
			_t_3341 = _t_3337 * _t_175;
			_t_3342 = _t_3341 * -1.0f;
			_t_3343 = -1.0f * ty1_7_1;
			_t_3344 = ty3_9_1 + _t_3343;
			_t_3345 = -1.0f * _t_3344;
			_t_3346 = _t_3345 < 0.0f;
			if(_t_3346)
				{
					float _t_3347;
					float _t_3348;
				
					_t_3347 = -1.0f * tx3_6_1;
					_t_3348 = tx1_4_1 + _t_3347;
					_t_3349 = _t_3348;
				
				}
		else
				{
					float _t_3350;
					float _t_3351;
					float _t_3352;
				
					_t_3350 = -1.0f * tx3_6_1;
					_t_3351 = tx1_4_1 + _t_3350;
					_t_3352 = -1.0f * _t_3351;
					_t_3349 = _t_3352;
				
				}
		
			_t_3353 = _t_3349 * _t_175;
			_t_3354 = _t_3353 * -1.0f;
			_t_3355 = 0.0f < _t_3354;
			if(_t_3355)
				{
				
					_t_3356 = px0_10_1;
				
				}
		else
				{
				
					_t_3356 = px1_11_1;
				
				}
		
			_t_3357 = _t_3342 * _t_3356;
			_t_3358 = -1.0f * ty1_7_1;
			_t_3359 = ty3_9_1 + _t_3358;
			_t_3360 = -1.0f * _t_3359;
			_t_3361 = _t_3360 < 0.0f;
			if(_t_3361)
				{
					float _t_3362;
					float _t_3363;
				
					_t_3362 = -1.0f * tx3_6_1;
					_t_3363 = tx1_4_1 + _t_3362;
					_t_3364 = _t_3363;
				
				}
		else
				{
					float _t_3365;
					float _t_3366;
					float _t_3367;
				
					_t_3365 = -1.0f * tx3_6_1;
					_t_3366 = tx1_4_1 + _t_3365;
					_t_3367 = -1.0f * _t_3366;
					_t_3364 = _t_3367;
				
				}
		
			_t_3368 = _t_3364 * _t_175;
			_t_3369 = -1.0f * ty1_7_1;
			_t_3370 = ty3_9_1 + _t_3369;
			_t_3371 = -1.0f * _t_3370;
			_t_3372 = _t_3371 < 0.0f;
			if(_t_3372)
				{
					float _t_3373;
					float _t_3374;
				
					_t_3373 = -1.0f * tx3_6_1;
					_t_3374 = tx1_4_1 + _t_3373;
					_t_3375 = _t_3374;
				
				}
		else
				{
					float _t_3376;
					float _t_3377;
					float _t_3378;
				
					_t_3376 = -1.0f * tx3_6_1;
					_t_3377 = tx1_4_1 + _t_3376;
					_t_3378 = -1.0f * _t_3377;
					_t_3375 = _t_3378;
				
				}
		
			_t_3379 = _t_3375 * _t_175;
			_t_3380 = _t_3368 * _t_3379;
			_t_3381 = -1.0f * ty1_7_1;
			_t_3382 = ty3_9_1 + _t_3381;
			_t_3383 = -1.0f * _t_3382;
			_t_3384 = _t_3383 < 0.0f;
			if(_t_3384)
				{
					float _t_3385;
					float _t_3386;
				
					_t_3385 = -1.0f * ty1_7_1;
					_t_3386 = ty3_9_1 + _t_3385;
					_t_3387 = _t_3386;
				
				}
		else
				{
					float _t_3388;
					float _t_3389;
					float _t_3390;
				
					_t_3388 = -1.0f * ty1_7_1;
					_t_3389 = ty3_9_1 + _t_3388;
					_t_3390 = -1.0f * _t_3389;
					_t_3387 = _t_3390;
				
				}
		
			_t_3391 = _t_3387 * _t_175;
			_t_3392 = 1.0f + _t_3391;
			_t_3393 = 1.0f / _t_3392;
			_t_3394 = _t_3380 * _t_3393;
			_t_3395 = _t_3394 * -1.0f;
			_t_3396 = 1.0f + _t_3395;
			_t_3397 = -1.0f * ty1_7_1;
			_t_3398 = ty3_9_1 + _t_3397;
			_t_3399 = -1.0f * _t_3398;
			_t_3400 = _t_3399 < 0.0f;
			if(_t_3400)
				{
					float _t_3401;
					float _t_3402;
				
					_t_3401 = -1.0f * tx3_6_1;
					_t_3402 = tx1_4_1 + _t_3401;
					_t_3403 = _t_3402;
				
				}
		else
				{
					float _t_3404;
					float _t_3405;
					float _t_3406;
				
					_t_3404 = -1.0f * tx3_6_1;
					_t_3405 = tx1_4_1 + _t_3404;
					_t_3406 = -1.0f * _t_3405;
					_t_3403 = _t_3406;
				
				}
		
			_t_3407 = _t_3403 * _t_175;
			_t_3408 = -1.0f * ty1_7_1;
			_t_3409 = ty3_9_1 + _t_3408;
			_t_3410 = -1.0f * _t_3409;
			_t_3411 = _t_3410 < 0.0f;
			if(_t_3411)
				{
					float _t_3412;
					float _t_3413;
				
					_t_3412 = -1.0f * tx3_6_1;
					_t_3413 = tx1_4_1 + _t_3412;
					_t_3414 = _t_3413;
				
				}
		else
				{
					float _t_3415;
					float _t_3416;
					float _t_3417;
				
					_t_3415 = -1.0f * tx3_6_1;
					_t_3416 = tx1_4_1 + _t_3415;
					_t_3417 = -1.0f * _t_3416;
					_t_3414 = _t_3417;
				
				}
		
			_t_3418 = _t_3414 * _t_175;
			_t_3419 = _t_3407 * _t_3418;
			_t_3420 = -1.0f * ty1_7_1;
			_t_3421 = ty3_9_1 + _t_3420;
			_t_3422 = -1.0f * _t_3421;
			_t_3423 = _t_3422 < 0.0f;
			if(_t_3423)
				{
					float _t_3424;
					float _t_3425;
				
					_t_3424 = -1.0f * ty1_7_1;
					_t_3425 = ty3_9_1 + _t_3424;
					_t_3426 = _t_3425;
				
				}
		else
				{
					float _t_3427;
					float _t_3428;
					float _t_3429;
				
					_t_3427 = -1.0f * ty1_7_1;
					_t_3428 = ty3_9_1 + _t_3427;
					_t_3429 = -1.0f * _t_3428;
					_t_3426 = _t_3429;
				
				}
		
			_t_3430 = _t_3426 * _t_175;
			_t_3431 = 1.0f + _t_3430;
			_t_3432 = 1.0f / _t_3431;
			_t_3433 = _t_3419 * _t_3432;
			_t_3434 = _t_3433 * -1.0f;
			_t_3435 = 1.0f + _t_3434;
			_t_3436 = 0.0f < _t_3435;
			if(_t_3436)
				{
				
					_t_3437 = py0_12_1;
				
				}
		else
				{
				
					_t_3437 = py1_13_1;
				
				}
		
			_t_3438 = _t_3396 * _t_3437;
			_t_3439 = _t_3357 + _t_3438;
			_t_3440 = _t_3439 < y__2721_1;
			_t_3441 = -1.0f * ty1_7_1;
			_t_3442 = ty3_9_1 + _t_3441;
			_t_3443 = -1.0f * _t_3442;
			_t_3444 = _t_3443 < 0.0f;
			if(_t_3444)
				{
					float _t_3445;
					float _t_3446;
				
					_t_3445 = -1.0f * tx3_6_1;
					_t_3446 = tx1_4_1 + _t_3445;
					_t_3447 = _t_3446;
				
				}
		else
				{
					float _t_3448;
					float _t_3449;
					float _t_3450;
				
					_t_3448 = -1.0f * tx3_6_1;
					_t_3449 = tx1_4_1 + _t_3448;
					_t_3450 = -1.0f * _t_3449;
					_t_3447 = _t_3450;
				
				}
		
			_t_3451 = _t_3447 * _t_175;
			_t_3452 = _t_3451 * -1.0f;
			_t_3453 = -1.0f * ty1_7_1;
			_t_3454 = ty3_9_1 + _t_3453;
			_t_3455 = -1.0f * _t_3454;
			_t_3456 = _t_3455 < 0.0f;
			if(_t_3456)
				{
					float _t_3457;
					float _t_3458;
				
					_t_3457 = -1.0f * tx3_6_1;
					_t_3458 = tx1_4_1 + _t_3457;
					_t_3459 = _t_3458;
				
				}
		else
				{
					float _t_3460;
					float _t_3461;
					float _t_3462;
				
					_t_3460 = -1.0f * tx3_6_1;
					_t_3461 = tx1_4_1 + _t_3460;
					_t_3462 = -1.0f * _t_3461;
					_t_3459 = _t_3462;
				
				}
		
			_t_3463 = _t_3459 * _t_175;
			_t_3464 = _t_3463 * -1.0f;
			_t_3465 = 0.0f < _t_3464;
			if(_t_3465)
				{
				
					_t_3466 = px1_11_1;
				
				}
		else
				{
				
					_t_3466 = px0_10_1;
				
				}
		
			_t_3467 = _t_3452 * _t_3466;
			_t_3468 = -1.0f * ty1_7_1;
			_t_3469 = ty3_9_1 + _t_3468;
			_t_3470 = -1.0f * _t_3469;
			_t_3471 = _t_3470 < 0.0f;
			if(_t_3471)
				{
					float _t_3472;
					float _t_3473;
				
					_t_3472 = -1.0f * tx3_6_1;
					_t_3473 = tx1_4_1 + _t_3472;
					_t_3474 = _t_3473;
				
				}
		else
				{
					float _t_3475;
					float _t_3476;
					float _t_3477;
				
					_t_3475 = -1.0f * tx3_6_1;
					_t_3476 = tx1_4_1 + _t_3475;
					_t_3477 = -1.0f * _t_3476;
					_t_3474 = _t_3477;
				
				}
		
			_t_3478 = _t_3474 * _t_175;
			_t_3479 = -1.0f * ty1_7_1;
			_t_3480 = ty3_9_1 + _t_3479;
			_t_3481 = -1.0f * _t_3480;
			_t_3482 = _t_3481 < 0.0f;
			if(_t_3482)
				{
					float _t_3483;
					float _t_3484;
				
					_t_3483 = -1.0f * tx3_6_1;
					_t_3484 = tx1_4_1 + _t_3483;
					_t_3485 = _t_3484;
				
				}
		else
				{
					float _t_3486;
					float _t_3487;
					float _t_3488;
				
					_t_3486 = -1.0f * tx3_6_1;
					_t_3487 = tx1_4_1 + _t_3486;
					_t_3488 = -1.0f * _t_3487;
					_t_3485 = _t_3488;
				
				}
		
			_t_3489 = _t_3485 * _t_175;
			_t_3490 = _t_3478 * _t_3489;
			_t_3491 = -1.0f * ty1_7_1;
			_t_3492 = ty3_9_1 + _t_3491;
			_t_3493 = -1.0f * _t_3492;
			_t_3494 = _t_3493 < 0.0f;
			if(_t_3494)
				{
					float _t_3495;
					float _t_3496;
				
					_t_3495 = -1.0f * ty1_7_1;
					_t_3496 = ty3_9_1 + _t_3495;
					_t_3497 = _t_3496;
				
				}
		else
				{
					float _t_3498;
					float _t_3499;
					float _t_3500;
				
					_t_3498 = -1.0f * ty1_7_1;
					_t_3499 = ty3_9_1 + _t_3498;
					_t_3500 = -1.0f * _t_3499;
					_t_3497 = _t_3500;
				
				}
		
			_t_3501 = _t_3497 * _t_175;
			_t_3502 = 1.0f + _t_3501;
			_t_3503 = 1.0f / _t_3502;
			_t_3504 = _t_3490 * _t_3503;
			_t_3505 = _t_3504 * -1.0f;
			_t_3506 = 1.0f + _t_3505;
			_t_3507 = -1.0f * ty1_7_1;
			_t_3508 = ty3_9_1 + _t_3507;
			_t_3509 = -1.0f * _t_3508;
			_t_3510 = _t_3509 < 0.0f;
			if(_t_3510)
				{
					float _t_3511;
					float _t_3512;
				
					_t_3511 = -1.0f * tx3_6_1;
					_t_3512 = tx1_4_1 + _t_3511;
					_t_3513 = _t_3512;
				
				}
		else
				{
					float _t_3514;
					float _t_3515;
					float _t_3516;
				
					_t_3514 = -1.0f * tx3_6_1;
					_t_3515 = tx1_4_1 + _t_3514;
					_t_3516 = -1.0f * _t_3515;
					_t_3513 = _t_3516;
				
				}
		
			_t_3517 = _t_3513 * _t_175;
			_t_3518 = -1.0f * ty1_7_1;
			_t_3519 = ty3_9_1 + _t_3518;
			_t_3520 = -1.0f * _t_3519;
			_t_3521 = _t_3520 < 0.0f;
			if(_t_3521)
				{
					float _t_3522;
					float _t_3523;
				
					_t_3522 = -1.0f * tx3_6_1;
					_t_3523 = tx1_4_1 + _t_3522;
					_t_3524 = _t_3523;
				
				}
		else
				{
					float _t_3525;
					float _t_3526;
					float _t_3527;
				
					_t_3525 = -1.0f * tx3_6_1;
					_t_3526 = tx1_4_1 + _t_3525;
					_t_3527 = -1.0f * _t_3526;
					_t_3524 = _t_3527;
				
				}
		
			_t_3528 = _t_3524 * _t_175;
			_t_3529 = _t_3517 * _t_3528;
			_t_3530 = -1.0f * ty1_7_1;
			_t_3531 = ty3_9_1 + _t_3530;
			_t_3532 = -1.0f * _t_3531;
			_t_3533 = _t_3532 < 0.0f;
			if(_t_3533)
				{
					float _t_3534;
					float _t_3535;
				
					_t_3534 = -1.0f * ty1_7_1;
					_t_3535 = ty3_9_1 + _t_3534;
					_t_3536 = _t_3535;
				
				}
		else
				{
					float _t_3537;
					float _t_3538;
					float _t_3539;
				
					_t_3537 = -1.0f * ty1_7_1;
					_t_3538 = ty3_9_1 + _t_3537;
					_t_3539 = -1.0f * _t_3538;
					_t_3536 = _t_3539;
				
				}
		
			_t_3540 = _t_3536 * _t_175;
			_t_3541 = 1.0f + _t_3540;
			_t_3542 = 1.0f / _t_3541;
			_t_3543 = _t_3529 * _t_3542;
			_t_3544 = _t_3543 * -1.0f;
			_t_3545 = 1.0f + _t_3544;
			_t_3546 = 0.0f < _t_3545;
			if(_t_3546)
				{
				
					_t_3547 = py1_13_1;
				
				}
		else
				{
				
					_t_3547 = py0_12_1;
				
				}
		
			_t_3548 = _t_3506 * _t_3547;
			_t_3549 = _t_3467 + _t_3548;
			_t_3550 = y__2721_1 < _t_3549;
			_t_3551 = _t_3440 && _t_3550;
			_t_3552 = -1.0f * ty1_7_1;
			_t_3553 = ty3_9_1 + _t_3552;
			_t_3554 = -1.0f * _t_3553;
			_t_3555 = _t_3554 < 0.0f;
			if(_t_3555)
				{
					float _t_3556;
					float _t_3557;
				
					_t_3556 = -1.0f * ty1_7_1;
					_t_3557 = ty3_9_1 + _t_3556;
					_t_3558 = _t_3557;
				
				}
		else
				{
					float _t_3559;
					float _t_3560;
					float _t_3561;
				
					_t_3559 = -1.0f * ty1_7_1;
					_t_3560 = ty3_9_1 + _t_3559;
					_t_3561 = -1.0f * _t_3560;
					_t_3558 = _t_3561;
				
				}
		
			_t_3562 = _t_3558 * _t_175;
			_t_3563 = -1.0f * ty1_7_1;
			_t_3564 = ty3_9_1 + _t_3563;
			_t_3565 = -1.0f * _t_3564;
			_t_3566 = _t_3565 < 0.0f;
			if(_t_3566)
				{
					float _t_3567;
					float _t_3568;
				
					_t_3567 = -1.0f * ty1_7_1;
					_t_3568 = ty3_9_1 + _t_3567;
					_t_3569 = _t_3568;
				
				}
		else
				{
					float _t_3570;
					float _t_3571;
					float _t_3572;
				
					_t_3570 = -1.0f * ty1_7_1;
					_t_3571 = ty3_9_1 + _t_3570;
					_t_3572 = -1.0f * _t_3571;
					_t_3569 = _t_3572;
				
				}
		
			_t_3573 = _t_3569 * _t_175;
			_t_3574 = 0.0f < _t_3573;
			if(_t_3574)
				{
				
					_t_3575 = px0_10_1;
				
				}
		else
				{
				
					_t_3575 = px1_11_1;
				
				}
		
			_t_3576 = _t_3562 * _t_3575;
			_t_3577 = -1.0f * ty1_7_1;
			_t_3578 = ty3_9_1 + _t_3577;
			_t_3579 = -1.0f * _t_3578;
			_t_3580 = _t_3579 < 0.0f;
			if(_t_3580)
				{
					float _t_3581;
					float _t_3582;
				
					_t_3581 = -1.0f * tx3_6_1;
					_t_3582 = tx1_4_1 + _t_3581;
					_t_3583 = _t_3582;
				
				}
		else
				{
					float _t_3584;
					float _t_3585;
					float _t_3586;
				
					_t_3584 = -1.0f * tx3_6_1;
					_t_3585 = tx1_4_1 + _t_3584;
					_t_3586 = -1.0f * _t_3585;
					_t_3583 = _t_3586;
				
				}
		
			_t_3587 = _t_3583 * _t_175;
			_t_3588 = -1.0f * ty1_7_1;
			_t_3589 = ty3_9_1 + _t_3588;
			_t_3590 = -1.0f * _t_3589;
			_t_3591 = _t_3590 < 0.0f;
			if(_t_3591)
				{
					float _t_3592;
					float _t_3593;
				
					_t_3592 = -1.0f * tx3_6_1;
					_t_3593 = tx1_4_1 + _t_3592;
					_t_3594 = _t_3593;
				
				}
		else
				{
					float _t_3595;
					float _t_3596;
					float _t_3597;
				
					_t_3595 = -1.0f * tx3_6_1;
					_t_3596 = tx1_4_1 + _t_3595;
					_t_3597 = -1.0f * _t_3596;
					_t_3594 = _t_3597;
				
				}
		
			_t_3598 = _t_3594 * _t_175;
			_t_3599 = 0.0f < _t_3598;
			if(_t_3599)
				{
				
					_t_3600 = py0_12_1;
				
				}
		else
				{
				
					_t_3600 = py1_13_1;
				
				}
		
			_t_3601 = _t_3587 * _t_3600;
			_t_3602 = _t_3576 + _t_3601;
			_t_3603 = _t_3602 < _t_3200;
			_t_3604 = -1.0f * ty1_7_1;
			_t_3605 = ty3_9_1 + _t_3604;
			_t_3606 = -1.0f * _t_3605;
			_t_3607 = _t_3606 < 0.0f;
			if(_t_3607)
				{
					float _t_3608;
					float _t_3609;
				
					_t_3608 = -1.0f * ty1_7_1;
					_t_3609 = ty3_9_1 + _t_3608;
					_t_3610 = _t_3609;
				
				}
		else
				{
					float _t_3611;
					float _t_3612;
					float _t_3613;
				
					_t_3611 = -1.0f * ty1_7_1;
					_t_3612 = ty3_9_1 + _t_3611;
					_t_3613 = -1.0f * _t_3612;
					_t_3610 = _t_3613;
				
				}
		
			_t_3614 = _t_3610 * _t_175;
			_t_3615 = -1.0f * ty1_7_1;
			_t_3616 = ty3_9_1 + _t_3615;
			_t_3617 = -1.0f * _t_3616;
			_t_3618 = _t_3617 < 0.0f;
			if(_t_3618)
				{
					float _t_3619;
					float _t_3620;
				
					_t_3619 = -1.0f * ty1_7_1;
					_t_3620 = ty3_9_1 + _t_3619;
					_t_3621 = _t_3620;
				
				}
		else
				{
					float _t_3622;
					float _t_3623;
					float _t_3624;
				
					_t_3622 = -1.0f * ty1_7_1;
					_t_3623 = ty3_9_1 + _t_3622;
					_t_3624 = -1.0f * _t_3623;
					_t_3621 = _t_3624;
				
				}
		
			_t_3625 = _t_3621 * _t_175;
			_t_3626 = 0.0f < _t_3625;
			if(_t_3626)
				{
				
					_t_3627 = px1_11_1;
				
				}
		else
				{
				
					_t_3627 = px0_10_1;
				
				}
		
			_t_3628 = _t_3614 * _t_3627;
			_t_3629 = -1.0f * ty1_7_1;
			_t_3630 = ty3_9_1 + _t_3629;
			_t_3631 = -1.0f * _t_3630;
			_t_3632 = _t_3631 < 0.0f;
			if(_t_3632)
				{
					float _t_3633;
					float _t_3634;
				
					_t_3633 = -1.0f * tx3_6_1;
					_t_3634 = tx1_4_1 + _t_3633;
					_t_3635 = _t_3634;
				
				}
		else
				{
					float _t_3636;
					float _t_3637;
					float _t_3638;
				
					_t_3636 = -1.0f * tx3_6_1;
					_t_3637 = tx1_4_1 + _t_3636;
					_t_3638 = -1.0f * _t_3637;
					_t_3635 = _t_3638;
				
				}
		
			_t_3639 = _t_3635 * _t_175;
			_t_3640 = -1.0f * ty1_7_1;
			_t_3641 = ty3_9_1 + _t_3640;
			_t_3642 = -1.0f * _t_3641;
			_t_3643 = _t_3642 < 0.0f;
			if(_t_3643)
				{
					float _t_3644;
					float _t_3645;
				
					_t_3644 = -1.0f * tx3_6_1;
					_t_3645 = tx1_4_1 + _t_3644;
					_t_3646 = _t_3645;
				
				}
		else
				{
					float _t_3647;
					float _t_3648;
					float _t_3649;
				
					_t_3647 = -1.0f * tx3_6_1;
					_t_3648 = tx1_4_1 + _t_3647;
					_t_3649 = -1.0f * _t_3648;
					_t_3646 = _t_3649;
				
				}
		
			_t_3650 = _t_3646 * _t_175;
			_t_3651 = 0.0f < _t_3650;
			if(_t_3651)
				{
				
					_t_3652 = py1_13_1;
				
				}
		else
				{
				
					_t_3652 = py0_12_1;
				
				}
		
			_t_3653 = _t_3639 * _t_3652;
			_t_3654 = _t_3628 + _t_3653;
			_t_3655 = _t_3200 < _t_3654;
			_t_3656 = _t_3603 && _t_3655;
			_t_3657 = _t_3551 && _t_3656;
			if(_t_3657)
				{
				
					_t_3658 = 1.0f;
				
				}
		else
				{
				
					_t_3658 = 0.0f;
				
				}
		
			_t_3659 = _t_3658 * _t_175;
			_t_3660 = _t_3659;
		
		}
else
		{
		
			_t_3660 = 0.0f;
		
		}

	_t_3201 = _t_3323 * _t_3660;

	return _t_3201;
}
__device__ float tegpixellet_block_23(float ty3_9_1,float ty1_7_1,float _t_175,float _t_3200,float tx1_4_1,float tx3_6_1,float y__2721_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1){
	float _t_3202;
	float _t_3203;
	float _t_3204;
	bool _t_3205;
	float _t_3208;
	float _t_3212;
	float _t_3213;
	float _t_3214;
	float _t_3215;
	float _t_3216;
	bool _t_3217;
	float _t_3220;
	float _t_3224;
	float _t_3225;
	float _t_3226;
	float _t_3227;
	float _t_3228;
	float _t_3229;
	float _t_3230;
	bool _t_3231;
	float _t_3234;
	float _t_3238;
	float _t_3239;
	float _t_3240;
	float _t_3241;
	bool _t_3242;
	float _t_3245;
	float _t_3249;
	float _t_3250;
	float _t_3251;
	float _t_3252;
	float _t_3253;
	bool _t_3254;
	float _t_3257;
	float _t_3261;
	float _t_3262;
	float _t_3263;
	float _t_3264;
	float _t_3265;
	float _t_3266;
	float _t_3267;
	float _t_3268;
	float _t_3269;
	float _t_3270;
	bool _t_3271;
	float _t_3274;
	float _t_3278;
	float _t_3279;
	float _t_3280;

	float _t_3201;

	_t_3202 = -1.0f * ty1_7_1;
	_t_3203 = ty3_9_1 + _t_3202;
	_t_3204 = -1.0f * _t_3203;
	_t_3205 = _t_3204 < 0.0f;
	if(_t_3205)
		{
			float _t_3206;
			float _t_3207;
		
			_t_3206 = -1.0f * ty1_7_1;
			_t_3207 = ty3_9_1 + _t_3206;
			_t_3208 = _t_3207;
		
		}
else
		{
			float _t_3209;
			float _t_3210;
			float _t_3211;
		
			_t_3209 = -1.0f * ty1_7_1;
			_t_3210 = ty3_9_1 + _t_3209;
			_t_3211 = -1.0f * _t_3210;
			_t_3208 = _t_3211;
		
		}

	_t_3212 = _t_3208 * _t_175;
	_t_3213 = _t_3212 * _t_3200;
	_t_3214 = -1.0f * ty1_7_1;
	_t_3215 = ty3_9_1 + _t_3214;
	_t_3216 = -1.0f * _t_3215;
	_t_3217 = _t_3216 < 0.0f;
	if(_t_3217)
		{
			float _t_3218;
			float _t_3219;
		
			_t_3218 = -1.0f * tx3_6_1;
			_t_3219 = tx1_4_1 + _t_3218;
			_t_3220 = _t_3219;
		
		}
else
		{
			float _t_3221;
			float _t_3222;
			float _t_3223;
		
			_t_3221 = -1.0f * tx3_6_1;
			_t_3222 = tx1_4_1 + _t_3221;
			_t_3223 = -1.0f * _t_3222;
			_t_3220 = _t_3223;
		
		}

	_t_3224 = _t_3220 * _t_175;
	_t_3225 = _t_3224 * -1.0f;
	_t_3226 = _t_3225 * y__2721_1;
	_t_3227 = _t_3213 + _t_3226;
	_t_3228 = -1.0f * ty1_7_1;
	_t_3229 = ty3_9_1 + _t_3228;
	_t_3230 = -1.0f * _t_3229;
	_t_3231 = _t_3230 < 0.0f;
	if(_t_3231)
		{
			float _t_3232;
			float _t_3233;
		
			_t_3232 = -1.0f * tx3_6_1;
			_t_3233 = tx1_4_1 + _t_3232;
			_t_3234 = _t_3233;
		
		}
else
		{
			float _t_3235;
			float _t_3236;
			float _t_3237;
		
			_t_3235 = -1.0f * tx3_6_1;
			_t_3236 = tx1_4_1 + _t_3235;
			_t_3237 = -1.0f * _t_3236;
			_t_3234 = _t_3237;
		
		}

	_t_3238 = _t_3234 * _t_175;
	_t_3239 = -1.0f * ty1_7_1;
	_t_3240 = ty3_9_1 + _t_3239;
	_t_3241 = -1.0f * _t_3240;
	_t_3242 = _t_3241 < 0.0f;
	if(_t_3242)
		{
			float _t_3243;
			float _t_3244;
		
			_t_3243 = -1.0f * tx3_6_1;
			_t_3244 = tx1_4_1 + _t_3243;
			_t_3245 = _t_3244;
		
		}
else
		{
			float _t_3246;
			float _t_3247;
			float _t_3248;
		
			_t_3246 = -1.0f * tx3_6_1;
			_t_3247 = tx1_4_1 + _t_3246;
			_t_3248 = -1.0f * _t_3247;
			_t_3245 = _t_3248;
		
		}

	_t_3249 = _t_3245 * _t_175;
	_t_3250 = _t_3238 * _t_3249;
	_t_3251 = -1.0f * ty1_7_1;
	_t_3252 = ty3_9_1 + _t_3251;
	_t_3253 = -1.0f * _t_3252;
	_t_3254 = _t_3253 < 0.0f;
	if(_t_3254)
		{
			float _t_3255;
			float _t_3256;
		
			_t_3255 = -1.0f * ty1_7_1;
			_t_3256 = ty3_9_1 + _t_3255;
			_t_3257 = _t_3256;
		
		}
else
		{
			float _t_3258;
			float _t_3259;
			float _t_3260;
		
			_t_3258 = -1.0f * ty1_7_1;
			_t_3259 = ty3_9_1 + _t_3258;
			_t_3260 = -1.0f * _t_3259;
			_t_3257 = _t_3260;
		
		}

	_t_3261 = _t_3257 * _t_175;
	_t_3262 = 1.0f + _t_3261;
	_t_3263 = 1.0f / _t_3262;
	_t_3264 = _t_3250 * _t_3263;
	_t_3265 = _t_3264 * -1.0f;
	_t_3266 = 1.0f + _t_3265;
	_t_3267 = _t_3266 * y__2721_1;
	_t_3268 = -1.0f * ty1_7_1;
	_t_3269 = ty3_9_1 + _t_3268;
	_t_3270 = -1.0f * _t_3269;
	_t_3271 = _t_3270 < 0.0f;
	if(_t_3271)
		{
			float _t_3272;
			float _t_3273;
		
			_t_3272 = -1.0f * tx3_6_1;
			_t_3273 = tx1_4_1 + _t_3272;
			_t_3274 = _t_3273;
		
		}
else
		{
			float _t_3275;
			float _t_3276;
			float _t_3277;
		
			_t_3275 = -1.0f * tx3_6_1;
			_t_3276 = tx1_4_1 + _t_3275;
			_t_3277 = -1.0f * _t_3276;
			_t_3274 = _t_3277;
		
		}

	_t_3278 = _t_3274 * _t_175;
	_t_3279 = _t_3278 * _t_3200;
	_t_3280 = _t_3267 + _t_3279;
	_t_3201 = tegpixellet_block_24(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,_t_3227,_t_3280,ty3_9_1,tx3_6_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,_t_175,y__2721_1,_t_3200);

	return _t_3201;
}
__device__ float tegpixelbody_block_19(float ty3_9_1,float ty1_7_1,float _t_175,float px0_10_1,float px1_11_1,float tx1_4_1,float tx3_6_1,float py0_12_1,float py1_13_1,float y__2721_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_3044;
	float _t_3045;
	float _t_3046;
	bool _t_3047;
	float _t_3050;
	float _t_3054;
	float _t_3055;
	float _t_3056;
	float _t_3057;
	bool _t_3058;
	float _t_3061;
	float _t_3065;
	bool _t_3066;
	float _t_3067;
	float _t_3068;
	float _t_3069;
	float _t_3070;
	float _t_3071;
	bool _t_3072;
	float _t_3075;
	float _t_3079;
	float _t_3080;
	float _t_3081;
	float _t_3082;
	bool _t_3083;
	float _t_3086;
	float _t_3090;
	bool _t_3091;
	float _t_3092;
	float _t_3093;
	float _t_3094;
	float _t_3095;
	float _t_3096;
	float _t_3097;
	bool _t_3098;
	float _t_3103;
	float _t_3109;
	float _t_3110;
	float _t_3111;
	float _t_3112;
	bool _t_3113;
	float _t_3114;
	float _t_3115;
	float _t_3116;
	bool _t_3117;
	float _t_3120;
	float _t_3124;
	float _t_3125;
	float _t_3126;
	float _t_3127;
	bool _t_3128;
	float _t_3131;
	float _t_3135;
	bool _t_3136;
	float _t_3137;
	float _t_3138;
	float _t_3139;
	float _t_3140;
	float _t_3141;
	bool _t_3142;
	float _t_3145;
	float _t_3149;
	float _t_3150;
	float _t_3151;
	float _t_3152;
	bool _t_3153;
	float _t_3156;
	float _t_3160;
	bool _t_3161;
	float _t_3162;
	float _t_3163;
	float _t_3164;
	float _t_3165;
	float _t_3166;
	float _t_3167;
	bool _t_3168;
	float _t_3173;
	float _t_3179;
	float _t_3180;
	float _t_3181;
	float _t_3182;
	bool _t_3183;
	bool _t_3184;

	float _t_3043;

	_t_3044 = -1.0f * ty1_7_1;
	_t_3045 = ty3_9_1 + _t_3044;
	_t_3046 = -1.0f * _t_3045;
	_t_3047 = _t_3046 < 0.0f;
	if(_t_3047)
		{
			float _t_3048;
			float _t_3049;
		
			_t_3048 = -1.0f * ty1_7_1;
			_t_3049 = ty3_9_1 + _t_3048;
			_t_3050 = _t_3049;
		
		}
else
		{
			float _t_3051;
			float _t_3052;
			float _t_3053;
		
			_t_3051 = -1.0f * ty1_7_1;
			_t_3052 = ty3_9_1 + _t_3051;
			_t_3053 = -1.0f * _t_3052;
			_t_3050 = _t_3053;
		
		}

	_t_3054 = _t_3050 * _t_175;
	_t_3055 = -1.0f * ty1_7_1;
	_t_3056 = ty3_9_1 + _t_3055;
	_t_3057 = -1.0f * _t_3056;
	_t_3058 = _t_3057 < 0.0f;
	if(_t_3058)
		{
			float _t_3059;
			float _t_3060;
		
			_t_3059 = -1.0f * ty1_7_1;
			_t_3060 = ty3_9_1 + _t_3059;
			_t_3061 = _t_3060;
		
		}
else
		{
			float _t_3062;
			float _t_3063;
			float _t_3064;
		
			_t_3062 = -1.0f * ty1_7_1;
			_t_3063 = ty3_9_1 + _t_3062;
			_t_3064 = -1.0f * _t_3063;
			_t_3061 = _t_3064;
		
		}

	_t_3065 = _t_3061 * _t_175;
	_t_3066 = 0.0f < _t_3065;
	if(_t_3066)
		{
		
			_t_3067 = px0_10_1;
		
		}
else
		{
		
			_t_3067 = px1_11_1;
		
		}

	_t_3068 = _t_3054 * _t_3067;
	_t_3069 = -1.0f * ty1_7_1;
	_t_3070 = ty3_9_1 + _t_3069;
	_t_3071 = -1.0f * _t_3070;
	_t_3072 = _t_3071 < 0.0f;
	if(_t_3072)
		{
			float _t_3073;
			float _t_3074;
		
			_t_3073 = -1.0f * tx3_6_1;
			_t_3074 = tx1_4_1 + _t_3073;
			_t_3075 = _t_3074;
		
		}
else
		{
			float _t_3076;
			float _t_3077;
			float _t_3078;
		
			_t_3076 = -1.0f * tx3_6_1;
			_t_3077 = tx1_4_1 + _t_3076;
			_t_3078 = -1.0f * _t_3077;
			_t_3075 = _t_3078;
		
		}

	_t_3079 = _t_3075 * _t_175;
	_t_3080 = -1.0f * ty1_7_1;
	_t_3081 = ty3_9_1 + _t_3080;
	_t_3082 = -1.0f * _t_3081;
	_t_3083 = _t_3082 < 0.0f;
	if(_t_3083)
		{
			float _t_3084;
			float _t_3085;
		
			_t_3084 = -1.0f * tx3_6_1;
			_t_3085 = tx1_4_1 + _t_3084;
			_t_3086 = _t_3085;
		
		}
else
		{
			float _t_3087;
			float _t_3088;
			float _t_3089;
		
			_t_3087 = -1.0f * tx3_6_1;
			_t_3088 = tx1_4_1 + _t_3087;
			_t_3089 = -1.0f * _t_3088;
			_t_3086 = _t_3089;
		
		}

	_t_3090 = _t_3086 * _t_175;
	_t_3091 = 0.0f < _t_3090;
	if(_t_3091)
		{
		
			_t_3092 = py0_12_1;
		
		}
else
		{
		
			_t_3092 = py1_13_1;
		
		}

	_t_3093 = _t_3079 * _t_3092;
	_t_3094 = _t_3068 + _t_3093;
	_t_3095 = -1.0f * ty1_7_1;
	_t_3096 = ty3_9_1 + _t_3095;
	_t_3097 = -1.0f * _t_3096;
	_t_3098 = _t_3097 < 0.0f;
	if(_t_3098)
		{
			float _t_3099;
			float _t_3100;
			float _t_3101;
			float _t_3102;
		
			_t_3099 = tx3_6_1 * ty1_7_1;
			_t_3100 = tx1_4_1 * ty3_9_1;
			_t_3101 = _t_3100 * -1.0f;
			_t_3102 = _t_3099 + _t_3101;
			_t_3103 = _t_3102;
		
		}
else
		{
			float _t_3104;
			float _t_3105;
			float _t_3106;
			float _t_3107;
			float _t_3108;
		
			_t_3104 = tx3_6_1 * ty1_7_1;
			_t_3105 = tx1_4_1 * ty3_9_1;
			_t_3106 = _t_3105 * -1.0f;
			_t_3107 = _t_3104 + _t_3106;
			_t_3108 = -1.0f * _t_3107;
			_t_3103 = _t_3108;
		
		}

	_t_3109 = -1.0f * _t_3103;
	_t_3110 = _t_3109 * _t_175;
	_t_3111 = _t_3110 * -1.0f;
	_t_3112 = _t_3094 + _t_3111;
	_t_3113 = _t_3112 < 0.0f;
	_t_3114 = -1.0f * ty1_7_1;
	_t_3115 = ty3_9_1 + _t_3114;
	_t_3116 = -1.0f * _t_3115;
	_t_3117 = _t_3116 < 0.0f;
	if(_t_3117)
		{
			float _t_3118;
			float _t_3119;
		
			_t_3118 = -1.0f * ty1_7_1;
			_t_3119 = ty3_9_1 + _t_3118;
			_t_3120 = _t_3119;
		
		}
else
		{
			float _t_3121;
			float _t_3122;
			float _t_3123;
		
			_t_3121 = -1.0f * ty1_7_1;
			_t_3122 = ty3_9_1 + _t_3121;
			_t_3123 = -1.0f * _t_3122;
			_t_3120 = _t_3123;
		
		}

	_t_3124 = _t_3120 * _t_175;
	_t_3125 = -1.0f * ty1_7_1;
	_t_3126 = ty3_9_1 + _t_3125;
	_t_3127 = -1.0f * _t_3126;
	_t_3128 = _t_3127 < 0.0f;
	if(_t_3128)
		{
			float _t_3129;
			float _t_3130;
		
			_t_3129 = -1.0f * ty1_7_1;
			_t_3130 = ty3_9_1 + _t_3129;
			_t_3131 = _t_3130;
		
		}
else
		{
			float _t_3132;
			float _t_3133;
			float _t_3134;
		
			_t_3132 = -1.0f * ty1_7_1;
			_t_3133 = ty3_9_1 + _t_3132;
			_t_3134 = -1.0f * _t_3133;
			_t_3131 = _t_3134;
		
		}

	_t_3135 = _t_3131 * _t_175;
	_t_3136 = 0.0f < _t_3135;
	if(_t_3136)
		{
		
			_t_3137 = px1_11_1;
		
		}
else
		{
		
			_t_3137 = px0_10_1;
		
		}

	_t_3138 = _t_3124 * _t_3137;
	_t_3139 = -1.0f * ty1_7_1;
	_t_3140 = ty3_9_1 + _t_3139;
	_t_3141 = -1.0f * _t_3140;
	_t_3142 = _t_3141 < 0.0f;
	if(_t_3142)
		{
			float _t_3143;
			float _t_3144;
		
			_t_3143 = -1.0f * tx3_6_1;
			_t_3144 = tx1_4_1 + _t_3143;
			_t_3145 = _t_3144;
		
		}
else
		{
			float _t_3146;
			float _t_3147;
			float _t_3148;
		
			_t_3146 = -1.0f * tx3_6_1;
			_t_3147 = tx1_4_1 + _t_3146;
			_t_3148 = -1.0f * _t_3147;
			_t_3145 = _t_3148;
		
		}

	_t_3149 = _t_3145 * _t_175;
	_t_3150 = -1.0f * ty1_7_1;
	_t_3151 = ty3_9_1 + _t_3150;
	_t_3152 = -1.0f * _t_3151;
	_t_3153 = _t_3152 < 0.0f;
	if(_t_3153)
		{
			float _t_3154;
			float _t_3155;
		
			_t_3154 = -1.0f * tx3_6_1;
			_t_3155 = tx1_4_1 + _t_3154;
			_t_3156 = _t_3155;
		
		}
else
		{
			float _t_3157;
			float _t_3158;
			float _t_3159;
		
			_t_3157 = -1.0f * tx3_6_1;
			_t_3158 = tx1_4_1 + _t_3157;
			_t_3159 = -1.0f * _t_3158;
			_t_3156 = _t_3159;
		
		}

	_t_3160 = _t_3156 * _t_175;
	_t_3161 = 0.0f < _t_3160;
	if(_t_3161)
		{
		
			_t_3162 = py1_13_1;
		
		}
else
		{
		
			_t_3162 = py0_12_1;
		
		}

	_t_3163 = _t_3149 * _t_3162;
	_t_3164 = _t_3138 + _t_3163;
	_t_3165 = -1.0f * ty1_7_1;
	_t_3166 = ty3_9_1 + _t_3165;
	_t_3167 = -1.0f * _t_3166;
	_t_3168 = _t_3167 < 0.0f;
	if(_t_3168)
		{
			float _t_3169;
			float _t_3170;
			float _t_3171;
			float _t_3172;
		
			_t_3169 = tx3_6_1 * ty1_7_1;
			_t_3170 = tx1_4_1 * ty3_9_1;
			_t_3171 = _t_3170 * -1.0f;
			_t_3172 = _t_3169 + _t_3171;
			_t_3173 = _t_3172;
		
		}
else
		{
			float _t_3174;
			float _t_3175;
			float _t_3176;
			float _t_3177;
			float _t_3178;
		
			_t_3174 = tx3_6_1 * ty1_7_1;
			_t_3175 = tx1_4_1 * ty3_9_1;
			_t_3176 = _t_3175 * -1.0f;
			_t_3177 = _t_3174 + _t_3176;
			_t_3178 = -1.0f * _t_3177;
			_t_3173 = _t_3178;
		
		}

	_t_3179 = -1.0f * _t_3173;
	_t_3180 = _t_3179 * _t_175;
	_t_3181 = _t_3180 * -1.0f;
	_t_3182 = _t_3164 + _t_3181;
	_t_3183 = 0.0f < _t_3182;
	_t_3184 = _t_3113 && _t_3183;
	if(_t_3184)
		{
			float _t_3185;
			float _t_3186;
			float _t_3187;
			bool _t_3188;
			float _t_3193;
			float _t_3199;
			float _t_3200;
			float _t_3201;
		
			_t_3185 = -1.0f * ty1_7_1;
			_t_3186 = ty3_9_1 + _t_3185;
			_t_3187 = -1.0f * _t_3186;
			_t_3188 = _t_3187 < 0.0f;
			if(_t_3188)
				{
					float _t_3189;
					float _t_3190;
					float _t_3191;
					float _t_3192;
				
					_t_3189 = tx3_6_1 * ty1_7_1;
					_t_3190 = tx1_4_1 * ty3_9_1;
					_t_3191 = _t_3190 * -1.0f;
					_t_3192 = _t_3189 + _t_3191;
					_t_3193 = _t_3192;
				
				}
		else
				{
					float _t_3194;
					float _t_3195;
					float _t_3196;
					float _t_3197;
					float _t_3198;
				
					_t_3194 = tx3_6_1 * ty1_7_1;
					_t_3195 = tx1_4_1 * ty3_9_1;
					_t_3196 = _t_3195 * -1.0f;
					_t_3197 = _t_3194 + _t_3196;
					_t_3198 = -1.0f * _t_3197;
					_t_3193 = _t_3198;
				
				}
		
			_t_3199 = -1.0f * _t_3193;
			_t_3200 = _t_3199 * _t_175;
			_t_3201 = tegpixellet_block_23(ty3_9_1,ty1_7_1,_t_175,_t_3200,tx1_4_1,tx3_6_1,y__2721_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1);
			_t_3043 = _t_3201;
		
		}
else
		{
		
			_t_3043 = 0.0f;
		
		}


	return _t_3043;
}
__device__ float tegpixelintegrator_19(float _t_175,float ty3_9_1,float pc1_15_1,float tc2_19_1,float ty2_8_1,float pc0_14_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float py1_13_1,float pc2_16_1,float tx2_5_1,float px1_11_1,float tc0_17_1,float py0_12_1,float _t_3042,float _t_2933,float tc1_18_1,float px0_10_1){
    float y__2721_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_3042 - _t_2933)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__2721_1 = _t_2933 + __step__ * (i + (float)(0.5));
        float _t_3043;
		_t_3043 = tegpixelbody_block_19(ty3_9_1,ty1_7_1,_t_175,px0_10_1,px1_11_1,tx1_4_1,tx3_6_1,py0_12_1,py1_13_1,y__2721_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);;
        __output__ = __output__ + _t_3043 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_3(float ty3_9_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float _t_175,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_2825;
	float _t_2826;
	float _t_2827;
	bool _t_2828;
	float _t_2831;
	float _t_2835;
	float _t_2836;
	float _t_2837;
	float _t_2838;
	float _t_2839;
	bool _t_2840;
	float _t_2843;
	float _t_2847;
	float _t_2848;
	bool _t_2849;
	float _t_2850;
	float _t_2851;
	float _t_2852;
	float _t_2853;
	float _t_2854;
	bool _t_2855;
	float _t_2858;
	float _t_2862;
	float _t_2863;
	float _t_2864;
	float _t_2865;
	bool _t_2866;
	float _t_2869;
	float _t_2873;
	float _t_2874;
	float _t_2875;
	float _t_2876;
	float _t_2877;
	bool _t_2878;
	float _t_2881;
	float _t_2885;
	float _t_2886;
	float _t_2887;
	float _t_2888;
	float _t_2889;
	float _t_2890;
	float _t_2891;
	float _t_2892;
	float _t_2893;
	bool _t_2894;
	float _t_2897;
	float _t_2901;
	float _t_2902;
	float _t_2903;
	float _t_2904;
	bool _t_2905;
	float _t_2908;
	float _t_2912;
	float _t_2913;
	float _t_2914;
	float _t_2915;
	float _t_2916;
	bool _t_2917;
	float _t_2920;
	float _t_2924;
	float _t_2925;
	float _t_2926;
	float _t_2927;
	float _t_2928;
	float _t_2929;
	bool _t_2930;
	float _t_2931;
	float _t_2932;
	float _t_2933;
	float _t_2934;
	float _t_2935;
	float _t_2936;
	bool _t_2937;
	float _t_2940;
	float _t_2944;
	float _t_2945;
	float _t_2946;
	float _t_2947;
	float _t_2948;
	bool _t_2949;
	float _t_2952;
	float _t_2956;
	float _t_2957;
	bool _t_2958;
	float _t_2959;
	float _t_2960;
	float _t_2961;
	float _t_2962;
	float _t_2963;
	bool _t_2964;
	float _t_2967;
	float _t_2971;
	float _t_2972;
	float _t_2973;
	float _t_2974;
	bool _t_2975;
	float _t_2978;
	float _t_2982;
	float _t_2983;
	float _t_2984;
	float _t_2985;
	float _t_2986;
	bool _t_2987;
	float _t_2990;
	float _t_2994;
	float _t_2995;
	float _t_2996;
	float _t_2997;
	float _t_2998;
	float _t_2999;
	float _t_3000;
	float _t_3001;
	float _t_3002;
	bool _t_3003;
	float _t_3006;
	float _t_3010;
	float _t_3011;
	float _t_3012;
	float _t_3013;
	bool _t_3014;
	float _t_3017;
	float _t_3021;
	float _t_3022;
	float _t_3023;
	float _t_3024;
	float _t_3025;
	bool _t_3026;
	float _t_3029;
	float _t_3033;
	float _t_3034;
	float _t_3035;
	float _t_3036;
	float _t_3037;
	float _t_3038;
	bool _t_3039;
	float _t_3040;
	float _t_3041;
	float _t_3042;

	float _t_176;

	_t_2825 = -1.0f * ty1_7_1;
	_t_2826 = ty3_9_1 + _t_2825;
	_t_2827 = -1.0f * _t_2826;
	_t_2828 = _t_2827 < 0.0f;
	if(_t_2828)
		{
			float _t_2829;
			float _t_2830;
		
			_t_2829 = -1.0f * tx3_6_1;
			_t_2830 = tx1_4_1 + _t_2829;
			_t_2831 = _t_2830;
		
		}
else
		{
			float _t_2832;
			float _t_2833;
			float _t_2834;
		
			_t_2832 = -1.0f * tx3_6_1;
			_t_2833 = tx1_4_1 + _t_2832;
			_t_2834 = -1.0f * _t_2833;
			_t_2831 = _t_2834;
		
		}

	_t_2835 = _t_2831 * _t_175;
	_t_2836 = _t_2835 * -1.0f;
	_t_2837 = -1.0f * ty1_7_1;
	_t_2838 = ty3_9_1 + _t_2837;
	_t_2839 = -1.0f * _t_2838;
	_t_2840 = _t_2839 < 0.0f;
	if(_t_2840)
		{
			float _t_2841;
			float _t_2842;
		
			_t_2841 = -1.0f * tx3_6_1;
			_t_2842 = tx1_4_1 + _t_2841;
			_t_2843 = _t_2842;
		
		}
else
		{
			float _t_2844;
			float _t_2845;
			float _t_2846;
		
			_t_2844 = -1.0f * tx3_6_1;
			_t_2845 = tx1_4_1 + _t_2844;
			_t_2846 = -1.0f * _t_2845;
			_t_2843 = _t_2846;
		
		}

	_t_2847 = _t_2843 * _t_175;
	_t_2848 = _t_2847 * -1.0f;
	_t_2849 = 0.0f < _t_2848;
	if(_t_2849)
		{
		
			_t_2850 = px0_10_1;
		
		}
else
		{
		
			_t_2850 = px1_11_1;
		
		}

	_t_2851 = _t_2836 * _t_2850;
	_t_2852 = -1.0f * ty1_7_1;
	_t_2853 = ty3_9_1 + _t_2852;
	_t_2854 = -1.0f * _t_2853;
	_t_2855 = _t_2854 < 0.0f;
	if(_t_2855)
		{
			float _t_2856;
			float _t_2857;
		
			_t_2856 = -1.0f * tx3_6_1;
			_t_2857 = tx1_4_1 + _t_2856;
			_t_2858 = _t_2857;
		
		}
else
		{
			float _t_2859;
			float _t_2860;
			float _t_2861;
		
			_t_2859 = -1.0f * tx3_6_1;
			_t_2860 = tx1_4_1 + _t_2859;
			_t_2861 = -1.0f * _t_2860;
			_t_2858 = _t_2861;
		
		}

	_t_2862 = _t_2858 * _t_175;
	_t_2863 = -1.0f * ty1_7_1;
	_t_2864 = ty3_9_1 + _t_2863;
	_t_2865 = -1.0f * _t_2864;
	_t_2866 = _t_2865 < 0.0f;
	if(_t_2866)
		{
			float _t_2867;
			float _t_2868;
		
			_t_2867 = -1.0f * tx3_6_1;
			_t_2868 = tx1_4_1 + _t_2867;
			_t_2869 = _t_2868;
		
		}
else
		{
			float _t_2870;
			float _t_2871;
			float _t_2872;
		
			_t_2870 = -1.0f * tx3_6_1;
			_t_2871 = tx1_4_1 + _t_2870;
			_t_2872 = -1.0f * _t_2871;
			_t_2869 = _t_2872;
		
		}

	_t_2873 = _t_2869 * _t_175;
	_t_2874 = _t_2862 * _t_2873;
	_t_2875 = -1.0f * ty1_7_1;
	_t_2876 = ty3_9_1 + _t_2875;
	_t_2877 = -1.0f * _t_2876;
	_t_2878 = _t_2877 < 0.0f;
	if(_t_2878)
		{
			float _t_2879;
			float _t_2880;
		
			_t_2879 = -1.0f * ty1_7_1;
			_t_2880 = ty3_9_1 + _t_2879;
			_t_2881 = _t_2880;
		
		}
else
		{
			float _t_2882;
			float _t_2883;
			float _t_2884;
		
			_t_2882 = -1.0f * ty1_7_1;
			_t_2883 = ty3_9_1 + _t_2882;
			_t_2884 = -1.0f * _t_2883;
			_t_2881 = _t_2884;
		
		}

	_t_2885 = _t_2881 * _t_175;
	_t_2886 = 1.0f + _t_2885;
	_t_2887 = 1.0f / _t_2886;
	_t_2888 = _t_2874 * _t_2887;
	_t_2889 = _t_2888 * -1.0f;
	_t_2890 = 1.0f + _t_2889;
	_t_2891 = -1.0f * ty1_7_1;
	_t_2892 = ty3_9_1 + _t_2891;
	_t_2893 = -1.0f * _t_2892;
	_t_2894 = _t_2893 < 0.0f;
	if(_t_2894)
		{
			float _t_2895;
			float _t_2896;
		
			_t_2895 = -1.0f * tx3_6_1;
			_t_2896 = tx1_4_1 + _t_2895;
			_t_2897 = _t_2896;
		
		}
else
		{
			float _t_2898;
			float _t_2899;
			float _t_2900;
		
			_t_2898 = -1.0f * tx3_6_1;
			_t_2899 = tx1_4_1 + _t_2898;
			_t_2900 = -1.0f * _t_2899;
			_t_2897 = _t_2900;
		
		}

	_t_2901 = _t_2897 * _t_175;
	_t_2902 = -1.0f * ty1_7_1;
	_t_2903 = ty3_9_1 + _t_2902;
	_t_2904 = -1.0f * _t_2903;
	_t_2905 = _t_2904 < 0.0f;
	if(_t_2905)
		{
			float _t_2906;
			float _t_2907;
		
			_t_2906 = -1.0f * tx3_6_1;
			_t_2907 = tx1_4_1 + _t_2906;
			_t_2908 = _t_2907;
		
		}
else
		{
			float _t_2909;
			float _t_2910;
			float _t_2911;
		
			_t_2909 = -1.0f * tx3_6_1;
			_t_2910 = tx1_4_1 + _t_2909;
			_t_2911 = -1.0f * _t_2910;
			_t_2908 = _t_2911;
		
		}

	_t_2912 = _t_2908 * _t_175;
	_t_2913 = _t_2901 * _t_2912;
	_t_2914 = -1.0f * ty1_7_1;
	_t_2915 = ty3_9_1 + _t_2914;
	_t_2916 = -1.0f * _t_2915;
	_t_2917 = _t_2916 < 0.0f;
	if(_t_2917)
		{
			float _t_2918;
			float _t_2919;
		
			_t_2918 = -1.0f * ty1_7_1;
			_t_2919 = ty3_9_1 + _t_2918;
			_t_2920 = _t_2919;
		
		}
else
		{
			float _t_2921;
			float _t_2922;
			float _t_2923;
		
			_t_2921 = -1.0f * ty1_7_1;
			_t_2922 = ty3_9_1 + _t_2921;
			_t_2923 = -1.0f * _t_2922;
			_t_2920 = _t_2923;
		
		}

	_t_2924 = _t_2920 * _t_175;
	_t_2925 = 1.0f + _t_2924;
	_t_2926 = 1.0f / _t_2925;
	_t_2927 = _t_2913 * _t_2926;
	_t_2928 = _t_2927 * -1.0f;
	_t_2929 = 1.0f + _t_2928;
	_t_2930 = 0.0f < _t_2929;
	if(_t_2930)
		{
		
			_t_2931 = py0_12_1;
		
		}
else
		{
		
			_t_2931 = py1_13_1;
		
		}

	_t_2932 = _t_2890 * _t_2931;
	_t_2933 = _t_2851 + _t_2932;
	_t_2934 = -1.0f * ty1_7_1;
	_t_2935 = ty3_9_1 + _t_2934;
	_t_2936 = -1.0f * _t_2935;
	_t_2937 = _t_2936 < 0.0f;
	if(_t_2937)
		{
			float _t_2938;
			float _t_2939;
		
			_t_2938 = -1.0f * tx3_6_1;
			_t_2939 = tx1_4_1 + _t_2938;
			_t_2940 = _t_2939;
		
		}
else
		{
			float _t_2941;
			float _t_2942;
			float _t_2943;
		
			_t_2941 = -1.0f * tx3_6_1;
			_t_2942 = tx1_4_1 + _t_2941;
			_t_2943 = -1.0f * _t_2942;
			_t_2940 = _t_2943;
		
		}

	_t_2944 = _t_2940 * _t_175;
	_t_2945 = _t_2944 * -1.0f;
	_t_2946 = -1.0f * ty1_7_1;
	_t_2947 = ty3_9_1 + _t_2946;
	_t_2948 = -1.0f * _t_2947;
	_t_2949 = _t_2948 < 0.0f;
	if(_t_2949)
		{
			float _t_2950;
			float _t_2951;
		
			_t_2950 = -1.0f * tx3_6_1;
			_t_2951 = tx1_4_1 + _t_2950;
			_t_2952 = _t_2951;
		
		}
else
		{
			float _t_2953;
			float _t_2954;
			float _t_2955;
		
			_t_2953 = -1.0f * tx3_6_1;
			_t_2954 = tx1_4_1 + _t_2953;
			_t_2955 = -1.0f * _t_2954;
			_t_2952 = _t_2955;
		
		}

	_t_2956 = _t_2952 * _t_175;
	_t_2957 = _t_2956 * -1.0f;
	_t_2958 = 0.0f < _t_2957;
	if(_t_2958)
		{
		
			_t_2959 = px1_11_1;
		
		}
else
		{
		
			_t_2959 = px0_10_1;
		
		}

	_t_2960 = _t_2945 * _t_2959;
	_t_2961 = -1.0f * ty1_7_1;
	_t_2962 = ty3_9_1 + _t_2961;
	_t_2963 = -1.0f * _t_2962;
	_t_2964 = _t_2963 < 0.0f;
	if(_t_2964)
		{
			float _t_2965;
			float _t_2966;
		
			_t_2965 = -1.0f * tx3_6_1;
			_t_2966 = tx1_4_1 + _t_2965;
			_t_2967 = _t_2966;
		
		}
else
		{
			float _t_2968;
			float _t_2969;
			float _t_2970;
		
			_t_2968 = -1.0f * tx3_6_1;
			_t_2969 = tx1_4_1 + _t_2968;
			_t_2970 = -1.0f * _t_2969;
			_t_2967 = _t_2970;
		
		}

	_t_2971 = _t_2967 * _t_175;
	_t_2972 = -1.0f * ty1_7_1;
	_t_2973 = ty3_9_1 + _t_2972;
	_t_2974 = -1.0f * _t_2973;
	_t_2975 = _t_2974 < 0.0f;
	if(_t_2975)
		{
			float _t_2976;
			float _t_2977;
		
			_t_2976 = -1.0f * tx3_6_1;
			_t_2977 = tx1_4_1 + _t_2976;
			_t_2978 = _t_2977;
		
		}
else
		{
			float _t_2979;
			float _t_2980;
			float _t_2981;
		
			_t_2979 = -1.0f * tx3_6_1;
			_t_2980 = tx1_4_1 + _t_2979;
			_t_2981 = -1.0f * _t_2980;
			_t_2978 = _t_2981;
		
		}

	_t_2982 = _t_2978 * _t_175;
	_t_2983 = _t_2971 * _t_2982;
	_t_2984 = -1.0f * ty1_7_1;
	_t_2985 = ty3_9_1 + _t_2984;
	_t_2986 = -1.0f * _t_2985;
	_t_2987 = _t_2986 < 0.0f;
	if(_t_2987)
		{
			float _t_2988;
			float _t_2989;
		
			_t_2988 = -1.0f * ty1_7_1;
			_t_2989 = ty3_9_1 + _t_2988;
			_t_2990 = _t_2989;
		
		}
else
		{
			float _t_2991;
			float _t_2992;
			float _t_2993;
		
			_t_2991 = -1.0f * ty1_7_1;
			_t_2992 = ty3_9_1 + _t_2991;
			_t_2993 = -1.0f * _t_2992;
			_t_2990 = _t_2993;
		
		}

	_t_2994 = _t_2990 * _t_175;
	_t_2995 = 1.0f + _t_2994;
	_t_2996 = 1.0f / _t_2995;
	_t_2997 = _t_2983 * _t_2996;
	_t_2998 = _t_2997 * -1.0f;
	_t_2999 = 1.0f + _t_2998;
	_t_3000 = -1.0f * ty1_7_1;
	_t_3001 = ty3_9_1 + _t_3000;
	_t_3002 = -1.0f * _t_3001;
	_t_3003 = _t_3002 < 0.0f;
	if(_t_3003)
		{
			float _t_3004;
			float _t_3005;
		
			_t_3004 = -1.0f * tx3_6_1;
			_t_3005 = tx1_4_1 + _t_3004;
			_t_3006 = _t_3005;
		
		}
else
		{
			float _t_3007;
			float _t_3008;
			float _t_3009;
		
			_t_3007 = -1.0f * tx3_6_1;
			_t_3008 = tx1_4_1 + _t_3007;
			_t_3009 = -1.0f * _t_3008;
			_t_3006 = _t_3009;
		
		}

	_t_3010 = _t_3006 * _t_175;
	_t_3011 = -1.0f * ty1_7_1;
	_t_3012 = ty3_9_1 + _t_3011;
	_t_3013 = -1.0f * _t_3012;
	_t_3014 = _t_3013 < 0.0f;
	if(_t_3014)
		{
			float _t_3015;
			float _t_3016;
		
			_t_3015 = -1.0f * tx3_6_1;
			_t_3016 = tx1_4_1 + _t_3015;
			_t_3017 = _t_3016;
		
		}
else
		{
			float _t_3018;
			float _t_3019;
			float _t_3020;
		
			_t_3018 = -1.0f * tx3_6_1;
			_t_3019 = tx1_4_1 + _t_3018;
			_t_3020 = -1.0f * _t_3019;
			_t_3017 = _t_3020;
		
		}

	_t_3021 = _t_3017 * _t_175;
	_t_3022 = _t_3010 * _t_3021;
	_t_3023 = -1.0f * ty1_7_1;
	_t_3024 = ty3_9_1 + _t_3023;
	_t_3025 = -1.0f * _t_3024;
	_t_3026 = _t_3025 < 0.0f;
	if(_t_3026)
		{
			float _t_3027;
			float _t_3028;
		
			_t_3027 = -1.0f * ty1_7_1;
			_t_3028 = ty3_9_1 + _t_3027;
			_t_3029 = _t_3028;
		
		}
else
		{
			float _t_3030;
			float _t_3031;
			float _t_3032;
		
			_t_3030 = -1.0f * ty1_7_1;
			_t_3031 = ty3_9_1 + _t_3030;
			_t_3032 = -1.0f * _t_3031;
			_t_3029 = _t_3032;
		
		}

	_t_3033 = _t_3029 * _t_175;
	_t_3034 = 1.0f + _t_3033;
	_t_3035 = 1.0f / _t_3034;
	_t_3036 = _t_3022 * _t_3035;
	_t_3037 = _t_3036 * -1.0f;
	_t_3038 = 1.0f + _t_3037;
	_t_3039 = 0.0f < _t_3038;
	if(_t_3039)
		{
		
			_t_3040 = py1_13_1;
		
		}
else
		{
		
			_t_3040 = py0_12_1;
		
		}

	_t_3041 = _t_2999 * _t_3040;
	_t_3042 = _t_2960 + _t_3041;
	_t_176 = tegpixelintegrator_19(_t_175,ty3_9_1,pc1_15_1,tc2_19_1,ty2_8_1,pc0_14_1,ty1_7_1,tx1_4_1,tx3_6_1,py1_13_1,pc2_16_1,tx2_5_1,px1_11_1,tc0_17_1,py0_12_1,_t_3042,_t_2933,tc1_18_1,px0_10_1);

	return _t_176;
}
__device__ float tegpixellet_block_26(float py0_12_1,float _t_4116,float py1_13_1,float px0_10_1,float _t_4063,float px1_11_1,float ty1_7_1,float ty2_8_1,float tx2_5_1,float tx1_4_1,float _t_203,float y__2795_1,float _t_4036,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	bool _t_4117;
	bool _t_4118;
	bool _t_4119;
	bool _t_4120;
	bool _t_4121;
	bool _t_4122;
	bool _t_4123;
	float _t_4453;
	float _t_4454;
	float _t_4455;
	float _t_4456;
	float _t_4457;
	float _t_4458;
	float _t_4459;
	float _t_4460;
	float _t_4461;
	float _t_4462;
	float _t_4463;
	float _t_4464;
	float _t_4465;
	float _t_4466;
	float _t_4467;
	float _t_4468;
	float _t_4469;
	float _t_4470;
	float _t_4471;
	float _t_4472;
	float _t_4473;
	float _t_4474;
	float _t_4475;
	float _t_4476;
	bool _t_4477;
	float _t_4478;
	float _t_4479;
	float _t_4480;
	float _t_4481;
	float _t_4482;
	float _t_4483;
	float _t_4484;
	float _t_4485;
	float _t_4486;
	float _t_4487;
	float _t_4488;
	float _t_4489;
	float _t_4490;
	float _t_4491;
	bool _t_4492;
	float _t_4493;
	float _t_4494;
	float _t_4495;
	float _t_4496;
	float _t_4497;
	float _t_4498;
	float _t_4499;
	float _t_4500;
	float _t_4501;
	float _t_4502;
	float _t_4503;
	float _t_4504;
	float _t_4505;
	float _t_4506;
	float _t_4507;
	float _t_4508;
	float _t_4509;
	float _t_4510;
	float _t_4511;
	float _t_4512;
	float _t_4513;
	float _t_4514;
	float _t_4515;
	float _t_4516;
	float _t_4517;
	float _t_4518;
	bool _t_4519;
	float _t_4520;
	float _t_4521;
	float _t_4522;
	float _t_4523;
	float _t_4524;
	float _t_4525;
	float _t_4526;
	float _t_4527;
	float _t_4528;
	float _t_4529;
	float _t_4530;
	float _t_4531;
	float _t_4532;
	float _t_4533;
	bool _t_4534;
	float _t_4535;
	float _t_4536;
	float _t_4537;
	float _t_4538;
	float _t_4539;

	float _t_4037;

	_t_4117 = py0_12_1 < _t_4116;
	_t_4118 = _t_4116 < py1_13_1;
	_t_4119 = _t_4117 && _t_4118;
	_t_4120 = px0_10_1 < _t_4063;
	_t_4121 = _t_4063 < px1_11_1;
	_t_4122 = _t_4120 && _t_4121;
	_t_4123 = _t_4119 && _t_4122;
	if(_t_4123)
		{
			float _t_4124;
			float _t_4125;
			float _t_4126;
			bool _t_4127;
			float _t_4130;
			float _t_4134;
			float _t_4135;
			float _t_4136;
			float _t_4137;
			float _t_4138;
			bool _t_4139;
			float _t_4142;
			float _t_4146;
			float _t_4147;
			bool _t_4148;
			float _t_4149;
			float _t_4150;
			float _t_4151;
			float _t_4152;
			float _t_4153;
			bool _t_4154;
			float _t_4157;
			float _t_4161;
			float _t_4162;
			float _t_4163;
			float _t_4164;
			bool _t_4165;
			float _t_4168;
			float _t_4172;
			float _t_4173;
			float _t_4174;
			float _t_4175;
			float _t_4176;
			bool _t_4177;
			float _t_4180;
			float _t_4184;
			float _t_4185;
			float _t_4186;
			float _t_4187;
			float _t_4188;
			float _t_4189;
			float _t_4190;
			float _t_4191;
			float _t_4192;
			bool _t_4193;
			float _t_4196;
			float _t_4200;
			float _t_4201;
			float _t_4202;
			float _t_4203;
			bool _t_4204;
			float _t_4207;
			float _t_4211;
			float _t_4212;
			float _t_4213;
			float _t_4214;
			float _t_4215;
			bool _t_4216;
			float _t_4219;
			float _t_4223;
			float _t_4224;
			float _t_4225;
			float _t_4226;
			float _t_4227;
			float _t_4228;
			bool _t_4229;
			float _t_4230;
			float _t_4231;
			float _t_4232;
			bool _t_4233;
			float _t_4234;
			float _t_4235;
			float _t_4236;
			bool _t_4237;
			float _t_4240;
			float _t_4244;
			float _t_4245;
			float _t_4246;
			float _t_4247;
			float _t_4248;
			bool _t_4249;
			float _t_4252;
			float _t_4256;
			float _t_4257;
			bool _t_4258;
			float _t_4259;
			float _t_4260;
			float _t_4261;
			float _t_4262;
			float _t_4263;
			bool _t_4264;
			float _t_4267;
			float _t_4271;
			float _t_4272;
			float _t_4273;
			float _t_4274;
			bool _t_4275;
			float _t_4278;
			float _t_4282;
			float _t_4283;
			float _t_4284;
			float _t_4285;
			float _t_4286;
			bool _t_4287;
			float _t_4290;
			float _t_4294;
			float _t_4295;
			float _t_4296;
			float _t_4297;
			float _t_4298;
			float _t_4299;
			float _t_4300;
			float _t_4301;
			float _t_4302;
			bool _t_4303;
			float _t_4306;
			float _t_4310;
			float _t_4311;
			float _t_4312;
			float _t_4313;
			bool _t_4314;
			float _t_4317;
			float _t_4321;
			float _t_4322;
			float _t_4323;
			float _t_4324;
			float _t_4325;
			bool _t_4326;
			float _t_4329;
			float _t_4333;
			float _t_4334;
			float _t_4335;
			float _t_4336;
			float _t_4337;
			float _t_4338;
			bool _t_4339;
			float _t_4340;
			float _t_4341;
			float _t_4342;
			bool _t_4343;
			bool _t_4344;
			float _t_4345;
			float _t_4346;
			float _t_4347;
			bool _t_4348;
			float _t_4351;
			float _t_4355;
			float _t_4356;
			float _t_4357;
			float _t_4358;
			bool _t_4359;
			float _t_4362;
			float _t_4366;
			bool _t_4367;
			float _t_4368;
			float _t_4369;
			float _t_4370;
			float _t_4371;
			float _t_4372;
			bool _t_4373;
			float _t_4376;
			float _t_4380;
			float _t_4381;
			float _t_4382;
			float _t_4383;
			bool _t_4384;
			float _t_4387;
			float _t_4391;
			bool _t_4392;
			float _t_4393;
			float _t_4394;
			float _t_4395;
			bool _t_4396;
			float _t_4397;
			float _t_4398;
			float _t_4399;
			bool _t_4400;
			float _t_4403;
			float _t_4407;
			float _t_4408;
			float _t_4409;
			float _t_4410;
			bool _t_4411;
			float _t_4414;
			float _t_4418;
			bool _t_4419;
			float _t_4420;
			float _t_4421;
			float _t_4422;
			float _t_4423;
			float _t_4424;
			bool _t_4425;
			float _t_4428;
			float _t_4432;
			float _t_4433;
			float _t_4434;
			float _t_4435;
			bool _t_4436;
			float _t_4439;
			float _t_4443;
			bool _t_4444;
			float _t_4445;
			float _t_4446;
			float _t_4447;
			bool _t_4448;
			bool _t_4449;
			bool _t_4450;
			float _t_4451;
			float _t_4452;
		
			_t_4124 = -1.0f * ty2_8_1;
			_t_4125 = ty1_7_1 + _t_4124;
			_t_4126 = -1.0f * _t_4125;
			_t_4127 = _t_4126 < 0.0f;
			if(_t_4127)
				{
					float _t_4128;
					float _t_4129;
				
					_t_4128 = -1.0f * tx1_4_1;
					_t_4129 = tx2_5_1 + _t_4128;
					_t_4130 = _t_4129;
				
				}
		else
				{
					float _t_4131;
					float _t_4132;
					float _t_4133;
				
					_t_4131 = -1.0f * tx1_4_1;
					_t_4132 = tx2_5_1 + _t_4131;
					_t_4133 = -1.0f * _t_4132;
					_t_4130 = _t_4133;
				
				}
		
			_t_4134 = _t_4130 * _t_203;
			_t_4135 = _t_4134 * -1.0f;
			_t_4136 = -1.0f * ty2_8_1;
			_t_4137 = ty1_7_1 + _t_4136;
			_t_4138 = -1.0f * _t_4137;
			_t_4139 = _t_4138 < 0.0f;
			if(_t_4139)
				{
					float _t_4140;
					float _t_4141;
				
					_t_4140 = -1.0f * tx1_4_1;
					_t_4141 = tx2_5_1 + _t_4140;
					_t_4142 = _t_4141;
				
				}
		else
				{
					float _t_4143;
					float _t_4144;
					float _t_4145;
				
					_t_4143 = -1.0f * tx1_4_1;
					_t_4144 = tx2_5_1 + _t_4143;
					_t_4145 = -1.0f * _t_4144;
					_t_4142 = _t_4145;
				
				}
		
			_t_4146 = _t_4142 * _t_203;
			_t_4147 = _t_4146 * -1.0f;
			_t_4148 = 0.0f < _t_4147;
			if(_t_4148)
				{
				
					_t_4149 = px0_10_1;
				
				}
		else
				{
				
					_t_4149 = px1_11_1;
				
				}
		
			_t_4150 = _t_4135 * _t_4149;
			_t_4151 = -1.0f * ty2_8_1;
			_t_4152 = ty1_7_1 + _t_4151;
			_t_4153 = -1.0f * _t_4152;
			_t_4154 = _t_4153 < 0.0f;
			if(_t_4154)
				{
					float _t_4155;
					float _t_4156;
				
					_t_4155 = -1.0f * tx1_4_1;
					_t_4156 = tx2_5_1 + _t_4155;
					_t_4157 = _t_4156;
				
				}
		else
				{
					float _t_4158;
					float _t_4159;
					float _t_4160;
				
					_t_4158 = -1.0f * tx1_4_1;
					_t_4159 = tx2_5_1 + _t_4158;
					_t_4160 = -1.0f * _t_4159;
					_t_4157 = _t_4160;
				
				}
		
			_t_4161 = _t_4157 * _t_203;
			_t_4162 = -1.0f * ty2_8_1;
			_t_4163 = ty1_7_1 + _t_4162;
			_t_4164 = -1.0f * _t_4163;
			_t_4165 = _t_4164 < 0.0f;
			if(_t_4165)
				{
					float _t_4166;
					float _t_4167;
				
					_t_4166 = -1.0f * tx1_4_1;
					_t_4167 = tx2_5_1 + _t_4166;
					_t_4168 = _t_4167;
				
				}
		else
				{
					float _t_4169;
					float _t_4170;
					float _t_4171;
				
					_t_4169 = -1.0f * tx1_4_1;
					_t_4170 = tx2_5_1 + _t_4169;
					_t_4171 = -1.0f * _t_4170;
					_t_4168 = _t_4171;
				
				}
		
			_t_4172 = _t_4168 * _t_203;
			_t_4173 = _t_4161 * _t_4172;
			_t_4174 = -1.0f * ty2_8_1;
			_t_4175 = ty1_7_1 + _t_4174;
			_t_4176 = -1.0f * _t_4175;
			_t_4177 = _t_4176 < 0.0f;
			if(_t_4177)
				{
					float _t_4178;
					float _t_4179;
				
					_t_4178 = -1.0f * ty2_8_1;
					_t_4179 = ty1_7_1 + _t_4178;
					_t_4180 = _t_4179;
				
				}
		else
				{
					float _t_4181;
					float _t_4182;
					float _t_4183;
				
					_t_4181 = -1.0f * ty2_8_1;
					_t_4182 = ty1_7_1 + _t_4181;
					_t_4183 = -1.0f * _t_4182;
					_t_4180 = _t_4183;
				
				}
		
			_t_4184 = _t_4180 * _t_203;
			_t_4185 = 1.0f + _t_4184;
			_t_4186 = 1.0f / _t_4185;
			_t_4187 = _t_4173 * _t_4186;
			_t_4188 = _t_4187 * -1.0f;
			_t_4189 = 1.0f + _t_4188;
			_t_4190 = -1.0f * ty2_8_1;
			_t_4191 = ty1_7_1 + _t_4190;
			_t_4192 = -1.0f * _t_4191;
			_t_4193 = _t_4192 < 0.0f;
			if(_t_4193)
				{
					float _t_4194;
					float _t_4195;
				
					_t_4194 = -1.0f * tx1_4_1;
					_t_4195 = tx2_5_1 + _t_4194;
					_t_4196 = _t_4195;
				
				}
		else
				{
					float _t_4197;
					float _t_4198;
					float _t_4199;
				
					_t_4197 = -1.0f * tx1_4_1;
					_t_4198 = tx2_5_1 + _t_4197;
					_t_4199 = -1.0f * _t_4198;
					_t_4196 = _t_4199;
				
				}
		
			_t_4200 = _t_4196 * _t_203;
			_t_4201 = -1.0f * ty2_8_1;
			_t_4202 = ty1_7_1 + _t_4201;
			_t_4203 = -1.0f * _t_4202;
			_t_4204 = _t_4203 < 0.0f;
			if(_t_4204)
				{
					float _t_4205;
					float _t_4206;
				
					_t_4205 = -1.0f * tx1_4_1;
					_t_4206 = tx2_5_1 + _t_4205;
					_t_4207 = _t_4206;
				
				}
		else
				{
					float _t_4208;
					float _t_4209;
					float _t_4210;
				
					_t_4208 = -1.0f * tx1_4_1;
					_t_4209 = tx2_5_1 + _t_4208;
					_t_4210 = -1.0f * _t_4209;
					_t_4207 = _t_4210;
				
				}
		
			_t_4211 = _t_4207 * _t_203;
			_t_4212 = _t_4200 * _t_4211;
			_t_4213 = -1.0f * ty2_8_1;
			_t_4214 = ty1_7_1 + _t_4213;
			_t_4215 = -1.0f * _t_4214;
			_t_4216 = _t_4215 < 0.0f;
			if(_t_4216)
				{
					float _t_4217;
					float _t_4218;
				
					_t_4217 = -1.0f * ty2_8_1;
					_t_4218 = ty1_7_1 + _t_4217;
					_t_4219 = _t_4218;
				
				}
		else
				{
					float _t_4220;
					float _t_4221;
					float _t_4222;
				
					_t_4220 = -1.0f * ty2_8_1;
					_t_4221 = ty1_7_1 + _t_4220;
					_t_4222 = -1.0f * _t_4221;
					_t_4219 = _t_4222;
				
				}
		
			_t_4223 = _t_4219 * _t_203;
			_t_4224 = 1.0f + _t_4223;
			_t_4225 = 1.0f / _t_4224;
			_t_4226 = _t_4212 * _t_4225;
			_t_4227 = _t_4226 * -1.0f;
			_t_4228 = 1.0f + _t_4227;
			_t_4229 = 0.0f < _t_4228;
			if(_t_4229)
				{
				
					_t_4230 = py0_12_1;
				
				}
		else
				{
				
					_t_4230 = py1_13_1;
				
				}
		
			_t_4231 = _t_4189 * _t_4230;
			_t_4232 = _t_4150 + _t_4231;
			_t_4233 = _t_4232 < y__2795_1;
			_t_4234 = -1.0f * ty2_8_1;
			_t_4235 = ty1_7_1 + _t_4234;
			_t_4236 = -1.0f * _t_4235;
			_t_4237 = _t_4236 < 0.0f;
			if(_t_4237)
				{
					float _t_4238;
					float _t_4239;
				
					_t_4238 = -1.0f * tx1_4_1;
					_t_4239 = tx2_5_1 + _t_4238;
					_t_4240 = _t_4239;
				
				}
		else
				{
					float _t_4241;
					float _t_4242;
					float _t_4243;
				
					_t_4241 = -1.0f * tx1_4_1;
					_t_4242 = tx2_5_1 + _t_4241;
					_t_4243 = -1.0f * _t_4242;
					_t_4240 = _t_4243;
				
				}
		
			_t_4244 = _t_4240 * _t_203;
			_t_4245 = _t_4244 * -1.0f;
			_t_4246 = -1.0f * ty2_8_1;
			_t_4247 = ty1_7_1 + _t_4246;
			_t_4248 = -1.0f * _t_4247;
			_t_4249 = _t_4248 < 0.0f;
			if(_t_4249)
				{
					float _t_4250;
					float _t_4251;
				
					_t_4250 = -1.0f * tx1_4_1;
					_t_4251 = tx2_5_1 + _t_4250;
					_t_4252 = _t_4251;
				
				}
		else
				{
					float _t_4253;
					float _t_4254;
					float _t_4255;
				
					_t_4253 = -1.0f * tx1_4_1;
					_t_4254 = tx2_5_1 + _t_4253;
					_t_4255 = -1.0f * _t_4254;
					_t_4252 = _t_4255;
				
				}
		
			_t_4256 = _t_4252 * _t_203;
			_t_4257 = _t_4256 * -1.0f;
			_t_4258 = 0.0f < _t_4257;
			if(_t_4258)
				{
				
					_t_4259 = px1_11_1;
				
				}
		else
				{
				
					_t_4259 = px0_10_1;
				
				}
		
			_t_4260 = _t_4245 * _t_4259;
			_t_4261 = -1.0f * ty2_8_1;
			_t_4262 = ty1_7_1 + _t_4261;
			_t_4263 = -1.0f * _t_4262;
			_t_4264 = _t_4263 < 0.0f;
			if(_t_4264)
				{
					float _t_4265;
					float _t_4266;
				
					_t_4265 = -1.0f * tx1_4_1;
					_t_4266 = tx2_5_1 + _t_4265;
					_t_4267 = _t_4266;
				
				}
		else
				{
					float _t_4268;
					float _t_4269;
					float _t_4270;
				
					_t_4268 = -1.0f * tx1_4_1;
					_t_4269 = tx2_5_1 + _t_4268;
					_t_4270 = -1.0f * _t_4269;
					_t_4267 = _t_4270;
				
				}
		
			_t_4271 = _t_4267 * _t_203;
			_t_4272 = -1.0f * ty2_8_1;
			_t_4273 = ty1_7_1 + _t_4272;
			_t_4274 = -1.0f * _t_4273;
			_t_4275 = _t_4274 < 0.0f;
			if(_t_4275)
				{
					float _t_4276;
					float _t_4277;
				
					_t_4276 = -1.0f * tx1_4_1;
					_t_4277 = tx2_5_1 + _t_4276;
					_t_4278 = _t_4277;
				
				}
		else
				{
					float _t_4279;
					float _t_4280;
					float _t_4281;
				
					_t_4279 = -1.0f * tx1_4_1;
					_t_4280 = tx2_5_1 + _t_4279;
					_t_4281 = -1.0f * _t_4280;
					_t_4278 = _t_4281;
				
				}
		
			_t_4282 = _t_4278 * _t_203;
			_t_4283 = _t_4271 * _t_4282;
			_t_4284 = -1.0f * ty2_8_1;
			_t_4285 = ty1_7_1 + _t_4284;
			_t_4286 = -1.0f * _t_4285;
			_t_4287 = _t_4286 < 0.0f;
			if(_t_4287)
				{
					float _t_4288;
					float _t_4289;
				
					_t_4288 = -1.0f * ty2_8_1;
					_t_4289 = ty1_7_1 + _t_4288;
					_t_4290 = _t_4289;
				
				}
		else
				{
					float _t_4291;
					float _t_4292;
					float _t_4293;
				
					_t_4291 = -1.0f * ty2_8_1;
					_t_4292 = ty1_7_1 + _t_4291;
					_t_4293 = -1.0f * _t_4292;
					_t_4290 = _t_4293;
				
				}
		
			_t_4294 = _t_4290 * _t_203;
			_t_4295 = 1.0f + _t_4294;
			_t_4296 = 1.0f / _t_4295;
			_t_4297 = _t_4283 * _t_4296;
			_t_4298 = _t_4297 * -1.0f;
			_t_4299 = 1.0f + _t_4298;
			_t_4300 = -1.0f * ty2_8_1;
			_t_4301 = ty1_7_1 + _t_4300;
			_t_4302 = -1.0f * _t_4301;
			_t_4303 = _t_4302 < 0.0f;
			if(_t_4303)
				{
					float _t_4304;
					float _t_4305;
				
					_t_4304 = -1.0f * tx1_4_1;
					_t_4305 = tx2_5_1 + _t_4304;
					_t_4306 = _t_4305;
				
				}
		else
				{
					float _t_4307;
					float _t_4308;
					float _t_4309;
				
					_t_4307 = -1.0f * tx1_4_1;
					_t_4308 = tx2_5_1 + _t_4307;
					_t_4309 = -1.0f * _t_4308;
					_t_4306 = _t_4309;
				
				}
		
			_t_4310 = _t_4306 * _t_203;
			_t_4311 = -1.0f * ty2_8_1;
			_t_4312 = ty1_7_1 + _t_4311;
			_t_4313 = -1.0f * _t_4312;
			_t_4314 = _t_4313 < 0.0f;
			if(_t_4314)
				{
					float _t_4315;
					float _t_4316;
				
					_t_4315 = -1.0f * tx1_4_1;
					_t_4316 = tx2_5_1 + _t_4315;
					_t_4317 = _t_4316;
				
				}
		else
				{
					float _t_4318;
					float _t_4319;
					float _t_4320;
				
					_t_4318 = -1.0f * tx1_4_1;
					_t_4319 = tx2_5_1 + _t_4318;
					_t_4320 = -1.0f * _t_4319;
					_t_4317 = _t_4320;
				
				}
		
			_t_4321 = _t_4317 * _t_203;
			_t_4322 = _t_4310 * _t_4321;
			_t_4323 = -1.0f * ty2_8_1;
			_t_4324 = ty1_7_1 + _t_4323;
			_t_4325 = -1.0f * _t_4324;
			_t_4326 = _t_4325 < 0.0f;
			if(_t_4326)
				{
					float _t_4327;
					float _t_4328;
				
					_t_4327 = -1.0f * ty2_8_1;
					_t_4328 = ty1_7_1 + _t_4327;
					_t_4329 = _t_4328;
				
				}
		else
				{
					float _t_4330;
					float _t_4331;
					float _t_4332;
				
					_t_4330 = -1.0f * ty2_8_1;
					_t_4331 = ty1_7_1 + _t_4330;
					_t_4332 = -1.0f * _t_4331;
					_t_4329 = _t_4332;
				
				}
		
			_t_4333 = _t_4329 * _t_203;
			_t_4334 = 1.0f + _t_4333;
			_t_4335 = 1.0f / _t_4334;
			_t_4336 = _t_4322 * _t_4335;
			_t_4337 = _t_4336 * -1.0f;
			_t_4338 = 1.0f + _t_4337;
			_t_4339 = 0.0f < _t_4338;
			if(_t_4339)
				{
				
					_t_4340 = py1_13_1;
				
				}
		else
				{
				
					_t_4340 = py0_12_1;
				
				}
		
			_t_4341 = _t_4299 * _t_4340;
			_t_4342 = _t_4260 + _t_4341;
			_t_4343 = y__2795_1 < _t_4342;
			_t_4344 = _t_4233 && _t_4343;
			_t_4345 = -1.0f * ty2_8_1;
			_t_4346 = ty1_7_1 + _t_4345;
			_t_4347 = -1.0f * _t_4346;
			_t_4348 = _t_4347 < 0.0f;
			if(_t_4348)
				{
					float _t_4349;
					float _t_4350;
				
					_t_4349 = -1.0f * ty2_8_1;
					_t_4350 = ty1_7_1 + _t_4349;
					_t_4351 = _t_4350;
				
				}
		else
				{
					float _t_4352;
					float _t_4353;
					float _t_4354;
				
					_t_4352 = -1.0f * ty2_8_1;
					_t_4353 = ty1_7_1 + _t_4352;
					_t_4354 = -1.0f * _t_4353;
					_t_4351 = _t_4354;
				
				}
		
			_t_4355 = _t_4351 * _t_203;
			_t_4356 = -1.0f * ty2_8_1;
			_t_4357 = ty1_7_1 + _t_4356;
			_t_4358 = -1.0f * _t_4357;
			_t_4359 = _t_4358 < 0.0f;
			if(_t_4359)
				{
					float _t_4360;
					float _t_4361;
				
					_t_4360 = -1.0f * ty2_8_1;
					_t_4361 = ty1_7_1 + _t_4360;
					_t_4362 = _t_4361;
				
				}
		else
				{
					float _t_4363;
					float _t_4364;
					float _t_4365;
				
					_t_4363 = -1.0f * ty2_8_1;
					_t_4364 = ty1_7_1 + _t_4363;
					_t_4365 = -1.0f * _t_4364;
					_t_4362 = _t_4365;
				
				}
		
			_t_4366 = _t_4362 * _t_203;
			_t_4367 = 0.0f < _t_4366;
			if(_t_4367)
				{
				
					_t_4368 = px0_10_1;
				
				}
		else
				{
				
					_t_4368 = px1_11_1;
				
				}
		
			_t_4369 = _t_4355 * _t_4368;
			_t_4370 = -1.0f * ty2_8_1;
			_t_4371 = ty1_7_1 + _t_4370;
			_t_4372 = -1.0f * _t_4371;
			_t_4373 = _t_4372 < 0.0f;
			if(_t_4373)
				{
					float _t_4374;
					float _t_4375;
				
					_t_4374 = -1.0f * tx1_4_1;
					_t_4375 = tx2_5_1 + _t_4374;
					_t_4376 = _t_4375;
				
				}
		else
				{
					float _t_4377;
					float _t_4378;
					float _t_4379;
				
					_t_4377 = -1.0f * tx1_4_1;
					_t_4378 = tx2_5_1 + _t_4377;
					_t_4379 = -1.0f * _t_4378;
					_t_4376 = _t_4379;
				
				}
		
			_t_4380 = _t_4376 * _t_203;
			_t_4381 = -1.0f * ty2_8_1;
			_t_4382 = ty1_7_1 + _t_4381;
			_t_4383 = -1.0f * _t_4382;
			_t_4384 = _t_4383 < 0.0f;
			if(_t_4384)
				{
					float _t_4385;
					float _t_4386;
				
					_t_4385 = -1.0f * tx1_4_1;
					_t_4386 = tx2_5_1 + _t_4385;
					_t_4387 = _t_4386;
				
				}
		else
				{
					float _t_4388;
					float _t_4389;
					float _t_4390;
				
					_t_4388 = -1.0f * tx1_4_1;
					_t_4389 = tx2_5_1 + _t_4388;
					_t_4390 = -1.0f * _t_4389;
					_t_4387 = _t_4390;
				
				}
		
			_t_4391 = _t_4387 * _t_203;
			_t_4392 = 0.0f < _t_4391;
			if(_t_4392)
				{
				
					_t_4393 = py0_12_1;
				
				}
		else
				{
				
					_t_4393 = py1_13_1;
				
				}
		
			_t_4394 = _t_4380 * _t_4393;
			_t_4395 = _t_4369 + _t_4394;
			_t_4396 = _t_4395 < _t_4036;
			_t_4397 = -1.0f * ty2_8_1;
			_t_4398 = ty1_7_1 + _t_4397;
			_t_4399 = -1.0f * _t_4398;
			_t_4400 = _t_4399 < 0.0f;
			if(_t_4400)
				{
					float _t_4401;
					float _t_4402;
				
					_t_4401 = -1.0f * ty2_8_1;
					_t_4402 = ty1_7_1 + _t_4401;
					_t_4403 = _t_4402;
				
				}
		else
				{
					float _t_4404;
					float _t_4405;
					float _t_4406;
				
					_t_4404 = -1.0f * ty2_8_1;
					_t_4405 = ty1_7_1 + _t_4404;
					_t_4406 = -1.0f * _t_4405;
					_t_4403 = _t_4406;
				
				}
		
			_t_4407 = _t_4403 * _t_203;
			_t_4408 = -1.0f * ty2_8_1;
			_t_4409 = ty1_7_1 + _t_4408;
			_t_4410 = -1.0f * _t_4409;
			_t_4411 = _t_4410 < 0.0f;
			if(_t_4411)
				{
					float _t_4412;
					float _t_4413;
				
					_t_4412 = -1.0f * ty2_8_1;
					_t_4413 = ty1_7_1 + _t_4412;
					_t_4414 = _t_4413;
				
				}
		else
				{
					float _t_4415;
					float _t_4416;
					float _t_4417;
				
					_t_4415 = -1.0f * ty2_8_1;
					_t_4416 = ty1_7_1 + _t_4415;
					_t_4417 = -1.0f * _t_4416;
					_t_4414 = _t_4417;
				
				}
		
			_t_4418 = _t_4414 * _t_203;
			_t_4419 = 0.0f < _t_4418;
			if(_t_4419)
				{
				
					_t_4420 = px1_11_1;
				
				}
		else
				{
				
					_t_4420 = px0_10_1;
				
				}
		
			_t_4421 = _t_4407 * _t_4420;
			_t_4422 = -1.0f * ty2_8_1;
			_t_4423 = ty1_7_1 + _t_4422;
			_t_4424 = -1.0f * _t_4423;
			_t_4425 = _t_4424 < 0.0f;
			if(_t_4425)
				{
					float _t_4426;
					float _t_4427;
				
					_t_4426 = -1.0f * tx1_4_1;
					_t_4427 = tx2_5_1 + _t_4426;
					_t_4428 = _t_4427;
				
				}
		else
				{
					float _t_4429;
					float _t_4430;
					float _t_4431;
				
					_t_4429 = -1.0f * tx1_4_1;
					_t_4430 = tx2_5_1 + _t_4429;
					_t_4431 = -1.0f * _t_4430;
					_t_4428 = _t_4431;
				
				}
		
			_t_4432 = _t_4428 * _t_203;
			_t_4433 = -1.0f * ty2_8_1;
			_t_4434 = ty1_7_1 + _t_4433;
			_t_4435 = -1.0f * _t_4434;
			_t_4436 = _t_4435 < 0.0f;
			if(_t_4436)
				{
					float _t_4437;
					float _t_4438;
				
					_t_4437 = -1.0f * tx1_4_1;
					_t_4438 = tx2_5_1 + _t_4437;
					_t_4439 = _t_4438;
				
				}
		else
				{
					float _t_4440;
					float _t_4441;
					float _t_4442;
				
					_t_4440 = -1.0f * tx1_4_1;
					_t_4441 = tx2_5_1 + _t_4440;
					_t_4442 = -1.0f * _t_4441;
					_t_4439 = _t_4442;
				
				}
		
			_t_4443 = _t_4439 * _t_203;
			_t_4444 = 0.0f < _t_4443;
			if(_t_4444)
				{
				
					_t_4445 = py1_13_1;
				
				}
		else
				{
				
					_t_4445 = py0_12_1;
				
				}
		
			_t_4446 = _t_4432 * _t_4445;
			_t_4447 = _t_4421 + _t_4446;
			_t_4448 = _t_4036 < _t_4447;
			_t_4449 = _t_4396 && _t_4448;
			_t_4450 = _t_4344 && _t_4449;
			if(_t_4450)
				{
				
					_t_4451 = 1.0f;
				
				}
		else
				{
				
					_t_4451 = 0.0f;
				
				}
		
			_t_4452 = _t_4451 * _t_203;
			_t_4453 = _t_4452;
		
		}
else
		{
		
			_t_4453 = 0.0f;
		
		}

	_t_4454 = -1.0f * pc0_14_1;
	_t_4455 = tc0_17_1 + _t_4454;
	_t_4456 = _t_4455 * _t_4455;
	_t_4457 = -1.0f * pc1_15_1;
	_t_4458 = tc1_18_1 + _t_4457;
	_t_4459 = _t_4458 * _t_4458;
	_t_4460 = _t_4456 + _t_4459;
	_t_4461 = -1.0f * pc2_16_1;
	_t_4462 = tc2_19_1 + _t_4461;
	_t_4463 = _t_4462 * _t_4462;
	_t_4464 = _t_4460 + _t_4463;
	_t_4465 = tx3_6_1 * ty1_7_1;
	_t_4466 = tx1_4_1 * ty3_9_1;
	_t_4467 = _t_4466 * -1.0f;
	_t_4468 = _t_4465 + _t_4467;
	_t_4469 = -1.0f * ty1_7_1;
	_t_4470 = ty3_9_1 + _t_4469;
	_t_4471 = _t_4470 * _t_4063;
	_t_4472 = _t_4468 + _t_4471;
	_t_4473 = -1.0f * tx3_6_1;
	_t_4474 = tx1_4_1 + _t_4473;
	_t_4475 = _t_4474 * _t_4116;
	_t_4476 = _t_4472 + _t_4475;
	_t_4477 = _t_4476 < 0.0f;
	if(_t_4477)
		{
		
			_t_4478 = 1.0f;
		
		}
else
		{
		
			_t_4478 = 0.0f;
		
		}

	_t_4479 = _t_4464 * _t_4478;
	_t_4480 = tx2_5_1 * ty3_9_1;
	_t_4481 = tx3_6_1 * ty2_8_1;
	_t_4482 = _t_4481 * -1.0f;
	_t_4483 = _t_4480 + _t_4482;
	_t_4484 = -1.0f * ty3_9_1;
	_t_4485 = ty2_8_1 + _t_4484;
	_t_4486 = _t_4485 * _t_4063;
	_t_4487 = _t_4483 + _t_4486;
	_t_4488 = -1.0f * tx2_5_1;
	_t_4489 = tx3_6_1 + _t_4488;
	_t_4490 = _t_4489 * _t_4116;
	_t_4491 = _t_4487 + _t_4490;
	_t_4492 = _t_4491 < 0.0f;
	if(_t_4492)
		{
		
			_t_4493 = 1.0f;
		
		}
else
		{
		
			_t_4493 = 0.0f;
		
		}

	_t_4494 = _t_4479 * _t_4493;
	_t_4495 = _t_4494 * ty1_7_1;
	_t_4496 = -1.0f * pc0_14_1;
	_t_4497 = tc0_17_1 + _t_4496;
	_t_4498 = _t_4497 * _t_4497;
	_t_4499 = -1.0f * pc1_15_1;
	_t_4500 = tc1_18_1 + _t_4499;
	_t_4501 = _t_4500 * _t_4500;
	_t_4502 = _t_4498 + _t_4501;
	_t_4503 = -1.0f * pc2_16_1;
	_t_4504 = tc2_19_1 + _t_4503;
	_t_4505 = _t_4504 * _t_4504;
	_t_4506 = _t_4502 + _t_4505;
	_t_4507 = tx3_6_1 * ty1_7_1;
	_t_4508 = tx1_4_1 * ty3_9_1;
	_t_4509 = _t_4508 * -1.0f;
	_t_4510 = _t_4507 + _t_4509;
	_t_4511 = -1.0f * ty1_7_1;
	_t_4512 = ty3_9_1 + _t_4511;
	_t_4513 = _t_4512 * _t_4063;
	_t_4514 = _t_4510 + _t_4513;
	_t_4515 = -1.0f * tx3_6_1;
	_t_4516 = tx1_4_1 + _t_4515;
	_t_4517 = _t_4516 * _t_4116;
	_t_4518 = _t_4514 + _t_4517;
	_t_4519 = _t_4518 < 0.0f;
	if(_t_4519)
		{
		
			_t_4520 = 1.0f;
		
		}
else
		{
		
			_t_4520 = 0.0f;
		
		}

	_t_4521 = _t_4506 * _t_4520;
	_t_4522 = tx2_5_1 * ty3_9_1;
	_t_4523 = tx3_6_1 * ty2_8_1;
	_t_4524 = _t_4523 * -1.0f;
	_t_4525 = _t_4522 + _t_4524;
	_t_4526 = -1.0f * ty3_9_1;
	_t_4527 = ty2_8_1 + _t_4526;
	_t_4528 = _t_4527 * _t_4063;
	_t_4529 = _t_4525 + _t_4528;
	_t_4530 = -1.0f * tx2_5_1;
	_t_4531 = tx3_6_1 + _t_4530;
	_t_4532 = _t_4531 * _t_4116;
	_t_4533 = _t_4529 + _t_4532;
	_t_4534 = _t_4533 < 0.0f;
	if(_t_4534)
		{
		
			_t_4535 = 1.0f;
		
		}
else
		{
		
			_t_4535 = 0.0f;
		
		}

	_t_4536 = _t_4521 * _t_4535;
	_t_4537 = _t_4536 * _t_4116;
	_t_4538 = _t_4537 * -1.0f;
	_t_4539 = _t_4495 + _t_4538;
	_t_4037 = _t_4453 * _t_4539;

	return _t_4037;
}
__device__ float tegpixellet_block_25(float ty1_7_1,float ty2_8_1,float _t_203,float _t_4036,float tx2_5_1,float tx1_4_1,float y__2795_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	float _t_4038;
	float _t_4039;
	float _t_4040;
	bool _t_4041;
	float _t_4044;
	float _t_4048;
	float _t_4049;
	float _t_4050;
	float _t_4051;
	float _t_4052;
	bool _t_4053;
	float _t_4056;
	float _t_4060;
	float _t_4061;
	float _t_4062;
	float _t_4063;
	float _t_4064;
	float _t_4065;
	float _t_4066;
	bool _t_4067;
	float _t_4070;
	float _t_4074;
	float _t_4075;
	float _t_4076;
	float _t_4077;
	bool _t_4078;
	float _t_4081;
	float _t_4085;
	float _t_4086;
	float _t_4087;
	float _t_4088;
	float _t_4089;
	bool _t_4090;
	float _t_4093;
	float _t_4097;
	float _t_4098;
	float _t_4099;
	float _t_4100;
	float _t_4101;
	float _t_4102;
	float _t_4103;
	float _t_4104;
	float _t_4105;
	float _t_4106;
	bool _t_4107;
	float _t_4110;
	float _t_4114;
	float _t_4115;
	float _t_4116;

	float _t_4037;

	_t_4038 = -1.0f * ty2_8_1;
	_t_4039 = ty1_7_1 + _t_4038;
	_t_4040 = -1.0f * _t_4039;
	_t_4041 = _t_4040 < 0.0f;
	if(_t_4041)
		{
			float _t_4042;
			float _t_4043;
		
			_t_4042 = -1.0f * ty2_8_1;
			_t_4043 = ty1_7_1 + _t_4042;
			_t_4044 = _t_4043;
		
		}
else
		{
			float _t_4045;
			float _t_4046;
			float _t_4047;
		
			_t_4045 = -1.0f * ty2_8_1;
			_t_4046 = ty1_7_1 + _t_4045;
			_t_4047 = -1.0f * _t_4046;
			_t_4044 = _t_4047;
		
		}

	_t_4048 = _t_4044 * _t_203;
	_t_4049 = _t_4048 * _t_4036;
	_t_4050 = -1.0f * ty2_8_1;
	_t_4051 = ty1_7_1 + _t_4050;
	_t_4052 = -1.0f * _t_4051;
	_t_4053 = _t_4052 < 0.0f;
	if(_t_4053)
		{
			float _t_4054;
			float _t_4055;
		
			_t_4054 = -1.0f * tx1_4_1;
			_t_4055 = tx2_5_1 + _t_4054;
			_t_4056 = _t_4055;
		
		}
else
		{
			float _t_4057;
			float _t_4058;
			float _t_4059;
		
			_t_4057 = -1.0f * tx1_4_1;
			_t_4058 = tx2_5_1 + _t_4057;
			_t_4059 = -1.0f * _t_4058;
			_t_4056 = _t_4059;
		
		}

	_t_4060 = _t_4056 * _t_203;
	_t_4061 = _t_4060 * -1.0f;
	_t_4062 = _t_4061 * y__2795_1;
	_t_4063 = _t_4049 + _t_4062;
	_t_4064 = -1.0f * ty2_8_1;
	_t_4065 = ty1_7_1 + _t_4064;
	_t_4066 = -1.0f * _t_4065;
	_t_4067 = _t_4066 < 0.0f;
	if(_t_4067)
		{
			float _t_4068;
			float _t_4069;
		
			_t_4068 = -1.0f * tx1_4_1;
			_t_4069 = tx2_5_1 + _t_4068;
			_t_4070 = _t_4069;
		
		}
else
		{
			float _t_4071;
			float _t_4072;
			float _t_4073;
		
			_t_4071 = -1.0f * tx1_4_1;
			_t_4072 = tx2_5_1 + _t_4071;
			_t_4073 = -1.0f * _t_4072;
			_t_4070 = _t_4073;
		
		}

	_t_4074 = _t_4070 * _t_203;
	_t_4075 = -1.0f * ty2_8_1;
	_t_4076 = ty1_7_1 + _t_4075;
	_t_4077 = -1.0f * _t_4076;
	_t_4078 = _t_4077 < 0.0f;
	if(_t_4078)
		{
			float _t_4079;
			float _t_4080;
		
			_t_4079 = -1.0f * tx1_4_1;
			_t_4080 = tx2_5_1 + _t_4079;
			_t_4081 = _t_4080;
		
		}
else
		{
			float _t_4082;
			float _t_4083;
			float _t_4084;
		
			_t_4082 = -1.0f * tx1_4_1;
			_t_4083 = tx2_5_1 + _t_4082;
			_t_4084 = -1.0f * _t_4083;
			_t_4081 = _t_4084;
		
		}

	_t_4085 = _t_4081 * _t_203;
	_t_4086 = _t_4074 * _t_4085;
	_t_4087 = -1.0f * ty2_8_1;
	_t_4088 = ty1_7_1 + _t_4087;
	_t_4089 = -1.0f * _t_4088;
	_t_4090 = _t_4089 < 0.0f;
	if(_t_4090)
		{
			float _t_4091;
			float _t_4092;
		
			_t_4091 = -1.0f * ty2_8_1;
			_t_4092 = ty1_7_1 + _t_4091;
			_t_4093 = _t_4092;
		
		}
else
		{
			float _t_4094;
			float _t_4095;
			float _t_4096;
		
			_t_4094 = -1.0f * ty2_8_1;
			_t_4095 = ty1_7_1 + _t_4094;
			_t_4096 = -1.0f * _t_4095;
			_t_4093 = _t_4096;
		
		}

	_t_4097 = _t_4093 * _t_203;
	_t_4098 = 1.0f + _t_4097;
	_t_4099 = 1.0f / _t_4098;
	_t_4100 = _t_4086 * _t_4099;
	_t_4101 = _t_4100 * -1.0f;
	_t_4102 = 1.0f + _t_4101;
	_t_4103 = _t_4102 * y__2795_1;
	_t_4104 = -1.0f * ty2_8_1;
	_t_4105 = ty1_7_1 + _t_4104;
	_t_4106 = -1.0f * _t_4105;
	_t_4107 = _t_4106 < 0.0f;
	if(_t_4107)
		{
			float _t_4108;
			float _t_4109;
		
			_t_4108 = -1.0f * tx1_4_1;
			_t_4109 = tx2_5_1 + _t_4108;
			_t_4110 = _t_4109;
		
		}
else
		{
			float _t_4111;
			float _t_4112;
			float _t_4113;
		
			_t_4111 = -1.0f * tx1_4_1;
			_t_4112 = tx2_5_1 + _t_4111;
			_t_4113 = -1.0f * _t_4112;
			_t_4110 = _t_4113;
		
		}

	_t_4114 = _t_4110 * _t_203;
	_t_4115 = _t_4114 * _t_4036;
	_t_4116 = _t_4103 + _t_4115;
	_t_4037 = tegpixellet_block_26(py0_12_1,_t_4116,py1_13_1,px0_10_1,_t_4063,px1_11_1,ty1_7_1,ty2_8_1,tx2_5_1,tx1_4_1,_t_203,y__2795_1,_t_4036,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);

	return _t_4037;
}
__device__ float tegpixelbody_block_20(float ty1_7_1,float ty2_8_1,float _t_203,float px0_10_1,float px1_11_1,float tx2_5_1,float tx1_4_1,float py0_12_1,float py1_13_1,float y__2795_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	float _t_3880;
	float _t_3881;
	float _t_3882;
	bool _t_3883;
	float _t_3886;
	float _t_3890;
	float _t_3891;
	float _t_3892;
	float _t_3893;
	bool _t_3894;
	float _t_3897;
	float _t_3901;
	bool _t_3902;
	float _t_3903;
	float _t_3904;
	float _t_3905;
	float _t_3906;
	float _t_3907;
	bool _t_3908;
	float _t_3911;
	float _t_3915;
	float _t_3916;
	float _t_3917;
	float _t_3918;
	bool _t_3919;
	float _t_3922;
	float _t_3926;
	bool _t_3927;
	float _t_3928;
	float _t_3929;
	float _t_3930;
	float _t_3931;
	float _t_3932;
	float _t_3933;
	bool _t_3934;
	float _t_3939;
	float _t_3945;
	float _t_3946;
	float _t_3947;
	float _t_3948;
	bool _t_3949;
	float _t_3950;
	float _t_3951;
	float _t_3952;
	bool _t_3953;
	float _t_3956;
	float _t_3960;
	float _t_3961;
	float _t_3962;
	float _t_3963;
	bool _t_3964;
	float _t_3967;
	float _t_3971;
	bool _t_3972;
	float _t_3973;
	float _t_3974;
	float _t_3975;
	float _t_3976;
	float _t_3977;
	bool _t_3978;
	float _t_3981;
	float _t_3985;
	float _t_3986;
	float _t_3987;
	float _t_3988;
	bool _t_3989;
	float _t_3992;
	float _t_3996;
	bool _t_3997;
	float _t_3998;
	float _t_3999;
	float _t_4000;
	float _t_4001;
	float _t_4002;
	float _t_4003;
	bool _t_4004;
	float _t_4009;
	float _t_4015;
	float _t_4016;
	float _t_4017;
	float _t_4018;
	bool _t_4019;
	bool _t_4020;

	float _t_3879;

	_t_3880 = -1.0f * ty2_8_1;
	_t_3881 = ty1_7_1 + _t_3880;
	_t_3882 = -1.0f * _t_3881;
	_t_3883 = _t_3882 < 0.0f;
	if(_t_3883)
		{
			float _t_3884;
			float _t_3885;
		
			_t_3884 = -1.0f * ty2_8_1;
			_t_3885 = ty1_7_1 + _t_3884;
			_t_3886 = _t_3885;
		
		}
else
		{
			float _t_3887;
			float _t_3888;
			float _t_3889;
		
			_t_3887 = -1.0f * ty2_8_1;
			_t_3888 = ty1_7_1 + _t_3887;
			_t_3889 = -1.0f * _t_3888;
			_t_3886 = _t_3889;
		
		}

	_t_3890 = _t_3886 * _t_203;
	_t_3891 = -1.0f * ty2_8_1;
	_t_3892 = ty1_7_1 + _t_3891;
	_t_3893 = -1.0f * _t_3892;
	_t_3894 = _t_3893 < 0.0f;
	if(_t_3894)
		{
			float _t_3895;
			float _t_3896;
		
			_t_3895 = -1.0f * ty2_8_1;
			_t_3896 = ty1_7_1 + _t_3895;
			_t_3897 = _t_3896;
		
		}
else
		{
			float _t_3898;
			float _t_3899;
			float _t_3900;
		
			_t_3898 = -1.0f * ty2_8_1;
			_t_3899 = ty1_7_1 + _t_3898;
			_t_3900 = -1.0f * _t_3899;
			_t_3897 = _t_3900;
		
		}

	_t_3901 = _t_3897 * _t_203;
	_t_3902 = 0.0f < _t_3901;
	if(_t_3902)
		{
		
			_t_3903 = px0_10_1;
		
		}
else
		{
		
			_t_3903 = px1_11_1;
		
		}

	_t_3904 = _t_3890 * _t_3903;
	_t_3905 = -1.0f * ty2_8_1;
	_t_3906 = ty1_7_1 + _t_3905;
	_t_3907 = -1.0f * _t_3906;
	_t_3908 = _t_3907 < 0.0f;
	if(_t_3908)
		{
			float _t_3909;
			float _t_3910;
		
			_t_3909 = -1.0f * tx1_4_1;
			_t_3910 = tx2_5_1 + _t_3909;
			_t_3911 = _t_3910;
		
		}
else
		{
			float _t_3912;
			float _t_3913;
			float _t_3914;
		
			_t_3912 = -1.0f * tx1_4_1;
			_t_3913 = tx2_5_1 + _t_3912;
			_t_3914 = -1.0f * _t_3913;
			_t_3911 = _t_3914;
		
		}

	_t_3915 = _t_3911 * _t_203;
	_t_3916 = -1.0f * ty2_8_1;
	_t_3917 = ty1_7_1 + _t_3916;
	_t_3918 = -1.0f * _t_3917;
	_t_3919 = _t_3918 < 0.0f;
	if(_t_3919)
		{
			float _t_3920;
			float _t_3921;
		
			_t_3920 = -1.0f * tx1_4_1;
			_t_3921 = tx2_5_1 + _t_3920;
			_t_3922 = _t_3921;
		
		}
else
		{
			float _t_3923;
			float _t_3924;
			float _t_3925;
		
			_t_3923 = -1.0f * tx1_4_1;
			_t_3924 = tx2_5_1 + _t_3923;
			_t_3925 = -1.0f * _t_3924;
			_t_3922 = _t_3925;
		
		}

	_t_3926 = _t_3922 * _t_203;
	_t_3927 = 0.0f < _t_3926;
	if(_t_3927)
		{
		
			_t_3928 = py0_12_1;
		
		}
else
		{
		
			_t_3928 = py1_13_1;
		
		}

	_t_3929 = _t_3915 * _t_3928;
	_t_3930 = _t_3904 + _t_3929;
	_t_3931 = -1.0f * ty2_8_1;
	_t_3932 = ty1_7_1 + _t_3931;
	_t_3933 = -1.0f * _t_3932;
	_t_3934 = _t_3933 < 0.0f;
	if(_t_3934)
		{
			float _t_3935;
			float _t_3936;
			float _t_3937;
			float _t_3938;
		
			_t_3935 = tx1_4_1 * ty2_8_1;
			_t_3936 = tx2_5_1 * ty1_7_1;
			_t_3937 = _t_3936 * -1.0f;
			_t_3938 = _t_3935 + _t_3937;
			_t_3939 = _t_3938;
		
		}
else
		{
			float _t_3940;
			float _t_3941;
			float _t_3942;
			float _t_3943;
			float _t_3944;
		
			_t_3940 = tx1_4_1 * ty2_8_1;
			_t_3941 = tx2_5_1 * ty1_7_1;
			_t_3942 = _t_3941 * -1.0f;
			_t_3943 = _t_3940 + _t_3942;
			_t_3944 = -1.0f * _t_3943;
			_t_3939 = _t_3944;
		
		}

	_t_3945 = -1.0f * _t_3939;
	_t_3946 = _t_3945 * _t_203;
	_t_3947 = _t_3946 * -1.0f;
	_t_3948 = _t_3930 + _t_3947;
	_t_3949 = _t_3948 < 0.0f;
	_t_3950 = -1.0f * ty2_8_1;
	_t_3951 = ty1_7_1 + _t_3950;
	_t_3952 = -1.0f * _t_3951;
	_t_3953 = _t_3952 < 0.0f;
	if(_t_3953)
		{
			float _t_3954;
			float _t_3955;
		
			_t_3954 = -1.0f * ty2_8_1;
			_t_3955 = ty1_7_1 + _t_3954;
			_t_3956 = _t_3955;
		
		}
else
		{
			float _t_3957;
			float _t_3958;
			float _t_3959;
		
			_t_3957 = -1.0f * ty2_8_1;
			_t_3958 = ty1_7_1 + _t_3957;
			_t_3959 = -1.0f * _t_3958;
			_t_3956 = _t_3959;
		
		}

	_t_3960 = _t_3956 * _t_203;
	_t_3961 = -1.0f * ty2_8_1;
	_t_3962 = ty1_7_1 + _t_3961;
	_t_3963 = -1.0f * _t_3962;
	_t_3964 = _t_3963 < 0.0f;
	if(_t_3964)
		{
			float _t_3965;
			float _t_3966;
		
			_t_3965 = -1.0f * ty2_8_1;
			_t_3966 = ty1_7_1 + _t_3965;
			_t_3967 = _t_3966;
		
		}
else
		{
			float _t_3968;
			float _t_3969;
			float _t_3970;
		
			_t_3968 = -1.0f * ty2_8_1;
			_t_3969 = ty1_7_1 + _t_3968;
			_t_3970 = -1.0f * _t_3969;
			_t_3967 = _t_3970;
		
		}

	_t_3971 = _t_3967 * _t_203;
	_t_3972 = 0.0f < _t_3971;
	if(_t_3972)
		{
		
			_t_3973 = px1_11_1;
		
		}
else
		{
		
			_t_3973 = px0_10_1;
		
		}

	_t_3974 = _t_3960 * _t_3973;
	_t_3975 = -1.0f * ty2_8_1;
	_t_3976 = ty1_7_1 + _t_3975;
	_t_3977 = -1.0f * _t_3976;
	_t_3978 = _t_3977 < 0.0f;
	if(_t_3978)
		{
			float _t_3979;
			float _t_3980;
		
			_t_3979 = -1.0f * tx1_4_1;
			_t_3980 = tx2_5_1 + _t_3979;
			_t_3981 = _t_3980;
		
		}
else
		{
			float _t_3982;
			float _t_3983;
			float _t_3984;
		
			_t_3982 = -1.0f * tx1_4_1;
			_t_3983 = tx2_5_1 + _t_3982;
			_t_3984 = -1.0f * _t_3983;
			_t_3981 = _t_3984;
		
		}

	_t_3985 = _t_3981 * _t_203;
	_t_3986 = -1.0f * ty2_8_1;
	_t_3987 = ty1_7_1 + _t_3986;
	_t_3988 = -1.0f * _t_3987;
	_t_3989 = _t_3988 < 0.0f;
	if(_t_3989)
		{
			float _t_3990;
			float _t_3991;
		
			_t_3990 = -1.0f * tx1_4_1;
			_t_3991 = tx2_5_1 + _t_3990;
			_t_3992 = _t_3991;
		
		}
else
		{
			float _t_3993;
			float _t_3994;
			float _t_3995;
		
			_t_3993 = -1.0f * tx1_4_1;
			_t_3994 = tx2_5_1 + _t_3993;
			_t_3995 = -1.0f * _t_3994;
			_t_3992 = _t_3995;
		
		}

	_t_3996 = _t_3992 * _t_203;
	_t_3997 = 0.0f < _t_3996;
	if(_t_3997)
		{
		
			_t_3998 = py1_13_1;
		
		}
else
		{
		
			_t_3998 = py0_12_1;
		
		}

	_t_3999 = _t_3985 * _t_3998;
	_t_4000 = _t_3974 + _t_3999;
	_t_4001 = -1.0f * ty2_8_1;
	_t_4002 = ty1_7_1 + _t_4001;
	_t_4003 = -1.0f * _t_4002;
	_t_4004 = _t_4003 < 0.0f;
	if(_t_4004)
		{
			float _t_4005;
			float _t_4006;
			float _t_4007;
			float _t_4008;
		
			_t_4005 = tx1_4_1 * ty2_8_1;
			_t_4006 = tx2_5_1 * ty1_7_1;
			_t_4007 = _t_4006 * -1.0f;
			_t_4008 = _t_4005 + _t_4007;
			_t_4009 = _t_4008;
		
		}
else
		{
			float _t_4010;
			float _t_4011;
			float _t_4012;
			float _t_4013;
			float _t_4014;
		
			_t_4010 = tx1_4_1 * ty2_8_1;
			_t_4011 = tx2_5_1 * ty1_7_1;
			_t_4012 = _t_4011 * -1.0f;
			_t_4013 = _t_4010 + _t_4012;
			_t_4014 = -1.0f * _t_4013;
			_t_4009 = _t_4014;
		
		}

	_t_4015 = -1.0f * _t_4009;
	_t_4016 = _t_4015 * _t_203;
	_t_4017 = _t_4016 * -1.0f;
	_t_4018 = _t_4000 + _t_4017;
	_t_4019 = 0.0f < _t_4018;
	_t_4020 = _t_3949 && _t_4019;
	if(_t_4020)
		{
			float _t_4021;
			float _t_4022;
			float _t_4023;
			bool _t_4024;
			float _t_4029;
			float _t_4035;
			float _t_4036;
			float _t_4037;
		
			_t_4021 = -1.0f * ty2_8_1;
			_t_4022 = ty1_7_1 + _t_4021;
			_t_4023 = -1.0f * _t_4022;
			_t_4024 = _t_4023 < 0.0f;
			if(_t_4024)
				{
					float _t_4025;
					float _t_4026;
					float _t_4027;
					float _t_4028;
				
					_t_4025 = tx1_4_1 * ty2_8_1;
					_t_4026 = tx2_5_1 * ty1_7_1;
					_t_4027 = _t_4026 * -1.0f;
					_t_4028 = _t_4025 + _t_4027;
					_t_4029 = _t_4028;
				
				}
		else
				{
					float _t_4030;
					float _t_4031;
					float _t_4032;
					float _t_4033;
					float _t_4034;
				
					_t_4030 = tx1_4_1 * ty2_8_1;
					_t_4031 = tx2_5_1 * ty1_7_1;
					_t_4032 = _t_4031 * -1.0f;
					_t_4033 = _t_4030 + _t_4032;
					_t_4034 = -1.0f * _t_4033;
					_t_4029 = _t_4034;
				
				}
		
			_t_4035 = -1.0f * _t_4029;
			_t_4036 = _t_4035 * _t_203;
			_t_4037 = tegpixellet_block_25(ty1_7_1,ty2_8_1,_t_203,_t_4036,tx2_5_1,tx1_4_1,y__2795_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);
			_t_3879 = _t_4037;
		
		}
else
		{
		
			_t_3879 = 0.0f;
		
		}


	return _t_3879;
}
__device__ float tegpixelintegrator_20(float pc1_15_1,float ty3_9_1,float tc2_19_1,float ty2_8_1,float pc0_14_1,float ty1_7_1,float _t_3878,float tx1_4_1,float tx3_6_1,float tx2_5_1,float py1_13_1,float pc2_16_1,float px1_11_1,float _t_203,float tc0_17_1,float py0_12_1,float tc1_18_1,float px0_10_1,float _t_3769){
    float y__2795_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_3878 - _t_3769)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__2795_1 = _t_3769 + __step__ * (i + (float)(0.5));
        float _t_3879;
		_t_3879 = tegpixelbody_block_20(ty1_7_1,ty2_8_1,_t_203,px0_10_1,px1_11_1,tx2_5_1,tx1_4_1,py0_12_1,py1_13_1,y__2795_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);;
        __output__ = __output__ + _t_3879 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_4(float ty1_7_1,float ty2_8_1,float tx2_5_1,float tx1_4_1,float _t_203,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	float _t_3661;
	float _t_3662;
	float _t_3663;
	bool _t_3664;
	float _t_3667;
	float _t_3671;
	float _t_3672;
	float _t_3673;
	float _t_3674;
	float _t_3675;
	bool _t_3676;
	float _t_3679;
	float _t_3683;
	float _t_3684;
	bool _t_3685;
	float _t_3686;
	float _t_3687;
	float _t_3688;
	float _t_3689;
	float _t_3690;
	bool _t_3691;
	float _t_3694;
	float _t_3698;
	float _t_3699;
	float _t_3700;
	float _t_3701;
	bool _t_3702;
	float _t_3705;
	float _t_3709;
	float _t_3710;
	float _t_3711;
	float _t_3712;
	float _t_3713;
	bool _t_3714;
	float _t_3717;
	float _t_3721;
	float _t_3722;
	float _t_3723;
	float _t_3724;
	float _t_3725;
	float _t_3726;
	float _t_3727;
	float _t_3728;
	float _t_3729;
	bool _t_3730;
	float _t_3733;
	float _t_3737;
	float _t_3738;
	float _t_3739;
	float _t_3740;
	bool _t_3741;
	float _t_3744;
	float _t_3748;
	float _t_3749;
	float _t_3750;
	float _t_3751;
	float _t_3752;
	bool _t_3753;
	float _t_3756;
	float _t_3760;
	float _t_3761;
	float _t_3762;
	float _t_3763;
	float _t_3764;
	float _t_3765;
	bool _t_3766;
	float _t_3767;
	float _t_3768;
	float _t_3769;
	float _t_3770;
	float _t_3771;
	float _t_3772;
	bool _t_3773;
	float _t_3776;
	float _t_3780;
	float _t_3781;
	float _t_3782;
	float _t_3783;
	float _t_3784;
	bool _t_3785;
	float _t_3788;
	float _t_3792;
	float _t_3793;
	bool _t_3794;
	float _t_3795;
	float _t_3796;
	float _t_3797;
	float _t_3798;
	float _t_3799;
	bool _t_3800;
	float _t_3803;
	float _t_3807;
	float _t_3808;
	float _t_3809;
	float _t_3810;
	bool _t_3811;
	float _t_3814;
	float _t_3818;
	float _t_3819;
	float _t_3820;
	float _t_3821;
	float _t_3822;
	bool _t_3823;
	float _t_3826;
	float _t_3830;
	float _t_3831;
	float _t_3832;
	float _t_3833;
	float _t_3834;
	float _t_3835;
	float _t_3836;
	float _t_3837;
	float _t_3838;
	bool _t_3839;
	float _t_3842;
	float _t_3846;
	float _t_3847;
	float _t_3848;
	float _t_3849;
	bool _t_3850;
	float _t_3853;
	float _t_3857;
	float _t_3858;
	float _t_3859;
	float _t_3860;
	float _t_3861;
	bool _t_3862;
	float _t_3865;
	float _t_3869;
	float _t_3870;
	float _t_3871;
	float _t_3872;
	float _t_3873;
	float _t_3874;
	bool _t_3875;
	float _t_3876;
	float _t_3877;
	float _t_3878;

	float _t_204;

	_t_3661 = -1.0f * ty2_8_1;
	_t_3662 = ty1_7_1 + _t_3661;
	_t_3663 = -1.0f * _t_3662;
	_t_3664 = _t_3663 < 0.0f;
	if(_t_3664)
		{
			float _t_3665;
			float _t_3666;
		
			_t_3665 = -1.0f * tx1_4_1;
			_t_3666 = tx2_5_1 + _t_3665;
			_t_3667 = _t_3666;
		
		}
else
		{
			float _t_3668;
			float _t_3669;
			float _t_3670;
		
			_t_3668 = -1.0f * tx1_4_1;
			_t_3669 = tx2_5_1 + _t_3668;
			_t_3670 = -1.0f * _t_3669;
			_t_3667 = _t_3670;
		
		}

	_t_3671 = _t_3667 * _t_203;
	_t_3672 = _t_3671 * -1.0f;
	_t_3673 = -1.0f * ty2_8_1;
	_t_3674 = ty1_7_1 + _t_3673;
	_t_3675 = -1.0f * _t_3674;
	_t_3676 = _t_3675 < 0.0f;
	if(_t_3676)
		{
			float _t_3677;
			float _t_3678;
		
			_t_3677 = -1.0f * tx1_4_1;
			_t_3678 = tx2_5_1 + _t_3677;
			_t_3679 = _t_3678;
		
		}
else
		{
			float _t_3680;
			float _t_3681;
			float _t_3682;
		
			_t_3680 = -1.0f * tx1_4_1;
			_t_3681 = tx2_5_1 + _t_3680;
			_t_3682 = -1.0f * _t_3681;
			_t_3679 = _t_3682;
		
		}

	_t_3683 = _t_3679 * _t_203;
	_t_3684 = _t_3683 * -1.0f;
	_t_3685 = 0.0f < _t_3684;
	if(_t_3685)
		{
		
			_t_3686 = px0_10_1;
		
		}
else
		{
		
			_t_3686 = px1_11_1;
		
		}

	_t_3687 = _t_3672 * _t_3686;
	_t_3688 = -1.0f * ty2_8_1;
	_t_3689 = ty1_7_1 + _t_3688;
	_t_3690 = -1.0f * _t_3689;
	_t_3691 = _t_3690 < 0.0f;
	if(_t_3691)
		{
			float _t_3692;
			float _t_3693;
		
			_t_3692 = -1.0f * tx1_4_1;
			_t_3693 = tx2_5_1 + _t_3692;
			_t_3694 = _t_3693;
		
		}
else
		{
			float _t_3695;
			float _t_3696;
			float _t_3697;
		
			_t_3695 = -1.0f * tx1_4_1;
			_t_3696 = tx2_5_1 + _t_3695;
			_t_3697 = -1.0f * _t_3696;
			_t_3694 = _t_3697;
		
		}

	_t_3698 = _t_3694 * _t_203;
	_t_3699 = -1.0f * ty2_8_1;
	_t_3700 = ty1_7_1 + _t_3699;
	_t_3701 = -1.0f * _t_3700;
	_t_3702 = _t_3701 < 0.0f;
	if(_t_3702)
		{
			float _t_3703;
			float _t_3704;
		
			_t_3703 = -1.0f * tx1_4_1;
			_t_3704 = tx2_5_1 + _t_3703;
			_t_3705 = _t_3704;
		
		}
else
		{
			float _t_3706;
			float _t_3707;
			float _t_3708;
		
			_t_3706 = -1.0f * tx1_4_1;
			_t_3707 = tx2_5_1 + _t_3706;
			_t_3708 = -1.0f * _t_3707;
			_t_3705 = _t_3708;
		
		}

	_t_3709 = _t_3705 * _t_203;
	_t_3710 = _t_3698 * _t_3709;
	_t_3711 = -1.0f * ty2_8_1;
	_t_3712 = ty1_7_1 + _t_3711;
	_t_3713 = -1.0f * _t_3712;
	_t_3714 = _t_3713 < 0.0f;
	if(_t_3714)
		{
			float _t_3715;
			float _t_3716;
		
			_t_3715 = -1.0f * ty2_8_1;
			_t_3716 = ty1_7_1 + _t_3715;
			_t_3717 = _t_3716;
		
		}
else
		{
			float _t_3718;
			float _t_3719;
			float _t_3720;
		
			_t_3718 = -1.0f * ty2_8_1;
			_t_3719 = ty1_7_1 + _t_3718;
			_t_3720 = -1.0f * _t_3719;
			_t_3717 = _t_3720;
		
		}

	_t_3721 = _t_3717 * _t_203;
	_t_3722 = 1.0f + _t_3721;
	_t_3723 = 1.0f / _t_3722;
	_t_3724 = _t_3710 * _t_3723;
	_t_3725 = _t_3724 * -1.0f;
	_t_3726 = 1.0f + _t_3725;
	_t_3727 = -1.0f * ty2_8_1;
	_t_3728 = ty1_7_1 + _t_3727;
	_t_3729 = -1.0f * _t_3728;
	_t_3730 = _t_3729 < 0.0f;
	if(_t_3730)
		{
			float _t_3731;
			float _t_3732;
		
			_t_3731 = -1.0f * tx1_4_1;
			_t_3732 = tx2_5_1 + _t_3731;
			_t_3733 = _t_3732;
		
		}
else
		{
			float _t_3734;
			float _t_3735;
			float _t_3736;
		
			_t_3734 = -1.0f * tx1_4_1;
			_t_3735 = tx2_5_1 + _t_3734;
			_t_3736 = -1.0f * _t_3735;
			_t_3733 = _t_3736;
		
		}

	_t_3737 = _t_3733 * _t_203;
	_t_3738 = -1.0f * ty2_8_1;
	_t_3739 = ty1_7_1 + _t_3738;
	_t_3740 = -1.0f * _t_3739;
	_t_3741 = _t_3740 < 0.0f;
	if(_t_3741)
		{
			float _t_3742;
			float _t_3743;
		
			_t_3742 = -1.0f * tx1_4_1;
			_t_3743 = tx2_5_1 + _t_3742;
			_t_3744 = _t_3743;
		
		}
else
		{
			float _t_3745;
			float _t_3746;
			float _t_3747;
		
			_t_3745 = -1.0f * tx1_4_1;
			_t_3746 = tx2_5_1 + _t_3745;
			_t_3747 = -1.0f * _t_3746;
			_t_3744 = _t_3747;
		
		}

	_t_3748 = _t_3744 * _t_203;
	_t_3749 = _t_3737 * _t_3748;
	_t_3750 = -1.0f * ty2_8_1;
	_t_3751 = ty1_7_1 + _t_3750;
	_t_3752 = -1.0f * _t_3751;
	_t_3753 = _t_3752 < 0.0f;
	if(_t_3753)
		{
			float _t_3754;
			float _t_3755;
		
			_t_3754 = -1.0f * ty2_8_1;
			_t_3755 = ty1_7_1 + _t_3754;
			_t_3756 = _t_3755;
		
		}
else
		{
			float _t_3757;
			float _t_3758;
			float _t_3759;
		
			_t_3757 = -1.0f * ty2_8_1;
			_t_3758 = ty1_7_1 + _t_3757;
			_t_3759 = -1.0f * _t_3758;
			_t_3756 = _t_3759;
		
		}

	_t_3760 = _t_3756 * _t_203;
	_t_3761 = 1.0f + _t_3760;
	_t_3762 = 1.0f / _t_3761;
	_t_3763 = _t_3749 * _t_3762;
	_t_3764 = _t_3763 * -1.0f;
	_t_3765 = 1.0f + _t_3764;
	_t_3766 = 0.0f < _t_3765;
	if(_t_3766)
		{
		
			_t_3767 = py0_12_1;
		
		}
else
		{
		
			_t_3767 = py1_13_1;
		
		}

	_t_3768 = _t_3726 * _t_3767;
	_t_3769 = _t_3687 + _t_3768;
	_t_3770 = -1.0f * ty2_8_1;
	_t_3771 = ty1_7_1 + _t_3770;
	_t_3772 = -1.0f * _t_3771;
	_t_3773 = _t_3772 < 0.0f;
	if(_t_3773)
		{
			float _t_3774;
			float _t_3775;
		
			_t_3774 = -1.0f * tx1_4_1;
			_t_3775 = tx2_5_1 + _t_3774;
			_t_3776 = _t_3775;
		
		}
else
		{
			float _t_3777;
			float _t_3778;
			float _t_3779;
		
			_t_3777 = -1.0f * tx1_4_1;
			_t_3778 = tx2_5_1 + _t_3777;
			_t_3779 = -1.0f * _t_3778;
			_t_3776 = _t_3779;
		
		}

	_t_3780 = _t_3776 * _t_203;
	_t_3781 = _t_3780 * -1.0f;
	_t_3782 = -1.0f * ty2_8_1;
	_t_3783 = ty1_7_1 + _t_3782;
	_t_3784 = -1.0f * _t_3783;
	_t_3785 = _t_3784 < 0.0f;
	if(_t_3785)
		{
			float _t_3786;
			float _t_3787;
		
			_t_3786 = -1.0f * tx1_4_1;
			_t_3787 = tx2_5_1 + _t_3786;
			_t_3788 = _t_3787;
		
		}
else
		{
			float _t_3789;
			float _t_3790;
			float _t_3791;
		
			_t_3789 = -1.0f * tx1_4_1;
			_t_3790 = tx2_5_1 + _t_3789;
			_t_3791 = -1.0f * _t_3790;
			_t_3788 = _t_3791;
		
		}

	_t_3792 = _t_3788 * _t_203;
	_t_3793 = _t_3792 * -1.0f;
	_t_3794 = 0.0f < _t_3793;
	if(_t_3794)
		{
		
			_t_3795 = px1_11_1;
		
		}
else
		{
		
			_t_3795 = px0_10_1;
		
		}

	_t_3796 = _t_3781 * _t_3795;
	_t_3797 = -1.0f * ty2_8_1;
	_t_3798 = ty1_7_1 + _t_3797;
	_t_3799 = -1.0f * _t_3798;
	_t_3800 = _t_3799 < 0.0f;
	if(_t_3800)
		{
			float _t_3801;
			float _t_3802;
		
			_t_3801 = -1.0f * tx1_4_1;
			_t_3802 = tx2_5_1 + _t_3801;
			_t_3803 = _t_3802;
		
		}
else
		{
			float _t_3804;
			float _t_3805;
			float _t_3806;
		
			_t_3804 = -1.0f * tx1_4_1;
			_t_3805 = tx2_5_1 + _t_3804;
			_t_3806 = -1.0f * _t_3805;
			_t_3803 = _t_3806;
		
		}

	_t_3807 = _t_3803 * _t_203;
	_t_3808 = -1.0f * ty2_8_1;
	_t_3809 = ty1_7_1 + _t_3808;
	_t_3810 = -1.0f * _t_3809;
	_t_3811 = _t_3810 < 0.0f;
	if(_t_3811)
		{
			float _t_3812;
			float _t_3813;
		
			_t_3812 = -1.0f * tx1_4_1;
			_t_3813 = tx2_5_1 + _t_3812;
			_t_3814 = _t_3813;
		
		}
else
		{
			float _t_3815;
			float _t_3816;
			float _t_3817;
		
			_t_3815 = -1.0f * tx1_4_1;
			_t_3816 = tx2_5_1 + _t_3815;
			_t_3817 = -1.0f * _t_3816;
			_t_3814 = _t_3817;
		
		}

	_t_3818 = _t_3814 * _t_203;
	_t_3819 = _t_3807 * _t_3818;
	_t_3820 = -1.0f * ty2_8_1;
	_t_3821 = ty1_7_1 + _t_3820;
	_t_3822 = -1.0f * _t_3821;
	_t_3823 = _t_3822 < 0.0f;
	if(_t_3823)
		{
			float _t_3824;
			float _t_3825;
		
			_t_3824 = -1.0f * ty2_8_1;
			_t_3825 = ty1_7_1 + _t_3824;
			_t_3826 = _t_3825;
		
		}
else
		{
			float _t_3827;
			float _t_3828;
			float _t_3829;
		
			_t_3827 = -1.0f * ty2_8_1;
			_t_3828 = ty1_7_1 + _t_3827;
			_t_3829 = -1.0f * _t_3828;
			_t_3826 = _t_3829;
		
		}

	_t_3830 = _t_3826 * _t_203;
	_t_3831 = 1.0f + _t_3830;
	_t_3832 = 1.0f / _t_3831;
	_t_3833 = _t_3819 * _t_3832;
	_t_3834 = _t_3833 * -1.0f;
	_t_3835 = 1.0f + _t_3834;
	_t_3836 = -1.0f * ty2_8_1;
	_t_3837 = ty1_7_1 + _t_3836;
	_t_3838 = -1.0f * _t_3837;
	_t_3839 = _t_3838 < 0.0f;
	if(_t_3839)
		{
			float _t_3840;
			float _t_3841;
		
			_t_3840 = -1.0f * tx1_4_1;
			_t_3841 = tx2_5_1 + _t_3840;
			_t_3842 = _t_3841;
		
		}
else
		{
			float _t_3843;
			float _t_3844;
			float _t_3845;
		
			_t_3843 = -1.0f * tx1_4_1;
			_t_3844 = tx2_5_1 + _t_3843;
			_t_3845 = -1.0f * _t_3844;
			_t_3842 = _t_3845;
		
		}

	_t_3846 = _t_3842 * _t_203;
	_t_3847 = -1.0f * ty2_8_1;
	_t_3848 = ty1_7_1 + _t_3847;
	_t_3849 = -1.0f * _t_3848;
	_t_3850 = _t_3849 < 0.0f;
	if(_t_3850)
		{
			float _t_3851;
			float _t_3852;
		
			_t_3851 = -1.0f * tx1_4_1;
			_t_3852 = tx2_5_1 + _t_3851;
			_t_3853 = _t_3852;
		
		}
else
		{
			float _t_3854;
			float _t_3855;
			float _t_3856;
		
			_t_3854 = -1.0f * tx1_4_1;
			_t_3855 = tx2_5_1 + _t_3854;
			_t_3856 = -1.0f * _t_3855;
			_t_3853 = _t_3856;
		
		}

	_t_3857 = _t_3853 * _t_203;
	_t_3858 = _t_3846 * _t_3857;
	_t_3859 = -1.0f * ty2_8_1;
	_t_3860 = ty1_7_1 + _t_3859;
	_t_3861 = -1.0f * _t_3860;
	_t_3862 = _t_3861 < 0.0f;
	if(_t_3862)
		{
			float _t_3863;
			float _t_3864;
		
			_t_3863 = -1.0f * ty2_8_1;
			_t_3864 = ty1_7_1 + _t_3863;
			_t_3865 = _t_3864;
		
		}
else
		{
			float _t_3866;
			float _t_3867;
			float _t_3868;
		
			_t_3866 = -1.0f * ty2_8_1;
			_t_3867 = ty1_7_1 + _t_3866;
			_t_3868 = -1.0f * _t_3867;
			_t_3865 = _t_3868;
		
		}

	_t_3869 = _t_3865 * _t_203;
	_t_3870 = 1.0f + _t_3869;
	_t_3871 = 1.0f / _t_3870;
	_t_3872 = _t_3858 * _t_3871;
	_t_3873 = _t_3872 * -1.0f;
	_t_3874 = 1.0f + _t_3873;
	_t_3875 = 0.0f < _t_3874;
	if(_t_3875)
		{
		
			_t_3876 = py1_13_1;
		
		}
else
		{
		
			_t_3876 = py0_12_1;
		
		}

	_t_3877 = _t_3835 * _t_3876;
	_t_3878 = _t_3796 + _t_3877;
	_t_204 = tegpixelintegrator_20(pc1_15_1,ty3_9_1,tc2_19_1,ty2_8_1,pc0_14_1,ty1_7_1,_t_3878,tx1_4_1,tx3_6_1,tx2_5_1,py1_13_1,pc2_16_1,px1_11_1,_t_203,tc0_17_1,py0_12_1,tc1_18_1,px0_10_1,_t_3769);

	return _t_204;
}
__device__ float tegpixellet_block_28(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty1_7_1,float tx1_4_1,float ty3_9_1,float _t_4942,float _t_4995,float ty2_8_1,float tx2_5_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float _t_231,float y__2869_1,float _t_4915){
	float _t_4996;
	float _t_4997;
	float _t_4998;
	float _t_4999;
	float _t_5000;
	float _t_5001;
	float _t_5002;
	float _t_5003;
	float _t_5004;
	float _t_5005;
	float _t_5006;
	float _t_5007;
	float _t_5008;
	float _t_5009;
	float _t_5010;
	float _t_5011;
	float _t_5012;
	float _t_5013;
	float _t_5014;
	float _t_5015;
	float _t_5016;
	float _t_5017;
	float _t_5018;
	bool _t_5019;
	float _t_5020;
	float _t_5021;
	float _t_5022;
	float _t_5023;
	float _t_5024;
	float _t_5025;
	float _t_5026;
	float _t_5027;
	float _t_5028;
	float _t_5029;
	float _t_5030;
	float _t_5031;
	float _t_5032;
	float _t_5033;
	bool _t_5034;
	float _t_5035;
	float _t_5036;
	float _t_5037;
	float _t_5038;
	bool _t_5039;
	bool _t_5040;
	bool _t_5041;
	bool _t_5042;
	bool _t_5043;
	bool _t_5044;
	bool _t_5045;
	float _t_5375;

	float _t_4916;

	_t_4996 = -1.0f * pc0_14_1;
	_t_4997 = tc0_17_1 + _t_4996;
	_t_4998 = _t_4997 * _t_4997;
	_t_4999 = -1.0f * pc1_15_1;
	_t_5000 = tc1_18_1 + _t_4999;
	_t_5001 = _t_5000 * _t_5000;
	_t_5002 = _t_4998 + _t_5001;
	_t_5003 = -1.0f * pc2_16_1;
	_t_5004 = tc2_19_1 + _t_5003;
	_t_5005 = _t_5004 * _t_5004;
	_t_5006 = _t_5002 + _t_5005;
	_t_5007 = tx3_6_1 * ty1_7_1;
	_t_5008 = tx1_4_1 * ty3_9_1;
	_t_5009 = _t_5008 * -1.0f;
	_t_5010 = _t_5007 + _t_5009;
	_t_5011 = -1.0f * ty1_7_1;
	_t_5012 = ty3_9_1 + _t_5011;
	_t_5013 = _t_5012 * _t_4942;
	_t_5014 = _t_5010 + _t_5013;
	_t_5015 = -1.0f * tx3_6_1;
	_t_5016 = tx1_4_1 + _t_5015;
	_t_5017 = _t_5016 * _t_4995;
	_t_5018 = _t_5014 + _t_5017;
	_t_5019 = _t_5018 < 0.0f;
	if(_t_5019)
		{
		
			_t_5020 = 1.0f;
		
		}
else
		{
		
			_t_5020 = 0.0f;
		
		}

	_t_5021 = _t_5006 * _t_5020;
	_t_5022 = tx1_4_1 * ty2_8_1;
	_t_5023 = tx2_5_1 * ty1_7_1;
	_t_5024 = _t_5023 * -1.0f;
	_t_5025 = _t_5022 + _t_5024;
	_t_5026 = -1.0f * ty2_8_1;
	_t_5027 = ty1_7_1 + _t_5026;
	_t_5028 = _t_5027 * _t_4942;
	_t_5029 = _t_5025 + _t_5028;
	_t_5030 = -1.0f * tx1_4_1;
	_t_5031 = tx2_5_1 + _t_5030;
	_t_5032 = _t_5031 * _t_4995;
	_t_5033 = _t_5029 + _t_5032;
	_t_5034 = _t_5033 < 0.0f;
	if(_t_5034)
		{
		
			_t_5035 = 1.0f;
		
		}
else
		{
		
			_t_5035 = 0.0f;
		
		}

	_t_5036 = _t_5021 * _t_5035;
	_t_5037 = _t_5036 * ty3_9_1;
	_t_5038 = _t_5037 * -1.0f;
	_t_5039 = py0_12_1 < _t_4995;
	_t_5040 = _t_4995 < py1_13_1;
	_t_5041 = _t_5039 && _t_5040;
	_t_5042 = px0_10_1 < _t_4942;
	_t_5043 = _t_4942 < px1_11_1;
	_t_5044 = _t_5042 && _t_5043;
	_t_5045 = _t_5041 && _t_5044;
	if(_t_5045)
		{
			float _t_5046;
			float _t_5047;
			float _t_5048;
			bool _t_5049;
			float _t_5052;
			float _t_5056;
			float _t_5057;
			float _t_5058;
			float _t_5059;
			float _t_5060;
			bool _t_5061;
			float _t_5064;
			float _t_5068;
			float _t_5069;
			bool _t_5070;
			float _t_5071;
			float _t_5072;
			float _t_5073;
			float _t_5074;
			float _t_5075;
			bool _t_5076;
			float _t_5079;
			float _t_5083;
			float _t_5084;
			float _t_5085;
			float _t_5086;
			bool _t_5087;
			float _t_5090;
			float _t_5094;
			float _t_5095;
			float _t_5096;
			float _t_5097;
			float _t_5098;
			bool _t_5099;
			float _t_5102;
			float _t_5106;
			float _t_5107;
			float _t_5108;
			float _t_5109;
			float _t_5110;
			float _t_5111;
			float _t_5112;
			float _t_5113;
			float _t_5114;
			bool _t_5115;
			float _t_5118;
			float _t_5122;
			float _t_5123;
			float _t_5124;
			float _t_5125;
			bool _t_5126;
			float _t_5129;
			float _t_5133;
			float _t_5134;
			float _t_5135;
			float _t_5136;
			float _t_5137;
			bool _t_5138;
			float _t_5141;
			float _t_5145;
			float _t_5146;
			float _t_5147;
			float _t_5148;
			float _t_5149;
			float _t_5150;
			bool _t_5151;
			float _t_5152;
			float _t_5153;
			float _t_5154;
			bool _t_5155;
			float _t_5156;
			float _t_5157;
			float _t_5158;
			bool _t_5159;
			float _t_5162;
			float _t_5166;
			float _t_5167;
			float _t_5168;
			float _t_5169;
			float _t_5170;
			bool _t_5171;
			float _t_5174;
			float _t_5178;
			float _t_5179;
			bool _t_5180;
			float _t_5181;
			float _t_5182;
			float _t_5183;
			float _t_5184;
			float _t_5185;
			bool _t_5186;
			float _t_5189;
			float _t_5193;
			float _t_5194;
			float _t_5195;
			float _t_5196;
			bool _t_5197;
			float _t_5200;
			float _t_5204;
			float _t_5205;
			float _t_5206;
			float _t_5207;
			float _t_5208;
			bool _t_5209;
			float _t_5212;
			float _t_5216;
			float _t_5217;
			float _t_5218;
			float _t_5219;
			float _t_5220;
			float _t_5221;
			float _t_5222;
			float _t_5223;
			float _t_5224;
			bool _t_5225;
			float _t_5228;
			float _t_5232;
			float _t_5233;
			float _t_5234;
			float _t_5235;
			bool _t_5236;
			float _t_5239;
			float _t_5243;
			float _t_5244;
			float _t_5245;
			float _t_5246;
			float _t_5247;
			bool _t_5248;
			float _t_5251;
			float _t_5255;
			float _t_5256;
			float _t_5257;
			float _t_5258;
			float _t_5259;
			float _t_5260;
			bool _t_5261;
			float _t_5262;
			float _t_5263;
			float _t_5264;
			bool _t_5265;
			bool _t_5266;
			float _t_5267;
			float _t_5268;
			float _t_5269;
			bool _t_5270;
			float _t_5273;
			float _t_5277;
			float _t_5278;
			float _t_5279;
			float _t_5280;
			bool _t_5281;
			float _t_5284;
			float _t_5288;
			bool _t_5289;
			float _t_5290;
			float _t_5291;
			float _t_5292;
			float _t_5293;
			float _t_5294;
			bool _t_5295;
			float _t_5298;
			float _t_5302;
			float _t_5303;
			float _t_5304;
			float _t_5305;
			bool _t_5306;
			float _t_5309;
			float _t_5313;
			bool _t_5314;
			float _t_5315;
			float _t_5316;
			float _t_5317;
			bool _t_5318;
			float _t_5319;
			float _t_5320;
			float _t_5321;
			bool _t_5322;
			float _t_5325;
			float _t_5329;
			float _t_5330;
			float _t_5331;
			float _t_5332;
			bool _t_5333;
			float _t_5336;
			float _t_5340;
			bool _t_5341;
			float _t_5342;
			float _t_5343;
			float _t_5344;
			float _t_5345;
			float _t_5346;
			bool _t_5347;
			float _t_5350;
			float _t_5354;
			float _t_5355;
			float _t_5356;
			float _t_5357;
			bool _t_5358;
			float _t_5361;
			float _t_5365;
			bool _t_5366;
			float _t_5367;
			float _t_5368;
			float _t_5369;
			bool _t_5370;
			bool _t_5371;
			bool _t_5372;
			float _t_5373;
			float _t_5374;
		
			_t_5046 = -1.0f * ty3_9_1;
			_t_5047 = ty2_8_1 + _t_5046;
			_t_5048 = -1.0f * _t_5047;
			_t_5049 = _t_5048 < 0.0f;
			if(_t_5049)
				{
					float _t_5050;
					float _t_5051;
				
					_t_5050 = -1.0f * tx2_5_1;
					_t_5051 = tx3_6_1 + _t_5050;
					_t_5052 = _t_5051;
				
				}
		else
				{
					float _t_5053;
					float _t_5054;
					float _t_5055;
				
					_t_5053 = -1.0f * tx2_5_1;
					_t_5054 = tx3_6_1 + _t_5053;
					_t_5055 = -1.0f * _t_5054;
					_t_5052 = _t_5055;
				
				}
		
			_t_5056 = _t_5052 * _t_231;
			_t_5057 = _t_5056 * -1.0f;
			_t_5058 = -1.0f * ty3_9_1;
			_t_5059 = ty2_8_1 + _t_5058;
			_t_5060 = -1.0f * _t_5059;
			_t_5061 = _t_5060 < 0.0f;
			if(_t_5061)
				{
					float _t_5062;
					float _t_5063;
				
					_t_5062 = -1.0f * tx2_5_1;
					_t_5063 = tx3_6_1 + _t_5062;
					_t_5064 = _t_5063;
				
				}
		else
				{
					float _t_5065;
					float _t_5066;
					float _t_5067;
				
					_t_5065 = -1.0f * tx2_5_1;
					_t_5066 = tx3_6_1 + _t_5065;
					_t_5067 = -1.0f * _t_5066;
					_t_5064 = _t_5067;
				
				}
		
			_t_5068 = _t_5064 * _t_231;
			_t_5069 = _t_5068 * -1.0f;
			_t_5070 = 0.0f < _t_5069;
			if(_t_5070)
				{
				
					_t_5071 = px0_10_1;
				
				}
		else
				{
				
					_t_5071 = px1_11_1;
				
				}
		
			_t_5072 = _t_5057 * _t_5071;
			_t_5073 = -1.0f * ty3_9_1;
			_t_5074 = ty2_8_1 + _t_5073;
			_t_5075 = -1.0f * _t_5074;
			_t_5076 = _t_5075 < 0.0f;
			if(_t_5076)
				{
					float _t_5077;
					float _t_5078;
				
					_t_5077 = -1.0f * tx2_5_1;
					_t_5078 = tx3_6_1 + _t_5077;
					_t_5079 = _t_5078;
				
				}
		else
				{
					float _t_5080;
					float _t_5081;
					float _t_5082;
				
					_t_5080 = -1.0f * tx2_5_1;
					_t_5081 = tx3_6_1 + _t_5080;
					_t_5082 = -1.0f * _t_5081;
					_t_5079 = _t_5082;
				
				}
		
			_t_5083 = _t_5079 * _t_231;
			_t_5084 = -1.0f * ty3_9_1;
			_t_5085 = ty2_8_1 + _t_5084;
			_t_5086 = -1.0f * _t_5085;
			_t_5087 = _t_5086 < 0.0f;
			if(_t_5087)
				{
					float _t_5088;
					float _t_5089;
				
					_t_5088 = -1.0f * tx2_5_1;
					_t_5089 = tx3_6_1 + _t_5088;
					_t_5090 = _t_5089;
				
				}
		else
				{
					float _t_5091;
					float _t_5092;
					float _t_5093;
				
					_t_5091 = -1.0f * tx2_5_1;
					_t_5092 = tx3_6_1 + _t_5091;
					_t_5093 = -1.0f * _t_5092;
					_t_5090 = _t_5093;
				
				}
		
			_t_5094 = _t_5090 * _t_231;
			_t_5095 = _t_5083 * _t_5094;
			_t_5096 = -1.0f * ty3_9_1;
			_t_5097 = ty2_8_1 + _t_5096;
			_t_5098 = -1.0f * _t_5097;
			_t_5099 = _t_5098 < 0.0f;
			if(_t_5099)
				{
					float _t_5100;
					float _t_5101;
				
					_t_5100 = -1.0f * ty3_9_1;
					_t_5101 = ty2_8_1 + _t_5100;
					_t_5102 = _t_5101;
				
				}
		else
				{
					float _t_5103;
					float _t_5104;
					float _t_5105;
				
					_t_5103 = -1.0f * ty3_9_1;
					_t_5104 = ty2_8_1 + _t_5103;
					_t_5105 = -1.0f * _t_5104;
					_t_5102 = _t_5105;
				
				}
		
			_t_5106 = _t_5102 * _t_231;
			_t_5107 = 1.0f + _t_5106;
			_t_5108 = 1.0f / _t_5107;
			_t_5109 = _t_5095 * _t_5108;
			_t_5110 = _t_5109 * -1.0f;
			_t_5111 = 1.0f + _t_5110;
			_t_5112 = -1.0f * ty3_9_1;
			_t_5113 = ty2_8_1 + _t_5112;
			_t_5114 = -1.0f * _t_5113;
			_t_5115 = _t_5114 < 0.0f;
			if(_t_5115)
				{
					float _t_5116;
					float _t_5117;
				
					_t_5116 = -1.0f * tx2_5_1;
					_t_5117 = tx3_6_1 + _t_5116;
					_t_5118 = _t_5117;
				
				}
		else
				{
					float _t_5119;
					float _t_5120;
					float _t_5121;
				
					_t_5119 = -1.0f * tx2_5_1;
					_t_5120 = tx3_6_1 + _t_5119;
					_t_5121 = -1.0f * _t_5120;
					_t_5118 = _t_5121;
				
				}
		
			_t_5122 = _t_5118 * _t_231;
			_t_5123 = -1.0f * ty3_9_1;
			_t_5124 = ty2_8_1 + _t_5123;
			_t_5125 = -1.0f * _t_5124;
			_t_5126 = _t_5125 < 0.0f;
			if(_t_5126)
				{
					float _t_5127;
					float _t_5128;
				
					_t_5127 = -1.0f * tx2_5_1;
					_t_5128 = tx3_6_1 + _t_5127;
					_t_5129 = _t_5128;
				
				}
		else
				{
					float _t_5130;
					float _t_5131;
					float _t_5132;
				
					_t_5130 = -1.0f * tx2_5_1;
					_t_5131 = tx3_6_1 + _t_5130;
					_t_5132 = -1.0f * _t_5131;
					_t_5129 = _t_5132;
				
				}
		
			_t_5133 = _t_5129 * _t_231;
			_t_5134 = _t_5122 * _t_5133;
			_t_5135 = -1.0f * ty3_9_1;
			_t_5136 = ty2_8_1 + _t_5135;
			_t_5137 = -1.0f * _t_5136;
			_t_5138 = _t_5137 < 0.0f;
			if(_t_5138)
				{
					float _t_5139;
					float _t_5140;
				
					_t_5139 = -1.0f * ty3_9_1;
					_t_5140 = ty2_8_1 + _t_5139;
					_t_5141 = _t_5140;
				
				}
		else
				{
					float _t_5142;
					float _t_5143;
					float _t_5144;
				
					_t_5142 = -1.0f * ty3_9_1;
					_t_5143 = ty2_8_1 + _t_5142;
					_t_5144 = -1.0f * _t_5143;
					_t_5141 = _t_5144;
				
				}
		
			_t_5145 = _t_5141 * _t_231;
			_t_5146 = 1.0f + _t_5145;
			_t_5147 = 1.0f / _t_5146;
			_t_5148 = _t_5134 * _t_5147;
			_t_5149 = _t_5148 * -1.0f;
			_t_5150 = 1.0f + _t_5149;
			_t_5151 = 0.0f < _t_5150;
			if(_t_5151)
				{
				
					_t_5152 = py0_12_1;
				
				}
		else
				{
				
					_t_5152 = py1_13_1;
				
				}
		
			_t_5153 = _t_5111 * _t_5152;
			_t_5154 = _t_5072 + _t_5153;
			_t_5155 = _t_5154 < y__2869_1;
			_t_5156 = -1.0f * ty3_9_1;
			_t_5157 = ty2_8_1 + _t_5156;
			_t_5158 = -1.0f * _t_5157;
			_t_5159 = _t_5158 < 0.0f;
			if(_t_5159)
				{
					float _t_5160;
					float _t_5161;
				
					_t_5160 = -1.0f * tx2_5_1;
					_t_5161 = tx3_6_1 + _t_5160;
					_t_5162 = _t_5161;
				
				}
		else
				{
					float _t_5163;
					float _t_5164;
					float _t_5165;
				
					_t_5163 = -1.0f * tx2_5_1;
					_t_5164 = tx3_6_1 + _t_5163;
					_t_5165 = -1.0f * _t_5164;
					_t_5162 = _t_5165;
				
				}
		
			_t_5166 = _t_5162 * _t_231;
			_t_5167 = _t_5166 * -1.0f;
			_t_5168 = -1.0f * ty3_9_1;
			_t_5169 = ty2_8_1 + _t_5168;
			_t_5170 = -1.0f * _t_5169;
			_t_5171 = _t_5170 < 0.0f;
			if(_t_5171)
				{
					float _t_5172;
					float _t_5173;
				
					_t_5172 = -1.0f * tx2_5_1;
					_t_5173 = tx3_6_1 + _t_5172;
					_t_5174 = _t_5173;
				
				}
		else
				{
					float _t_5175;
					float _t_5176;
					float _t_5177;
				
					_t_5175 = -1.0f * tx2_5_1;
					_t_5176 = tx3_6_1 + _t_5175;
					_t_5177 = -1.0f * _t_5176;
					_t_5174 = _t_5177;
				
				}
		
			_t_5178 = _t_5174 * _t_231;
			_t_5179 = _t_5178 * -1.0f;
			_t_5180 = 0.0f < _t_5179;
			if(_t_5180)
				{
				
					_t_5181 = px1_11_1;
				
				}
		else
				{
				
					_t_5181 = px0_10_1;
				
				}
		
			_t_5182 = _t_5167 * _t_5181;
			_t_5183 = -1.0f * ty3_9_1;
			_t_5184 = ty2_8_1 + _t_5183;
			_t_5185 = -1.0f * _t_5184;
			_t_5186 = _t_5185 < 0.0f;
			if(_t_5186)
				{
					float _t_5187;
					float _t_5188;
				
					_t_5187 = -1.0f * tx2_5_1;
					_t_5188 = tx3_6_1 + _t_5187;
					_t_5189 = _t_5188;
				
				}
		else
				{
					float _t_5190;
					float _t_5191;
					float _t_5192;
				
					_t_5190 = -1.0f * tx2_5_1;
					_t_5191 = tx3_6_1 + _t_5190;
					_t_5192 = -1.0f * _t_5191;
					_t_5189 = _t_5192;
				
				}
		
			_t_5193 = _t_5189 * _t_231;
			_t_5194 = -1.0f * ty3_9_1;
			_t_5195 = ty2_8_1 + _t_5194;
			_t_5196 = -1.0f * _t_5195;
			_t_5197 = _t_5196 < 0.0f;
			if(_t_5197)
				{
					float _t_5198;
					float _t_5199;
				
					_t_5198 = -1.0f * tx2_5_1;
					_t_5199 = tx3_6_1 + _t_5198;
					_t_5200 = _t_5199;
				
				}
		else
				{
					float _t_5201;
					float _t_5202;
					float _t_5203;
				
					_t_5201 = -1.0f * tx2_5_1;
					_t_5202 = tx3_6_1 + _t_5201;
					_t_5203 = -1.0f * _t_5202;
					_t_5200 = _t_5203;
				
				}
		
			_t_5204 = _t_5200 * _t_231;
			_t_5205 = _t_5193 * _t_5204;
			_t_5206 = -1.0f * ty3_9_1;
			_t_5207 = ty2_8_1 + _t_5206;
			_t_5208 = -1.0f * _t_5207;
			_t_5209 = _t_5208 < 0.0f;
			if(_t_5209)
				{
					float _t_5210;
					float _t_5211;
				
					_t_5210 = -1.0f * ty3_9_1;
					_t_5211 = ty2_8_1 + _t_5210;
					_t_5212 = _t_5211;
				
				}
		else
				{
					float _t_5213;
					float _t_5214;
					float _t_5215;
				
					_t_5213 = -1.0f * ty3_9_1;
					_t_5214 = ty2_8_1 + _t_5213;
					_t_5215 = -1.0f * _t_5214;
					_t_5212 = _t_5215;
				
				}
		
			_t_5216 = _t_5212 * _t_231;
			_t_5217 = 1.0f + _t_5216;
			_t_5218 = 1.0f / _t_5217;
			_t_5219 = _t_5205 * _t_5218;
			_t_5220 = _t_5219 * -1.0f;
			_t_5221 = 1.0f + _t_5220;
			_t_5222 = -1.0f * ty3_9_1;
			_t_5223 = ty2_8_1 + _t_5222;
			_t_5224 = -1.0f * _t_5223;
			_t_5225 = _t_5224 < 0.0f;
			if(_t_5225)
				{
					float _t_5226;
					float _t_5227;
				
					_t_5226 = -1.0f * tx2_5_1;
					_t_5227 = tx3_6_1 + _t_5226;
					_t_5228 = _t_5227;
				
				}
		else
				{
					float _t_5229;
					float _t_5230;
					float _t_5231;
				
					_t_5229 = -1.0f * tx2_5_1;
					_t_5230 = tx3_6_1 + _t_5229;
					_t_5231 = -1.0f * _t_5230;
					_t_5228 = _t_5231;
				
				}
		
			_t_5232 = _t_5228 * _t_231;
			_t_5233 = -1.0f * ty3_9_1;
			_t_5234 = ty2_8_1 + _t_5233;
			_t_5235 = -1.0f * _t_5234;
			_t_5236 = _t_5235 < 0.0f;
			if(_t_5236)
				{
					float _t_5237;
					float _t_5238;
				
					_t_5237 = -1.0f * tx2_5_1;
					_t_5238 = tx3_6_1 + _t_5237;
					_t_5239 = _t_5238;
				
				}
		else
				{
					float _t_5240;
					float _t_5241;
					float _t_5242;
				
					_t_5240 = -1.0f * tx2_5_1;
					_t_5241 = tx3_6_1 + _t_5240;
					_t_5242 = -1.0f * _t_5241;
					_t_5239 = _t_5242;
				
				}
		
			_t_5243 = _t_5239 * _t_231;
			_t_5244 = _t_5232 * _t_5243;
			_t_5245 = -1.0f * ty3_9_1;
			_t_5246 = ty2_8_1 + _t_5245;
			_t_5247 = -1.0f * _t_5246;
			_t_5248 = _t_5247 < 0.0f;
			if(_t_5248)
				{
					float _t_5249;
					float _t_5250;
				
					_t_5249 = -1.0f * ty3_9_1;
					_t_5250 = ty2_8_1 + _t_5249;
					_t_5251 = _t_5250;
				
				}
		else
				{
					float _t_5252;
					float _t_5253;
					float _t_5254;
				
					_t_5252 = -1.0f * ty3_9_1;
					_t_5253 = ty2_8_1 + _t_5252;
					_t_5254 = -1.0f * _t_5253;
					_t_5251 = _t_5254;
				
				}
		
			_t_5255 = _t_5251 * _t_231;
			_t_5256 = 1.0f + _t_5255;
			_t_5257 = 1.0f / _t_5256;
			_t_5258 = _t_5244 * _t_5257;
			_t_5259 = _t_5258 * -1.0f;
			_t_5260 = 1.0f + _t_5259;
			_t_5261 = 0.0f < _t_5260;
			if(_t_5261)
				{
				
					_t_5262 = py1_13_1;
				
				}
		else
				{
				
					_t_5262 = py0_12_1;
				
				}
		
			_t_5263 = _t_5221 * _t_5262;
			_t_5264 = _t_5182 + _t_5263;
			_t_5265 = y__2869_1 < _t_5264;
			_t_5266 = _t_5155 && _t_5265;
			_t_5267 = -1.0f * ty3_9_1;
			_t_5268 = ty2_8_1 + _t_5267;
			_t_5269 = -1.0f * _t_5268;
			_t_5270 = _t_5269 < 0.0f;
			if(_t_5270)
				{
					float _t_5271;
					float _t_5272;
				
					_t_5271 = -1.0f * ty3_9_1;
					_t_5272 = ty2_8_1 + _t_5271;
					_t_5273 = _t_5272;
				
				}
		else
				{
					float _t_5274;
					float _t_5275;
					float _t_5276;
				
					_t_5274 = -1.0f * ty3_9_1;
					_t_5275 = ty2_8_1 + _t_5274;
					_t_5276 = -1.0f * _t_5275;
					_t_5273 = _t_5276;
				
				}
		
			_t_5277 = _t_5273 * _t_231;
			_t_5278 = -1.0f * ty3_9_1;
			_t_5279 = ty2_8_1 + _t_5278;
			_t_5280 = -1.0f * _t_5279;
			_t_5281 = _t_5280 < 0.0f;
			if(_t_5281)
				{
					float _t_5282;
					float _t_5283;
				
					_t_5282 = -1.0f * ty3_9_1;
					_t_5283 = ty2_8_1 + _t_5282;
					_t_5284 = _t_5283;
				
				}
		else
				{
					float _t_5285;
					float _t_5286;
					float _t_5287;
				
					_t_5285 = -1.0f * ty3_9_1;
					_t_5286 = ty2_8_1 + _t_5285;
					_t_5287 = -1.0f * _t_5286;
					_t_5284 = _t_5287;
				
				}
		
			_t_5288 = _t_5284 * _t_231;
			_t_5289 = 0.0f < _t_5288;
			if(_t_5289)
				{
				
					_t_5290 = px0_10_1;
				
				}
		else
				{
				
					_t_5290 = px1_11_1;
				
				}
		
			_t_5291 = _t_5277 * _t_5290;
			_t_5292 = -1.0f * ty3_9_1;
			_t_5293 = ty2_8_1 + _t_5292;
			_t_5294 = -1.0f * _t_5293;
			_t_5295 = _t_5294 < 0.0f;
			if(_t_5295)
				{
					float _t_5296;
					float _t_5297;
				
					_t_5296 = -1.0f * tx2_5_1;
					_t_5297 = tx3_6_1 + _t_5296;
					_t_5298 = _t_5297;
				
				}
		else
				{
					float _t_5299;
					float _t_5300;
					float _t_5301;
				
					_t_5299 = -1.0f * tx2_5_1;
					_t_5300 = tx3_6_1 + _t_5299;
					_t_5301 = -1.0f * _t_5300;
					_t_5298 = _t_5301;
				
				}
		
			_t_5302 = _t_5298 * _t_231;
			_t_5303 = -1.0f * ty3_9_1;
			_t_5304 = ty2_8_1 + _t_5303;
			_t_5305 = -1.0f * _t_5304;
			_t_5306 = _t_5305 < 0.0f;
			if(_t_5306)
				{
					float _t_5307;
					float _t_5308;
				
					_t_5307 = -1.0f * tx2_5_1;
					_t_5308 = tx3_6_1 + _t_5307;
					_t_5309 = _t_5308;
				
				}
		else
				{
					float _t_5310;
					float _t_5311;
					float _t_5312;
				
					_t_5310 = -1.0f * tx2_5_1;
					_t_5311 = tx3_6_1 + _t_5310;
					_t_5312 = -1.0f * _t_5311;
					_t_5309 = _t_5312;
				
				}
		
			_t_5313 = _t_5309 * _t_231;
			_t_5314 = 0.0f < _t_5313;
			if(_t_5314)
				{
				
					_t_5315 = py0_12_1;
				
				}
		else
				{
				
					_t_5315 = py1_13_1;
				
				}
		
			_t_5316 = _t_5302 * _t_5315;
			_t_5317 = _t_5291 + _t_5316;
			_t_5318 = _t_5317 < _t_4915;
			_t_5319 = -1.0f * ty3_9_1;
			_t_5320 = ty2_8_1 + _t_5319;
			_t_5321 = -1.0f * _t_5320;
			_t_5322 = _t_5321 < 0.0f;
			if(_t_5322)
				{
					float _t_5323;
					float _t_5324;
				
					_t_5323 = -1.0f * ty3_9_1;
					_t_5324 = ty2_8_1 + _t_5323;
					_t_5325 = _t_5324;
				
				}
		else
				{
					float _t_5326;
					float _t_5327;
					float _t_5328;
				
					_t_5326 = -1.0f * ty3_9_1;
					_t_5327 = ty2_8_1 + _t_5326;
					_t_5328 = -1.0f * _t_5327;
					_t_5325 = _t_5328;
				
				}
		
			_t_5329 = _t_5325 * _t_231;
			_t_5330 = -1.0f * ty3_9_1;
			_t_5331 = ty2_8_1 + _t_5330;
			_t_5332 = -1.0f * _t_5331;
			_t_5333 = _t_5332 < 0.0f;
			if(_t_5333)
				{
					float _t_5334;
					float _t_5335;
				
					_t_5334 = -1.0f * ty3_9_1;
					_t_5335 = ty2_8_1 + _t_5334;
					_t_5336 = _t_5335;
				
				}
		else
				{
					float _t_5337;
					float _t_5338;
					float _t_5339;
				
					_t_5337 = -1.0f * ty3_9_1;
					_t_5338 = ty2_8_1 + _t_5337;
					_t_5339 = -1.0f * _t_5338;
					_t_5336 = _t_5339;
				
				}
		
			_t_5340 = _t_5336 * _t_231;
			_t_5341 = 0.0f < _t_5340;
			if(_t_5341)
				{
				
					_t_5342 = px1_11_1;
				
				}
		else
				{
				
					_t_5342 = px0_10_1;
				
				}
		
			_t_5343 = _t_5329 * _t_5342;
			_t_5344 = -1.0f * ty3_9_1;
			_t_5345 = ty2_8_1 + _t_5344;
			_t_5346 = -1.0f * _t_5345;
			_t_5347 = _t_5346 < 0.0f;
			if(_t_5347)
				{
					float _t_5348;
					float _t_5349;
				
					_t_5348 = -1.0f * tx2_5_1;
					_t_5349 = tx3_6_1 + _t_5348;
					_t_5350 = _t_5349;
				
				}
		else
				{
					float _t_5351;
					float _t_5352;
					float _t_5353;
				
					_t_5351 = -1.0f * tx2_5_1;
					_t_5352 = tx3_6_1 + _t_5351;
					_t_5353 = -1.0f * _t_5352;
					_t_5350 = _t_5353;
				
				}
		
			_t_5354 = _t_5350 * _t_231;
			_t_5355 = -1.0f * ty3_9_1;
			_t_5356 = ty2_8_1 + _t_5355;
			_t_5357 = -1.0f * _t_5356;
			_t_5358 = _t_5357 < 0.0f;
			if(_t_5358)
				{
					float _t_5359;
					float _t_5360;
				
					_t_5359 = -1.0f * tx2_5_1;
					_t_5360 = tx3_6_1 + _t_5359;
					_t_5361 = _t_5360;
				
				}
		else
				{
					float _t_5362;
					float _t_5363;
					float _t_5364;
				
					_t_5362 = -1.0f * tx2_5_1;
					_t_5363 = tx3_6_1 + _t_5362;
					_t_5364 = -1.0f * _t_5363;
					_t_5361 = _t_5364;
				
				}
		
			_t_5365 = _t_5361 * _t_231;
			_t_5366 = 0.0f < _t_5365;
			if(_t_5366)
				{
				
					_t_5367 = py1_13_1;
				
				}
		else
				{
				
					_t_5367 = py0_12_1;
				
				}
		
			_t_5368 = _t_5354 * _t_5367;
			_t_5369 = _t_5343 + _t_5368;
			_t_5370 = _t_4915 < _t_5369;
			_t_5371 = _t_5318 && _t_5370;
			_t_5372 = _t_5266 && _t_5371;
			if(_t_5372)
				{
				
					_t_5373 = 1.0f;
				
				}
		else
				{
				
					_t_5373 = 0.0f;
				
				}
		
			_t_5374 = _t_5373 * _t_231;
			_t_5375 = _t_5374;
		
		}
else
		{
		
			_t_5375 = 0.0f;
		
		}

	_t_4916 = _t_5038 * _t_5375;

	return _t_4916;
}
__device__ float tegpixellet_block_27(float ty2_8_1,float ty3_9_1,float _t_231,float _t_4915,float tx3_6_1,float tx2_5_1,float y__2869_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1){
	float _t_4917;
	float _t_4918;
	float _t_4919;
	bool _t_4920;
	float _t_4923;
	float _t_4927;
	float _t_4928;
	float _t_4929;
	float _t_4930;
	float _t_4931;
	bool _t_4932;
	float _t_4935;
	float _t_4939;
	float _t_4940;
	float _t_4941;
	float _t_4942;
	float _t_4943;
	float _t_4944;
	float _t_4945;
	bool _t_4946;
	float _t_4949;
	float _t_4953;
	float _t_4954;
	float _t_4955;
	float _t_4956;
	bool _t_4957;
	float _t_4960;
	float _t_4964;
	float _t_4965;
	float _t_4966;
	float _t_4967;
	float _t_4968;
	bool _t_4969;
	float _t_4972;
	float _t_4976;
	float _t_4977;
	float _t_4978;
	float _t_4979;
	float _t_4980;
	float _t_4981;
	float _t_4982;
	float _t_4983;
	float _t_4984;
	float _t_4985;
	bool _t_4986;
	float _t_4989;
	float _t_4993;
	float _t_4994;
	float _t_4995;

	float _t_4916;

	_t_4917 = -1.0f * ty3_9_1;
	_t_4918 = ty2_8_1 + _t_4917;
	_t_4919 = -1.0f * _t_4918;
	_t_4920 = _t_4919 < 0.0f;
	if(_t_4920)
		{
			float _t_4921;
			float _t_4922;
		
			_t_4921 = -1.0f * ty3_9_1;
			_t_4922 = ty2_8_1 + _t_4921;
			_t_4923 = _t_4922;
		
		}
else
		{
			float _t_4924;
			float _t_4925;
			float _t_4926;
		
			_t_4924 = -1.0f * ty3_9_1;
			_t_4925 = ty2_8_1 + _t_4924;
			_t_4926 = -1.0f * _t_4925;
			_t_4923 = _t_4926;
		
		}

	_t_4927 = _t_4923 * _t_231;
	_t_4928 = _t_4927 * _t_4915;
	_t_4929 = -1.0f * ty3_9_1;
	_t_4930 = ty2_8_1 + _t_4929;
	_t_4931 = -1.0f * _t_4930;
	_t_4932 = _t_4931 < 0.0f;
	if(_t_4932)
		{
			float _t_4933;
			float _t_4934;
		
			_t_4933 = -1.0f * tx2_5_1;
			_t_4934 = tx3_6_1 + _t_4933;
			_t_4935 = _t_4934;
		
		}
else
		{
			float _t_4936;
			float _t_4937;
			float _t_4938;
		
			_t_4936 = -1.0f * tx2_5_1;
			_t_4937 = tx3_6_1 + _t_4936;
			_t_4938 = -1.0f * _t_4937;
			_t_4935 = _t_4938;
		
		}

	_t_4939 = _t_4935 * _t_231;
	_t_4940 = _t_4939 * -1.0f;
	_t_4941 = _t_4940 * y__2869_1;
	_t_4942 = _t_4928 + _t_4941;
	_t_4943 = -1.0f * ty3_9_1;
	_t_4944 = ty2_8_1 + _t_4943;
	_t_4945 = -1.0f * _t_4944;
	_t_4946 = _t_4945 < 0.0f;
	if(_t_4946)
		{
			float _t_4947;
			float _t_4948;
		
			_t_4947 = -1.0f * tx2_5_1;
			_t_4948 = tx3_6_1 + _t_4947;
			_t_4949 = _t_4948;
		
		}
else
		{
			float _t_4950;
			float _t_4951;
			float _t_4952;
		
			_t_4950 = -1.0f * tx2_5_1;
			_t_4951 = tx3_6_1 + _t_4950;
			_t_4952 = -1.0f * _t_4951;
			_t_4949 = _t_4952;
		
		}

	_t_4953 = _t_4949 * _t_231;
	_t_4954 = -1.0f * ty3_9_1;
	_t_4955 = ty2_8_1 + _t_4954;
	_t_4956 = -1.0f * _t_4955;
	_t_4957 = _t_4956 < 0.0f;
	if(_t_4957)
		{
			float _t_4958;
			float _t_4959;
		
			_t_4958 = -1.0f * tx2_5_1;
			_t_4959 = tx3_6_1 + _t_4958;
			_t_4960 = _t_4959;
		
		}
else
		{
			float _t_4961;
			float _t_4962;
			float _t_4963;
		
			_t_4961 = -1.0f * tx2_5_1;
			_t_4962 = tx3_6_1 + _t_4961;
			_t_4963 = -1.0f * _t_4962;
			_t_4960 = _t_4963;
		
		}

	_t_4964 = _t_4960 * _t_231;
	_t_4965 = _t_4953 * _t_4964;
	_t_4966 = -1.0f * ty3_9_1;
	_t_4967 = ty2_8_1 + _t_4966;
	_t_4968 = -1.0f * _t_4967;
	_t_4969 = _t_4968 < 0.0f;
	if(_t_4969)
		{
			float _t_4970;
			float _t_4971;
		
			_t_4970 = -1.0f * ty3_9_1;
			_t_4971 = ty2_8_1 + _t_4970;
			_t_4972 = _t_4971;
		
		}
else
		{
			float _t_4973;
			float _t_4974;
			float _t_4975;
		
			_t_4973 = -1.0f * ty3_9_1;
			_t_4974 = ty2_8_1 + _t_4973;
			_t_4975 = -1.0f * _t_4974;
			_t_4972 = _t_4975;
		
		}

	_t_4976 = _t_4972 * _t_231;
	_t_4977 = 1.0f + _t_4976;
	_t_4978 = 1.0f / _t_4977;
	_t_4979 = _t_4965 * _t_4978;
	_t_4980 = _t_4979 * -1.0f;
	_t_4981 = 1.0f + _t_4980;
	_t_4982 = _t_4981 * y__2869_1;
	_t_4983 = -1.0f * ty3_9_1;
	_t_4984 = ty2_8_1 + _t_4983;
	_t_4985 = -1.0f * _t_4984;
	_t_4986 = _t_4985 < 0.0f;
	if(_t_4986)
		{
			float _t_4987;
			float _t_4988;
		
			_t_4987 = -1.0f * tx2_5_1;
			_t_4988 = tx3_6_1 + _t_4987;
			_t_4989 = _t_4988;
		
		}
else
		{
			float _t_4990;
			float _t_4991;
			float _t_4992;
		
			_t_4990 = -1.0f * tx2_5_1;
			_t_4991 = tx3_6_1 + _t_4990;
			_t_4992 = -1.0f * _t_4991;
			_t_4989 = _t_4992;
		
		}

	_t_4993 = _t_4989 * _t_231;
	_t_4994 = _t_4993 * _t_4915;
	_t_4995 = _t_4982 + _t_4994;
	_t_4916 = tegpixellet_block_28(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty1_7_1,tx1_4_1,ty3_9_1,_t_4942,_t_4995,ty2_8_1,tx2_5_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,_t_231,y__2869_1,_t_4915);

	return _t_4916;
}
__device__ float tegpixelbody_block_21(float ty2_8_1,float ty3_9_1,float _t_231,float px0_10_1,float px1_11_1,float tx3_6_1,float tx2_5_1,float py0_12_1,float py1_13_1,float y__2869_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_4759;
	float _t_4760;
	float _t_4761;
	bool _t_4762;
	float _t_4765;
	float _t_4769;
	float _t_4770;
	float _t_4771;
	float _t_4772;
	bool _t_4773;
	float _t_4776;
	float _t_4780;
	bool _t_4781;
	float _t_4782;
	float _t_4783;
	float _t_4784;
	float _t_4785;
	float _t_4786;
	bool _t_4787;
	float _t_4790;
	float _t_4794;
	float _t_4795;
	float _t_4796;
	float _t_4797;
	bool _t_4798;
	float _t_4801;
	float _t_4805;
	bool _t_4806;
	float _t_4807;
	float _t_4808;
	float _t_4809;
	float _t_4810;
	float _t_4811;
	float _t_4812;
	bool _t_4813;
	float _t_4818;
	float _t_4824;
	float _t_4825;
	float _t_4826;
	float _t_4827;
	bool _t_4828;
	float _t_4829;
	float _t_4830;
	float _t_4831;
	bool _t_4832;
	float _t_4835;
	float _t_4839;
	float _t_4840;
	float _t_4841;
	float _t_4842;
	bool _t_4843;
	float _t_4846;
	float _t_4850;
	bool _t_4851;
	float _t_4852;
	float _t_4853;
	float _t_4854;
	float _t_4855;
	float _t_4856;
	bool _t_4857;
	float _t_4860;
	float _t_4864;
	float _t_4865;
	float _t_4866;
	float _t_4867;
	bool _t_4868;
	float _t_4871;
	float _t_4875;
	bool _t_4876;
	float _t_4877;
	float _t_4878;
	float _t_4879;
	float _t_4880;
	float _t_4881;
	float _t_4882;
	bool _t_4883;
	float _t_4888;
	float _t_4894;
	float _t_4895;
	float _t_4896;
	float _t_4897;
	bool _t_4898;
	bool _t_4899;

	float _t_4758;

	_t_4759 = -1.0f * ty3_9_1;
	_t_4760 = ty2_8_1 + _t_4759;
	_t_4761 = -1.0f * _t_4760;
	_t_4762 = _t_4761 < 0.0f;
	if(_t_4762)
		{
			float _t_4763;
			float _t_4764;
		
			_t_4763 = -1.0f * ty3_9_1;
			_t_4764 = ty2_8_1 + _t_4763;
			_t_4765 = _t_4764;
		
		}
else
		{
			float _t_4766;
			float _t_4767;
			float _t_4768;
		
			_t_4766 = -1.0f * ty3_9_1;
			_t_4767 = ty2_8_1 + _t_4766;
			_t_4768 = -1.0f * _t_4767;
			_t_4765 = _t_4768;
		
		}

	_t_4769 = _t_4765 * _t_231;
	_t_4770 = -1.0f * ty3_9_1;
	_t_4771 = ty2_8_1 + _t_4770;
	_t_4772 = -1.0f * _t_4771;
	_t_4773 = _t_4772 < 0.0f;
	if(_t_4773)
		{
			float _t_4774;
			float _t_4775;
		
			_t_4774 = -1.0f * ty3_9_1;
			_t_4775 = ty2_8_1 + _t_4774;
			_t_4776 = _t_4775;
		
		}
else
		{
			float _t_4777;
			float _t_4778;
			float _t_4779;
		
			_t_4777 = -1.0f * ty3_9_1;
			_t_4778 = ty2_8_1 + _t_4777;
			_t_4779 = -1.0f * _t_4778;
			_t_4776 = _t_4779;
		
		}

	_t_4780 = _t_4776 * _t_231;
	_t_4781 = 0.0f < _t_4780;
	if(_t_4781)
		{
		
			_t_4782 = px0_10_1;
		
		}
else
		{
		
			_t_4782 = px1_11_1;
		
		}

	_t_4783 = _t_4769 * _t_4782;
	_t_4784 = -1.0f * ty3_9_1;
	_t_4785 = ty2_8_1 + _t_4784;
	_t_4786 = -1.0f * _t_4785;
	_t_4787 = _t_4786 < 0.0f;
	if(_t_4787)
		{
			float _t_4788;
			float _t_4789;
		
			_t_4788 = -1.0f * tx2_5_1;
			_t_4789 = tx3_6_1 + _t_4788;
			_t_4790 = _t_4789;
		
		}
else
		{
			float _t_4791;
			float _t_4792;
			float _t_4793;
		
			_t_4791 = -1.0f * tx2_5_1;
			_t_4792 = tx3_6_1 + _t_4791;
			_t_4793 = -1.0f * _t_4792;
			_t_4790 = _t_4793;
		
		}

	_t_4794 = _t_4790 * _t_231;
	_t_4795 = -1.0f * ty3_9_1;
	_t_4796 = ty2_8_1 + _t_4795;
	_t_4797 = -1.0f * _t_4796;
	_t_4798 = _t_4797 < 0.0f;
	if(_t_4798)
		{
			float _t_4799;
			float _t_4800;
		
			_t_4799 = -1.0f * tx2_5_1;
			_t_4800 = tx3_6_1 + _t_4799;
			_t_4801 = _t_4800;
		
		}
else
		{
			float _t_4802;
			float _t_4803;
			float _t_4804;
		
			_t_4802 = -1.0f * tx2_5_1;
			_t_4803 = tx3_6_1 + _t_4802;
			_t_4804 = -1.0f * _t_4803;
			_t_4801 = _t_4804;
		
		}

	_t_4805 = _t_4801 * _t_231;
	_t_4806 = 0.0f < _t_4805;
	if(_t_4806)
		{
		
			_t_4807 = py0_12_1;
		
		}
else
		{
		
			_t_4807 = py1_13_1;
		
		}

	_t_4808 = _t_4794 * _t_4807;
	_t_4809 = _t_4783 + _t_4808;
	_t_4810 = -1.0f * ty3_9_1;
	_t_4811 = ty2_8_1 + _t_4810;
	_t_4812 = -1.0f * _t_4811;
	_t_4813 = _t_4812 < 0.0f;
	if(_t_4813)
		{
			float _t_4814;
			float _t_4815;
			float _t_4816;
			float _t_4817;
		
			_t_4814 = tx2_5_1 * ty3_9_1;
			_t_4815 = tx3_6_1 * ty2_8_1;
			_t_4816 = _t_4815 * -1.0f;
			_t_4817 = _t_4814 + _t_4816;
			_t_4818 = _t_4817;
		
		}
else
		{
			float _t_4819;
			float _t_4820;
			float _t_4821;
			float _t_4822;
			float _t_4823;
		
			_t_4819 = tx2_5_1 * ty3_9_1;
			_t_4820 = tx3_6_1 * ty2_8_1;
			_t_4821 = _t_4820 * -1.0f;
			_t_4822 = _t_4819 + _t_4821;
			_t_4823 = -1.0f * _t_4822;
			_t_4818 = _t_4823;
		
		}

	_t_4824 = -1.0f * _t_4818;
	_t_4825 = _t_4824 * _t_231;
	_t_4826 = _t_4825 * -1.0f;
	_t_4827 = _t_4809 + _t_4826;
	_t_4828 = _t_4827 < 0.0f;
	_t_4829 = -1.0f * ty3_9_1;
	_t_4830 = ty2_8_1 + _t_4829;
	_t_4831 = -1.0f * _t_4830;
	_t_4832 = _t_4831 < 0.0f;
	if(_t_4832)
		{
			float _t_4833;
			float _t_4834;
		
			_t_4833 = -1.0f * ty3_9_1;
			_t_4834 = ty2_8_1 + _t_4833;
			_t_4835 = _t_4834;
		
		}
else
		{
			float _t_4836;
			float _t_4837;
			float _t_4838;
		
			_t_4836 = -1.0f * ty3_9_1;
			_t_4837 = ty2_8_1 + _t_4836;
			_t_4838 = -1.0f * _t_4837;
			_t_4835 = _t_4838;
		
		}

	_t_4839 = _t_4835 * _t_231;
	_t_4840 = -1.0f * ty3_9_1;
	_t_4841 = ty2_8_1 + _t_4840;
	_t_4842 = -1.0f * _t_4841;
	_t_4843 = _t_4842 < 0.0f;
	if(_t_4843)
		{
			float _t_4844;
			float _t_4845;
		
			_t_4844 = -1.0f * ty3_9_1;
			_t_4845 = ty2_8_1 + _t_4844;
			_t_4846 = _t_4845;
		
		}
else
		{
			float _t_4847;
			float _t_4848;
			float _t_4849;
		
			_t_4847 = -1.0f * ty3_9_1;
			_t_4848 = ty2_8_1 + _t_4847;
			_t_4849 = -1.0f * _t_4848;
			_t_4846 = _t_4849;
		
		}

	_t_4850 = _t_4846 * _t_231;
	_t_4851 = 0.0f < _t_4850;
	if(_t_4851)
		{
		
			_t_4852 = px1_11_1;
		
		}
else
		{
		
			_t_4852 = px0_10_1;
		
		}

	_t_4853 = _t_4839 * _t_4852;
	_t_4854 = -1.0f * ty3_9_1;
	_t_4855 = ty2_8_1 + _t_4854;
	_t_4856 = -1.0f * _t_4855;
	_t_4857 = _t_4856 < 0.0f;
	if(_t_4857)
		{
			float _t_4858;
			float _t_4859;
		
			_t_4858 = -1.0f * tx2_5_1;
			_t_4859 = tx3_6_1 + _t_4858;
			_t_4860 = _t_4859;
		
		}
else
		{
			float _t_4861;
			float _t_4862;
			float _t_4863;
		
			_t_4861 = -1.0f * tx2_5_1;
			_t_4862 = tx3_6_1 + _t_4861;
			_t_4863 = -1.0f * _t_4862;
			_t_4860 = _t_4863;
		
		}

	_t_4864 = _t_4860 * _t_231;
	_t_4865 = -1.0f * ty3_9_1;
	_t_4866 = ty2_8_1 + _t_4865;
	_t_4867 = -1.0f * _t_4866;
	_t_4868 = _t_4867 < 0.0f;
	if(_t_4868)
		{
			float _t_4869;
			float _t_4870;
		
			_t_4869 = -1.0f * tx2_5_1;
			_t_4870 = tx3_6_1 + _t_4869;
			_t_4871 = _t_4870;
		
		}
else
		{
			float _t_4872;
			float _t_4873;
			float _t_4874;
		
			_t_4872 = -1.0f * tx2_5_1;
			_t_4873 = tx3_6_1 + _t_4872;
			_t_4874 = -1.0f * _t_4873;
			_t_4871 = _t_4874;
		
		}

	_t_4875 = _t_4871 * _t_231;
	_t_4876 = 0.0f < _t_4875;
	if(_t_4876)
		{
		
			_t_4877 = py1_13_1;
		
		}
else
		{
		
			_t_4877 = py0_12_1;
		
		}

	_t_4878 = _t_4864 * _t_4877;
	_t_4879 = _t_4853 + _t_4878;
	_t_4880 = -1.0f * ty3_9_1;
	_t_4881 = ty2_8_1 + _t_4880;
	_t_4882 = -1.0f * _t_4881;
	_t_4883 = _t_4882 < 0.0f;
	if(_t_4883)
		{
			float _t_4884;
			float _t_4885;
			float _t_4886;
			float _t_4887;
		
			_t_4884 = tx2_5_1 * ty3_9_1;
			_t_4885 = tx3_6_1 * ty2_8_1;
			_t_4886 = _t_4885 * -1.0f;
			_t_4887 = _t_4884 + _t_4886;
			_t_4888 = _t_4887;
		
		}
else
		{
			float _t_4889;
			float _t_4890;
			float _t_4891;
			float _t_4892;
			float _t_4893;
		
			_t_4889 = tx2_5_1 * ty3_9_1;
			_t_4890 = tx3_6_1 * ty2_8_1;
			_t_4891 = _t_4890 * -1.0f;
			_t_4892 = _t_4889 + _t_4891;
			_t_4893 = -1.0f * _t_4892;
			_t_4888 = _t_4893;
		
		}

	_t_4894 = -1.0f * _t_4888;
	_t_4895 = _t_4894 * _t_231;
	_t_4896 = _t_4895 * -1.0f;
	_t_4897 = _t_4879 + _t_4896;
	_t_4898 = 0.0f < _t_4897;
	_t_4899 = _t_4828 && _t_4898;
	if(_t_4899)
		{
			float _t_4900;
			float _t_4901;
			float _t_4902;
			bool _t_4903;
			float _t_4908;
			float _t_4914;
			float _t_4915;
			float _t_4916;
		
			_t_4900 = -1.0f * ty3_9_1;
			_t_4901 = ty2_8_1 + _t_4900;
			_t_4902 = -1.0f * _t_4901;
			_t_4903 = _t_4902 < 0.0f;
			if(_t_4903)
				{
					float _t_4904;
					float _t_4905;
					float _t_4906;
					float _t_4907;
				
					_t_4904 = tx2_5_1 * ty3_9_1;
					_t_4905 = tx3_6_1 * ty2_8_1;
					_t_4906 = _t_4905 * -1.0f;
					_t_4907 = _t_4904 + _t_4906;
					_t_4908 = _t_4907;
				
				}
		else
				{
					float _t_4909;
					float _t_4910;
					float _t_4911;
					float _t_4912;
					float _t_4913;
				
					_t_4909 = tx2_5_1 * ty3_9_1;
					_t_4910 = tx3_6_1 * ty2_8_1;
					_t_4911 = _t_4910 * -1.0f;
					_t_4912 = _t_4909 + _t_4911;
					_t_4913 = -1.0f * _t_4912;
					_t_4908 = _t_4913;
				
				}
		
			_t_4914 = -1.0f * _t_4908;
			_t_4915 = _t_4914 * _t_231;
			_t_4916 = tegpixellet_block_27(ty2_8_1,ty3_9_1,_t_231,_t_4915,tx3_6_1,tx2_5_1,y__2869_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1);
			_t_4758 = _t_4916;
		
		}
else
		{
		
			_t_4758 = 0.0f;
		
		}


	return _t_4758;
}
__device__ float tegpixelintegrator_21(float ty3_9_1,float pc1_15_1,float _t_4757,float _t_231,float tc2_19_1,float ty2_8_1,float ty1_7_1,float pc0_14_1,float tx3_6_1,float tx1_4_1,float tx2_5_1,float py1_13_1,float pc2_16_1,float px1_11_1,float tc0_17_1,float py0_12_1,float _t_4648,float tc1_18_1,float px0_10_1){
    float y__2869_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_4757 - _t_4648)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__2869_1 = _t_4648 + __step__ * (i + (float)(0.5));
        float _t_4758;
		_t_4758 = tegpixelbody_block_21(ty2_8_1,ty3_9_1,_t_231,px0_10_1,px1_11_1,tx3_6_1,tx2_5_1,py0_12_1,py1_13_1,y__2869_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);;
        __output__ = __output__ + _t_4758 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_5(float ty2_8_1,float ty3_9_1,float tx3_6_1,float tx2_5_1,float _t_231,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_4540;
	float _t_4541;
	float _t_4542;
	bool _t_4543;
	float _t_4546;
	float _t_4550;
	float _t_4551;
	float _t_4552;
	float _t_4553;
	float _t_4554;
	bool _t_4555;
	float _t_4558;
	float _t_4562;
	float _t_4563;
	bool _t_4564;
	float _t_4565;
	float _t_4566;
	float _t_4567;
	float _t_4568;
	float _t_4569;
	bool _t_4570;
	float _t_4573;
	float _t_4577;
	float _t_4578;
	float _t_4579;
	float _t_4580;
	bool _t_4581;
	float _t_4584;
	float _t_4588;
	float _t_4589;
	float _t_4590;
	float _t_4591;
	float _t_4592;
	bool _t_4593;
	float _t_4596;
	float _t_4600;
	float _t_4601;
	float _t_4602;
	float _t_4603;
	float _t_4604;
	float _t_4605;
	float _t_4606;
	float _t_4607;
	float _t_4608;
	bool _t_4609;
	float _t_4612;
	float _t_4616;
	float _t_4617;
	float _t_4618;
	float _t_4619;
	bool _t_4620;
	float _t_4623;
	float _t_4627;
	float _t_4628;
	float _t_4629;
	float _t_4630;
	float _t_4631;
	bool _t_4632;
	float _t_4635;
	float _t_4639;
	float _t_4640;
	float _t_4641;
	float _t_4642;
	float _t_4643;
	float _t_4644;
	bool _t_4645;
	float _t_4646;
	float _t_4647;
	float _t_4648;
	float _t_4649;
	float _t_4650;
	float _t_4651;
	bool _t_4652;
	float _t_4655;
	float _t_4659;
	float _t_4660;
	float _t_4661;
	float _t_4662;
	float _t_4663;
	bool _t_4664;
	float _t_4667;
	float _t_4671;
	float _t_4672;
	bool _t_4673;
	float _t_4674;
	float _t_4675;
	float _t_4676;
	float _t_4677;
	float _t_4678;
	bool _t_4679;
	float _t_4682;
	float _t_4686;
	float _t_4687;
	float _t_4688;
	float _t_4689;
	bool _t_4690;
	float _t_4693;
	float _t_4697;
	float _t_4698;
	float _t_4699;
	float _t_4700;
	float _t_4701;
	bool _t_4702;
	float _t_4705;
	float _t_4709;
	float _t_4710;
	float _t_4711;
	float _t_4712;
	float _t_4713;
	float _t_4714;
	float _t_4715;
	float _t_4716;
	float _t_4717;
	bool _t_4718;
	float _t_4721;
	float _t_4725;
	float _t_4726;
	float _t_4727;
	float _t_4728;
	bool _t_4729;
	float _t_4732;
	float _t_4736;
	float _t_4737;
	float _t_4738;
	float _t_4739;
	float _t_4740;
	bool _t_4741;
	float _t_4744;
	float _t_4748;
	float _t_4749;
	float _t_4750;
	float _t_4751;
	float _t_4752;
	float _t_4753;
	bool _t_4754;
	float _t_4755;
	float _t_4756;
	float _t_4757;

	float _t_232;

	_t_4540 = -1.0f * ty3_9_1;
	_t_4541 = ty2_8_1 + _t_4540;
	_t_4542 = -1.0f * _t_4541;
	_t_4543 = _t_4542 < 0.0f;
	if(_t_4543)
		{
			float _t_4544;
			float _t_4545;
		
			_t_4544 = -1.0f * tx2_5_1;
			_t_4545 = tx3_6_1 + _t_4544;
			_t_4546 = _t_4545;
		
		}
else
		{
			float _t_4547;
			float _t_4548;
			float _t_4549;
		
			_t_4547 = -1.0f * tx2_5_1;
			_t_4548 = tx3_6_1 + _t_4547;
			_t_4549 = -1.0f * _t_4548;
			_t_4546 = _t_4549;
		
		}

	_t_4550 = _t_4546 * _t_231;
	_t_4551 = _t_4550 * -1.0f;
	_t_4552 = -1.0f * ty3_9_1;
	_t_4553 = ty2_8_1 + _t_4552;
	_t_4554 = -1.0f * _t_4553;
	_t_4555 = _t_4554 < 0.0f;
	if(_t_4555)
		{
			float _t_4556;
			float _t_4557;
		
			_t_4556 = -1.0f * tx2_5_1;
			_t_4557 = tx3_6_1 + _t_4556;
			_t_4558 = _t_4557;
		
		}
else
		{
			float _t_4559;
			float _t_4560;
			float _t_4561;
		
			_t_4559 = -1.0f * tx2_5_1;
			_t_4560 = tx3_6_1 + _t_4559;
			_t_4561 = -1.0f * _t_4560;
			_t_4558 = _t_4561;
		
		}

	_t_4562 = _t_4558 * _t_231;
	_t_4563 = _t_4562 * -1.0f;
	_t_4564 = 0.0f < _t_4563;
	if(_t_4564)
		{
		
			_t_4565 = px0_10_1;
		
		}
else
		{
		
			_t_4565 = px1_11_1;
		
		}

	_t_4566 = _t_4551 * _t_4565;
	_t_4567 = -1.0f * ty3_9_1;
	_t_4568 = ty2_8_1 + _t_4567;
	_t_4569 = -1.0f * _t_4568;
	_t_4570 = _t_4569 < 0.0f;
	if(_t_4570)
		{
			float _t_4571;
			float _t_4572;
		
			_t_4571 = -1.0f * tx2_5_1;
			_t_4572 = tx3_6_1 + _t_4571;
			_t_4573 = _t_4572;
		
		}
else
		{
			float _t_4574;
			float _t_4575;
			float _t_4576;
		
			_t_4574 = -1.0f * tx2_5_1;
			_t_4575 = tx3_6_1 + _t_4574;
			_t_4576 = -1.0f * _t_4575;
			_t_4573 = _t_4576;
		
		}

	_t_4577 = _t_4573 * _t_231;
	_t_4578 = -1.0f * ty3_9_1;
	_t_4579 = ty2_8_1 + _t_4578;
	_t_4580 = -1.0f * _t_4579;
	_t_4581 = _t_4580 < 0.0f;
	if(_t_4581)
		{
			float _t_4582;
			float _t_4583;
		
			_t_4582 = -1.0f * tx2_5_1;
			_t_4583 = tx3_6_1 + _t_4582;
			_t_4584 = _t_4583;
		
		}
else
		{
			float _t_4585;
			float _t_4586;
			float _t_4587;
		
			_t_4585 = -1.0f * tx2_5_1;
			_t_4586 = tx3_6_1 + _t_4585;
			_t_4587 = -1.0f * _t_4586;
			_t_4584 = _t_4587;
		
		}

	_t_4588 = _t_4584 * _t_231;
	_t_4589 = _t_4577 * _t_4588;
	_t_4590 = -1.0f * ty3_9_1;
	_t_4591 = ty2_8_1 + _t_4590;
	_t_4592 = -1.0f * _t_4591;
	_t_4593 = _t_4592 < 0.0f;
	if(_t_4593)
		{
			float _t_4594;
			float _t_4595;
		
			_t_4594 = -1.0f * ty3_9_1;
			_t_4595 = ty2_8_1 + _t_4594;
			_t_4596 = _t_4595;
		
		}
else
		{
			float _t_4597;
			float _t_4598;
			float _t_4599;
		
			_t_4597 = -1.0f * ty3_9_1;
			_t_4598 = ty2_8_1 + _t_4597;
			_t_4599 = -1.0f * _t_4598;
			_t_4596 = _t_4599;
		
		}

	_t_4600 = _t_4596 * _t_231;
	_t_4601 = 1.0f + _t_4600;
	_t_4602 = 1.0f / _t_4601;
	_t_4603 = _t_4589 * _t_4602;
	_t_4604 = _t_4603 * -1.0f;
	_t_4605 = 1.0f + _t_4604;
	_t_4606 = -1.0f * ty3_9_1;
	_t_4607 = ty2_8_1 + _t_4606;
	_t_4608 = -1.0f * _t_4607;
	_t_4609 = _t_4608 < 0.0f;
	if(_t_4609)
		{
			float _t_4610;
			float _t_4611;
		
			_t_4610 = -1.0f * tx2_5_1;
			_t_4611 = tx3_6_1 + _t_4610;
			_t_4612 = _t_4611;
		
		}
else
		{
			float _t_4613;
			float _t_4614;
			float _t_4615;
		
			_t_4613 = -1.0f * tx2_5_1;
			_t_4614 = tx3_6_1 + _t_4613;
			_t_4615 = -1.0f * _t_4614;
			_t_4612 = _t_4615;
		
		}

	_t_4616 = _t_4612 * _t_231;
	_t_4617 = -1.0f * ty3_9_1;
	_t_4618 = ty2_8_1 + _t_4617;
	_t_4619 = -1.0f * _t_4618;
	_t_4620 = _t_4619 < 0.0f;
	if(_t_4620)
		{
			float _t_4621;
			float _t_4622;
		
			_t_4621 = -1.0f * tx2_5_1;
			_t_4622 = tx3_6_1 + _t_4621;
			_t_4623 = _t_4622;
		
		}
else
		{
			float _t_4624;
			float _t_4625;
			float _t_4626;
		
			_t_4624 = -1.0f * tx2_5_1;
			_t_4625 = tx3_6_1 + _t_4624;
			_t_4626 = -1.0f * _t_4625;
			_t_4623 = _t_4626;
		
		}

	_t_4627 = _t_4623 * _t_231;
	_t_4628 = _t_4616 * _t_4627;
	_t_4629 = -1.0f * ty3_9_1;
	_t_4630 = ty2_8_1 + _t_4629;
	_t_4631 = -1.0f * _t_4630;
	_t_4632 = _t_4631 < 0.0f;
	if(_t_4632)
		{
			float _t_4633;
			float _t_4634;
		
			_t_4633 = -1.0f * ty3_9_1;
			_t_4634 = ty2_8_1 + _t_4633;
			_t_4635 = _t_4634;
		
		}
else
		{
			float _t_4636;
			float _t_4637;
			float _t_4638;
		
			_t_4636 = -1.0f * ty3_9_1;
			_t_4637 = ty2_8_1 + _t_4636;
			_t_4638 = -1.0f * _t_4637;
			_t_4635 = _t_4638;
		
		}

	_t_4639 = _t_4635 * _t_231;
	_t_4640 = 1.0f + _t_4639;
	_t_4641 = 1.0f / _t_4640;
	_t_4642 = _t_4628 * _t_4641;
	_t_4643 = _t_4642 * -1.0f;
	_t_4644 = 1.0f + _t_4643;
	_t_4645 = 0.0f < _t_4644;
	if(_t_4645)
		{
		
			_t_4646 = py0_12_1;
		
		}
else
		{
		
			_t_4646 = py1_13_1;
		
		}

	_t_4647 = _t_4605 * _t_4646;
	_t_4648 = _t_4566 + _t_4647;
	_t_4649 = -1.0f * ty3_9_1;
	_t_4650 = ty2_8_1 + _t_4649;
	_t_4651 = -1.0f * _t_4650;
	_t_4652 = _t_4651 < 0.0f;
	if(_t_4652)
		{
			float _t_4653;
			float _t_4654;
		
			_t_4653 = -1.0f * tx2_5_1;
			_t_4654 = tx3_6_1 + _t_4653;
			_t_4655 = _t_4654;
		
		}
else
		{
			float _t_4656;
			float _t_4657;
			float _t_4658;
		
			_t_4656 = -1.0f * tx2_5_1;
			_t_4657 = tx3_6_1 + _t_4656;
			_t_4658 = -1.0f * _t_4657;
			_t_4655 = _t_4658;
		
		}

	_t_4659 = _t_4655 * _t_231;
	_t_4660 = _t_4659 * -1.0f;
	_t_4661 = -1.0f * ty3_9_1;
	_t_4662 = ty2_8_1 + _t_4661;
	_t_4663 = -1.0f * _t_4662;
	_t_4664 = _t_4663 < 0.0f;
	if(_t_4664)
		{
			float _t_4665;
			float _t_4666;
		
			_t_4665 = -1.0f * tx2_5_1;
			_t_4666 = tx3_6_1 + _t_4665;
			_t_4667 = _t_4666;
		
		}
else
		{
			float _t_4668;
			float _t_4669;
			float _t_4670;
		
			_t_4668 = -1.0f * tx2_5_1;
			_t_4669 = tx3_6_1 + _t_4668;
			_t_4670 = -1.0f * _t_4669;
			_t_4667 = _t_4670;
		
		}

	_t_4671 = _t_4667 * _t_231;
	_t_4672 = _t_4671 * -1.0f;
	_t_4673 = 0.0f < _t_4672;
	if(_t_4673)
		{
		
			_t_4674 = px1_11_1;
		
		}
else
		{
		
			_t_4674 = px0_10_1;
		
		}

	_t_4675 = _t_4660 * _t_4674;
	_t_4676 = -1.0f * ty3_9_1;
	_t_4677 = ty2_8_1 + _t_4676;
	_t_4678 = -1.0f * _t_4677;
	_t_4679 = _t_4678 < 0.0f;
	if(_t_4679)
		{
			float _t_4680;
			float _t_4681;
		
			_t_4680 = -1.0f * tx2_5_1;
			_t_4681 = tx3_6_1 + _t_4680;
			_t_4682 = _t_4681;
		
		}
else
		{
			float _t_4683;
			float _t_4684;
			float _t_4685;
		
			_t_4683 = -1.0f * tx2_5_1;
			_t_4684 = tx3_6_1 + _t_4683;
			_t_4685 = -1.0f * _t_4684;
			_t_4682 = _t_4685;
		
		}

	_t_4686 = _t_4682 * _t_231;
	_t_4687 = -1.0f * ty3_9_1;
	_t_4688 = ty2_8_1 + _t_4687;
	_t_4689 = -1.0f * _t_4688;
	_t_4690 = _t_4689 < 0.0f;
	if(_t_4690)
		{
			float _t_4691;
			float _t_4692;
		
			_t_4691 = -1.0f * tx2_5_1;
			_t_4692 = tx3_6_1 + _t_4691;
			_t_4693 = _t_4692;
		
		}
else
		{
			float _t_4694;
			float _t_4695;
			float _t_4696;
		
			_t_4694 = -1.0f * tx2_5_1;
			_t_4695 = tx3_6_1 + _t_4694;
			_t_4696 = -1.0f * _t_4695;
			_t_4693 = _t_4696;
		
		}

	_t_4697 = _t_4693 * _t_231;
	_t_4698 = _t_4686 * _t_4697;
	_t_4699 = -1.0f * ty3_9_1;
	_t_4700 = ty2_8_1 + _t_4699;
	_t_4701 = -1.0f * _t_4700;
	_t_4702 = _t_4701 < 0.0f;
	if(_t_4702)
		{
			float _t_4703;
			float _t_4704;
		
			_t_4703 = -1.0f * ty3_9_1;
			_t_4704 = ty2_8_1 + _t_4703;
			_t_4705 = _t_4704;
		
		}
else
		{
			float _t_4706;
			float _t_4707;
			float _t_4708;
		
			_t_4706 = -1.0f * ty3_9_1;
			_t_4707 = ty2_8_1 + _t_4706;
			_t_4708 = -1.0f * _t_4707;
			_t_4705 = _t_4708;
		
		}

	_t_4709 = _t_4705 * _t_231;
	_t_4710 = 1.0f + _t_4709;
	_t_4711 = 1.0f / _t_4710;
	_t_4712 = _t_4698 * _t_4711;
	_t_4713 = _t_4712 * -1.0f;
	_t_4714 = 1.0f + _t_4713;
	_t_4715 = -1.0f * ty3_9_1;
	_t_4716 = ty2_8_1 + _t_4715;
	_t_4717 = -1.0f * _t_4716;
	_t_4718 = _t_4717 < 0.0f;
	if(_t_4718)
		{
			float _t_4719;
			float _t_4720;
		
			_t_4719 = -1.0f * tx2_5_1;
			_t_4720 = tx3_6_1 + _t_4719;
			_t_4721 = _t_4720;
		
		}
else
		{
			float _t_4722;
			float _t_4723;
			float _t_4724;
		
			_t_4722 = -1.0f * tx2_5_1;
			_t_4723 = tx3_6_1 + _t_4722;
			_t_4724 = -1.0f * _t_4723;
			_t_4721 = _t_4724;
		
		}

	_t_4725 = _t_4721 * _t_231;
	_t_4726 = -1.0f * ty3_9_1;
	_t_4727 = ty2_8_1 + _t_4726;
	_t_4728 = -1.0f * _t_4727;
	_t_4729 = _t_4728 < 0.0f;
	if(_t_4729)
		{
			float _t_4730;
			float _t_4731;
		
			_t_4730 = -1.0f * tx2_5_1;
			_t_4731 = tx3_6_1 + _t_4730;
			_t_4732 = _t_4731;
		
		}
else
		{
			float _t_4733;
			float _t_4734;
			float _t_4735;
		
			_t_4733 = -1.0f * tx2_5_1;
			_t_4734 = tx3_6_1 + _t_4733;
			_t_4735 = -1.0f * _t_4734;
			_t_4732 = _t_4735;
		
		}

	_t_4736 = _t_4732 * _t_231;
	_t_4737 = _t_4725 * _t_4736;
	_t_4738 = -1.0f * ty3_9_1;
	_t_4739 = ty2_8_1 + _t_4738;
	_t_4740 = -1.0f * _t_4739;
	_t_4741 = _t_4740 < 0.0f;
	if(_t_4741)
		{
			float _t_4742;
			float _t_4743;
		
			_t_4742 = -1.0f * ty3_9_1;
			_t_4743 = ty2_8_1 + _t_4742;
			_t_4744 = _t_4743;
		
		}
else
		{
			float _t_4745;
			float _t_4746;
			float _t_4747;
		
			_t_4745 = -1.0f * ty3_9_1;
			_t_4746 = ty2_8_1 + _t_4745;
			_t_4747 = -1.0f * _t_4746;
			_t_4744 = _t_4747;
		
		}

	_t_4748 = _t_4744 * _t_231;
	_t_4749 = 1.0f + _t_4748;
	_t_4750 = 1.0f / _t_4749;
	_t_4751 = _t_4737 * _t_4750;
	_t_4752 = _t_4751 * -1.0f;
	_t_4753 = 1.0f + _t_4752;
	_t_4754 = 0.0f < _t_4753;
	if(_t_4754)
		{
		
			_t_4755 = py1_13_1;
		
		}
else
		{
		
			_t_4755 = py0_12_1;
		
		}

	_t_4756 = _t_4714 * _t_4755;
	_t_4757 = _t_4675 + _t_4756;
	_t_232 = tegpixelintegrator_21(ty3_9_1,pc1_15_1,_t_4757,_t_231,tc2_19_1,ty2_8_1,ty1_7_1,pc0_14_1,tx3_6_1,tx1_4_1,tx2_5_1,py1_13_1,pc2_16_1,px1_11_1,tc0_17_1,py0_12_1,_t_4648,tc1_18_1,px0_10_1);

	return _t_232;
}
__device__ float tegpixellet_block_30(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty1_7_1,float tx1_4_1,float ty3_9_1,float _t_5778,float _t_5831,float ty2_8_1,float tx2_5_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float _t_259,float y__2943_1,float _t_5751){
	float _t_5832;
	float _t_5833;
	float _t_5834;
	float _t_5835;
	float _t_5836;
	float _t_5837;
	float _t_5838;
	float _t_5839;
	float _t_5840;
	float _t_5841;
	float _t_5842;
	float _t_5843;
	float _t_5844;
	float _t_5845;
	float _t_5846;
	float _t_5847;
	float _t_5848;
	float _t_5849;
	float _t_5850;
	float _t_5851;
	float _t_5852;
	float _t_5853;
	float _t_5854;
	bool _t_5855;
	float _t_5856;
	float _t_5857;
	float _t_5858;
	float _t_5859;
	float _t_5860;
	float _t_5861;
	float _t_5862;
	float _t_5863;
	float _t_5864;
	float _t_5865;
	float _t_5866;
	float _t_5867;
	float _t_5868;
	float _t_5869;
	bool _t_5870;
	float _t_5871;
	float _t_5872;
	float _t_5873;
	bool _t_5874;
	bool _t_5875;
	bool _t_5876;
	bool _t_5877;
	bool _t_5878;
	bool _t_5879;
	bool _t_5880;
	float _t_6210;

	float _t_5752;

	_t_5832 = -1.0f * pc0_14_1;
	_t_5833 = tc0_17_1 + _t_5832;
	_t_5834 = _t_5833 * _t_5833;
	_t_5835 = -1.0f * pc1_15_1;
	_t_5836 = tc1_18_1 + _t_5835;
	_t_5837 = _t_5836 * _t_5836;
	_t_5838 = _t_5834 + _t_5837;
	_t_5839 = -1.0f * pc2_16_1;
	_t_5840 = tc2_19_1 + _t_5839;
	_t_5841 = _t_5840 * _t_5840;
	_t_5842 = _t_5838 + _t_5841;
	_t_5843 = tx3_6_1 * ty1_7_1;
	_t_5844 = tx1_4_1 * ty3_9_1;
	_t_5845 = _t_5844 * -1.0f;
	_t_5846 = _t_5843 + _t_5845;
	_t_5847 = -1.0f * ty1_7_1;
	_t_5848 = ty3_9_1 + _t_5847;
	_t_5849 = _t_5848 * _t_5778;
	_t_5850 = _t_5846 + _t_5849;
	_t_5851 = -1.0f * tx3_6_1;
	_t_5852 = tx1_4_1 + _t_5851;
	_t_5853 = _t_5852 * _t_5831;
	_t_5854 = _t_5850 + _t_5853;
	_t_5855 = _t_5854 < 0.0f;
	if(_t_5855)
		{
		
			_t_5856 = 1.0f;
		
		}
else
		{
		
			_t_5856 = 0.0f;
		
		}

	_t_5857 = _t_5842 * _t_5856;
	_t_5858 = tx1_4_1 * ty2_8_1;
	_t_5859 = tx2_5_1 * ty1_7_1;
	_t_5860 = _t_5859 * -1.0f;
	_t_5861 = _t_5858 + _t_5860;
	_t_5862 = -1.0f * ty2_8_1;
	_t_5863 = ty1_7_1 + _t_5862;
	_t_5864 = _t_5863 * _t_5778;
	_t_5865 = _t_5861 + _t_5864;
	_t_5866 = -1.0f * tx1_4_1;
	_t_5867 = tx2_5_1 + _t_5866;
	_t_5868 = _t_5867 * _t_5831;
	_t_5869 = _t_5865 + _t_5868;
	_t_5870 = _t_5869 < 0.0f;
	if(_t_5870)
		{
		
			_t_5871 = 1.0f;
		
		}
else
		{
		
			_t_5871 = 0.0f;
		
		}

	_t_5872 = _t_5857 * _t_5871;
	_t_5873 = _t_5872 * _t_5831;
	_t_5874 = py0_12_1 < _t_5831;
	_t_5875 = _t_5831 < py1_13_1;
	_t_5876 = _t_5874 && _t_5875;
	_t_5877 = px0_10_1 < _t_5778;
	_t_5878 = _t_5778 < px1_11_1;
	_t_5879 = _t_5877 && _t_5878;
	_t_5880 = _t_5876 && _t_5879;
	if(_t_5880)
		{
			float _t_5881;
			float _t_5882;
			float _t_5883;
			bool _t_5884;
			float _t_5887;
			float _t_5891;
			float _t_5892;
			float _t_5893;
			float _t_5894;
			float _t_5895;
			bool _t_5896;
			float _t_5899;
			float _t_5903;
			float _t_5904;
			bool _t_5905;
			float _t_5906;
			float _t_5907;
			float _t_5908;
			float _t_5909;
			float _t_5910;
			bool _t_5911;
			float _t_5914;
			float _t_5918;
			float _t_5919;
			float _t_5920;
			float _t_5921;
			bool _t_5922;
			float _t_5925;
			float _t_5929;
			float _t_5930;
			float _t_5931;
			float _t_5932;
			float _t_5933;
			bool _t_5934;
			float _t_5937;
			float _t_5941;
			float _t_5942;
			float _t_5943;
			float _t_5944;
			float _t_5945;
			float _t_5946;
			float _t_5947;
			float _t_5948;
			float _t_5949;
			bool _t_5950;
			float _t_5953;
			float _t_5957;
			float _t_5958;
			float _t_5959;
			float _t_5960;
			bool _t_5961;
			float _t_5964;
			float _t_5968;
			float _t_5969;
			float _t_5970;
			float _t_5971;
			float _t_5972;
			bool _t_5973;
			float _t_5976;
			float _t_5980;
			float _t_5981;
			float _t_5982;
			float _t_5983;
			float _t_5984;
			float _t_5985;
			bool _t_5986;
			float _t_5987;
			float _t_5988;
			float _t_5989;
			bool _t_5990;
			float _t_5991;
			float _t_5992;
			float _t_5993;
			bool _t_5994;
			float _t_5997;
			float _t_6001;
			float _t_6002;
			float _t_6003;
			float _t_6004;
			float _t_6005;
			bool _t_6006;
			float _t_6009;
			float _t_6013;
			float _t_6014;
			bool _t_6015;
			float _t_6016;
			float _t_6017;
			float _t_6018;
			float _t_6019;
			float _t_6020;
			bool _t_6021;
			float _t_6024;
			float _t_6028;
			float _t_6029;
			float _t_6030;
			float _t_6031;
			bool _t_6032;
			float _t_6035;
			float _t_6039;
			float _t_6040;
			float _t_6041;
			float _t_6042;
			float _t_6043;
			bool _t_6044;
			float _t_6047;
			float _t_6051;
			float _t_6052;
			float _t_6053;
			float _t_6054;
			float _t_6055;
			float _t_6056;
			float _t_6057;
			float _t_6058;
			float _t_6059;
			bool _t_6060;
			float _t_6063;
			float _t_6067;
			float _t_6068;
			float _t_6069;
			float _t_6070;
			bool _t_6071;
			float _t_6074;
			float _t_6078;
			float _t_6079;
			float _t_6080;
			float _t_6081;
			float _t_6082;
			bool _t_6083;
			float _t_6086;
			float _t_6090;
			float _t_6091;
			float _t_6092;
			float _t_6093;
			float _t_6094;
			float _t_6095;
			bool _t_6096;
			float _t_6097;
			float _t_6098;
			float _t_6099;
			bool _t_6100;
			bool _t_6101;
			float _t_6102;
			float _t_6103;
			float _t_6104;
			bool _t_6105;
			float _t_6108;
			float _t_6112;
			float _t_6113;
			float _t_6114;
			float _t_6115;
			bool _t_6116;
			float _t_6119;
			float _t_6123;
			bool _t_6124;
			float _t_6125;
			float _t_6126;
			float _t_6127;
			float _t_6128;
			float _t_6129;
			bool _t_6130;
			float _t_6133;
			float _t_6137;
			float _t_6138;
			float _t_6139;
			float _t_6140;
			bool _t_6141;
			float _t_6144;
			float _t_6148;
			bool _t_6149;
			float _t_6150;
			float _t_6151;
			float _t_6152;
			bool _t_6153;
			float _t_6154;
			float _t_6155;
			float _t_6156;
			bool _t_6157;
			float _t_6160;
			float _t_6164;
			float _t_6165;
			float _t_6166;
			float _t_6167;
			bool _t_6168;
			float _t_6171;
			float _t_6175;
			bool _t_6176;
			float _t_6177;
			float _t_6178;
			float _t_6179;
			float _t_6180;
			float _t_6181;
			bool _t_6182;
			float _t_6185;
			float _t_6189;
			float _t_6190;
			float _t_6191;
			float _t_6192;
			bool _t_6193;
			float _t_6196;
			float _t_6200;
			bool _t_6201;
			float _t_6202;
			float _t_6203;
			float _t_6204;
			bool _t_6205;
			bool _t_6206;
			bool _t_6207;
			float _t_6208;
			float _t_6209;
		
			_t_5881 = -1.0f * ty3_9_1;
			_t_5882 = ty2_8_1 + _t_5881;
			_t_5883 = -1.0f * _t_5882;
			_t_5884 = _t_5883 < 0.0f;
			if(_t_5884)
				{
					float _t_5885;
					float _t_5886;
				
					_t_5885 = -1.0f * tx2_5_1;
					_t_5886 = tx3_6_1 + _t_5885;
					_t_5887 = _t_5886;
				
				}
		else
				{
					float _t_5888;
					float _t_5889;
					float _t_5890;
				
					_t_5888 = -1.0f * tx2_5_1;
					_t_5889 = tx3_6_1 + _t_5888;
					_t_5890 = -1.0f * _t_5889;
					_t_5887 = _t_5890;
				
				}
		
			_t_5891 = _t_5887 * _t_259;
			_t_5892 = _t_5891 * -1.0f;
			_t_5893 = -1.0f * ty3_9_1;
			_t_5894 = ty2_8_1 + _t_5893;
			_t_5895 = -1.0f * _t_5894;
			_t_5896 = _t_5895 < 0.0f;
			if(_t_5896)
				{
					float _t_5897;
					float _t_5898;
				
					_t_5897 = -1.0f * tx2_5_1;
					_t_5898 = tx3_6_1 + _t_5897;
					_t_5899 = _t_5898;
				
				}
		else
				{
					float _t_5900;
					float _t_5901;
					float _t_5902;
				
					_t_5900 = -1.0f * tx2_5_1;
					_t_5901 = tx3_6_1 + _t_5900;
					_t_5902 = -1.0f * _t_5901;
					_t_5899 = _t_5902;
				
				}
		
			_t_5903 = _t_5899 * _t_259;
			_t_5904 = _t_5903 * -1.0f;
			_t_5905 = 0.0f < _t_5904;
			if(_t_5905)
				{
				
					_t_5906 = px0_10_1;
				
				}
		else
				{
				
					_t_5906 = px1_11_1;
				
				}
		
			_t_5907 = _t_5892 * _t_5906;
			_t_5908 = -1.0f * ty3_9_1;
			_t_5909 = ty2_8_1 + _t_5908;
			_t_5910 = -1.0f * _t_5909;
			_t_5911 = _t_5910 < 0.0f;
			if(_t_5911)
				{
					float _t_5912;
					float _t_5913;
				
					_t_5912 = -1.0f * tx2_5_1;
					_t_5913 = tx3_6_1 + _t_5912;
					_t_5914 = _t_5913;
				
				}
		else
				{
					float _t_5915;
					float _t_5916;
					float _t_5917;
				
					_t_5915 = -1.0f * tx2_5_1;
					_t_5916 = tx3_6_1 + _t_5915;
					_t_5917 = -1.0f * _t_5916;
					_t_5914 = _t_5917;
				
				}
		
			_t_5918 = _t_5914 * _t_259;
			_t_5919 = -1.0f * ty3_9_1;
			_t_5920 = ty2_8_1 + _t_5919;
			_t_5921 = -1.0f * _t_5920;
			_t_5922 = _t_5921 < 0.0f;
			if(_t_5922)
				{
					float _t_5923;
					float _t_5924;
				
					_t_5923 = -1.0f * tx2_5_1;
					_t_5924 = tx3_6_1 + _t_5923;
					_t_5925 = _t_5924;
				
				}
		else
				{
					float _t_5926;
					float _t_5927;
					float _t_5928;
				
					_t_5926 = -1.0f * tx2_5_1;
					_t_5927 = tx3_6_1 + _t_5926;
					_t_5928 = -1.0f * _t_5927;
					_t_5925 = _t_5928;
				
				}
		
			_t_5929 = _t_5925 * _t_259;
			_t_5930 = _t_5918 * _t_5929;
			_t_5931 = -1.0f * ty3_9_1;
			_t_5932 = ty2_8_1 + _t_5931;
			_t_5933 = -1.0f * _t_5932;
			_t_5934 = _t_5933 < 0.0f;
			if(_t_5934)
				{
					float _t_5935;
					float _t_5936;
				
					_t_5935 = -1.0f * ty3_9_1;
					_t_5936 = ty2_8_1 + _t_5935;
					_t_5937 = _t_5936;
				
				}
		else
				{
					float _t_5938;
					float _t_5939;
					float _t_5940;
				
					_t_5938 = -1.0f * ty3_9_1;
					_t_5939 = ty2_8_1 + _t_5938;
					_t_5940 = -1.0f * _t_5939;
					_t_5937 = _t_5940;
				
				}
		
			_t_5941 = _t_5937 * _t_259;
			_t_5942 = 1.0f + _t_5941;
			_t_5943 = 1.0f / _t_5942;
			_t_5944 = _t_5930 * _t_5943;
			_t_5945 = _t_5944 * -1.0f;
			_t_5946 = 1.0f + _t_5945;
			_t_5947 = -1.0f * ty3_9_1;
			_t_5948 = ty2_8_1 + _t_5947;
			_t_5949 = -1.0f * _t_5948;
			_t_5950 = _t_5949 < 0.0f;
			if(_t_5950)
				{
					float _t_5951;
					float _t_5952;
				
					_t_5951 = -1.0f * tx2_5_1;
					_t_5952 = tx3_6_1 + _t_5951;
					_t_5953 = _t_5952;
				
				}
		else
				{
					float _t_5954;
					float _t_5955;
					float _t_5956;
				
					_t_5954 = -1.0f * tx2_5_1;
					_t_5955 = tx3_6_1 + _t_5954;
					_t_5956 = -1.0f * _t_5955;
					_t_5953 = _t_5956;
				
				}
		
			_t_5957 = _t_5953 * _t_259;
			_t_5958 = -1.0f * ty3_9_1;
			_t_5959 = ty2_8_1 + _t_5958;
			_t_5960 = -1.0f * _t_5959;
			_t_5961 = _t_5960 < 0.0f;
			if(_t_5961)
				{
					float _t_5962;
					float _t_5963;
				
					_t_5962 = -1.0f * tx2_5_1;
					_t_5963 = tx3_6_1 + _t_5962;
					_t_5964 = _t_5963;
				
				}
		else
				{
					float _t_5965;
					float _t_5966;
					float _t_5967;
				
					_t_5965 = -1.0f * tx2_5_1;
					_t_5966 = tx3_6_1 + _t_5965;
					_t_5967 = -1.0f * _t_5966;
					_t_5964 = _t_5967;
				
				}
		
			_t_5968 = _t_5964 * _t_259;
			_t_5969 = _t_5957 * _t_5968;
			_t_5970 = -1.0f * ty3_9_1;
			_t_5971 = ty2_8_1 + _t_5970;
			_t_5972 = -1.0f * _t_5971;
			_t_5973 = _t_5972 < 0.0f;
			if(_t_5973)
				{
					float _t_5974;
					float _t_5975;
				
					_t_5974 = -1.0f * ty3_9_1;
					_t_5975 = ty2_8_1 + _t_5974;
					_t_5976 = _t_5975;
				
				}
		else
				{
					float _t_5977;
					float _t_5978;
					float _t_5979;
				
					_t_5977 = -1.0f * ty3_9_1;
					_t_5978 = ty2_8_1 + _t_5977;
					_t_5979 = -1.0f * _t_5978;
					_t_5976 = _t_5979;
				
				}
		
			_t_5980 = _t_5976 * _t_259;
			_t_5981 = 1.0f + _t_5980;
			_t_5982 = 1.0f / _t_5981;
			_t_5983 = _t_5969 * _t_5982;
			_t_5984 = _t_5983 * -1.0f;
			_t_5985 = 1.0f + _t_5984;
			_t_5986 = 0.0f < _t_5985;
			if(_t_5986)
				{
				
					_t_5987 = py0_12_1;
				
				}
		else
				{
				
					_t_5987 = py1_13_1;
				
				}
		
			_t_5988 = _t_5946 * _t_5987;
			_t_5989 = _t_5907 + _t_5988;
			_t_5990 = _t_5989 < y__2943_1;
			_t_5991 = -1.0f * ty3_9_1;
			_t_5992 = ty2_8_1 + _t_5991;
			_t_5993 = -1.0f * _t_5992;
			_t_5994 = _t_5993 < 0.0f;
			if(_t_5994)
				{
					float _t_5995;
					float _t_5996;
				
					_t_5995 = -1.0f * tx2_5_1;
					_t_5996 = tx3_6_1 + _t_5995;
					_t_5997 = _t_5996;
				
				}
		else
				{
					float _t_5998;
					float _t_5999;
					float _t_6000;
				
					_t_5998 = -1.0f * tx2_5_1;
					_t_5999 = tx3_6_1 + _t_5998;
					_t_6000 = -1.0f * _t_5999;
					_t_5997 = _t_6000;
				
				}
		
			_t_6001 = _t_5997 * _t_259;
			_t_6002 = _t_6001 * -1.0f;
			_t_6003 = -1.0f * ty3_9_1;
			_t_6004 = ty2_8_1 + _t_6003;
			_t_6005 = -1.0f * _t_6004;
			_t_6006 = _t_6005 < 0.0f;
			if(_t_6006)
				{
					float _t_6007;
					float _t_6008;
				
					_t_6007 = -1.0f * tx2_5_1;
					_t_6008 = tx3_6_1 + _t_6007;
					_t_6009 = _t_6008;
				
				}
		else
				{
					float _t_6010;
					float _t_6011;
					float _t_6012;
				
					_t_6010 = -1.0f * tx2_5_1;
					_t_6011 = tx3_6_1 + _t_6010;
					_t_6012 = -1.0f * _t_6011;
					_t_6009 = _t_6012;
				
				}
		
			_t_6013 = _t_6009 * _t_259;
			_t_6014 = _t_6013 * -1.0f;
			_t_6015 = 0.0f < _t_6014;
			if(_t_6015)
				{
				
					_t_6016 = px1_11_1;
				
				}
		else
				{
				
					_t_6016 = px0_10_1;
				
				}
		
			_t_6017 = _t_6002 * _t_6016;
			_t_6018 = -1.0f * ty3_9_1;
			_t_6019 = ty2_8_1 + _t_6018;
			_t_6020 = -1.0f * _t_6019;
			_t_6021 = _t_6020 < 0.0f;
			if(_t_6021)
				{
					float _t_6022;
					float _t_6023;
				
					_t_6022 = -1.0f * tx2_5_1;
					_t_6023 = tx3_6_1 + _t_6022;
					_t_6024 = _t_6023;
				
				}
		else
				{
					float _t_6025;
					float _t_6026;
					float _t_6027;
				
					_t_6025 = -1.0f * tx2_5_1;
					_t_6026 = tx3_6_1 + _t_6025;
					_t_6027 = -1.0f * _t_6026;
					_t_6024 = _t_6027;
				
				}
		
			_t_6028 = _t_6024 * _t_259;
			_t_6029 = -1.0f * ty3_9_1;
			_t_6030 = ty2_8_1 + _t_6029;
			_t_6031 = -1.0f * _t_6030;
			_t_6032 = _t_6031 < 0.0f;
			if(_t_6032)
				{
					float _t_6033;
					float _t_6034;
				
					_t_6033 = -1.0f * tx2_5_1;
					_t_6034 = tx3_6_1 + _t_6033;
					_t_6035 = _t_6034;
				
				}
		else
				{
					float _t_6036;
					float _t_6037;
					float _t_6038;
				
					_t_6036 = -1.0f * tx2_5_1;
					_t_6037 = tx3_6_1 + _t_6036;
					_t_6038 = -1.0f * _t_6037;
					_t_6035 = _t_6038;
				
				}
		
			_t_6039 = _t_6035 * _t_259;
			_t_6040 = _t_6028 * _t_6039;
			_t_6041 = -1.0f * ty3_9_1;
			_t_6042 = ty2_8_1 + _t_6041;
			_t_6043 = -1.0f * _t_6042;
			_t_6044 = _t_6043 < 0.0f;
			if(_t_6044)
				{
					float _t_6045;
					float _t_6046;
				
					_t_6045 = -1.0f * ty3_9_1;
					_t_6046 = ty2_8_1 + _t_6045;
					_t_6047 = _t_6046;
				
				}
		else
				{
					float _t_6048;
					float _t_6049;
					float _t_6050;
				
					_t_6048 = -1.0f * ty3_9_1;
					_t_6049 = ty2_8_1 + _t_6048;
					_t_6050 = -1.0f * _t_6049;
					_t_6047 = _t_6050;
				
				}
		
			_t_6051 = _t_6047 * _t_259;
			_t_6052 = 1.0f + _t_6051;
			_t_6053 = 1.0f / _t_6052;
			_t_6054 = _t_6040 * _t_6053;
			_t_6055 = _t_6054 * -1.0f;
			_t_6056 = 1.0f + _t_6055;
			_t_6057 = -1.0f * ty3_9_1;
			_t_6058 = ty2_8_1 + _t_6057;
			_t_6059 = -1.0f * _t_6058;
			_t_6060 = _t_6059 < 0.0f;
			if(_t_6060)
				{
					float _t_6061;
					float _t_6062;
				
					_t_6061 = -1.0f * tx2_5_1;
					_t_6062 = tx3_6_1 + _t_6061;
					_t_6063 = _t_6062;
				
				}
		else
				{
					float _t_6064;
					float _t_6065;
					float _t_6066;
				
					_t_6064 = -1.0f * tx2_5_1;
					_t_6065 = tx3_6_1 + _t_6064;
					_t_6066 = -1.0f * _t_6065;
					_t_6063 = _t_6066;
				
				}
		
			_t_6067 = _t_6063 * _t_259;
			_t_6068 = -1.0f * ty3_9_1;
			_t_6069 = ty2_8_1 + _t_6068;
			_t_6070 = -1.0f * _t_6069;
			_t_6071 = _t_6070 < 0.0f;
			if(_t_6071)
				{
					float _t_6072;
					float _t_6073;
				
					_t_6072 = -1.0f * tx2_5_1;
					_t_6073 = tx3_6_1 + _t_6072;
					_t_6074 = _t_6073;
				
				}
		else
				{
					float _t_6075;
					float _t_6076;
					float _t_6077;
				
					_t_6075 = -1.0f * tx2_5_1;
					_t_6076 = tx3_6_1 + _t_6075;
					_t_6077 = -1.0f * _t_6076;
					_t_6074 = _t_6077;
				
				}
		
			_t_6078 = _t_6074 * _t_259;
			_t_6079 = _t_6067 * _t_6078;
			_t_6080 = -1.0f * ty3_9_1;
			_t_6081 = ty2_8_1 + _t_6080;
			_t_6082 = -1.0f * _t_6081;
			_t_6083 = _t_6082 < 0.0f;
			if(_t_6083)
				{
					float _t_6084;
					float _t_6085;
				
					_t_6084 = -1.0f * ty3_9_1;
					_t_6085 = ty2_8_1 + _t_6084;
					_t_6086 = _t_6085;
				
				}
		else
				{
					float _t_6087;
					float _t_6088;
					float _t_6089;
				
					_t_6087 = -1.0f * ty3_9_1;
					_t_6088 = ty2_8_1 + _t_6087;
					_t_6089 = -1.0f * _t_6088;
					_t_6086 = _t_6089;
				
				}
		
			_t_6090 = _t_6086 * _t_259;
			_t_6091 = 1.0f + _t_6090;
			_t_6092 = 1.0f / _t_6091;
			_t_6093 = _t_6079 * _t_6092;
			_t_6094 = _t_6093 * -1.0f;
			_t_6095 = 1.0f + _t_6094;
			_t_6096 = 0.0f < _t_6095;
			if(_t_6096)
				{
				
					_t_6097 = py1_13_1;
				
				}
		else
				{
				
					_t_6097 = py0_12_1;
				
				}
		
			_t_6098 = _t_6056 * _t_6097;
			_t_6099 = _t_6017 + _t_6098;
			_t_6100 = y__2943_1 < _t_6099;
			_t_6101 = _t_5990 && _t_6100;
			_t_6102 = -1.0f * ty3_9_1;
			_t_6103 = ty2_8_1 + _t_6102;
			_t_6104 = -1.0f * _t_6103;
			_t_6105 = _t_6104 < 0.0f;
			if(_t_6105)
				{
					float _t_6106;
					float _t_6107;
				
					_t_6106 = -1.0f * ty3_9_1;
					_t_6107 = ty2_8_1 + _t_6106;
					_t_6108 = _t_6107;
				
				}
		else
				{
					float _t_6109;
					float _t_6110;
					float _t_6111;
				
					_t_6109 = -1.0f * ty3_9_1;
					_t_6110 = ty2_8_1 + _t_6109;
					_t_6111 = -1.0f * _t_6110;
					_t_6108 = _t_6111;
				
				}
		
			_t_6112 = _t_6108 * _t_259;
			_t_6113 = -1.0f * ty3_9_1;
			_t_6114 = ty2_8_1 + _t_6113;
			_t_6115 = -1.0f * _t_6114;
			_t_6116 = _t_6115 < 0.0f;
			if(_t_6116)
				{
					float _t_6117;
					float _t_6118;
				
					_t_6117 = -1.0f * ty3_9_1;
					_t_6118 = ty2_8_1 + _t_6117;
					_t_6119 = _t_6118;
				
				}
		else
				{
					float _t_6120;
					float _t_6121;
					float _t_6122;
				
					_t_6120 = -1.0f * ty3_9_1;
					_t_6121 = ty2_8_1 + _t_6120;
					_t_6122 = -1.0f * _t_6121;
					_t_6119 = _t_6122;
				
				}
		
			_t_6123 = _t_6119 * _t_259;
			_t_6124 = 0.0f < _t_6123;
			if(_t_6124)
				{
				
					_t_6125 = px0_10_1;
				
				}
		else
				{
				
					_t_6125 = px1_11_1;
				
				}
		
			_t_6126 = _t_6112 * _t_6125;
			_t_6127 = -1.0f * ty3_9_1;
			_t_6128 = ty2_8_1 + _t_6127;
			_t_6129 = -1.0f * _t_6128;
			_t_6130 = _t_6129 < 0.0f;
			if(_t_6130)
				{
					float _t_6131;
					float _t_6132;
				
					_t_6131 = -1.0f * tx2_5_1;
					_t_6132 = tx3_6_1 + _t_6131;
					_t_6133 = _t_6132;
				
				}
		else
				{
					float _t_6134;
					float _t_6135;
					float _t_6136;
				
					_t_6134 = -1.0f * tx2_5_1;
					_t_6135 = tx3_6_1 + _t_6134;
					_t_6136 = -1.0f * _t_6135;
					_t_6133 = _t_6136;
				
				}
		
			_t_6137 = _t_6133 * _t_259;
			_t_6138 = -1.0f * ty3_9_1;
			_t_6139 = ty2_8_1 + _t_6138;
			_t_6140 = -1.0f * _t_6139;
			_t_6141 = _t_6140 < 0.0f;
			if(_t_6141)
				{
					float _t_6142;
					float _t_6143;
				
					_t_6142 = -1.0f * tx2_5_1;
					_t_6143 = tx3_6_1 + _t_6142;
					_t_6144 = _t_6143;
				
				}
		else
				{
					float _t_6145;
					float _t_6146;
					float _t_6147;
				
					_t_6145 = -1.0f * tx2_5_1;
					_t_6146 = tx3_6_1 + _t_6145;
					_t_6147 = -1.0f * _t_6146;
					_t_6144 = _t_6147;
				
				}
		
			_t_6148 = _t_6144 * _t_259;
			_t_6149 = 0.0f < _t_6148;
			if(_t_6149)
				{
				
					_t_6150 = py0_12_1;
				
				}
		else
				{
				
					_t_6150 = py1_13_1;
				
				}
		
			_t_6151 = _t_6137 * _t_6150;
			_t_6152 = _t_6126 + _t_6151;
			_t_6153 = _t_6152 < _t_5751;
			_t_6154 = -1.0f * ty3_9_1;
			_t_6155 = ty2_8_1 + _t_6154;
			_t_6156 = -1.0f * _t_6155;
			_t_6157 = _t_6156 < 0.0f;
			if(_t_6157)
				{
					float _t_6158;
					float _t_6159;
				
					_t_6158 = -1.0f * ty3_9_1;
					_t_6159 = ty2_8_1 + _t_6158;
					_t_6160 = _t_6159;
				
				}
		else
				{
					float _t_6161;
					float _t_6162;
					float _t_6163;
				
					_t_6161 = -1.0f * ty3_9_1;
					_t_6162 = ty2_8_1 + _t_6161;
					_t_6163 = -1.0f * _t_6162;
					_t_6160 = _t_6163;
				
				}
		
			_t_6164 = _t_6160 * _t_259;
			_t_6165 = -1.0f * ty3_9_1;
			_t_6166 = ty2_8_1 + _t_6165;
			_t_6167 = -1.0f * _t_6166;
			_t_6168 = _t_6167 < 0.0f;
			if(_t_6168)
				{
					float _t_6169;
					float _t_6170;
				
					_t_6169 = -1.0f * ty3_9_1;
					_t_6170 = ty2_8_1 + _t_6169;
					_t_6171 = _t_6170;
				
				}
		else
				{
					float _t_6172;
					float _t_6173;
					float _t_6174;
				
					_t_6172 = -1.0f * ty3_9_1;
					_t_6173 = ty2_8_1 + _t_6172;
					_t_6174 = -1.0f * _t_6173;
					_t_6171 = _t_6174;
				
				}
		
			_t_6175 = _t_6171 * _t_259;
			_t_6176 = 0.0f < _t_6175;
			if(_t_6176)
				{
				
					_t_6177 = px1_11_1;
				
				}
		else
				{
				
					_t_6177 = px0_10_1;
				
				}
		
			_t_6178 = _t_6164 * _t_6177;
			_t_6179 = -1.0f * ty3_9_1;
			_t_6180 = ty2_8_1 + _t_6179;
			_t_6181 = -1.0f * _t_6180;
			_t_6182 = _t_6181 < 0.0f;
			if(_t_6182)
				{
					float _t_6183;
					float _t_6184;
				
					_t_6183 = -1.0f * tx2_5_1;
					_t_6184 = tx3_6_1 + _t_6183;
					_t_6185 = _t_6184;
				
				}
		else
				{
					float _t_6186;
					float _t_6187;
					float _t_6188;
				
					_t_6186 = -1.0f * tx2_5_1;
					_t_6187 = tx3_6_1 + _t_6186;
					_t_6188 = -1.0f * _t_6187;
					_t_6185 = _t_6188;
				
				}
		
			_t_6189 = _t_6185 * _t_259;
			_t_6190 = -1.0f * ty3_9_1;
			_t_6191 = ty2_8_1 + _t_6190;
			_t_6192 = -1.0f * _t_6191;
			_t_6193 = _t_6192 < 0.0f;
			if(_t_6193)
				{
					float _t_6194;
					float _t_6195;
				
					_t_6194 = -1.0f * tx2_5_1;
					_t_6195 = tx3_6_1 + _t_6194;
					_t_6196 = _t_6195;
				
				}
		else
				{
					float _t_6197;
					float _t_6198;
					float _t_6199;
				
					_t_6197 = -1.0f * tx2_5_1;
					_t_6198 = tx3_6_1 + _t_6197;
					_t_6199 = -1.0f * _t_6198;
					_t_6196 = _t_6199;
				
				}
		
			_t_6200 = _t_6196 * _t_259;
			_t_6201 = 0.0f < _t_6200;
			if(_t_6201)
				{
				
					_t_6202 = py1_13_1;
				
				}
		else
				{
				
					_t_6202 = py0_12_1;
				
				}
		
			_t_6203 = _t_6189 * _t_6202;
			_t_6204 = _t_6178 + _t_6203;
			_t_6205 = _t_5751 < _t_6204;
			_t_6206 = _t_6153 && _t_6205;
			_t_6207 = _t_6101 && _t_6206;
			if(_t_6207)
				{
				
					_t_6208 = 1.0f;
				
				}
		else
				{
				
					_t_6208 = 0.0f;
				
				}
		
			_t_6209 = _t_6208 * _t_259;
			_t_6210 = _t_6209;
		
		}
else
		{
		
			_t_6210 = 0.0f;
		
		}

	_t_5752 = _t_5873 * _t_6210;

	return _t_5752;
}
__device__ float tegpixellet_block_29(float ty2_8_1,float ty3_9_1,float _t_259,float _t_5751,float tx3_6_1,float tx2_5_1,float y__2943_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1){
	float _t_5753;
	float _t_5754;
	float _t_5755;
	bool _t_5756;
	float _t_5759;
	float _t_5763;
	float _t_5764;
	float _t_5765;
	float _t_5766;
	float _t_5767;
	bool _t_5768;
	float _t_5771;
	float _t_5775;
	float _t_5776;
	float _t_5777;
	float _t_5778;
	float _t_5779;
	float _t_5780;
	float _t_5781;
	bool _t_5782;
	float _t_5785;
	float _t_5789;
	float _t_5790;
	float _t_5791;
	float _t_5792;
	bool _t_5793;
	float _t_5796;
	float _t_5800;
	float _t_5801;
	float _t_5802;
	float _t_5803;
	float _t_5804;
	bool _t_5805;
	float _t_5808;
	float _t_5812;
	float _t_5813;
	float _t_5814;
	float _t_5815;
	float _t_5816;
	float _t_5817;
	float _t_5818;
	float _t_5819;
	float _t_5820;
	float _t_5821;
	bool _t_5822;
	float _t_5825;
	float _t_5829;
	float _t_5830;
	float _t_5831;

	float _t_5752;

	_t_5753 = -1.0f * ty3_9_1;
	_t_5754 = ty2_8_1 + _t_5753;
	_t_5755 = -1.0f * _t_5754;
	_t_5756 = _t_5755 < 0.0f;
	if(_t_5756)
		{
			float _t_5757;
			float _t_5758;
		
			_t_5757 = -1.0f * ty3_9_1;
			_t_5758 = ty2_8_1 + _t_5757;
			_t_5759 = _t_5758;
		
		}
else
		{
			float _t_5760;
			float _t_5761;
			float _t_5762;
		
			_t_5760 = -1.0f * ty3_9_1;
			_t_5761 = ty2_8_1 + _t_5760;
			_t_5762 = -1.0f * _t_5761;
			_t_5759 = _t_5762;
		
		}

	_t_5763 = _t_5759 * _t_259;
	_t_5764 = _t_5763 * _t_5751;
	_t_5765 = -1.0f * ty3_9_1;
	_t_5766 = ty2_8_1 + _t_5765;
	_t_5767 = -1.0f * _t_5766;
	_t_5768 = _t_5767 < 0.0f;
	if(_t_5768)
		{
			float _t_5769;
			float _t_5770;
		
			_t_5769 = -1.0f * tx2_5_1;
			_t_5770 = tx3_6_1 + _t_5769;
			_t_5771 = _t_5770;
		
		}
else
		{
			float _t_5772;
			float _t_5773;
			float _t_5774;
		
			_t_5772 = -1.0f * tx2_5_1;
			_t_5773 = tx3_6_1 + _t_5772;
			_t_5774 = -1.0f * _t_5773;
			_t_5771 = _t_5774;
		
		}

	_t_5775 = _t_5771 * _t_259;
	_t_5776 = _t_5775 * -1.0f;
	_t_5777 = _t_5776 * y__2943_1;
	_t_5778 = _t_5764 + _t_5777;
	_t_5779 = -1.0f * ty3_9_1;
	_t_5780 = ty2_8_1 + _t_5779;
	_t_5781 = -1.0f * _t_5780;
	_t_5782 = _t_5781 < 0.0f;
	if(_t_5782)
		{
			float _t_5783;
			float _t_5784;
		
			_t_5783 = -1.0f * tx2_5_1;
			_t_5784 = tx3_6_1 + _t_5783;
			_t_5785 = _t_5784;
		
		}
else
		{
			float _t_5786;
			float _t_5787;
			float _t_5788;
		
			_t_5786 = -1.0f * tx2_5_1;
			_t_5787 = tx3_6_1 + _t_5786;
			_t_5788 = -1.0f * _t_5787;
			_t_5785 = _t_5788;
		
		}

	_t_5789 = _t_5785 * _t_259;
	_t_5790 = -1.0f * ty3_9_1;
	_t_5791 = ty2_8_1 + _t_5790;
	_t_5792 = -1.0f * _t_5791;
	_t_5793 = _t_5792 < 0.0f;
	if(_t_5793)
		{
			float _t_5794;
			float _t_5795;
		
			_t_5794 = -1.0f * tx2_5_1;
			_t_5795 = tx3_6_1 + _t_5794;
			_t_5796 = _t_5795;
		
		}
else
		{
			float _t_5797;
			float _t_5798;
			float _t_5799;
		
			_t_5797 = -1.0f * tx2_5_1;
			_t_5798 = tx3_6_1 + _t_5797;
			_t_5799 = -1.0f * _t_5798;
			_t_5796 = _t_5799;
		
		}

	_t_5800 = _t_5796 * _t_259;
	_t_5801 = _t_5789 * _t_5800;
	_t_5802 = -1.0f * ty3_9_1;
	_t_5803 = ty2_8_1 + _t_5802;
	_t_5804 = -1.0f * _t_5803;
	_t_5805 = _t_5804 < 0.0f;
	if(_t_5805)
		{
			float _t_5806;
			float _t_5807;
		
			_t_5806 = -1.0f * ty3_9_1;
			_t_5807 = ty2_8_1 + _t_5806;
			_t_5808 = _t_5807;
		
		}
else
		{
			float _t_5809;
			float _t_5810;
			float _t_5811;
		
			_t_5809 = -1.0f * ty3_9_1;
			_t_5810 = ty2_8_1 + _t_5809;
			_t_5811 = -1.0f * _t_5810;
			_t_5808 = _t_5811;
		
		}

	_t_5812 = _t_5808 * _t_259;
	_t_5813 = 1.0f + _t_5812;
	_t_5814 = 1.0f / _t_5813;
	_t_5815 = _t_5801 * _t_5814;
	_t_5816 = _t_5815 * -1.0f;
	_t_5817 = 1.0f + _t_5816;
	_t_5818 = _t_5817 * y__2943_1;
	_t_5819 = -1.0f * ty3_9_1;
	_t_5820 = ty2_8_1 + _t_5819;
	_t_5821 = -1.0f * _t_5820;
	_t_5822 = _t_5821 < 0.0f;
	if(_t_5822)
		{
			float _t_5823;
			float _t_5824;
		
			_t_5823 = -1.0f * tx2_5_1;
			_t_5824 = tx3_6_1 + _t_5823;
			_t_5825 = _t_5824;
		
		}
else
		{
			float _t_5826;
			float _t_5827;
			float _t_5828;
		
			_t_5826 = -1.0f * tx2_5_1;
			_t_5827 = tx3_6_1 + _t_5826;
			_t_5828 = -1.0f * _t_5827;
			_t_5825 = _t_5828;
		
		}

	_t_5829 = _t_5825 * _t_259;
	_t_5830 = _t_5829 * _t_5751;
	_t_5831 = _t_5818 + _t_5830;
	_t_5752 = tegpixellet_block_30(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty1_7_1,tx1_4_1,ty3_9_1,_t_5778,_t_5831,ty2_8_1,tx2_5_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,_t_259,y__2943_1,_t_5751);

	return _t_5752;
}
__device__ float tegpixelbody_block_22(float ty2_8_1,float ty3_9_1,float _t_259,float px0_10_1,float px1_11_1,float tx3_6_1,float tx2_5_1,float py0_12_1,float py1_13_1,float y__2943_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_5595;
	float _t_5596;
	float _t_5597;
	bool _t_5598;
	float _t_5601;
	float _t_5605;
	float _t_5606;
	float _t_5607;
	float _t_5608;
	bool _t_5609;
	float _t_5612;
	float _t_5616;
	bool _t_5617;
	float _t_5618;
	float _t_5619;
	float _t_5620;
	float _t_5621;
	float _t_5622;
	bool _t_5623;
	float _t_5626;
	float _t_5630;
	float _t_5631;
	float _t_5632;
	float _t_5633;
	bool _t_5634;
	float _t_5637;
	float _t_5641;
	bool _t_5642;
	float _t_5643;
	float _t_5644;
	float _t_5645;
	float _t_5646;
	float _t_5647;
	float _t_5648;
	bool _t_5649;
	float _t_5654;
	float _t_5660;
	float _t_5661;
	float _t_5662;
	float _t_5663;
	bool _t_5664;
	float _t_5665;
	float _t_5666;
	float _t_5667;
	bool _t_5668;
	float _t_5671;
	float _t_5675;
	float _t_5676;
	float _t_5677;
	float _t_5678;
	bool _t_5679;
	float _t_5682;
	float _t_5686;
	bool _t_5687;
	float _t_5688;
	float _t_5689;
	float _t_5690;
	float _t_5691;
	float _t_5692;
	bool _t_5693;
	float _t_5696;
	float _t_5700;
	float _t_5701;
	float _t_5702;
	float _t_5703;
	bool _t_5704;
	float _t_5707;
	float _t_5711;
	bool _t_5712;
	float _t_5713;
	float _t_5714;
	float _t_5715;
	float _t_5716;
	float _t_5717;
	float _t_5718;
	bool _t_5719;
	float _t_5724;
	float _t_5730;
	float _t_5731;
	float _t_5732;
	float _t_5733;
	bool _t_5734;
	bool _t_5735;

	float _t_5594;

	_t_5595 = -1.0f * ty3_9_1;
	_t_5596 = ty2_8_1 + _t_5595;
	_t_5597 = -1.0f * _t_5596;
	_t_5598 = _t_5597 < 0.0f;
	if(_t_5598)
		{
			float _t_5599;
			float _t_5600;
		
			_t_5599 = -1.0f * ty3_9_1;
			_t_5600 = ty2_8_1 + _t_5599;
			_t_5601 = _t_5600;
		
		}
else
		{
			float _t_5602;
			float _t_5603;
			float _t_5604;
		
			_t_5602 = -1.0f * ty3_9_1;
			_t_5603 = ty2_8_1 + _t_5602;
			_t_5604 = -1.0f * _t_5603;
			_t_5601 = _t_5604;
		
		}

	_t_5605 = _t_5601 * _t_259;
	_t_5606 = -1.0f * ty3_9_1;
	_t_5607 = ty2_8_1 + _t_5606;
	_t_5608 = -1.0f * _t_5607;
	_t_5609 = _t_5608 < 0.0f;
	if(_t_5609)
		{
			float _t_5610;
			float _t_5611;
		
			_t_5610 = -1.0f * ty3_9_1;
			_t_5611 = ty2_8_1 + _t_5610;
			_t_5612 = _t_5611;
		
		}
else
		{
			float _t_5613;
			float _t_5614;
			float _t_5615;
		
			_t_5613 = -1.0f * ty3_9_1;
			_t_5614 = ty2_8_1 + _t_5613;
			_t_5615 = -1.0f * _t_5614;
			_t_5612 = _t_5615;
		
		}

	_t_5616 = _t_5612 * _t_259;
	_t_5617 = 0.0f < _t_5616;
	if(_t_5617)
		{
		
			_t_5618 = px0_10_1;
		
		}
else
		{
		
			_t_5618 = px1_11_1;
		
		}

	_t_5619 = _t_5605 * _t_5618;
	_t_5620 = -1.0f * ty3_9_1;
	_t_5621 = ty2_8_1 + _t_5620;
	_t_5622 = -1.0f * _t_5621;
	_t_5623 = _t_5622 < 0.0f;
	if(_t_5623)
		{
			float _t_5624;
			float _t_5625;
		
			_t_5624 = -1.0f * tx2_5_1;
			_t_5625 = tx3_6_1 + _t_5624;
			_t_5626 = _t_5625;
		
		}
else
		{
			float _t_5627;
			float _t_5628;
			float _t_5629;
		
			_t_5627 = -1.0f * tx2_5_1;
			_t_5628 = tx3_6_1 + _t_5627;
			_t_5629 = -1.0f * _t_5628;
			_t_5626 = _t_5629;
		
		}

	_t_5630 = _t_5626 * _t_259;
	_t_5631 = -1.0f * ty3_9_1;
	_t_5632 = ty2_8_1 + _t_5631;
	_t_5633 = -1.0f * _t_5632;
	_t_5634 = _t_5633 < 0.0f;
	if(_t_5634)
		{
			float _t_5635;
			float _t_5636;
		
			_t_5635 = -1.0f * tx2_5_1;
			_t_5636 = tx3_6_1 + _t_5635;
			_t_5637 = _t_5636;
		
		}
else
		{
			float _t_5638;
			float _t_5639;
			float _t_5640;
		
			_t_5638 = -1.0f * tx2_5_1;
			_t_5639 = tx3_6_1 + _t_5638;
			_t_5640 = -1.0f * _t_5639;
			_t_5637 = _t_5640;
		
		}

	_t_5641 = _t_5637 * _t_259;
	_t_5642 = 0.0f < _t_5641;
	if(_t_5642)
		{
		
			_t_5643 = py0_12_1;
		
		}
else
		{
		
			_t_5643 = py1_13_1;
		
		}

	_t_5644 = _t_5630 * _t_5643;
	_t_5645 = _t_5619 + _t_5644;
	_t_5646 = -1.0f * ty3_9_1;
	_t_5647 = ty2_8_1 + _t_5646;
	_t_5648 = -1.0f * _t_5647;
	_t_5649 = _t_5648 < 0.0f;
	if(_t_5649)
		{
			float _t_5650;
			float _t_5651;
			float _t_5652;
			float _t_5653;
		
			_t_5650 = tx2_5_1 * ty3_9_1;
			_t_5651 = tx3_6_1 * ty2_8_1;
			_t_5652 = _t_5651 * -1.0f;
			_t_5653 = _t_5650 + _t_5652;
			_t_5654 = _t_5653;
		
		}
else
		{
			float _t_5655;
			float _t_5656;
			float _t_5657;
			float _t_5658;
			float _t_5659;
		
			_t_5655 = tx2_5_1 * ty3_9_1;
			_t_5656 = tx3_6_1 * ty2_8_1;
			_t_5657 = _t_5656 * -1.0f;
			_t_5658 = _t_5655 + _t_5657;
			_t_5659 = -1.0f * _t_5658;
			_t_5654 = _t_5659;
		
		}

	_t_5660 = -1.0f * _t_5654;
	_t_5661 = _t_5660 * _t_259;
	_t_5662 = _t_5661 * -1.0f;
	_t_5663 = _t_5645 + _t_5662;
	_t_5664 = _t_5663 < 0.0f;
	_t_5665 = -1.0f * ty3_9_1;
	_t_5666 = ty2_8_1 + _t_5665;
	_t_5667 = -1.0f * _t_5666;
	_t_5668 = _t_5667 < 0.0f;
	if(_t_5668)
		{
			float _t_5669;
			float _t_5670;
		
			_t_5669 = -1.0f * ty3_9_1;
			_t_5670 = ty2_8_1 + _t_5669;
			_t_5671 = _t_5670;
		
		}
else
		{
			float _t_5672;
			float _t_5673;
			float _t_5674;
		
			_t_5672 = -1.0f * ty3_9_1;
			_t_5673 = ty2_8_1 + _t_5672;
			_t_5674 = -1.0f * _t_5673;
			_t_5671 = _t_5674;
		
		}

	_t_5675 = _t_5671 * _t_259;
	_t_5676 = -1.0f * ty3_9_1;
	_t_5677 = ty2_8_1 + _t_5676;
	_t_5678 = -1.0f * _t_5677;
	_t_5679 = _t_5678 < 0.0f;
	if(_t_5679)
		{
			float _t_5680;
			float _t_5681;
		
			_t_5680 = -1.0f * ty3_9_1;
			_t_5681 = ty2_8_1 + _t_5680;
			_t_5682 = _t_5681;
		
		}
else
		{
			float _t_5683;
			float _t_5684;
			float _t_5685;
		
			_t_5683 = -1.0f * ty3_9_1;
			_t_5684 = ty2_8_1 + _t_5683;
			_t_5685 = -1.0f * _t_5684;
			_t_5682 = _t_5685;
		
		}

	_t_5686 = _t_5682 * _t_259;
	_t_5687 = 0.0f < _t_5686;
	if(_t_5687)
		{
		
			_t_5688 = px1_11_1;
		
		}
else
		{
		
			_t_5688 = px0_10_1;
		
		}

	_t_5689 = _t_5675 * _t_5688;
	_t_5690 = -1.0f * ty3_9_1;
	_t_5691 = ty2_8_1 + _t_5690;
	_t_5692 = -1.0f * _t_5691;
	_t_5693 = _t_5692 < 0.0f;
	if(_t_5693)
		{
			float _t_5694;
			float _t_5695;
		
			_t_5694 = -1.0f * tx2_5_1;
			_t_5695 = tx3_6_1 + _t_5694;
			_t_5696 = _t_5695;
		
		}
else
		{
			float _t_5697;
			float _t_5698;
			float _t_5699;
		
			_t_5697 = -1.0f * tx2_5_1;
			_t_5698 = tx3_6_1 + _t_5697;
			_t_5699 = -1.0f * _t_5698;
			_t_5696 = _t_5699;
		
		}

	_t_5700 = _t_5696 * _t_259;
	_t_5701 = -1.0f * ty3_9_1;
	_t_5702 = ty2_8_1 + _t_5701;
	_t_5703 = -1.0f * _t_5702;
	_t_5704 = _t_5703 < 0.0f;
	if(_t_5704)
		{
			float _t_5705;
			float _t_5706;
		
			_t_5705 = -1.0f * tx2_5_1;
			_t_5706 = tx3_6_1 + _t_5705;
			_t_5707 = _t_5706;
		
		}
else
		{
			float _t_5708;
			float _t_5709;
			float _t_5710;
		
			_t_5708 = -1.0f * tx2_5_1;
			_t_5709 = tx3_6_1 + _t_5708;
			_t_5710 = -1.0f * _t_5709;
			_t_5707 = _t_5710;
		
		}

	_t_5711 = _t_5707 * _t_259;
	_t_5712 = 0.0f < _t_5711;
	if(_t_5712)
		{
		
			_t_5713 = py1_13_1;
		
		}
else
		{
		
			_t_5713 = py0_12_1;
		
		}

	_t_5714 = _t_5700 * _t_5713;
	_t_5715 = _t_5689 + _t_5714;
	_t_5716 = -1.0f * ty3_9_1;
	_t_5717 = ty2_8_1 + _t_5716;
	_t_5718 = -1.0f * _t_5717;
	_t_5719 = _t_5718 < 0.0f;
	if(_t_5719)
		{
			float _t_5720;
			float _t_5721;
			float _t_5722;
			float _t_5723;
		
			_t_5720 = tx2_5_1 * ty3_9_1;
			_t_5721 = tx3_6_1 * ty2_8_1;
			_t_5722 = _t_5721 * -1.0f;
			_t_5723 = _t_5720 + _t_5722;
			_t_5724 = _t_5723;
		
		}
else
		{
			float _t_5725;
			float _t_5726;
			float _t_5727;
			float _t_5728;
			float _t_5729;
		
			_t_5725 = tx2_5_1 * ty3_9_1;
			_t_5726 = tx3_6_1 * ty2_8_1;
			_t_5727 = _t_5726 * -1.0f;
			_t_5728 = _t_5725 + _t_5727;
			_t_5729 = -1.0f * _t_5728;
			_t_5724 = _t_5729;
		
		}

	_t_5730 = -1.0f * _t_5724;
	_t_5731 = _t_5730 * _t_259;
	_t_5732 = _t_5731 * -1.0f;
	_t_5733 = _t_5715 + _t_5732;
	_t_5734 = 0.0f < _t_5733;
	_t_5735 = _t_5664 && _t_5734;
	if(_t_5735)
		{
			float _t_5736;
			float _t_5737;
			float _t_5738;
			bool _t_5739;
			float _t_5744;
			float _t_5750;
			float _t_5751;
			float _t_5752;
		
			_t_5736 = -1.0f * ty3_9_1;
			_t_5737 = ty2_8_1 + _t_5736;
			_t_5738 = -1.0f * _t_5737;
			_t_5739 = _t_5738 < 0.0f;
			if(_t_5739)
				{
					float _t_5740;
					float _t_5741;
					float _t_5742;
					float _t_5743;
				
					_t_5740 = tx2_5_1 * ty3_9_1;
					_t_5741 = tx3_6_1 * ty2_8_1;
					_t_5742 = _t_5741 * -1.0f;
					_t_5743 = _t_5740 + _t_5742;
					_t_5744 = _t_5743;
				
				}
		else
				{
					float _t_5745;
					float _t_5746;
					float _t_5747;
					float _t_5748;
					float _t_5749;
				
					_t_5745 = tx2_5_1 * ty3_9_1;
					_t_5746 = tx3_6_1 * ty2_8_1;
					_t_5747 = _t_5746 * -1.0f;
					_t_5748 = _t_5745 + _t_5747;
					_t_5749 = -1.0f * _t_5748;
					_t_5744 = _t_5749;
				
				}
		
			_t_5750 = -1.0f * _t_5744;
			_t_5751 = _t_5750 * _t_259;
			_t_5752 = tegpixellet_block_29(ty2_8_1,ty3_9_1,_t_259,_t_5751,tx3_6_1,tx2_5_1,y__2943_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1);
			_t_5594 = _t_5752;
		
		}
else
		{
		
			_t_5594 = 0.0f;
		
		}


	return _t_5594;
}
__device__ float tegpixelintegrator_22(float ty3_9_1,float pc1_15_1,float _t_259,float tc2_19_1,float ty2_8_1,float ty1_7_1,float pc0_14_1,float tx3_6_1,float tx1_4_1,float _t_5484,float tx2_5_1,float py1_13_1,float pc2_16_1,float px1_11_1,float tc0_17_1,float py0_12_1,float tc1_18_1,float px0_10_1,float _t_5593){
    float y__2943_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_5593 - _t_5484)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__2943_1 = _t_5484 + __step__ * (i + (float)(0.5));
        float _t_5594;
		_t_5594 = tegpixelbody_block_22(ty2_8_1,ty3_9_1,_t_259,px0_10_1,px1_11_1,tx3_6_1,tx2_5_1,py0_12_1,py1_13_1,y__2943_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);;
        __output__ = __output__ + _t_5594 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_6(float ty2_8_1,float ty3_9_1,float tx3_6_1,float tx2_5_1,float _t_259,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_5376;
	float _t_5377;
	float _t_5378;
	bool _t_5379;
	float _t_5382;
	float _t_5386;
	float _t_5387;
	float _t_5388;
	float _t_5389;
	float _t_5390;
	bool _t_5391;
	float _t_5394;
	float _t_5398;
	float _t_5399;
	bool _t_5400;
	float _t_5401;
	float _t_5402;
	float _t_5403;
	float _t_5404;
	float _t_5405;
	bool _t_5406;
	float _t_5409;
	float _t_5413;
	float _t_5414;
	float _t_5415;
	float _t_5416;
	bool _t_5417;
	float _t_5420;
	float _t_5424;
	float _t_5425;
	float _t_5426;
	float _t_5427;
	float _t_5428;
	bool _t_5429;
	float _t_5432;
	float _t_5436;
	float _t_5437;
	float _t_5438;
	float _t_5439;
	float _t_5440;
	float _t_5441;
	float _t_5442;
	float _t_5443;
	float _t_5444;
	bool _t_5445;
	float _t_5448;
	float _t_5452;
	float _t_5453;
	float _t_5454;
	float _t_5455;
	bool _t_5456;
	float _t_5459;
	float _t_5463;
	float _t_5464;
	float _t_5465;
	float _t_5466;
	float _t_5467;
	bool _t_5468;
	float _t_5471;
	float _t_5475;
	float _t_5476;
	float _t_5477;
	float _t_5478;
	float _t_5479;
	float _t_5480;
	bool _t_5481;
	float _t_5482;
	float _t_5483;
	float _t_5484;
	float _t_5485;
	float _t_5486;
	float _t_5487;
	bool _t_5488;
	float _t_5491;
	float _t_5495;
	float _t_5496;
	float _t_5497;
	float _t_5498;
	float _t_5499;
	bool _t_5500;
	float _t_5503;
	float _t_5507;
	float _t_5508;
	bool _t_5509;
	float _t_5510;
	float _t_5511;
	float _t_5512;
	float _t_5513;
	float _t_5514;
	bool _t_5515;
	float _t_5518;
	float _t_5522;
	float _t_5523;
	float _t_5524;
	float _t_5525;
	bool _t_5526;
	float _t_5529;
	float _t_5533;
	float _t_5534;
	float _t_5535;
	float _t_5536;
	float _t_5537;
	bool _t_5538;
	float _t_5541;
	float _t_5545;
	float _t_5546;
	float _t_5547;
	float _t_5548;
	float _t_5549;
	float _t_5550;
	float _t_5551;
	float _t_5552;
	float _t_5553;
	bool _t_5554;
	float _t_5557;
	float _t_5561;
	float _t_5562;
	float _t_5563;
	float _t_5564;
	bool _t_5565;
	float _t_5568;
	float _t_5572;
	float _t_5573;
	float _t_5574;
	float _t_5575;
	float _t_5576;
	bool _t_5577;
	float _t_5580;
	float _t_5584;
	float _t_5585;
	float _t_5586;
	float _t_5587;
	float _t_5588;
	float _t_5589;
	bool _t_5590;
	float _t_5591;
	float _t_5592;
	float _t_5593;

	float _t_260;

	_t_5376 = -1.0f * ty3_9_1;
	_t_5377 = ty2_8_1 + _t_5376;
	_t_5378 = -1.0f * _t_5377;
	_t_5379 = _t_5378 < 0.0f;
	if(_t_5379)
		{
			float _t_5380;
			float _t_5381;
		
			_t_5380 = -1.0f * tx2_5_1;
			_t_5381 = tx3_6_1 + _t_5380;
			_t_5382 = _t_5381;
		
		}
else
		{
			float _t_5383;
			float _t_5384;
			float _t_5385;
		
			_t_5383 = -1.0f * tx2_5_1;
			_t_5384 = tx3_6_1 + _t_5383;
			_t_5385 = -1.0f * _t_5384;
			_t_5382 = _t_5385;
		
		}

	_t_5386 = _t_5382 * _t_259;
	_t_5387 = _t_5386 * -1.0f;
	_t_5388 = -1.0f * ty3_9_1;
	_t_5389 = ty2_8_1 + _t_5388;
	_t_5390 = -1.0f * _t_5389;
	_t_5391 = _t_5390 < 0.0f;
	if(_t_5391)
		{
			float _t_5392;
			float _t_5393;
		
			_t_5392 = -1.0f * tx2_5_1;
			_t_5393 = tx3_6_1 + _t_5392;
			_t_5394 = _t_5393;
		
		}
else
		{
			float _t_5395;
			float _t_5396;
			float _t_5397;
		
			_t_5395 = -1.0f * tx2_5_1;
			_t_5396 = tx3_6_1 + _t_5395;
			_t_5397 = -1.0f * _t_5396;
			_t_5394 = _t_5397;
		
		}

	_t_5398 = _t_5394 * _t_259;
	_t_5399 = _t_5398 * -1.0f;
	_t_5400 = 0.0f < _t_5399;
	if(_t_5400)
		{
		
			_t_5401 = px0_10_1;
		
		}
else
		{
		
			_t_5401 = px1_11_1;
		
		}

	_t_5402 = _t_5387 * _t_5401;
	_t_5403 = -1.0f * ty3_9_1;
	_t_5404 = ty2_8_1 + _t_5403;
	_t_5405 = -1.0f * _t_5404;
	_t_5406 = _t_5405 < 0.0f;
	if(_t_5406)
		{
			float _t_5407;
			float _t_5408;
		
			_t_5407 = -1.0f * tx2_5_1;
			_t_5408 = tx3_6_1 + _t_5407;
			_t_5409 = _t_5408;
		
		}
else
		{
			float _t_5410;
			float _t_5411;
			float _t_5412;
		
			_t_5410 = -1.0f * tx2_5_1;
			_t_5411 = tx3_6_1 + _t_5410;
			_t_5412 = -1.0f * _t_5411;
			_t_5409 = _t_5412;
		
		}

	_t_5413 = _t_5409 * _t_259;
	_t_5414 = -1.0f * ty3_9_1;
	_t_5415 = ty2_8_1 + _t_5414;
	_t_5416 = -1.0f * _t_5415;
	_t_5417 = _t_5416 < 0.0f;
	if(_t_5417)
		{
			float _t_5418;
			float _t_5419;
		
			_t_5418 = -1.0f * tx2_5_1;
			_t_5419 = tx3_6_1 + _t_5418;
			_t_5420 = _t_5419;
		
		}
else
		{
			float _t_5421;
			float _t_5422;
			float _t_5423;
		
			_t_5421 = -1.0f * tx2_5_1;
			_t_5422 = tx3_6_1 + _t_5421;
			_t_5423 = -1.0f * _t_5422;
			_t_5420 = _t_5423;
		
		}

	_t_5424 = _t_5420 * _t_259;
	_t_5425 = _t_5413 * _t_5424;
	_t_5426 = -1.0f * ty3_9_1;
	_t_5427 = ty2_8_1 + _t_5426;
	_t_5428 = -1.0f * _t_5427;
	_t_5429 = _t_5428 < 0.0f;
	if(_t_5429)
		{
			float _t_5430;
			float _t_5431;
		
			_t_5430 = -1.0f * ty3_9_1;
			_t_5431 = ty2_8_1 + _t_5430;
			_t_5432 = _t_5431;
		
		}
else
		{
			float _t_5433;
			float _t_5434;
			float _t_5435;
		
			_t_5433 = -1.0f * ty3_9_1;
			_t_5434 = ty2_8_1 + _t_5433;
			_t_5435 = -1.0f * _t_5434;
			_t_5432 = _t_5435;
		
		}

	_t_5436 = _t_5432 * _t_259;
	_t_5437 = 1.0f + _t_5436;
	_t_5438 = 1.0f / _t_5437;
	_t_5439 = _t_5425 * _t_5438;
	_t_5440 = _t_5439 * -1.0f;
	_t_5441 = 1.0f + _t_5440;
	_t_5442 = -1.0f * ty3_9_1;
	_t_5443 = ty2_8_1 + _t_5442;
	_t_5444 = -1.0f * _t_5443;
	_t_5445 = _t_5444 < 0.0f;
	if(_t_5445)
		{
			float _t_5446;
			float _t_5447;
		
			_t_5446 = -1.0f * tx2_5_1;
			_t_5447 = tx3_6_1 + _t_5446;
			_t_5448 = _t_5447;
		
		}
else
		{
			float _t_5449;
			float _t_5450;
			float _t_5451;
		
			_t_5449 = -1.0f * tx2_5_1;
			_t_5450 = tx3_6_1 + _t_5449;
			_t_5451 = -1.0f * _t_5450;
			_t_5448 = _t_5451;
		
		}

	_t_5452 = _t_5448 * _t_259;
	_t_5453 = -1.0f * ty3_9_1;
	_t_5454 = ty2_8_1 + _t_5453;
	_t_5455 = -1.0f * _t_5454;
	_t_5456 = _t_5455 < 0.0f;
	if(_t_5456)
		{
			float _t_5457;
			float _t_5458;
		
			_t_5457 = -1.0f * tx2_5_1;
			_t_5458 = tx3_6_1 + _t_5457;
			_t_5459 = _t_5458;
		
		}
else
		{
			float _t_5460;
			float _t_5461;
			float _t_5462;
		
			_t_5460 = -1.0f * tx2_5_1;
			_t_5461 = tx3_6_1 + _t_5460;
			_t_5462 = -1.0f * _t_5461;
			_t_5459 = _t_5462;
		
		}

	_t_5463 = _t_5459 * _t_259;
	_t_5464 = _t_5452 * _t_5463;
	_t_5465 = -1.0f * ty3_9_1;
	_t_5466 = ty2_8_1 + _t_5465;
	_t_5467 = -1.0f * _t_5466;
	_t_5468 = _t_5467 < 0.0f;
	if(_t_5468)
		{
			float _t_5469;
			float _t_5470;
		
			_t_5469 = -1.0f * ty3_9_1;
			_t_5470 = ty2_8_1 + _t_5469;
			_t_5471 = _t_5470;
		
		}
else
		{
			float _t_5472;
			float _t_5473;
			float _t_5474;
		
			_t_5472 = -1.0f * ty3_9_1;
			_t_5473 = ty2_8_1 + _t_5472;
			_t_5474 = -1.0f * _t_5473;
			_t_5471 = _t_5474;
		
		}

	_t_5475 = _t_5471 * _t_259;
	_t_5476 = 1.0f + _t_5475;
	_t_5477 = 1.0f / _t_5476;
	_t_5478 = _t_5464 * _t_5477;
	_t_5479 = _t_5478 * -1.0f;
	_t_5480 = 1.0f + _t_5479;
	_t_5481 = 0.0f < _t_5480;
	if(_t_5481)
		{
		
			_t_5482 = py0_12_1;
		
		}
else
		{
		
			_t_5482 = py1_13_1;
		
		}

	_t_5483 = _t_5441 * _t_5482;
	_t_5484 = _t_5402 + _t_5483;
	_t_5485 = -1.0f * ty3_9_1;
	_t_5486 = ty2_8_1 + _t_5485;
	_t_5487 = -1.0f * _t_5486;
	_t_5488 = _t_5487 < 0.0f;
	if(_t_5488)
		{
			float _t_5489;
			float _t_5490;
		
			_t_5489 = -1.0f * tx2_5_1;
			_t_5490 = tx3_6_1 + _t_5489;
			_t_5491 = _t_5490;
		
		}
else
		{
			float _t_5492;
			float _t_5493;
			float _t_5494;
		
			_t_5492 = -1.0f * tx2_5_1;
			_t_5493 = tx3_6_1 + _t_5492;
			_t_5494 = -1.0f * _t_5493;
			_t_5491 = _t_5494;
		
		}

	_t_5495 = _t_5491 * _t_259;
	_t_5496 = _t_5495 * -1.0f;
	_t_5497 = -1.0f * ty3_9_1;
	_t_5498 = ty2_8_1 + _t_5497;
	_t_5499 = -1.0f * _t_5498;
	_t_5500 = _t_5499 < 0.0f;
	if(_t_5500)
		{
			float _t_5501;
			float _t_5502;
		
			_t_5501 = -1.0f * tx2_5_1;
			_t_5502 = tx3_6_1 + _t_5501;
			_t_5503 = _t_5502;
		
		}
else
		{
			float _t_5504;
			float _t_5505;
			float _t_5506;
		
			_t_5504 = -1.0f * tx2_5_1;
			_t_5505 = tx3_6_1 + _t_5504;
			_t_5506 = -1.0f * _t_5505;
			_t_5503 = _t_5506;
		
		}

	_t_5507 = _t_5503 * _t_259;
	_t_5508 = _t_5507 * -1.0f;
	_t_5509 = 0.0f < _t_5508;
	if(_t_5509)
		{
		
			_t_5510 = px1_11_1;
		
		}
else
		{
		
			_t_5510 = px0_10_1;
		
		}

	_t_5511 = _t_5496 * _t_5510;
	_t_5512 = -1.0f * ty3_9_1;
	_t_5513 = ty2_8_1 + _t_5512;
	_t_5514 = -1.0f * _t_5513;
	_t_5515 = _t_5514 < 0.0f;
	if(_t_5515)
		{
			float _t_5516;
			float _t_5517;
		
			_t_5516 = -1.0f * tx2_5_1;
			_t_5517 = tx3_6_1 + _t_5516;
			_t_5518 = _t_5517;
		
		}
else
		{
			float _t_5519;
			float _t_5520;
			float _t_5521;
		
			_t_5519 = -1.0f * tx2_5_1;
			_t_5520 = tx3_6_1 + _t_5519;
			_t_5521 = -1.0f * _t_5520;
			_t_5518 = _t_5521;
		
		}

	_t_5522 = _t_5518 * _t_259;
	_t_5523 = -1.0f * ty3_9_1;
	_t_5524 = ty2_8_1 + _t_5523;
	_t_5525 = -1.0f * _t_5524;
	_t_5526 = _t_5525 < 0.0f;
	if(_t_5526)
		{
			float _t_5527;
			float _t_5528;
		
			_t_5527 = -1.0f * tx2_5_1;
			_t_5528 = tx3_6_1 + _t_5527;
			_t_5529 = _t_5528;
		
		}
else
		{
			float _t_5530;
			float _t_5531;
			float _t_5532;
		
			_t_5530 = -1.0f * tx2_5_1;
			_t_5531 = tx3_6_1 + _t_5530;
			_t_5532 = -1.0f * _t_5531;
			_t_5529 = _t_5532;
		
		}

	_t_5533 = _t_5529 * _t_259;
	_t_5534 = _t_5522 * _t_5533;
	_t_5535 = -1.0f * ty3_9_1;
	_t_5536 = ty2_8_1 + _t_5535;
	_t_5537 = -1.0f * _t_5536;
	_t_5538 = _t_5537 < 0.0f;
	if(_t_5538)
		{
			float _t_5539;
			float _t_5540;
		
			_t_5539 = -1.0f * ty3_9_1;
			_t_5540 = ty2_8_1 + _t_5539;
			_t_5541 = _t_5540;
		
		}
else
		{
			float _t_5542;
			float _t_5543;
			float _t_5544;
		
			_t_5542 = -1.0f * ty3_9_1;
			_t_5543 = ty2_8_1 + _t_5542;
			_t_5544 = -1.0f * _t_5543;
			_t_5541 = _t_5544;
		
		}

	_t_5545 = _t_5541 * _t_259;
	_t_5546 = 1.0f + _t_5545;
	_t_5547 = 1.0f / _t_5546;
	_t_5548 = _t_5534 * _t_5547;
	_t_5549 = _t_5548 * -1.0f;
	_t_5550 = 1.0f + _t_5549;
	_t_5551 = -1.0f * ty3_9_1;
	_t_5552 = ty2_8_1 + _t_5551;
	_t_5553 = -1.0f * _t_5552;
	_t_5554 = _t_5553 < 0.0f;
	if(_t_5554)
		{
			float _t_5555;
			float _t_5556;
		
			_t_5555 = -1.0f * tx2_5_1;
			_t_5556 = tx3_6_1 + _t_5555;
			_t_5557 = _t_5556;
		
		}
else
		{
			float _t_5558;
			float _t_5559;
			float _t_5560;
		
			_t_5558 = -1.0f * tx2_5_1;
			_t_5559 = tx3_6_1 + _t_5558;
			_t_5560 = -1.0f * _t_5559;
			_t_5557 = _t_5560;
		
		}

	_t_5561 = _t_5557 * _t_259;
	_t_5562 = -1.0f * ty3_9_1;
	_t_5563 = ty2_8_1 + _t_5562;
	_t_5564 = -1.0f * _t_5563;
	_t_5565 = _t_5564 < 0.0f;
	if(_t_5565)
		{
			float _t_5566;
			float _t_5567;
		
			_t_5566 = -1.0f * tx2_5_1;
			_t_5567 = tx3_6_1 + _t_5566;
			_t_5568 = _t_5567;
		
		}
else
		{
			float _t_5569;
			float _t_5570;
			float _t_5571;
		
			_t_5569 = -1.0f * tx2_5_1;
			_t_5570 = tx3_6_1 + _t_5569;
			_t_5571 = -1.0f * _t_5570;
			_t_5568 = _t_5571;
		
		}

	_t_5572 = _t_5568 * _t_259;
	_t_5573 = _t_5561 * _t_5572;
	_t_5574 = -1.0f * ty3_9_1;
	_t_5575 = ty2_8_1 + _t_5574;
	_t_5576 = -1.0f * _t_5575;
	_t_5577 = _t_5576 < 0.0f;
	if(_t_5577)
		{
			float _t_5578;
			float _t_5579;
		
			_t_5578 = -1.0f * ty3_9_1;
			_t_5579 = ty2_8_1 + _t_5578;
			_t_5580 = _t_5579;
		
		}
else
		{
			float _t_5581;
			float _t_5582;
			float _t_5583;
		
			_t_5581 = -1.0f * ty3_9_1;
			_t_5582 = ty2_8_1 + _t_5581;
			_t_5583 = -1.0f * _t_5582;
			_t_5580 = _t_5583;
		
		}

	_t_5584 = _t_5580 * _t_259;
	_t_5585 = 1.0f + _t_5584;
	_t_5586 = 1.0f / _t_5585;
	_t_5587 = _t_5573 * _t_5586;
	_t_5588 = _t_5587 * -1.0f;
	_t_5589 = 1.0f + _t_5588;
	_t_5590 = 0.0f < _t_5589;
	if(_t_5590)
		{
		
			_t_5591 = py1_13_1;
		
		}
else
		{
		
			_t_5591 = py0_12_1;
		
		}

	_t_5592 = _t_5550 * _t_5591;
	_t_5593 = _t_5511 + _t_5592;
	_t_260 = tegpixelintegrator_22(ty3_9_1,pc1_15_1,_t_259,tc2_19_1,ty2_8_1,ty1_7_1,pc0_14_1,tx3_6_1,tx1_4_1,_t_5484,tx2_5_1,py1_13_1,pc2_16_1,px1_11_1,tc0_17_1,py0_12_1,tc1_18_1,px0_10_1,_t_5593);

	return _t_260;
}
__device__ float tegpixellet_block_32(float py0_12_1,float _t_6666,float py1_13_1,float px0_10_1,float _t_6613,float px1_11_1,float ty2_8_1,float ty3_9_1,float tx3_6_1,float tx2_5_1,float _t_287,float y__3017_1,float _t_6586,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	bool _t_6667;
	bool _t_6668;
	bool _t_6669;
	bool _t_6670;
	bool _t_6671;
	bool _t_6672;
	bool _t_6673;
	float _t_7003;
	float _t_7004;
	float _t_7005;
	float _t_7006;
	float _t_7007;
	float _t_7008;
	float _t_7009;
	float _t_7010;
	float _t_7011;
	float _t_7012;
	float _t_7013;
	float _t_7014;
	float _t_7015;
	float _t_7016;
	float _t_7017;
	float _t_7018;
	float _t_7019;
	float _t_7020;
	float _t_7021;
	float _t_7022;
	float _t_7023;
	float _t_7024;
	float _t_7025;
	float _t_7026;
	bool _t_7027;
	float _t_7028;
	float _t_7029;
	float _t_7030;
	float _t_7031;
	float _t_7032;
	float _t_7033;
	float _t_7034;
	float _t_7035;
	float _t_7036;
	float _t_7037;
	float _t_7038;
	float _t_7039;
	float _t_7040;
	float _t_7041;
	bool _t_7042;
	float _t_7043;
	float _t_7044;
	float _t_7045;
	float _t_7046;
	float _t_7047;
	float _t_7048;
	float _t_7049;
	float _t_7050;
	float _t_7051;
	float _t_7052;
	float _t_7053;
	float _t_7054;
	float _t_7055;
	float _t_7056;
	float _t_7057;
	float _t_7058;
	float _t_7059;
	float _t_7060;
	float _t_7061;
	float _t_7062;
	float _t_7063;
	float _t_7064;
	float _t_7065;
	float _t_7066;
	float _t_7067;
	float _t_7068;
	bool _t_7069;
	float _t_7070;
	float _t_7071;
	float _t_7072;
	float _t_7073;
	float _t_7074;
	float _t_7075;
	float _t_7076;
	float _t_7077;
	float _t_7078;
	float _t_7079;
	float _t_7080;
	float _t_7081;
	float _t_7082;
	float _t_7083;
	bool _t_7084;
	float _t_7085;
	float _t_7086;
	float _t_7087;
	float _t_7088;
	float _t_7089;

	float _t_6587;

	_t_6667 = py0_12_1 < _t_6666;
	_t_6668 = _t_6666 < py1_13_1;
	_t_6669 = _t_6667 && _t_6668;
	_t_6670 = px0_10_1 < _t_6613;
	_t_6671 = _t_6613 < px1_11_1;
	_t_6672 = _t_6670 && _t_6671;
	_t_6673 = _t_6669 && _t_6672;
	if(_t_6673)
		{
			float _t_6674;
			float _t_6675;
			float _t_6676;
			bool _t_6677;
			float _t_6680;
			float _t_6684;
			float _t_6685;
			float _t_6686;
			float _t_6687;
			float _t_6688;
			bool _t_6689;
			float _t_6692;
			float _t_6696;
			float _t_6697;
			bool _t_6698;
			float _t_6699;
			float _t_6700;
			float _t_6701;
			float _t_6702;
			float _t_6703;
			bool _t_6704;
			float _t_6707;
			float _t_6711;
			float _t_6712;
			float _t_6713;
			float _t_6714;
			bool _t_6715;
			float _t_6718;
			float _t_6722;
			float _t_6723;
			float _t_6724;
			float _t_6725;
			float _t_6726;
			bool _t_6727;
			float _t_6730;
			float _t_6734;
			float _t_6735;
			float _t_6736;
			float _t_6737;
			float _t_6738;
			float _t_6739;
			float _t_6740;
			float _t_6741;
			float _t_6742;
			bool _t_6743;
			float _t_6746;
			float _t_6750;
			float _t_6751;
			float _t_6752;
			float _t_6753;
			bool _t_6754;
			float _t_6757;
			float _t_6761;
			float _t_6762;
			float _t_6763;
			float _t_6764;
			float _t_6765;
			bool _t_6766;
			float _t_6769;
			float _t_6773;
			float _t_6774;
			float _t_6775;
			float _t_6776;
			float _t_6777;
			float _t_6778;
			bool _t_6779;
			float _t_6780;
			float _t_6781;
			float _t_6782;
			bool _t_6783;
			float _t_6784;
			float _t_6785;
			float _t_6786;
			bool _t_6787;
			float _t_6790;
			float _t_6794;
			float _t_6795;
			float _t_6796;
			float _t_6797;
			float _t_6798;
			bool _t_6799;
			float _t_6802;
			float _t_6806;
			float _t_6807;
			bool _t_6808;
			float _t_6809;
			float _t_6810;
			float _t_6811;
			float _t_6812;
			float _t_6813;
			bool _t_6814;
			float _t_6817;
			float _t_6821;
			float _t_6822;
			float _t_6823;
			float _t_6824;
			bool _t_6825;
			float _t_6828;
			float _t_6832;
			float _t_6833;
			float _t_6834;
			float _t_6835;
			float _t_6836;
			bool _t_6837;
			float _t_6840;
			float _t_6844;
			float _t_6845;
			float _t_6846;
			float _t_6847;
			float _t_6848;
			float _t_6849;
			float _t_6850;
			float _t_6851;
			float _t_6852;
			bool _t_6853;
			float _t_6856;
			float _t_6860;
			float _t_6861;
			float _t_6862;
			float _t_6863;
			bool _t_6864;
			float _t_6867;
			float _t_6871;
			float _t_6872;
			float _t_6873;
			float _t_6874;
			float _t_6875;
			bool _t_6876;
			float _t_6879;
			float _t_6883;
			float _t_6884;
			float _t_6885;
			float _t_6886;
			float _t_6887;
			float _t_6888;
			bool _t_6889;
			float _t_6890;
			float _t_6891;
			float _t_6892;
			bool _t_6893;
			bool _t_6894;
			float _t_6895;
			float _t_6896;
			float _t_6897;
			bool _t_6898;
			float _t_6901;
			float _t_6905;
			float _t_6906;
			float _t_6907;
			float _t_6908;
			bool _t_6909;
			float _t_6912;
			float _t_6916;
			bool _t_6917;
			float _t_6918;
			float _t_6919;
			float _t_6920;
			float _t_6921;
			float _t_6922;
			bool _t_6923;
			float _t_6926;
			float _t_6930;
			float _t_6931;
			float _t_6932;
			float _t_6933;
			bool _t_6934;
			float _t_6937;
			float _t_6941;
			bool _t_6942;
			float _t_6943;
			float _t_6944;
			float _t_6945;
			bool _t_6946;
			float _t_6947;
			float _t_6948;
			float _t_6949;
			bool _t_6950;
			float _t_6953;
			float _t_6957;
			float _t_6958;
			float _t_6959;
			float _t_6960;
			bool _t_6961;
			float _t_6964;
			float _t_6968;
			bool _t_6969;
			float _t_6970;
			float _t_6971;
			float _t_6972;
			float _t_6973;
			float _t_6974;
			bool _t_6975;
			float _t_6978;
			float _t_6982;
			float _t_6983;
			float _t_6984;
			float _t_6985;
			bool _t_6986;
			float _t_6989;
			float _t_6993;
			bool _t_6994;
			float _t_6995;
			float _t_6996;
			float _t_6997;
			bool _t_6998;
			bool _t_6999;
			bool _t_7000;
			float _t_7001;
			float _t_7002;
		
			_t_6674 = -1.0f * ty3_9_1;
			_t_6675 = ty2_8_1 + _t_6674;
			_t_6676 = -1.0f * _t_6675;
			_t_6677 = _t_6676 < 0.0f;
			if(_t_6677)
				{
					float _t_6678;
					float _t_6679;
				
					_t_6678 = -1.0f * tx2_5_1;
					_t_6679 = tx3_6_1 + _t_6678;
					_t_6680 = _t_6679;
				
				}
		else
				{
					float _t_6681;
					float _t_6682;
					float _t_6683;
				
					_t_6681 = -1.0f * tx2_5_1;
					_t_6682 = tx3_6_1 + _t_6681;
					_t_6683 = -1.0f * _t_6682;
					_t_6680 = _t_6683;
				
				}
		
			_t_6684 = _t_6680 * _t_287;
			_t_6685 = _t_6684 * -1.0f;
			_t_6686 = -1.0f * ty3_9_1;
			_t_6687 = ty2_8_1 + _t_6686;
			_t_6688 = -1.0f * _t_6687;
			_t_6689 = _t_6688 < 0.0f;
			if(_t_6689)
				{
					float _t_6690;
					float _t_6691;
				
					_t_6690 = -1.0f * tx2_5_1;
					_t_6691 = tx3_6_1 + _t_6690;
					_t_6692 = _t_6691;
				
				}
		else
				{
					float _t_6693;
					float _t_6694;
					float _t_6695;
				
					_t_6693 = -1.0f * tx2_5_1;
					_t_6694 = tx3_6_1 + _t_6693;
					_t_6695 = -1.0f * _t_6694;
					_t_6692 = _t_6695;
				
				}
		
			_t_6696 = _t_6692 * _t_287;
			_t_6697 = _t_6696 * -1.0f;
			_t_6698 = 0.0f < _t_6697;
			if(_t_6698)
				{
				
					_t_6699 = px0_10_1;
				
				}
		else
				{
				
					_t_6699 = px1_11_1;
				
				}
		
			_t_6700 = _t_6685 * _t_6699;
			_t_6701 = -1.0f * ty3_9_1;
			_t_6702 = ty2_8_1 + _t_6701;
			_t_6703 = -1.0f * _t_6702;
			_t_6704 = _t_6703 < 0.0f;
			if(_t_6704)
				{
					float _t_6705;
					float _t_6706;
				
					_t_6705 = -1.0f * tx2_5_1;
					_t_6706 = tx3_6_1 + _t_6705;
					_t_6707 = _t_6706;
				
				}
		else
				{
					float _t_6708;
					float _t_6709;
					float _t_6710;
				
					_t_6708 = -1.0f * tx2_5_1;
					_t_6709 = tx3_6_1 + _t_6708;
					_t_6710 = -1.0f * _t_6709;
					_t_6707 = _t_6710;
				
				}
		
			_t_6711 = _t_6707 * _t_287;
			_t_6712 = -1.0f * ty3_9_1;
			_t_6713 = ty2_8_1 + _t_6712;
			_t_6714 = -1.0f * _t_6713;
			_t_6715 = _t_6714 < 0.0f;
			if(_t_6715)
				{
					float _t_6716;
					float _t_6717;
				
					_t_6716 = -1.0f * tx2_5_1;
					_t_6717 = tx3_6_1 + _t_6716;
					_t_6718 = _t_6717;
				
				}
		else
				{
					float _t_6719;
					float _t_6720;
					float _t_6721;
				
					_t_6719 = -1.0f * tx2_5_1;
					_t_6720 = tx3_6_1 + _t_6719;
					_t_6721 = -1.0f * _t_6720;
					_t_6718 = _t_6721;
				
				}
		
			_t_6722 = _t_6718 * _t_287;
			_t_6723 = _t_6711 * _t_6722;
			_t_6724 = -1.0f * ty3_9_1;
			_t_6725 = ty2_8_1 + _t_6724;
			_t_6726 = -1.0f * _t_6725;
			_t_6727 = _t_6726 < 0.0f;
			if(_t_6727)
				{
					float _t_6728;
					float _t_6729;
				
					_t_6728 = -1.0f * ty3_9_1;
					_t_6729 = ty2_8_1 + _t_6728;
					_t_6730 = _t_6729;
				
				}
		else
				{
					float _t_6731;
					float _t_6732;
					float _t_6733;
				
					_t_6731 = -1.0f * ty3_9_1;
					_t_6732 = ty2_8_1 + _t_6731;
					_t_6733 = -1.0f * _t_6732;
					_t_6730 = _t_6733;
				
				}
		
			_t_6734 = _t_6730 * _t_287;
			_t_6735 = 1.0f + _t_6734;
			_t_6736 = 1.0f / _t_6735;
			_t_6737 = _t_6723 * _t_6736;
			_t_6738 = _t_6737 * -1.0f;
			_t_6739 = 1.0f + _t_6738;
			_t_6740 = -1.0f * ty3_9_1;
			_t_6741 = ty2_8_1 + _t_6740;
			_t_6742 = -1.0f * _t_6741;
			_t_6743 = _t_6742 < 0.0f;
			if(_t_6743)
				{
					float _t_6744;
					float _t_6745;
				
					_t_6744 = -1.0f * tx2_5_1;
					_t_6745 = tx3_6_1 + _t_6744;
					_t_6746 = _t_6745;
				
				}
		else
				{
					float _t_6747;
					float _t_6748;
					float _t_6749;
				
					_t_6747 = -1.0f * tx2_5_1;
					_t_6748 = tx3_6_1 + _t_6747;
					_t_6749 = -1.0f * _t_6748;
					_t_6746 = _t_6749;
				
				}
		
			_t_6750 = _t_6746 * _t_287;
			_t_6751 = -1.0f * ty3_9_1;
			_t_6752 = ty2_8_1 + _t_6751;
			_t_6753 = -1.0f * _t_6752;
			_t_6754 = _t_6753 < 0.0f;
			if(_t_6754)
				{
					float _t_6755;
					float _t_6756;
				
					_t_6755 = -1.0f * tx2_5_1;
					_t_6756 = tx3_6_1 + _t_6755;
					_t_6757 = _t_6756;
				
				}
		else
				{
					float _t_6758;
					float _t_6759;
					float _t_6760;
				
					_t_6758 = -1.0f * tx2_5_1;
					_t_6759 = tx3_6_1 + _t_6758;
					_t_6760 = -1.0f * _t_6759;
					_t_6757 = _t_6760;
				
				}
		
			_t_6761 = _t_6757 * _t_287;
			_t_6762 = _t_6750 * _t_6761;
			_t_6763 = -1.0f * ty3_9_1;
			_t_6764 = ty2_8_1 + _t_6763;
			_t_6765 = -1.0f * _t_6764;
			_t_6766 = _t_6765 < 0.0f;
			if(_t_6766)
				{
					float _t_6767;
					float _t_6768;
				
					_t_6767 = -1.0f * ty3_9_1;
					_t_6768 = ty2_8_1 + _t_6767;
					_t_6769 = _t_6768;
				
				}
		else
				{
					float _t_6770;
					float _t_6771;
					float _t_6772;
				
					_t_6770 = -1.0f * ty3_9_1;
					_t_6771 = ty2_8_1 + _t_6770;
					_t_6772 = -1.0f * _t_6771;
					_t_6769 = _t_6772;
				
				}
		
			_t_6773 = _t_6769 * _t_287;
			_t_6774 = 1.0f + _t_6773;
			_t_6775 = 1.0f / _t_6774;
			_t_6776 = _t_6762 * _t_6775;
			_t_6777 = _t_6776 * -1.0f;
			_t_6778 = 1.0f + _t_6777;
			_t_6779 = 0.0f < _t_6778;
			if(_t_6779)
				{
				
					_t_6780 = py0_12_1;
				
				}
		else
				{
				
					_t_6780 = py1_13_1;
				
				}
		
			_t_6781 = _t_6739 * _t_6780;
			_t_6782 = _t_6700 + _t_6781;
			_t_6783 = _t_6782 < y__3017_1;
			_t_6784 = -1.0f * ty3_9_1;
			_t_6785 = ty2_8_1 + _t_6784;
			_t_6786 = -1.0f * _t_6785;
			_t_6787 = _t_6786 < 0.0f;
			if(_t_6787)
				{
					float _t_6788;
					float _t_6789;
				
					_t_6788 = -1.0f * tx2_5_1;
					_t_6789 = tx3_6_1 + _t_6788;
					_t_6790 = _t_6789;
				
				}
		else
				{
					float _t_6791;
					float _t_6792;
					float _t_6793;
				
					_t_6791 = -1.0f * tx2_5_1;
					_t_6792 = tx3_6_1 + _t_6791;
					_t_6793 = -1.0f * _t_6792;
					_t_6790 = _t_6793;
				
				}
		
			_t_6794 = _t_6790 * _t_287;
			_t_6795 = _t_6794 * -1.0f;
			_t_6796 = -1.0f * ty3_9_1;
			_t_6797 = ty2_8_1 + _t_6796;
			_t_6798 = -1.0f * _t_6797;
			_t_6799 = _t_6798 < 0.0f;
			if(_t_6799)
				{
					float _t_6800;
					float _t_6801;
				
					_t_6800 = -1.0f * tx2_5_1;
					_t_6801 = tx3_6_1 + _t_6800;
					_t_6802 = _t_6801;
				
				}
		else
				{
					float _t_6803;
					float _t_6804;
					float _t_6805;
				
					_t_6803 = -1.0f * tx2_5_1;
					_t_6804 = tx3_6_1 + _t_6803;
					_t_6805 = -1.0f * _t_6804;
					_t_6802 = _t_6805;
				
				}
		
			_t_6806 = _t_6802 * _t_287;
			_t_6807 = _t_6806 * -1.0f;
			_t_6808 = 0.0f < _t_6807;
			if(_t_6808)
				{
				
					_t_6809 = px1_11_1;
				
				}
		else
				{
				
					_t_6809 = px0_10_1;
				
				}
		
			_t_6810 = _t_6795 * _t_6809;
			_t_6811 = -1.0f * ty3_9_1;
			_t_6812 = ty2_8_1 + _t_6811;
			_t_6813 = -1.0f * _t_6812;
			_t_6814 = _t_6813 < 0.0f;
			if(_t_6814)
				{
					float _t_6815;
					float _t_6816;
				
					_t_6815 = -1.0f * tx2_5_1;
					_t_6816 = tx3_6_1 + _t_6815;
					_t_6817 = _t_6816;
				
				}
		else
				{
					float _t_6818;
					float _t_6819;
					float _t_6820;
				
					_t_6818 = -1.0f * tx2_5_1;
					_t_6819 = tx3_6_1 + _t_6818;
					_t_6820 = -1.0f * _t_6819;
					_t_6817 = _t_6820;
				
				}
		
			_t_6821 = _t_6817 * _t_287;
			_t_6822 = -1.0f * ty3_9_1;
			_t_6823 = ty2_8_1 + _t_6822;
			_t_6824 = -1.0f * _t_6823;
			_t_6825 = _t_6824 < 0.0f;
			if(_t_6825)
				{
					float _t_6826;
					float _t_6827;
				
					_t_6826 = -1.0f * tx2_5_1;
					_t_6827 = tx3_6_1 + _t_6826;
					_t_6828 = _t_6827;
				
				}
		else
				{
					float _t_6829;
					float _t_6830;
					float _t_6831;
				
					_t_6829 = -1.0f * tx2_5_1;
					_t_6830 = tx3_6_1 + _t_6829;
					_t_6831 = -1.0f * _t_6830;
					_t_6828 = _t_6831;
				
				}
		
			_t_6832 = _t_6828 * _t_287;
			_t_6833 = _t_6821 * _t_6832;
			_t_6834 = -1.0f * ty3_9_1;
			_t_6835 = ty2_8_1 + _t_6834;
			_t_6836 = -1.0f * _t_6835;
			_t_6837 = _t_6836 < 0.0f;
			if(_t_6837)
				{
					float _t_6838;
					float _t_6839;
				
					_t_6838 = -1.0f * ty3_9_1;
					_t_6839 = ty2_8_1 + _t_6838;
					_t_6840 = _t_6839;
				
				}
		else
				{
					float _t_6841;
					float _t_6842;
					float _t_6843;
				
					_t_6841 = -1.0f * ty3_9_1;
					_t_6842 = ty2_8_1 + _t_6841;
					_t_6843 = -1.0f * _t_6842;
					_t_6840 = _t_6843;
				
				}
		
			_t_6844 = _t_6840 * _t_287;
			_t_6845 = 1.0f + _t_6844;
			_t_6846 = 1.0f / _t_6845;
			_t_6847 = _t_6833 * _t_6846;
			_t_6848 = _t_6847 * -1.0f;
			_t_6849 = 1.0f + _t_6848;
			_t_6850 = -1.0f * ty3_9_1;
			_t_6851 = ty2_8_1 + _t_6850;
			_t_6852 = -1.0f * _t_6851;
			_t_6853 = _t_6852 < 0.0f;
			if(_t_6853)
				{
					float _t_6854;
					float _t_6855;
				
					_t_6854 = -1.0f * tx2_5_1;
					_t_6855 = tx3_6_1 + _t_6854;
					_t_6856 = _t_6855;
				
				}
		else
				{
					float _t_6857;
					float _t_6858;
					float _t_6859;
				
					_t_6857 = -1.0f * tx2_5_1;
					_t_6858 = tx3_6_1 + _t_6857;
					_t_6859 = -1.0f * _t_6858;
					_t_6856 = _t_6859;
				
				}
		
			_t_6860 = _t_6856 * _t_287;
			_t_6861 = -1.0f * ty3_9_1;
			_t_6862 = ty2_8_1 + _t_6861;
			_t_6863 = -1.0f * _t_6862;
			_t_6864 = _t_6863 < 0.0f;
			if(_t_6864)
				{
					float _t_6865;
					float _t_6866;
				
					_t_6865 = -1.0f * tx2_5_1;
					_t_6866 = tx3_6_1 + _t_6865;
					_t_6867 = _t_6866;
				
				}
		else
				{
					float _t_6868;
					float _t_6869;
					float _t_6870;
				
					_t_6868 = -1.0f * tx2_5_1;
					_t_6869 = tx3_6_1 + _t_6868;
					_t_6870 = -1.0f * _t_6869;
					_t_6867 = _t_6870;
				
				}
		
			_t_6871 = _t_6867 * _t_287;
			_t_6872 = _t_6860 * _t_6871;
			_t_6873 = -1.0f * ty3_9_1;
			_t_6874 = ty2_8_1 + _t_6873;
			_t_6875 = -1.0f * _t_6874;
			_t_6876 = _t_6875 < 0.0f;
			if(_t_6876)
				{
					float _t_6877;
					float _t_6878;
				
					_t_6877 = -1.0f * ty3_9_1;
					_t_6878 = ty2_8_1 + _t_6877;
					_t_6879 = _t_6878;
				
				}
		else
				{
					float _t_6880;
					float _t_6881;
					float _t_6882;
				
					_t_6880 = -1.0f * ty3_9_1;
					_t_6881 = ty2_8_1 + _t_6880;
					_t_6882 = -1.0f * _t_6881;
					_t_6879 = _t_6882;
				
				}
		
			_t_6883 = _t_6879 * _t_287;
			_t_6884 = 1.0f + _t_6883;
			_t_6885 = 1.0f / _t_6884;
			_t_6886 = _t_6872 * _t_6885;
			_t_6887 = _t_6886 * -1.0f;
			_t_6888 = 1.0f + _t_6887;
			_t_6889 = 0.0f < _t_6888;
			if(_t_6889)
				{
				
					_t_6890 = py1_13_1;
				
				}
		else
				{
				
					_t_6890 = py0_12_1;
				
				}
		
			_t_6891 = _t_6849 * _t_6890;
			_t_6892 = _t_6810 + _t_6891;
			_t_6893 = y__3017_1 < _t_6892;
			_t_6894 = _t_6783 && _t_6893;
			_t_6895 = -1.0f * ty3_9_1;
			_t_6896 = ty2_8_1 + _t_6895;
			_t_6897 = -1.0f * _t_6896;
			_t_6898 = _t_6897 < 0.0f;
			if(_t_6898)
				{
					float _t_6899;
					float _t_6900;
				
					_t_6899 = -1.0f * ty3_9_1;
					_t_6900 = ty2_8_1 + _t_6899;
					_t_6901 = _t_6900;
				
				}
		else
				{
					float _t_6902;
					float _t_6903;
					float _t_6904;
				
					_t_6902 = -1.0f * ty3_9_1;
					_t_6903 = ty2_8_1 + _t_6902;
					_t_6904 = -1.0f * _t_6903;
					_t_6901 = _t_6904;
				
				}
		
			_t_6905 = _t_6901 * _t_287;
			_t_6906 = -1.0f * ty3_9_1;
			_t_6907 = ty2_8_1 + _t_6906;
			_t_6908 = -1.0f * _t_6907;
			_t_6909 = _t_6908 < 0.0f;
			if(_t_6909)
				{
					float _t_6910;
					float _t_6911;
				
					_t_6910 = -1.0f * ty3_9_1;
					_t_6911 = ty2_8_1 + _t_6910;
					_t_6912 = _t_6911;
				
				}
		else
				{
					float _t_6913;
					float _t_6914;
					float _t_6915;
				
					_t_6913 = -1.0f * ty3_9_1;
					_t_6914 = ty2_8_1 + _t_6913;
					_t_6915 = -1.0f * _t_6914;
					_t_6912 = _t_6915;
				
				}
		
			_t_6916 = _t_6912 * _t_287;
			_t_6917 = 0.0f < _t_6916;
			if(_t_6917)
				{
				
					_t_6918 = px0_10_1;
				
				}
		else
				{
				
					_t_6918 = px1_11_1;
				
				}
		
			_t_6919 = _t_6905 * _t_6918;
			_t_6920 = -1.0f * ty3_9_1;
			_t_6921 = ty2_8_1 + _t_6920;
			_t_6922 = -1.0f * _t_6921;
			_t_6923 = _t_6922 < 0.0f;
			if(_t_6923)
				{
					float _t_6924;
					float _t_6925;
				
					_t_6924 = -1.0f * tx2_5_1;
					_t_6925 = tx3_6_1 + _t_6924;
					_t_6926 = _t_6925;
				
				}
		else
				{
					float _t_6927;
					float _t_6928;
					float _t_6929;
				
					_t_6927 = -1.0f * tx2_5_1;
					_t_6928 = tx3_6_1 + _t_6927;
					_t_6929 = -1.0f * _t_6928;
					_t_6926 = _t_6929;
				
				}
		
			_t_6930 = _t_6926 * _t_287;
			_t_6931 = -1.0f * ty3_9_1;
			_t_6932 = ty2_8_1 + _t_6931;
			_t_6933 = -1.0f * _t_6932;
			_t_6934 = _t_6933 < 0.0f;
			if(_t_6934)
				{
					float _t_6935;
					float _t_6936;
				
					_t_6935 = -1.0f * tx2_5_1;
					_t_6936 = tx3_6_1 + _t_6935;
					_t_6937 = _t_6936;
				
				}
		else
				{
					float _t_6938;
					float _t_6939;
					float _t_6940;
				
					_t_6938 = -1.0f * tx2_5_1;
					_t_6939 = tx3_6_1 + _t_6938;
					_t_6940 = -1.0f * _t_6939;
					_t_6937 = _t_6940;
				
				}
		
			_t_6941 = _t_6937 * _t_287;
			_t_6942 = 0.0f < _t_6941;
			if(_t_6942)
				{
				
					_t_6943 = py0_12_1;
				
				}
		else
				{
				
					_t_6943 = py1_13_1;
				
				}
		
			_t_6944 = _t_6930 * _t_6943;
			_t_6945 = _t_6919 + _t_6944;
			_t_6946 = _t_6945 < _t_6586;
			_t_6947 = -1.0f * ty3_9_1;
			_t_6948 = ty2_8_1 + _t_6947;
			_t_6949 = -1.0f * _t_6948;
			_t_6950 = _t_6949 < 0.0f;
			if(_t_6950)
				{
					float _t_6951;
					float _t_6952;
				
					_t_6951 = -1.0f * ty3_9_1;
					_t_6952 = ty2_8_1 + _t_6951;
					_t_6953 = _t_6952;
				
				}
		else
				{
					float _t_6954;
					float _t_6955;
					float _t_6956;
				
					_t_6954 = -1.0f * ty3_9_1;
					_t_6955 = ty2_8_1 + _t_6954;
					_t_6956 = -1.0f * _t_6955;
					_t_6953 = _t_6956;
				
				}
		
			_t_6957 = _t_6953 * _t_287;
			_t_6958 = -1.0f * ty3_9_1;
			_t_6959 = ty2_8_1 + _t_6958;
			_t_6960 = -1.0f * _t_6959;
			_t_6961 = _t_6960 < 0.0f;
			if(_t_6961)
				{
					float _t_6962;
					float _t_6963;
				
					_t_6962 = -1.0f * ty3_9_1;
					_t_6963 = ty2_8_1 + _t_6962;
					_t_6964 = _t_6963;
				
				}
		else
				{
					float _t_6965;
					float _t_6966;
					float _t_6967;
				
					_t_6965 = -1.0f * ty3_9_1;
					_t_6966 = ty2_8_1 + _t_6965;
					_t_6967 = -1.0f * _t_6966;
					_t_6964 = _t_6967;
				
				}
		
			_t_6968 = _t_6964 * _t_287;
			_t_6969 = 0.0f < _t_6968;
			if(_t_6969)
				{
				
					_t_6970 = px1_11_1;
				
				}
		else
				{
				
					_t_6970 = px0_10_1;
				
				}
		
			_t_6971 = _t_6957 * _t_6970;
			_t_6972 = -1.0f * ty3_9_1;
			_t_6973 = ty2_8_1 + _t_6972;
			_t_6974 = -1.0f * _t_6973;
			_t_6975 = _t_6974 < 0.0f;
			if(_t_6975)
				{
					float _t_6976;
					float _t_6977;
				
					_t_6976 = -1.0f * tx2_5_1;
					_t_6977 = tx3_6_1 + _t_6976;
					_t_6978 = _t_6977;
				
				}
		else
				{
					float _t_6979;
					float _t_6980;
					float _t_6981;
				
					_t_6979 = -1.0f * tx2_5_1;
					_t_6980 = tx3_6_1 + _t_6979;
					_t_6981 = -1.0f * _t_6980;
					_t_6978 = _t_6981;
				
				}
		
			_t_6982 = _t_6978 * _t_287;
			_t_6983 = -1.0f * ty3_9_1;
			_t_6984 = ty2_8_1 + _t_6983;
			_t_6985 = -1.0f * _t_6984;
			_t_6986 = _t_6985 < 0.0f;
			if(_t_6986)
				{
					float _t_6987;
					float _t_6988;
				
					_t_6987 = -1.0f * tx2_5_1;
					_t_6988 = tx3_6_1 + _t_6987;
					_t_6989 = _t_6988;
				
				}
		else
				{
					float _t_6990;
					float _t_6991;
					float _t_6992;
				
					_t_6990 = -1.0f * tx2_5_1;
					_t_6991 = tx3_6_1 + _t_6990;
					_t_6992 = -1.0f * _t_6991;
					_t_6989 = _t_6992;
				
				}
		
			_t_6993 = _t_6989 * _t_287;
			_t_6994 = 0.0f < _t_6993;
			if(_t_6994)
				{
				
					_t_6995 = py1_13_1;
				
				}
		else
				{
				
					_t_6995 = py0_12_1;
				
				}
		
			_t_6996 = _t_6982 * _t_6995;
			_t_6997 = _t_6971 + _t_6996;
			_t_6998 = _t_6586 < _t_6997;
			_t_6999 = _t_6946 && _t_6998;
			_t_7000 = _t_6894 && _t_6999;
			if(_t_7000)
				{
				
					_t_7001 = 1.0f;
				
				}
		else
				{
				
					_t_7001 = 0.0f;
				
				}
		
			_t_7002 = _t_7001 * _t_287;
			_t_7003 = _t_7002;
		
		}
else
		{
		
			_t_7003 = 0.0f;
		
		}

	_t_7004 = -1.0f * pc0_14_1;
	_t_7005 = tc0_17_1 + _t_7004;
	_t_7006 = _t_7005 * _t_7005;
	_t_7007 = -1.0f * pc1_15_1;
	_t_7008 = tc1_18_1 + _t_7007;
	_t_7009 = _t_7008 * _t_7008;
	_t_7010 = _t_7006 + _t_7009;
	_t_7011 = -1.0f * pc2_16_1;
	_t_7012 = tc2_19_1 + _t_7011;
	_t_7013 = _t_7012 * _t_7012;
	_t_7014 = _t_7010 + _t_7013;
	_t_7015 = tx3_6_1 * ty1_7_1;
	_t_7016 = tx1_4_1 * ty3_9_1;
	_t_7017 = _t_7016 * -1.0f;
	_t_7018 = _t_7015 + _t_7017;
	_t_7019 = -1.0f * ty1_7_1;
	_t_7020 = ty3_9_1 + _t_7019;
	_t_7021 = _t_7020 * _t_6613;
	_t_7022 = _t_7018 + _t_7021;
	_t_7023 = -1.0f * tx3_6_1;
	_t_7024 = tx1_4_1 + _t_7023;
	_t_7025 = _t_7024 * _t_6666;
	_t_7026 = _t_7022 + _t_7025;
	_t_7027 = _t_7026 < 0.0f;
	if(_t_7027)
		{
		
			_t_7028 = 1.0f;
		
		}
else
		{
		
			_t_7028 = 0.0f;
		
		}

	_t_7029 = _t_7014 * _t_7028;
	_t_7030 = tx1_4_1 * ty2_8_1;
	_t_7031 = tx2_5_1 * ty1_7_1;
	_t_7032 = _t_7031 * -1.0f;
	_t_7033 = _t_7030 + _t_7032;
	_t_7034 = -1.0f * ty2_8_1;
	_t_7035 = ty1_7_1 + _t_7034;
	_t_7036 = _t_7035 * _t_6613;
	_t_7037 = _t_7033 + _t_7036;
	_t_7038 = -1.0f * tx1_4_1;
	_t_7039 = tx2_5_1 + _t_7038;
	_t_7040 = _t_7039 * _t_6666;
	_t_7041 = _t_7037 + _t_7040;
	_t_7042 = _t_7041 < 0.0f;
	if(_t_7042)
		{
		
			_t_7043 = 1.0f;
		
		}
else
		{
		
			_t_7043 = 0.0f;
		
		}

	_t_7044 = _t_7029 * _t_7043;
	_t_7045 = _t_7044 * ty2_8_1;
	_t_7046 = -1.0f * pc0_14_1;
	_t_7047 = tc0_17_1 + _t_7046;
	_t_7048 = _t_7047 * _t_7047;
	_t_7049 = -1.0f * pc1_15_1;
	_t_7050 = tc1_18_1 + _t_7049;
	_t_7051 = _t_7050 * _t_7050;
	_t_7052 = _t_7048 + _t_7051;
	_t_7053 = -1.0f * pc2_16_1;
	_t_7054 = tc2_19_1 + _t_7053;
	_t_7055 = _t_7054 * _t_7054;
	_t_7056 = _t_7052 + _t_7055;
	_t_7057 = tx3_6_1 * ty1_7_1;
	_t_7058 = tx1_4_1 * ty3_9_1;
	_t_7059 = _t_7058 * -1.0f;
	_t_7060 = _t_7057 + _t_7059;
	_t_7061 = -1.0f * ty1_7_1;
	_t_7062 = ty3_9_1 + _t_7061;
	_t_7063 = _t_7062 * _t_6613;
	_t_7064 = _t_7060 + _t_7063;
	_t_7065 = -1.0f * tx3_6_1;
	_t_7066 = tx1_4_1 + _t_7065;
	_t_7067 = _t_7066 * _t_6666;
	_t_7068 = _t_7064 + _t_7067;
	_t_7069 = _t_7068 < 0.0f;
	if(_t_7069)
		{
		
			_t_7070 = 1.0f;
		
		}
else
		{
		
			_t_7070 = 0.0f;
		
		}

	_t_7071 = _t_7056 * _t_7070;
	_t_7072 = tx1_4_1 * ty2_8_1;
	_t_7073 = tx2_5_1 * ty1_7_1;
	_t_7074 = _t_7073 * -1.0f;
	_t_7075 = _t_7072 + _t_7074;
	_t_7076 = -1.0f * ty2_8_1;
	_t_7077 = ty1_7_1 + _t_7076;
	_t_7078 = _t_7077 * _t_6613;
	_t_7079 = _t_7075 + _t_7078;
	_t_7080 = -1.0f * tx1_4_1;
	_t_7081 = tx2_5_1 + _t_7080;
	_t_7082 = _t_7081 * _t_6666;
	_t_7083 = _t_7079 + _t_7082;
	_t_7084 = _t_7083 < 0.0f;
	if(_t_7084)
		{
		
			_t_7085 = 1.0f;
		
		}
else
		{
		
			_t_7085 = 0.0f;
		
		}

	_t_7086 = _t_7071 * _t_7085;
	_t_7087 = _t_7086 * _t_6666;
	_t_7088 = _t_7087 * -1.0f;
	_t_7089 = _t_7045 + _t_7088;
	_t_6587 = _t_7003 * _t_7089;

	return _t_6587;
}
__device__ float tegpixellet_block_31(float ty2_8_1,float ty3_9_1,float _t_287,float _t_6586,float tx3_6_1,float tx2_5_1,float y__3017_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_6588;
	float _t_6589;
	float _t_6590;
	bool _t_6591;
	float _t_6594;
	float _t_6598;
	float _t_6599;
	float _t_6600;
	float _t_6601;
	float _t_6602;
	bool _t_6603;
	float _t_6606;
	float _t_6610;
	float _t_6611;
	float _t_6612;
	float _t_6613;
	float _t_6614;
	float _t_6615;
	float _t_6616;
	bool _t_6617;
	float _t_6620;
	float _t_6624;
	float _t_6625;
	float _t_6626;
	float _t_6627;
	bool _t_6628;
	float _t_6631;
	float _t_6635;
	float _t_6636;
	float _t_6637;
	float _t_6638;
	float _t_6639;
	bool _t_6640;
	float _t_6643;
	float _t_6647;
	float _t_6648;
	float _t_6649;
	float _t_6650;
	float _t_6651;
	float _t_6652;
	float _t_6653;
	float _t_6654;
	float _t_6655;
	float _t_6656;
	bool _t_6657;
	float _t_6660;
	float _t_6664;
	float _t_6665;
	float _t_6666;

	float _t_6587;

	_t_6588 = -1.0f * ty3_9_1;
	_t_6589 = ty2_8_1 + _t_6588;
	_t_6590 = -1.0f * _t_6589;
	_t_6591 = _t_6590 < 0.0f;
	if(_t_6591)
		{
			float _t_6592;
			float _t_6593;
		
			_t_6592 = -1.0f * ty3_9_1;
			_t_6593 = ty2_8_1 + _t_6592;
			_t_6594 = _t_6593;
		
		}
else
		{
			float _t_6595;
			float _t_6596;
			float _t_6597;
		
			_t_6595 = -1.0f * ty3_9_1;
			_t_6596 = ty2_8_1 + _t_6595;
			_t_6597 = -1.0f * _t_6596;
			_t_6594 = _t_6597;
		
		}

	_t_6598 = _t_6594 * _t_287;
	_t_6599 = _t_6598 * _t_6586;
	_t_6600 = -1.0f * ty3_9_1;
	_t_6601 = ty2_8_1 + _t_6600;
	_t_6602 = -1.0f * _t_6601;
	_t_6603 = _t_6602 < 0.0f;
	if(_t_6603)
		{
			float _t_6604;
			float _t_6605;
		
			_t_6604 = -1.0f * tx2_5_1;
			_t_6605 = tx3_6_1 + _t_6604;
			_t_6606 = _t_6605;
		
		}
else
		{
			float _t_6607;
			float _t_6608;
			float _t_6609;
		
			_t_6607 = -1.0f * tx2_5_1;
			_t_6608 = tx3_6_1 + _t_6607;
			_t_6609 = -1.0f * _t_6608;
			_t_6606 = _t_6609;
		
		}

	_t_6610 = _t_6606 * _t_287;
	_t_6611 = _t_6610 * -1.0f;
	_t_6612 = _t_6611 * y__3017_1;
	_t_6613 = _t_6599 + _t_6612;
	_t_6614 = -1.0f * ty3_9_1;
	_t_6615 = ty2_8_1 + _t_6614;
	_t_6616 = -1.0f * _t_6615;
	_t_6617 = _t_6616 < 0.0f;
	if(_t_6617)
		{
			float _t_6618;
			float _t_6619;
		
			_t_6618 = -1.0f * tx2_5_1;
			_t_6619 = tx3_6_1 + _t_6618;
			_t_6620 = _t_6619;
		
		}
else
		{
			float _t_6621;
			float _t_6622;
			float _t_6623;
		
			_t_6621 = -1.0f * tx2_5_1;
			_t_6622 = tx3_6_1 + _t_6621;
			_t_6623 = -1.0f * _t_6622;
			_t_6620 = _t_6623;
		
		}

	_t_6624 = _t_6620 * _t_287;
	_t_6625 = -1.0f * ty3_9_1;
	_t_6626 = ty2_8_1 + _t_6625;
	_t_6627 = -1.0f * _t_6626;
	_t_6628 = _t_6627 < 0.0f;
	if(_t_6628)
		{
			float _t_6629;
			float _t_6630;
		
			_t_6629 = -1.0f * tx2_5_1;
			_t_6630 = tx3_6_1 + _t_6629;
			_t_6631 = _t_6630;
		
		}
else
		{
			float _t_6632;
			float _t_6633;
			float _t_6634;
		
			_t_6632 = -1.0f * tx2_5_1;
			_t_6633 = tx3_6_1 + _t_6632;
			_t_6634 = -1.0f * _t_6633;
			_t_6631 = _t_6634;
		
		}

	_t_6635 = _t_6631 * _t_287;
	_t_6636 = _t_6624 * _t_6635;
	_t_6637 = -1.0f * ty3_9_1;
	_t_6638 = ty2_8_1 + _t_6637;
	_t_6639 = -1.0f * _t_6638;
	_t_6640 = _t_6639 < 0.0f;
	if(_t_6640)
		{
			float _t_6641;
			float _t_6642;
		
			_t_6641 = -1.0f * ty3_9_1;
			_t_6642 = ty2_8_1 + _t_6641;
			_t_6643 = _t_6642;
		
		}
else
		{
			float _t_6644;
			float _t_6645;
			float _t_6646;
		
			_t_6644 = -1.0f * ty3_9_1;
			_t_6645 = ty2_8_1 + _t_6644;
			_t_6646 = -1.0f * _t_6645;
			_t_6643 = _t_6646;
		
		}

	_t_6647 = _t_6643 * _t_287;
	_t_6648 = 1.0f + _t_6647;
	_t_6649 = 1.0f / _t_6648;
	_t_6650 = _t_6636 * _t_6649;
	_t_6651 = _t_6650 * -1.0f;
	_t_6652 = 1.0f + _t_6651;
	_t_6653 = _t_6652 * y__3017_1;
	_t_6654 = -1.0f * ty3_9_1;
	_t_6655 = ty2_8_1 + _t_6654;
	_t_6656 = -1.0f * _t_6655;
	_t_6657 = _t_6656 < 0.0f;
	if(_t_6657)
		{
			float _t_6658;
			float _t_6659;
		
			_t_6658 = -1.0f * tx2_5_1;
			_t_6659 = tx3_6_1 + _t_6658;
			_t_6660 = _t_6659;
		
		}
else
		{
			float _t_6661;
			float _t_6662;
			float _t_6663;
		
			_t_6661 = -1.0f * tx2_5_1;
			_t_6662 = tx3_6_1 + _t_6661;
			_t_6663 = -1.0f * _t_6662;
			_t_6660 = _t_6663;
		
		}

	_t_6664 = _t_6660 * _t_287;
	_t_6665 = _t_6664 * _t_6586;
	_t_6666 = _t_6653 + _t_6665;
	_t_6587 = tegpixellet_block_32(py0_12_1,_t_6666,py1_13_1,px0_10_1,_t_6613,px1_11_1,ty2_8_1,ty3_9_1,tx3_6_1,tx2_5_1,_t_287,y__3017_1,_t_6586,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);

	return _t_6587;
}
__device__ float tegpixelbody_block_23(float ty2_8_1,float ty3_9_1,float _t_287,float px0_10_1,float px1_11_1,float tx3_6_1,float tx2_5_1,float py0_12_1,float py1_13_1,float y__3017_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_6430;
	float _t_6431;
	float _t_6432;
	bool _t_6433;
	float _t_6436;
	float _t_6440;
	float _t_6441;
	float _t_6442;
	float _t_6443;
	bool _t_6444;
	float _t_6447;
	float _t_6451;
	bool _t_6452;
	float _t_6453;
	float _t_6454;
	float _t_6455;
	float _t_6456;
	float _t_6457;
	bool _t_6458;
	float _t_6461;
	float _t_6465;
	float _t_6466;
	float _t_6467;
	float _t_6468;
	bool _t_6469;
	float _t_6472;
	float _t_6476;
	bool _t_6477;
	float _t_6478;
	float _t_6479;
	float _t_6480;
	float _t_6481;
	float _t_6482;
	float _t_6483;
	bool _t_6484;
	float _t_6489;
	float _t_6495;
	float _t_6496;
	float _t_6497;
	float _t_6498;
	bool _t_6499;
	float _t_6500;
	float _t_6501;
	float _t_6502;
	bool _t_6503;
	float _t_6506;
	float _t_6510;
	float _t_6511;
	float _t_6512;
	float _t_6513;
	bool _t_6514;
	float _t_6517;
	float _t_6521;
	bool _t_6522;
	float _t_6523;
	float _t_6524;
	float _t_6525;
	float _t_6526;
	float _t_6527;
	bool _t_6528;
	float _t_6531;
	float _t_6535;
	float _t_6536;
	float _t_6537;
	float _t_6538;
	bool _t_6539;
	float _t_6542;
	float _t_6546;
	bool _t_6547;
	float _t_6548;
	float _t_6549;
	float _t_6550;
	float _t_6551;
	float _t_6552;
	float _t_6553;
	bool _t_6554;
	float _t_6559;
	float _t_6565;
	float _t_6566;
	float _t_6567;
	float _t_6568;
	bool _t_6569;
	bool _t_6570;

	float _t_6429;

	_t_6430 = -1.0f * ty3_9_1;
	_t_6431 = ty2_8_1 + _t_6430;
	_t_6432 = -1.0f * _t_6431;
	_t_6433 = _t_6432 < 0.0f;
	if(_t_6433)
		{
			float _t_6434;
			float _t_6435;
		
			_t_6434 = -1.0f * ty3_9_1;
			_t_6435 = ty2_8_1 + _t_6434;
			_t_6436 = _t_6435;
		
		}
else
		{
			float _t_6437;
			float _t_6438;
			float _t_6439;
		
			_t_6437 = -1.0f * ty3_9_1;
			_t_6438 = ty2_8_1 + _t_6437;
			_t_6439 = -1.0f * _t_6438;
			_t_6436 = _t_6439;
		
		}

	_t_6440 = _t_6436 * _t_287;
	_t_6441 = -1.0f * ty3_9_1;
	_t_6442 = ty2_8_1 + _t_6441;
	_t_6443 = -1.0f * _t_6442;
	_t_6444 = _t_6443 < 0.0f;
	if(_t_6444)
		{
			float _t_6445;
			float _t_6446;
		
			_t_6445 = -1.0f * ty3_9_1;
			_t_6446 = ty2_8_1 + _t_6445;
			_t_6447 = _t_6446;
		
		}
else
		{
			float _t_6448;
			float _t_6449;
			float _t_6450;
		
			_t_6448 = -1.0f * ty3_9_1;
			_t_6449 = ty2_8_1 + _t_6448;
			_t_6450 = -1.0f * _t_6449;
			_t_6447 = _t_6450;
		
		}

	_t_6451 = _t_6447 * _t_287;
	_t_6452 = 0.0f < _t_6451;
	if(_t_6452)
		{
		
			_t_6453 = px0_10_1;
		
		}
else
		{
		
			_t_6453 = px1_11_1;
		
		}

	_t_6454 = _t_6440 * _t_6453;
	_t_6455 = -1.0f * ty3_9_1;
	_t_6456 = ty2_8_1 + _t_6455;
	_t_6457 = -1.0f * _t_6456;
	_t_6458 = _t_6457 < 0.0f;
	if(_t_6458)
		{
			float _t_6459;
			float _t_6460;
		
			_t_6459 = -1.0f * tx2_5_1;
			_t_6460 = tx3_6_1 + _t_6459;
			_t_6461 = _t_6460;
		
		}
else
		{
			float _t_6462;
			float _t_6463;
			float _t_6464;
		
			_t_6462 = -1.0f * tx2_5_1;
			_t_6463 = tx3_6_1 + _t_6462;
			_t_6464 = -1.0f * _t_6463;
			_t_6461 = _t_6464;
		
		}

	_t_6465 = _t_6461 * _t_287;
	_t_6466 = -1.0f * ty3_9_1;
	_t_6467 = ty2_8_1 + _t_6466;
	_t_6468 = -1.0f * _t_6467;
	_t_6469 = _t_6468 < 0.0f;
	if(_t_6469)
		{
			float _t_6470;
			float _t_6471;
		
			_t_6470 = -1.0f * tx2_5_1;
			_t_6471 = tx3_6_1 + _t_6470;
			_t_6472 = _t_6471;
		
		}
else
		{
			float _t_6473;
			float _t_6474;
			float _t_6475;
		
			_t_6473 = -1.0f * tx2_5_1;
			_t_6474 = tx3_6_1 + _t_6473;
			_t_6475 = -1.0f * _t_6474;
			_t_6472 = _t_6475;
		
		}

	_t_6476 = _t_6472 * _t_287;
	_t_6477 = 0.0f < _t_6476;
	if(_t_6477)
		{
		
			_t_6478 = py0_12_1;
		
		}
else
		{
		
			_t_6478 = py1_13_1;
		
		}

	_t_6479 = _t_6465 * _t_6478;
	_t_6480 = _t_6454 + _t_6479;
	_t_6481 = -1.0f * ty3_9_1;
	_t_6482 = ty2_8_1 + _t_6481;
	_t_6483 = -1.0f * _t_6482;
	_t_6484 = _t_6483 < 0.0f;
	if(_t_6484)
		{
			float _t_6485;
			float _t_6486;
			float _t_6487;
			float _t_6488;
		
			_t_6485 = tx2_5_1 * ty3_9_1;
			_t_6486 = tx3_6_1 * ty2_8_1;
			_t_6487 = _t_6486 * -1.0f;
			_t_6488 = _t_6485 + _t_6487;
			_t_6489 = _t_6488;
		
		}
else
		{
			float _t_6490;
			float _t_6491;
			float _t_6492;
			float _t_6493;
			float _t_6494;
		
			_t_6490 = tx2_5_1 * ty3_9_1;
			_t_6491 = tx3_6_1 * ty2_8_1;
			_t_6492 = _t_6491 * -1.0f;
			_t_6493 = _t_6490 + _t_6492;
			_t_6494 = -1.0f * _t_6493;
			_t_6489 = _t_6494;
		
		}

	_t_6495 = -1.0f * _t_6489;
	_t_6496 = _t_6495 * _t_287;
	_t_6497 = _t_6496 * -1.0f;
	_t_6498 = _t_6480 + _t_6497;
	_t_6499 = _t_6498 < 0.0f;
	_t_6500 = -1.0f * ty3_9_1;
	_t_6501 = ty2_8_1 + _t_6500;
	_t_6502 = -1.0f * _t_6501;
	_t_6503 = _t_6502 < 0.0f;
	if(_t_6503)
		{
			float _t_6504;
			float _t_6505;
		
			_t_6504 = -1.0f * ty3_9_1;
			_t_6505 = ty2_8_1 + _t_6504;
			_t_6506 = _t_6505;
		
		}
else
		{
			float _t_6507;
			float _t_6508;
			float _t_6509;
		
			_t_6507 = -1.0f * ty3_9_1;
			_t_6508 = ty2_8_1 + _t_6507;
			_t_6509 = -1.0f * _t_6508;
			_t_6506 = _t_6509;
		
		}

	_t_6510 = _t_6506 * _t_287;
	_t_6511 = -1.0f * ty3_9_1;
	_t_6512 = ty2_8_1 + _t_6511;
	_t_6513 = -1.0f * _t_6512;
	_t_6514 = _t_6513 < 0.0f;
	if(_t_6514)
		{
			float _t_6515;
			float _t_6516;
		
			_t_6515 = -1.0f * ty3_9_1;
			_t_6516 = ty2_8_1 + _t_6515;
			_t_6517 = _t_6516;
		
		}
else
		{
			float _t_6518;
			float _t_6519;
			float _t_6520;
		
			_t_6518 = -1.0f * ty3_9_1;
			_t_6519 = ty2_8_1 + _t_6518;
			_t_6520 = -1.0f * _t_6519;
			_t_6517 = _t_6520;
		
		}

	_t_6521 = _t_6517 * _t_287;
	_t_6522 = 0.0f < _t_6521;
	if(_t_6522)
		{
		
			_t_6523 = px1_11_1;
		
		}
else
		{
		
			_t_6523 = px0_10_1;
		
		}

	_t_6524 = _t_6510 * _t_6523;
	_t_6525 = -1.0f * ty3_9_1;
	_t_6526 = ty2_8_1 + _t_6525;
	_t_6527 = -1.0f * _t_6526;
	_t_6528 = _t_6527 < 0.0f;
	if(_t_6528)
		{
			float _t_6529;
			float _t_6530;
		
			_t_6529 = -1.0f * tx2_5_1;
			_t_6530 = tx3_6_1 + _t_6529;
			_t_6531 = _t_6530;
		
		}
else
		{
			float _t_6532;
			float _t_6533;
			float _t_6534;
		
			_t_6532 = -1.0f * tx2_5_1;
			_t_6533 = tx3_6_1 + _t_6532;
			_t_6534 = -1.0f * _t_6533;
			_t_6531 = _t_6534;
		
		}

	_t_6535 = _t_6531 * _t_287;
	_t_6536 = -1.0f * ty3_9_1;
	_t_6537 = ty2_8_1 + _t_6536;
	_t_6538 = -1.0f * _t_6537;
	_t_6539 = _t_6538 < 0.0f;
	if(_t_6539)
		{
			float _t_6540;
			float _t_6541;
		
			_t_6540 = -1.0f * tx2_5_1;
			_t_6541 = tx3_6_1 + _t_6540;
			_t_6542 = _t_6541;
		
		}
else
		{
			float _t_6543;
			float _t_6544;
			float _t_6545;
		
			_t_6543 = -1.0f * tx2_5_1;
			_t_6544 = tx3_6_1 + _t_6543;
			_t_6545 = -1.0f * _t_6544;
			_t_6542 = _t_6545;
		
		}

	_t_6546 = _t_6542 * _t_287;
	_t_6547 = 0.0f < _t_6546;
	if(_t_6547)
		{
		
			_t_6548 = py1_13_1;
		
		}
else
		{
		
			_t_6548 = py0_12_1;
		
		}

	_t_6549 = _t_6535 * _t_6548;
	_t_6550 = _t_6524 + _t_6549;
	_t_6551 = -1.0f * ty3_9_1;
	_t_6552 = ty2_8_1 + _t_6551;
	_t_6553 = -1.0f * _t_6552;
	_t_6554 = _t_6553 < 0.0f;
	if(_t_6554)
		{
			float _t_6555;
			float _t_6556;
			float _t_6557;
			float _t_6558;
		
			_t_6555 = tx2_5_1 * ty3_9_1;
			_t_6556 = tx3_6_1 * ty2_8_1;
			_t_6557 = _t_6556 * -1.0f;
			_t_6558 = _t_6555 + _t_6557;
			_t_6559 = _t_6558;
		
		}
else
		{
			float _t_6560;
			float _t_6561;
			float _t_6562;
			float _t_6563;
			float _t_6564;
		
			_t_6560 = tx2_5_1 * ty3_9_1;
			_t_6561 = tx3_6_1 * ty2_8_1;
			_t_6562 = _t_6561 * -1.0f;
			_t_6563 = _t_6560 + _t_6562;
			_t_6564 = -1.0f * _t_6563;
			_t_6559 = _t_6564;
		
		}

	_t_6565 = -1.0f * _t_6559;
	_t_6566 = _t_6565 * _t_287;
	_t_6567 = _t_6566 * -1.0f;
	_t_6568 = _t_6550 + _t_6567;
	_t_6569 = 0.0f < _t_6568;
	_t_6570 = _t_6499 && _t_6569;
	if(_t_6570)
		{
			float _t_6571;
			float _t_6572;
			float _t_6573;
			bool _t_6574;
			float _t_6579;
			float _t_6585;
			float _t_6586;
			float _t_6587;
		
			_t_6571 = -1.0f * ty3_9_1;
			_t_6572 = ty2_8_1 + _t_6571;
			_t_6573 = -1.0f * _t_6572;
			_t_6574 = _t_6573 < 0.0f;
			if(_t_6574)
				{
					float _t_6575;
					float _t_6576;
					float _t_6577;
					float _t_6578;
				
					_t_6575 = tx2_5_1 * ty3_9_1;
					_t_6576 = tx3_6_1 * ty2_8_1;
					_t_6577 = _t_6576 * -1.0f;
					_t_6578 = _t_6575 + _t_6577;
					_t_6579 = _t_6578;
				
				}
		else
				{
					float _t_6580;
					float _t_6581;
					float _t_6582;
					float _t_6583;
					float _t_6584;
				
					_t_6580 = tx2_5_1 * ty3_9_1;
					_t_6581 = tx3_6_1 * ty2_8_1;
					_t_6582 = _t_6581 * -1.0f;
					_t_6583 = _t_6580 + _t_6582;
					_t_6584 = -1.0f * _t_6583;
					_t_6579 = _t_6584;
				
				}
		
			_t_6585 = -1.0f * _t_6579;
			_t_6586 = _t_6585 * _t_287;
			_t_6587 = tegpixellet_block_31(ty2_8_1,ty3_9_1,_t_287,_t_6586,tx3_6_1,tx2_5_1,y__3017_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);
			_t_6429 = _t_6587;
		
		}
else
		{
		
			_t_6429 = 0.0f;
		
		}


	return _t_6429;
}
__device__ float tegpixelintegrator_23(float ty3_9_1,float pc1_15_1,float _t_6319,float tc2_19_1,float ty2_8_1,float ty1_7_1,float pc0_14_1,float tx3_6_1,float tx1_4_1,float tx2_5_1,float py1_13_1,float pc2_16_1,float px1_11_1,float tc0_17_1,float _t_6428,float py0_12_1,float tc1_18_1,float px0_10_1,float _t_287){
    float y__3017_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_6428 - _t_6319)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__3017_1 = _t_6319 + __step__ * (i + (float)(0.5));
        float _t_6429;
		_t_6429 = tegpixelbody_block_23(ty2_8_1,ty3_9_1,_t_287,px0_10_1,px1_11_1,tx3_6_1,tx2_5_1,py0_12_1,py1_13_1,y__3017_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);;
        __output__ = __output__ + _t_6429 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_7(float ty2_8_1,float ty3_9_1,float tx3_6_1,float tx2_5_1,float _t_287,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_6211;
	float _t_6212;
	float _t_6213;
	bool _t_6214;
	float _t_6217;
	float _t_6221;
	float _t_6222;
	float _t_6223;
	float _t_6224;
	float _t_6225;
	bool _t_6226;
	float _t_6229;
	float _t_6233;
	float _t_6234;
	bool _t_6235;
	float _t_6236;
	float _t_6237;
	float _t_6238;
	float _t_6239;
	float _t_6240;
	bool _t_6241;
	float _t_6244;
	float _t_6248;
	float _t_6249;
	float _t_6250;
	float _t_6251;
	bool _t_6252;
	float _t_6255;
	float _t_6259;
	float _t_6260;
	float _t_6261;
	float _t_6262;
	float _t_6263;
	bool _t_6264;
	float _t_6267;
	float _t_6271;
	float _t_6272;
	float _t_6273;
	float _t_6274;
	float _t_6275;
	float _t_6276;
	float _t_6277;
	float _t_6278;
	float _t_6279;
	bool _t_6280;
	float _t_6283;
	float _t_6287;
	float _t_6288;
	float _t_6289;
	float _t_6290;
	bool _t_6291;
	float _t_6294;
	float _t_6298;
	float _t_6299;
	float _t_6300;
	float _t_6301;
	float _t_6302;
	bool _t_6303;
	float _t_6306;
	float _t_6310;
	float _t_6311;
	float _t_6312;
	float _t_6313;
	float _t_6314;
	float _t_6315;
	bool _t_6316;
	float _t_6317;
	float _t_6318;
	float _t_6319;
	float _t_6320;
	float _t_6321;
	float _t_6322;
	bool _t_6323;
	float _t_6326;
	float _t_6330;
	float _t_6331;
	float _t_6332;
	float _t_6333;
	float _t_6334;
	bool _t_6335;
	float _t_6338;
	float _t_6342;
	float _t_6343;
	bool _t_6344;
	float _t_6345;
	float _t_6346;
	float _t_6347;
	float _t_6348;
	float _t_6349;
	bool _t_6350;
	float _t_6353;
	float _t_6357;
	float _t_6358;
	float _t_6359;
	float _t_6360;
	bool _t_6361;
	float _t_6364;
	float _t_6368;
	float _t_6369;
	float _t_6370;
	float _t_6371;
	float _t_6372;
	bool _t_6373;
	float _t_6376;
	float _t_6380;
	float _t_6381;
	float _t_6382;
	float _t_6383;
	float _t_6384;
	float _t_6385;
	float _t_6386;
	float _t_6387;
	float _t_6388;
	bool _t_6389;
	float _t_6392;
	float _t_6396;
	float _t_6397;
	float _t_6398;
	float _t_6399;
	bool _t_6400;
	float _t_6403;
	float _t_6407;
	float _t_6408;
	float _t_6409;
	float _t_6410;
	float _t_6411;
	bool _t_6412;
	float _t_6415;
	float _t_6419;
	float _t_6420;
	float _t_6421;
	float _t_6422;
	float _t_6423;
	float _t_6424;
	bool _t_6425;
	float _t_6426;
	float _t_6427;
	float _t_6428;

	float _t_288;

	_t_6211 = -1.0f * ty3_9_1;
	_t_6212 = ty2_8_1 + _t_6211;
	_t_6213 = -1.0f * _t_6212;
	_t_6214 = _t_6213 < 0.0f;
	if(_t_6214)
		{
			float _t_6215;
			float _t_6216;
		
			_t_6215 = -1.0f * tx2_5_1;
			_t_6216 = tx3_6_1 + _t_6215;
			_t_6217 = _t_6216;
		
		}
else
		{
			float _t_6218;
			float _t_6219;
			float _t_6220;
		
			_t_6218 = -1.0f * tx2_5_1;
			_t_6219 = tx3_6_1 + _t_6218;
			_t_6220 = -1.0f * _t_6219;
			_t_6217 = _t_6220;
		
		}

	_t_6221 = _t_6217 * _t_287;
	_t_6222 = _t_6221 * -1.0f;
	_t_6223 = -1.0f * ty3_9_1;
	_t_6224 = ty2_8_1 + _t_6223;
	_t_6225 = -1.0f * _t_6224;
	_t_6226 = _t_6225 < 0.0f;
	if(_t_6226)
		{
			float _t_6227;
			float _t_6228;
		
			_t_6227 = -1.0f * tx2_5_1;
			_t_6228 = tx3_6_1 + _t_6227;
			_t_6229 = _t_6228;
		
		}
else
		{
			float _t_6230;
			float _t_6231;
			float _t_6232;
		
			_t_6230 = -1.0f * tx2_5_1;
			_t_6231 = tx3_6_1 + _t_6230;
			_t_6232 = -1.0f * _t_6231;
			_t_6229 = _t_6232;
		
		}

	_t_6233 = _t_6229 * _t_287;
	_t_6234 = _t_6233 * -1.0f;
	_t_6235 = 0.0f < _t_6234;
	if(_t_6235)
		{
		
			_t_6236 = px0_10_1;
		
		}
else
		{
		
			_t_6236 = px1_11_1;
		
		}

	_t_6237 = _t_6222 * _t_6236;
	_t_6238 = -1.0f * ty3_9_1;
	_t_6239 = ty2_8_1 + _t_6238;
	_t_6240 = -1.0f * _t_6239;
	_t_6241 = _t_6240 < 0.0f;
	if(_t_6241)
		{
			float _t_6242;
			float _t_6243;
		
			_t_6242 = -1.0f * tx2_5_1;
			_t_6243 = tx3_6_1 + _t_6242;
			_t_6244 = _t_6243;
		
		}
else
		{
			float _t_6245;
			float _t_6246;
			float _t_6247;
		
			_t_6245 = -1.0f * tx2_5_1;
			_t_6246 = tx3_6_1 + _t_6245;
			_t_6247 = -1.0f * _t_6246;
			_t_6244 = _t_6247;
		
		}

	_t_6248 = _t_6244 * _t_287;
	_t_6249 = -1.0f * ty3_9_1;
	_t_6250 = ty2_8_1 + _t_6249;
	_t_6251 = -1.0f * _t_6250;
	_t_6252 = _t_6251 < 0.0f;
	if(_t_6252)
		{
			float _t_6253;
			float _t_6254;
		
			_t_6253 = -1.0f * tx2_5_1;
			_t_6254 = tx3_6_1 + _t_6253;
			_t_6255 = _t_6254;
		
		}
else
		{
			float _t_6256;
			float _t_6257;
			float _t_6258;
		
			_t_6256 = -1.0f * tx2_5_1;
			_t_6257 = tx3_6_1 + _t_6256;
			_t_6258 = -1.0f * _t_6257;
			_t_6255 = _t_6258;
		
		}

	_t_6259 = _t_6255 * _t_287;
	_t_6260 = _t_6248 * _t_6259;
	_t_6261 = -1.0f * ty3_9_1;
	_t_6262 = ty2_8_1 + _t_6261;
	_t_6263 = -1.0f * _t_6262;
	_t_6264 = _t_6263 < 0.0f;
	if(_t_6264)
		{
			float _t_6265;
			float _t_6266;
		
			_t_6265 = -1.0f * ty3_9_1;
			_t_6266 = ty2_8_1 + _t_6265;
			_t_6267 = _t_6266;
		
		}
else
		{
			float _t_6268;
			float _t_6269;
			float _t_6270;
		
			_t_6268 = -1.0f * ty3_9_1;
			_t_6269 = ty2_8_1 + _t_6268;
			_t_6270 = -1.0f * _t_6269;
			_t_6267 = _t_6270;
		
		}

	_t_6271 = _t_6267 * _t_287;
	_t_6272 = 1.0f + _t_6271;
	_t_6273 = 1.0f / _t_6272;
	_t_6274 = _t_6260 * _t_6273;
	_t_6275 = _t_6274 * -1.0f;
	_t_6276 = 1.0f + _t_6275;
	_t_6277 = -1.0f * ty3_9_1;
	_t_6278 = ty2_8_1 + _t_6277;
	_t_6279 = -1.0f * _t_6278;
	_t_6280 = _t_6279 < 0.0f;
	if(_t_6280)
		{
			float _t_6281;
			float _t_6282;
		
			_t_6281 = -1.0f * tx2_5_1;
			_t_6282 = tx3_6_1 + _t_6281;
			_t_6283 = _t_6282;
		
		}
else
		{
			float _t_6284;
			float _t_6285;
			float _t_6286;
		
			_t_6284 = -1.0f * tx2_5_1;
			_t_6285 = tx3_6_1 + _t_6284;
			_t_6286 = -1.0f * _t_6285;
			_t_6283 = _t_6286;
		
		}

	_t_6287 = _t_6283 * _t_287;
	_t_6288 = -1.0f * ty3_9_1;
	_t_6289 = ty2_8_1 + _t_6288;
	_t_6290 = -1.0f * _t_6289;
	_t_6291 = _t_6290 < 0.0f;
	if(_t_6291)
		{
			float _t_6292;
			float _t_6293;
		
			_t_6292 = -1.0f * tx2_5_1;
			_t_6293 = tx3_6_1 + _t_6292;
			_t_6294 = _t_6293;
		
		}
else
		{
			float _t_6295;
			float _t_6296;
			float _t_6297;
		
			_t_6295 = -1.0f * tx2_5_1;
			_t_6296 = tx3_6_1 + _t_6295;
			_t_6297 = -1.0f * _t_6296;
			_t_6294 = _t_6297;
		
		}

	_t_6298 = _t_6294 * _t_287;
	_t_6299 = _t_6287 * _t_6298;
	_t_6300 = -1.0f * ty3_9_1;
	_t_6301 = ty2_8_1 + _t_6300;
	_t_6302 = -1.0f * _t_6301;
	_t_6303 = _t_6302 < 0.0f;
	if(_t_6303)
		{
			float _t_6304;
			float _t_6305;
		
			_t_6304 = -1.0f * ty3_9_1;
			_t_6305 = ty2_8_1 + _t_6304;
			_t_6306 = _t_6305;
		
		}
else
		{
			float _t_6307;
			float _t_6308;
			float _t_6309;
		
			_t_6307 = -1.0f * ty3_9_1;
			_t_6308 = ty2_8_1 + _t_6307;
			_t_6309 = -1.0f * _t_6308;
			_t_6306 = _t_6309;
		
		}

	_t_6310 = _t_6306 * _t_287;
	_t_6311 = 1.0f + _t_6310;
	_t_6312 = 1.0f / _t_6311;
	_t_6313 = _t_6299 * _t_6312;
	_t_6314 = _t_6313 * -1.0f;
	_t_6315 = 1.0f + _t_6314;
	_t_6316 = 0.0f < _t_6315;
	if(_t_6316)
		{
		
			_t_6317 = py0_12_1;
		
		}
else
		{
		
			_t_6317 = py1_13_1;
		
		}

	_t_6318 = _t_6276 * _t_6317;
	_t_6319 = _t_6237 + _t_6318;
	_t_6320 = -1.0f * ty3_9_1;
	_t_6321 = ty2_8_1 + _t_6320;
	_t_6322 = -1.0f * _t_6321;
	_t_6323 = _t_6322 < 0.0f;
	if(_t_6323)
		{
			float _t_6324;
			float _t_6325;
		
			_t_6324 = -1.0f * tx2_5_1;
			_t_6325 = tx3_6_1 + _t_6324;
			_t_6326 = _t_6325;
		
		}
else
		{
			float _t_6327;
			float _t_6328;
			float _t_6329;
		
			_t_6327 = -1.0f * tx2_5_1;
			_t_6328 = tx3_6_1 + _t_6327;
			_t_6329 = -1.0f * _t_6328;
			_t_6326 = _t_6329;
		
		}

	_t_6330 = _t_6326 * _t_287;
	_t_6331 = _t_6330 * -1.0f;
	_t_6332 = -1.0f * ty3_9_1;
	_t_6333 = ty2_8_1 + _t_6332;
	_t_6334 = -1.0f * _t_6333;
	_t_6335 = _t_6334 < 0.0f;
	if(_t_6335)
		{
			float _t_6336;
			float _t_6337;
		
			_t_6336 = -1.0f * tx2_5_1;
			_t_6337 = tx3_6_1 + _t_6336;
			_t_6338 = _t_6337;
		
		}
else
		{
			float _t_6339;
			float _t_6340;
			float _t_6341;
		
			_t_6339 = -1.0f * tx2_5_1;
			_t_6340 = tx3_6_1 + _t_6339;
			_t_6341 = -1.0f * _t_6340;
			_t_6338 = _t_6341;
		
		}

	_t_6342 = _t_6338 * _t_287;
	_t_6343 = _t_6342 * -1.0f;
	_t_6344 = 0.0f < _t_6343;
	if(_t_6344)
		{
		
			_t_6345 = px1_11_1;
		
		}
else
		{
		
			_t_6345 = px0_10_1;
		
		}

	_t_6346 = _t_6331 * _t_6345;
	_t_6347 = -1.0f * ty3_9_1;
	_t_6348 = ty2_8_1 + _t_6347;
	_t_6349 = -1.0f * _t_6348;
	_t_6350 = _t_6349 < 0.0f;
	if(_t_6350)
		{
			float _t_6351;
			float _t_6352;
		
			_t_6351 = -1.0f * tx2_5_1;
			_t_6352 = tx3_6_1 + _t_6351;
			_t_6353 = _t_6352;
		
		}
else
		{
			float _t_6354;
			float _t_6355;
			float _t_6356;
		
			_t_6354 = -1.0f * tx2_5_1;
			_t_6355 = tx3_6_1 + _t_6354;
			_t_6356 = -1.0f * _t_6355;
			_t_6353 = _t_6356;
		
		}

	_t_6357 = _t_6353 * _t_287;
	_t_6358 = -1.0f * ty3_9_1;
	_t_6359 = ty2_8_1 + _t_6358;
	_t_6360 = -1.0f * _t_6359;
	_t_6361 = _t_6360 < 0.0f;
	if(_t_6361)
		{
			float _t_6362;
			float _t_6363;
		
			_t_6362 = -1.0f * tx2_5_1;
			_t_6363 = tx3_6_1 + _t_6362;
			_t_6364 = _t_6363;
		
		}
else
		{
			float _t_6365;
			float _t_6366;
			float _t_6367;
		
			_t_6365 = -1.0f * tx2_5_1;
			_t_6366 = tx3_6_1 + _t_6365;
			_t_6367 = -1.0f * _t_6366;
			_t_6364 = _t_6367;
		
		}

	_t_6368 = _t_6364 * _t_287;
	_t_6369 = _t_6357 * _t_6368;
	_t_6370 = -1.0f * ty3_9_1;
	_t_6371 = ty2_8_1 + _t_6370;
	_t_6372 = -1.0f * _t_6371;
	_t_6373 = _t_6372 < 0.0f;
	if(_t_6373)
		{
			float _t_6374;
			float _t_6375;
		
			_t_6374 = -1.0f * ty3_9_1;
			_t_6375 = ty2_8_1 + _t_6374;
			_t_6376 = _t_6375;
		
		}
else
		{
			float _t_6377;
			float _t_6378;
			float _t_6379;
		
			_t_6377 = -1.0f * ty3_9_1;
			_t_6378 = ty2_8_1 + _t_6377;
			_t_6379 = -1.0f * _t_6378;
			_t_6376 = _t_6379;
		
		}

	_t_6380 = _t_6376 * _t_287;
	_t_6381 = 1.0f + _t_6380;
	_t_6382 = 1.0f / _t_6381;
	_t_6383 = _t_6369 * _t_6382;
	_t_6384 = _t_6383 * -1.0f;
	_t_6385 = 1.0f + _t_6384;
	_t_6386 = -1.0f * ty3_9_1;
	_t_6387 = ty2_8_1 + _t_6386;
	_t_6388 = -1.0f * _t_6387;
	_t_6389 = _t_6388 < 0.0f;
	if(_t_6389)
		{
			float _t_6390;
			float _t_6391;
		
			_t_6390 = -1.0f * tx2_5_1;
			_t_6391 = tx3_6_1 + _t_6390;
			_t_6392 = _t_6391;
		
		}
else
		{
			float _t_6393;
			float _t_6394;
			float _t_6395;
		
			_t_6393 = -1.0f * tx2_5_1;
			_t_6394 = tx3_6_1 + _t_6393;
			_t_6395 = -1.0f * _t_6394;
			_t_6392 = _t_6395;
		
		}

	_t_6396 = _t_6392 * _t_287;
	_t_6397 = -1.0f * ty3_9_1;
	_t_6398 = ty2_8_1 + _t_6397;
	_t_6399 = -1.0f * _t_6398;
	_t_6400 = _t_6399 < 0.0f;
	if(_t_6400)
		{
			float _t_6401;
			float _t_6402;
		
			_t_6401 = -1.0f * tx2_5_1;
			_t_6402 = tx3_6_1 + _t_6401;
			_t_6403 = _t_6402;
		
		}
else
		{
			float _t_6404;
			float _t_6405;
			float _t_6406;
		
			_t_6404 = -1.0f * tx2_5_1;
			_t_6405 = tx3_6_1 + _t_6404;
			_t_6406 = -1.0f * _t_6405;
			_t_6403 = _t_6406;
		
		}

	_t_6407 = _t_6403 * _t_287;
	_t_6408 = _t_6396 * _t_6407;
	_t_6409 = -1.0f * ty3_9_1;
	_t_6410 = ty2_8_1 + _t_6409;
	_t_6411 = -1.0f * _t_6410;
	_t_6412 = _t_6411 < 0.0f;
	if(_t_6412)
		{
			float _t_6413;
			float _t_6414;
		
			_t_6413 = -1.0f * ty3_9_1;
			_t_6414 = ty2_8_1 + _t_6413;
			_t_6415 = _t_6414;
		
		}
else
		{
			float _t_6416;
			float _t_6417;
			float _t_6418;
		
			_t_6416 = -1.0f * ty3_9_1;
			_t_6417 = ty2_8_1 + _t_6416;
			_t_6418 = -1.0f * _t_6417;
			_t_6415 = _t_6418;
		
		}

	_t_6419 = _t_6415 * _t_287;
	_t_6420 = 1.0f + _t_6419;
	_t_6421 = 1.0f / _t_6420;
	_t_6422 = _t_6408 * _t_6421;
	_t_6423 = _t_6422 * -1.0f;
	_t_6424 = 1.0f + _t_6423;
	_t_6425 = 0.0f < _t_6424;
	if(_t_6425)
		{
		
			_t_6426 = py1_13_1;
		
		}
else
		{
		
			_t_6426 = py0_12_1;
		
		}

	_t_6427 = _t_6385 * _t_6426;
	_t_6428 = _t_6346 + _t_6427;
	_t_288 = tegpixelintegrator_23(ty3_9_1,pc1_15_1,_t_6319,tc2_19_1,ty2_8_1,ty1_7_1,pc0_14_1,tx3_6_1,tx1_4_1,tx2_5_1,py1_13_1,pc2_16_1,px1_11_1,tc0_17_1,_t_6428,py0_12_1,tc1_18_1,px0_10_1,_t_287);

	return _t_288;
}
__device__ float tegpixellet_block_34(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float _t_7492,float _t_7545,float ty3_9_1,float tx3_6_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float _t_315,float y__3091_1,float _t_7465){
	float _t_7546;
	float _t_7547;
	float _t_7548;
	float _t_7549;
	float _t_7550;
	float _t_7551;
	float _t_7552;
	float _t_7553;
	float _t_7554;
	float _t_7555;
	float _t_7556;
	float _t_7557;
	float _t_7558;
	float _t_7559;
	float _t_7560;
	float _t_7561;
	float _t_7562;
	float _t_7563;
	float _t_7564;
	float _t_7565;
	float _t_7566;
	float _t_7567;
	float _t_7568;
	bool _t_7569;
	float _t_7570;
	float _t_7571;
	float _t_7572;
	float _t_7573;
	float _t_7574;
	float _t_7575;
	float _t_7576;
	float _t_7577;
	float _t_7578;
	float _t_7579;
	float _t_7580;
	float _t_7581;
	float _t_7582;
	bool _t_7583;
	float _t_7584;
	float _t_7585;
	float _t_7586;
	float _t_7587;
	float _t_7588;
	bool _t_7589;
	bool _t_7590;
	bool _t_7591;
	bool _t_7592;
	bool _t_7593;
	bool _t_7594;
	bool _t_7595;
	float _t_7925;

	float _t_7466;

	_t_7546 = -1.0f * pc0_14_1;
	_t_7547 = tc0_17_1 + _t_7546;
	_t_7548 = _t_7547 * _t_7547;
	_t_7549 = -1.0f * pc1_15_1;
	_t_7550 = tc1_18_1 + _t_7549;
	_t_7551 = _t_7550 * _t_7550;
	_t_7552 = _t_7548 + _t_7551;
	_t_7553 = -1.0f * pc2_16_1;
	_t_7554 = tc2_19_1 + _t_7553;
	_t_7555 = _t_7554 * _t_7554;
	_t_7556 = _t_7552 + _t_7555;
	_t_7557 = tx1_4_1 * ty2_8_1;
	_t_7558 = tx2_5_1 * ty1_7_1;
	_t_7559 = _t_7558 * -1.0f;
	_t_7560 = _t_7557 + _t_7559;
	_t_7561 = -1.0f * ty2_8_1;
	_t_7562 = ty1_7_1 + _t_7561;
	_t_7563 = _t_7562 * _t_7492;
	_t_7564 = _t_7560 + _t_7563;
	_t_7565 = -1.0f * tx1_4_1;
	_t_7566 = tx2_5_1 + _t_7565;
	_t_7567 = _t_7566 * _t_7545;
	_t_7568 = _t_7564 + _t_7567;
	_t_7569 = _t_7568 < 0.0f;
	if(_t_7569)
		{
		
			_t_7570 = 1.0f;
		
		}
else
		{
		
			_t_7570 = 0.0f;
		
		}

	_t_7571 = tx2_5_1 * ty3_9_1;
	_t_7572 = tx3_6_1 * ty2_8_1;
	_t_7573 = _t_7572 * -1.0f;
	_t_7574 = _t_7571 + _t_7573;
	_t_7575 = -1.0f * ty3_9_1;
	_t_7576 = ty2_8_1 + _t_7575;
	_t_7577 = _t_7576 * _t_7492;
	_t_7578 = _t_7574 + _t_7577;
	_t_7579 = -1.0f * tx2_5_1;
	_t_7580 = tx3_6_1 + _t_7579;
	_t_7581 = _t_7580 * _t_7545;
	_t_7582 = _t_7578 + _t_7581;
	_t_7583 = _t_7582 < 0.0f;
	if(_t_7583)
		{
		
			_t_7584 = 1.0f;
		
		}
else
		{
		
			_t_7584 = 0.0f;
		
		}

	_t_7585 = _t_7570 * _t_7584;
	_t_7586 = _t_7556 * _t_7585;
	_t_7587 = _t_7586 * ty1_7_1;
	_t_7588 = _t_7587 * -1.0f;
	_t_7589 = py0_12_1 < _t_7545;
	_t_7590 = _t_7545 < py1_13_1;
	_t_7591 = _t_7589 && _t_7590;
	_t_7592 = px0_10_1 < _t_7492;
	_t_7593 = _t_7492 < px1_11_1;
	_t_7594 = _t_7592 && _t_7593;
	_t_7595 = _t_7591 && _t_7594;
	if(_t_7595)
		{
			float _t_7596;
			float _t_7597;
			float _t_7598;
			bool _t_7599;
			float _t_7602;
			float _t_7606;
			float _t_7607;
			float _t_7608;
			float _t_7609;
			float _t_7610;
			bool _t_7611;
			float _t_7614;
			float _t_7618;
			float _t_7619;
			bool _t_7620;
			float _t_7621;
			float _t_7622;
			float _t_7623;
			float _t_7624;
			float _t_7625;
			bool _t_7626;
			float _t_7629;
			float _t_7633;
			float _t_7634;
			float _t_7635;
			float _t_7636;
			bool _t_7637;
			float _t_7640;
			float _t_7644;
			float _t_7645;
			float _t_7646;
			float _t_7647;
			float _t_7648;
			bool _t_7649;
			float _t_7652;
			float _t_7656;
			float _t_7657;
			float _t_7658;
			float _t_7659;
			float _t_7660;
			float _t_7661;
			float _t_7662;
			float _t_7663;
			float _t_7664;
			bool _t_7665;
			float _t_7668;
			float _t_7672;
			float _t_7673;
			float _t_7674;
			float _t_7675;
			bool _t_7676;
			float _t_7679;
			float _t_7683;
			float _t_7684;
			float _t_7685;
			float _t_7686;
			float _t_7687;
			bool _t_7688;
			float _t_7691;
			float _t_7695;
			float _t_7696;
			float _t_7697;
			float _t_7698;
			float _t_7699;
			float _t_7700;
			bool _t_7701;
			float _t_7702;
			float _t_7703;
			float _t_7704;
			bool _t_7705;
			float _t_7706;
			float _t_7707;
			float _t_7708;
			bool _t_7709;
			float _t_7712;
			float _t_7716;
			float _t_7717;
			float _t_7718;
			float _t_7719;
			float _t_7720;
			bool _t_7721;
			float _t_7724;
			float _t_7728;
			float _t_7729;
			bool _t_7730;
			float _t_7731;
			float _t_7732;
			float _t_7733;
			float _t_7734;
			float _t_7735;
			bool _t_7736;
			float _t_7739;
			float _t_7743;
			float _t_7744;
			float _t_7745;
			float _t_7746;
			bool _t_7747;
			float _t_7750;
			float _t_7754;
			float _t_7755;
			float _t_7756;
			float _t_7757;
			float _t_7758;
			bool _t_7759;
			float _t_7762;
			float _t_7766;
			float _t_7767;
			float _t_7768;
			float _t_7769;
			float _t_7770;
			float _t_7771;
			float _t_7772;
			float _t_7773;
			float _t_7774;
			bool _t_7775;
			float _t_7778;
			float _t_7782;
			float _t_7783;
			float _t_7784;
			float _t_7785;
			bool _t_7786;
			float _t_7789;
			float _t_7793;
			float _t_7794;
			float _t_7795;
			float _t_7796;
			float _t_7797;
			bool _t_7798;
			float _t_7801;
			float _t_7805;
			float _t_7806;
			float _t_7807;
			float _t_7808;
			float _t_7809;
			float _t_7810;
			bool _t_7811;
			float _t_7812;
			float _t_7813;
			float _t_7814;
			bool _t_7815;
			bool _t_7816;
			float _t_7817;
			float _t_7818;
			float _t_7819;
			bool _t_7820;
			float _t_7823;
			float _t_7827;
			float _t_7828;
			float _t_7829;
			float _t_7830;
			bool _t_7831;
			float _t_7834;
			float _t_7838;
			bool _t_7839;
			float _t_7840;
			float _t_7841;
			float _t_7842;
			float _t_7843;
			float _t_7844;
			bool _t_7845;
			float _t_7848;
			float _t_7852;
			float _t_7853;
			float _t_7854;
			float _t_7855;
			bool _t_7856;
			float _t_7859;
			float _t_7863;
			bool _t_7864;
			float _t_7865;
			float _t_7866;
			float _t_7867;
			bool _t_7868;
			float _t_7869;
			float _t_7870;
			float _t_7871;
			bool _t_7872;
			float _t_7875;
			float _t_7879;
			float _t_7880;
			float _t_7881;
			float _t_7882;
			bool _t_7883;
			float _t_7886;
			float _t_7890;
			bool _t_7891;
			float _t_7892;
			float _t_7893;
			float _t_7894;
			float _t_7895;
			float _t_7896;
			bool _t_7897;
			float _t_7900;
			float _t_7904;
			float _t_7905;
			float _t_7906;
			float _t_7907;
			bool _t_7908;
			float _t_7911;
			float _t_7915;
			bool _t_7916;
			float _t_7917;
			float _t_7918;
			float _t_7919;
			bool _t_7920;
			bool _t_7921;
			bool _t_7922;
			float _t_7923;
			float _t_7924;
		
			_t_7596 = -1.0f * ty1_7_1;
			_t_7597 = ty3_9_1 + _t_7596;
			_t_7598 = -1.0f * _t_7597;
			_t_7599 = _t_7598 < 0.0f;
			if(_t_7599)
				{
					float _t_7600;
					float _t_7601;
				
					_t_7600 = -1.0f * tx3_6_1;
					_t_7601 = tx1_4_1 + _t_7600;
					_t_7602 = _t_7601;
				
				}
		else
				{
					float _t_7603;
					float _t_7604;
					float _t_7605;
				
					_t_7603 = -1.0f * tx3_6_1;
					_t_7604 = tx1_4_1 + _t_7603;
					_t_7605 = -1.0f * _t_7604;
					_t_7602 = _t_7605;
				
				}
		
			_t_7606 = _t_7602 * _t_315;
			_t_7607 = _t_7606 * -1.0f;
			_t_7608 = -1.0f * ty1_7_1;
			_t_7609 = ty3_9_1 + _t_7608;
			_t_7610 = -1.0f * _t_7609;
			_t_7611 = _t_7610 < 0.0f;
			if(_t_7611)
				{
					float _t_7612;
					float _t_7613;
				
					_t_7612 = -1.0f * tx3_6_1;
					_t_7613 = tx1_4_1 + _t_7612;
					_t_7614 = _t_7613;
				
				}
		else
				{
					float _t_7615;
					float _t_7616;
					float _t_7617;
				
					_t_7615 = -1.0f * tx3_6_1;
					_t_7616 = tx1_4_1 + _t_7615;
					_t_7617 = -1.0f * _t_7616;
					_t_7614 = _t_7617;
				
				}
		
			_t_7618 = _t_7614 * _t_315;
			_t_7619 = _t_7618 * -1.0f;
			_t_7620 = 0.0f < _t_7619;
			if(_t_7620)
				{
				
					_t_7621 = px0_10_1;
				
				}
		else
				{
				
					_t_7621 = px1_11_1;
				
				}
		
			_t_7622 = _t_7607 * _t_7621;
			_t_7623 = -1.0f * ty1_7_1;
			_t_7624 = ty3_9_1 + _t_7623;
			_t_7625 = -1.0f * _t_7624;
			_t_7626 = _t_7625 < 0.0f;
			if(_t_7626)
				{
					float _t_7627;
					float _t_7628;
				
					_t_7627 = -1.0f * tx3_6_1;
					_t_7628 = tx1_4_1 + _t_7627;
					_t_7629 = _t_7628;
				
				}
		else
				{
					float _t_7630;
					float _t_7631;
					float _t_7632;
				
					_t_7630 = -1.0f * tx3_6_1;
					_t_7631 = tx1_4_1 + _t_7630;
					_t_7632 = -1.0f * _t_7631;
					_t_7629 = _t_7632;
				
				}
		
			_t_7633 = _t_7629 * _t_315;
			_t_7634 = -1.0f * ty1_7_1;
			_t_7635 = ty3_9_1 + _t_7634;
			_t_7636 = -1.0f * _t_7635;
			_t_7637 = _t_7636 < 0.0f;
			if(_t_7637)
				{
					float _t_7638;
					float _t_7639;
				
					_t_7638 = -1.0f * tx3_6_1;
					_t_7639 = tx1_4_1 + _t_7638;
					_t_7640 = _t_7639;
				
				}
		else
				{
					float _t_7641;
					float _t_7642;
					float _t_7643;
				
					_t_7641 = -1.0f * tx3_6_1;
					_t_7642 = tx1_4_1 + _t_7641;
					_t_7643 = -1.0f * _t_7642;
					_t_7640 = _t_7643;
				
				}
		
			_t_7644 = _t_7640 * _t_315;
			_t_7645 = _t_7633 * _t_7644;
			_t_7646 = -1.0f * ty1_7_1;
			_t_7647 = ty3_9_1 + _t_7646;
			_t_7648 = -1.0f * _t_7647;
			_t_7649 = _t_7648 < 0.0f;
			if(_t_7649)
				{
					float _t_7650;
					float _t_7651;
				
					_t_7650 = -1.0f * ty1_7_1;
					_t_7651 = ty3_9_1 + _t_7650;
					_t_7652 = _t_7651;
				
				}
		else
				{
					float _t_7653;
					float _t_7654;
					float _t_7655;
				
					_t_7653 = -1.0f * ty1_7_1;
					_t_7654 = ty3_9_1 + _t_7653;
					_t_7655 = -1.0f * _t_7654;
					_t_7652 = _t_7655;
				
				}
		
			_t_7656 = _t_7652 * _t_315;
			_t_7657 = 1.0f + _t_7656;
			_t_7658 = 1.0f / _t_7657;
			_t_7659 = _t_7645 * _t_7658;
			_t_7660 = _t_7659 * -1.0f;
			_t_7661 = 1.0f + _t_7660;
			_t_7662 = -1.0f * ty1_7_1;
			_t_7663 = ty3_9_1 + _t_7662;
			_t_7664 = -1.0f * _t_7663;
			_t_7665 = _t_7664 < 0.0f;
			if(_t_7665)
				{
					float _t_7666;
					float _t_7667;
				
					_t_7666 = -1.0f * tx3_6_1;
					_t_7667 = tx1_4_1 + _t_7666;
					_t_7668 = _t_7667;
				
				}
		else
				{
					float _t_7669;
					float _t_7670;
					float _t_7671;
				
					_t_7669 = -1.0f * tx3_6_1;
					_t_7670 = tx1_4_1 + _t_7669;
					_t_7671 = -1.0f * _t_7670;
					_t_7668 = _t_7671;
				
				}
		
			_t_7672 = _t_7668 * _t_315;
			_t_7673 = -1.0f * ty1_7_1;
			_t_7674 = ty3_9_1 + _t_7673;
			_t_7675 = -1.0f * _t_7674;
			_t_7676 = _t_7675 < 0.0f;
			if(_t_7676)
				{
					float _t_7677;
					float _t_7678;
				
					_t_7677 = -1.0f * tx3_6_1;
					_t_7678 = tx1_4_1 + _t_7677;
					_t_7679 = _t_7678;
				
				}
		else
				{
					float _t_7680;
					float _t_7681;
					float _t_7682;
				
					_t_7680 = -1.0f * tx3_6_1;
					_t_7681 = tx1_4_1 + _t_7680;
					_t_7682 = -1.0f * _t_7681;
					_t_7679 = _t_7682;
				
				}
		
			_t_7683 = _t_7679 * _t_315;
			_t_7684 = _t_7672 * _t_7683;
			_t_7685 = -1.0f * ty1_7_1;
			_t_7686 = ty3_9_1 + _t_7685;
			_t_7687 = -1.0f * _t_7686;
			_t_7688 = _t_7687 < 0.0f;
			if(_t_7688)
				{
					float _t_7689;
					float _t_7690;
				
					_t_7689 = -1.0f * ty1_7_1;
					_t_7690 = ty3_9_1 + _t_7689;
					_t_7691 = _t_7690;
				
				}
		else
				{
					float _t_7692;
					float _t_7693;
					float _t_7694;
				
					_t_7692 = -1.0f * ty1_7_1;
					_t_7693 = ty3_9_1 + _t_7692;
					_t_7694 = -1.0f * _t_7693;
					_t_7691 = _t_7694;
				
				}
		
			_t_7695 = _t_7691 * _t_315;
			_t_7696 = 1.0f + _t_7695;
			_t_7697 = 1.0f / _t_7696;
			_t_7698 = _t_7684 * _t_7697;
			_t_7699 = _t_7698 * -1.0f;
			_t_7700 = 1.0f + _t_7699;
			_t_7701 = 0.0f < _t_7700;
			if(_t_7701)
				{
				
					_t_7702 = py0_12_1;
				
				}
		else
				{
				
					_t_7702 = py1_13_1;
				
				}
		
			_t_7703 = _t_7661 * _t_7702;
			_t_7704 = _t_7622 + _t_7703;
			_t_7705 = _t_7704 < y__3091_1;
			_t_7706 = -1.0f * ty1_7_1;
			_t_7707 = ty3_9_1 + _t_7706;
			_t_7708 = -1.0f * _t_7707;
			_t_7709 = _t_7708 < 0.0f;
			if(_t_7709)
				{
					float _t_7710;
					float _t_7711;
				
					_t_7710 = -1.0f * tx3_6_1;
					_t_7711 = tx1_4_1 + _t_7710;
					_t_7712 = _t_7711;
				
				}
		else
				{
					float _t_7713;
					float _t_7714;
					float _t_7715;
				
					_t_7713 = -1.0f * tx3_6_1;
					_t_7714 = tx1_4_1 + _t_7713;
					_t_7715 = -1.0f * _t_7714;
					_t_7712 = _t_7715;
				
				}
		
			_t_7716 = _t_7712 * _t_315;
			_t_7717 = _t_7716 * -1.0f;
			_t_7718 = -1.0f * ty1_7_1;
			_t_7719 = ty3_9_1 + _t_7718;
			_t_7720 = -1.0f * _t_7719;
			_t_7721 = _t_7720 < 0.0f;
			if(_t_7721)
				{
					float _t_7722;
					float _t_7723;
				
					_t_7722 = -1.0f * tx3_6_1;
					_t_7723 = tx1_4_1 + _t_7722;
					_t_7724 = _t_7723;
				
				}
		else
				{
					float _t_7725;
					float _t_7726;
					float _t_7727;
				
					_t_7725 = -1.0f * tx3_6_1;
					_t_7726 = tx1_4_1 + _t_7725;
					_t_7727 = -1.0f * _t_7726;
					_t_7724 = _t_7727;
				
				}
		
			_t_7728 = _t_7724 * _t_315;
			_t_7729 = _t_7728 * -1.0f;
			_t_7730 = 0.0f < _t_7729;
			if(_t_7730)
				{
				
					_t_7731 = px1_11_1;
				
				}
		else
				{
				
					_t_7731 = px0_10_1;
				
				}
		
			_t_7732 = _t_7717 * _t_7731;
			_t_7733 = -1.0f * ty1_7_1;
			_t_7734 = ty3_9_1 + _t_7733;
			_t_7735 = -1.0f * _t_7734;
			_t_7736 = _t_7735 < 0.0f;
			if(_t_7736)
				{
					float _t_7737;
					float _t_7738;
				
					_t_7737 = -1.0f * tx3_6_1;
					_t_7738 = tx1_4_1 + _t_7737;
					_t_7739 = _t_7738;
				
				}
		else
				{
					float _t_7740;
					float _t_7741;
					float _t_7742;
				
					_t_7740 = -1.0f * tx3_6_1;
					_t_7741 = tx1_4_1 + _t_7740;
					_t_7742 = -1.0f * _t_7741;
					_t_7739 = _t_7742;
				
				}
		
			_t_7743 = _t_7739 * _t_315;
			_t_7744 = -1.0f * ty1_7_1;
			_t_7745 = ty3_9_1 + _t_7744;
			_t_7746 = -1.0f * _t_7745;
			_t_7747 = _t_7746 < 0.0f;
			if(_t_7747)
				{
					float _t_7748;
					float _t_7749;
				
					_t_7748 = -1.0f * tx3_6_1;
					_t_7749 = tx1_4_1 + _t_7748;
					_t_7750 = _t_7749;
				
				}
		else
				{
					float _t_7751;
					float _t_7752;
					float _t_7753;
				
					_t_7751 = -1.0f * tx3_6_1;
					_t_7752 = tx1_4_1 + _t_7751;
					_t_7753 = -1.0f * _t_7752;
					_t_7750 = _t_7753;
				
				}
		
			_t_7754 = _t_7750 * _t_315;
			_t_7755 = _t_7743 * _t_7754;
			_t_7756 = -1.0f * ty1_7_1;
			_t_7757 = ty3_9_1 + _t_7756;
			_t_7758 = -1.0f * _t_7757;
			_t_7759 = _t_7758 < 0.0f;
			if(_t_7759)
				{
					float _t_7760;
					float _t_7761;
				
					_t_7760 = -1.0f * ty1_7_1;
					_t_7761 = ty3_9_1 + _t_7760;
					_t_7762 = _t_7761;
				
				}
		else
				{
					float _t_7763;
					float _t_7764;
					float _t_7765;
				
					_t_7763 = -1.0f * ty1_7_1;
					_t_7764 = ty3_9_1 + _t_7763;
					_t_7765 = -1.0f * _t_7764;
					_t_7762 = _t_7765;
				
				}
		
			_t_7766 = _t_7762 * _t_315;
			_t_7767 = 1.0f + _t_7766;
			_t_7768 = 1.0f / _t_7767;
			_t_7769 = _t_7755 * _t_7768;
			_t_7770 = _t_7769 * -1.0f;
			_t_7771 = 1.0f + _t_7770;
			_t_7772 = -1.0f * ty1_7_1;
			_t_7773 = ty3_9_1 + _t_7772;
			_t_7774 = -1.0f * _t_7773;
			_t_7775 = _t_7774 < 0.0f;
			if(_t_7775)
				{
					float _t_7776;
					float _t_7777;
				
					_t_7776 = -1.0f * tx3_6_1;
					_t_7777 = tx1_4_1 + _t_7776;
					_t_7778 = _t_7777;
				
				}
		else
				{
					float _t_7779;
					float _t_7780;
					float _t_7781;
				
					_t_7779 = -1.0f * tx3_6_1;
					_t_7780 = tx1_4_1 + _t_7779;
					_t_7781 = -1.0f * _t_7780;
					_t_7778 = _t_7781;
				
				}
		
			_t_7782 = _t_7778 * _t_315;
			_t_7783 = -1.0f * ty1_7_1;
			_t_7784 = ty3_9_1 + _t_7783;
			_t_7785 = -1.0f * _t_7784;
			_t_7786 = _t_7785 < 0.0f;
			if(_t_7786)
				{
					float _t_7787;
					float _t_7788;
				
					_t_7787 = -1.0f * tx3_6_1;
					_t_7788 = tx1_4_1 + _t_7787;
					_t_7789 = _t_7788;
				
				}
		else
				{
					float _t_7790;
					float _t_7791;
					float _t_7792;
				
					_t_7790 = -1.0f * tx3_6_1;
					_t_7791 = tx1_4_1 + _t_7790;
					_t_7792 = -1.0f * _t_7791;
					_t_7789 = _t_7792;
				
				}
		
			_t_7793 = _t_7789 * _t_315;
			_t_7794 = _t_7782 * _t_7793;
			_t_7795 = -1.0f * ty1_7_1;
			_t_7796 = ty3_9_1 + _t_7795;
			_t_7797 = -1.0f * _t_7796;
			_t_7798 = _t_7797 < 0.0f;
			if(_t_7798)
				{
					float _t_7799;
					float _t_7800;
				
					_t_7799 = -1.0f * ty1_7_1;
					_t_7800 = ty3_9_1 + _t_7799;
					_t_7801 = _t_7800;
				
				}
		else
				{
					float _t_7802;
					float _t_7803;
					float _t_7804;
				
					_t_7802 = -1.0f * ty1_7_1;
					_t_7803 = ty3_9_1 + _t_7802;
					_t_7804 = -1.0f * _t_7803;
					_t_7801 = _t_7804;
				
				}
		
			_t_7805 = _t_7801 * _t_315;
			_t_7806 = 1.0f + _t_7805;
			_t_7807 = 1.0f / _t_7806;
			_t_7808 = _t_7794 * _t_7807;
			_t_7809 = _t_7808 * -1.0f;
			_t_7810 = 1.0f + _t_7809;
			_t_7811 = 0.0f < _t_7810;
			if(_t_7811)
				{
				
					_t_7812 = py1_13_1;
				
				}
		else
				{
				
					_t_7812 = py0_12_1;
				
				}
		
			_t_7813 = _t_7771 * _t_7812;
			_t_7814 = _t_7732 + _t_7813;
			_t_7815 = y__3091_1 < _t_7814;
			_t_7816 = _t_7705 && _t_7815;
			_t_7817 = -1.0f * ty1_7_1;
			_t_7818 = ty3_9_1 + _t_7817;
			_t_7819 = -1.0f * _t_7818;
			_t_7820 = _t_7819 < 0.0f;
			if(_t_7820)
				{
					float _t_7821;
					float _t_7822;
				
					_t_7821 = -1.0f * ty1_7_1;
					_t_7822 = ty3_9_1 + _t_7821;
					_t_7823 = _t_7822;
				
				}
		else
				{
					float _t_7824;
					float _t_7825;
					float _t_7826;
				
					_t_7824 = -1.0f * ty1_7_1;
					_t_7825 = ty3_9_1 + _t_7824;
					_t_7826 = -1.0f * _t_7825;
					_t_7823 = _t_7826;
				
				}
		
			_t_7827 = _t_7823 * _t_315;
			_t_7828 = -1.0f * ty1_7_1;
			_t_7829 = ty3_9_1 + _t_7828;
			_t_7830 = -1.0f * _t_7829;
			_t_7831 = _t_7830 < 0.0f;
			if(_t_7831)
				{
					float _t_7832;
					float _t_7833;
				
					_t_7832 = -1.0f * ty1_7_1;
					_t_7833 = ty3_9_1 + _t_7832;
					_t_7834 = _t_7833;
				
				}
		else
				{
					float _t_7835;
					float _t_7836;
					float _t_7837;
				
					_t_7835 = -1.0f * ty1_7_1;
					_t_7836 = ty3_9_1 + _t_7835;
					_t_7837 = -1.0f * _t_7836;
					_t_7834 = _t_7837;
				
				}
		
			_t_7838 = _t_7834 * _t_315;
			_t_7839 = 0.0f < _t_7838;
			if(_t_7839)
				{
				
					_t_7840 = px0_10_1;
				
				}
		else
				{
				
					_t_7840 = px1_11_1;
				
				}
		
			_t_7841 = _t_7827 * _t_7840;
			_t_7842 = -1.0f * ty1_7_1;
			_t_7843 = ty3_9_1 + _t_7842;
			_t_7844 = -1.0f * _t_7843;
			_t_7845 = _t_7844 < 0.0f;
			if(_t_7845)
				{
					float _t_7846;
					float _t_7847;
				
					_t_7846 = -1.0f * tx3_6_1;
					_t_7847 = tx1_4_1 + _t_7846;
					_t_7848 = _t_7847;
				
				}
		else
				{
					float _t_7849;
					float _t_7850;
					float _t_7851;
				
					_t_7849 = -1.0f * tx3_6_1;
					_t_7850 = tx1_4_1 + _t_7849;
					_t_7851 = -1.0f * _t_7850;
					_t_7848 = _t_7851;
				
				}
		
			_t_7852 = _t_7848 * _t_315;
			_t_7853 = -1.0f * ty1_7_1;
			_t_7854 = ty3_9_1 + _t_7853;
			_t_7855 = -1.0f * _t_7854;
			_t_7856 = _t_7855 < 0.0f;
			if(_t_7856)
				{
					float _t_7857;
					float _t_7858;
				
					_t_7857 = -1.0f * tx3_6_1;
					_t_7858 = tx1_4_1 + _t_7857;
					_t_7859 = _t_7858;
				
				}
		else
				{
					float _t_7860;
					float _t_7861;
					float _t_7862;
				
					_t_7860 = -1.0f * tx3_6_1;
					_t_7861 = tx1_4_1 + _t_7860;
					_t_7862 = -1.0f * _t_7861;
					_t_7859 = _t_7862;
				
				}
		
			_t_7863 = _t_7859 * _t_315;
			_t_7864 = 0.0f < _t_7863;
			if(_t_7864)
				{
				
					_t_7865 = py0_12_1;
				
				}
		else
				{
				
					_t_7865 = py1_13_1;
				
				}
		
			_t_7866 = _t_7852 * _t_7865;
			_t_7867 = _t_7841 + _t_7866;
			_t_7868 = _t_7867 < _t_7465;
			_t_7869 = -1.0f * ty1_7_1;
			_t_7870 = ty3_9_1 + _t_7869;
			_t_7871 = -1.0f * _t_7870;
			_t_7872 = _t_7871 < 0.0f;
			if(_t_7872)
				{
					float _t_7873;
					float _t_7874;
				
					_t_7873 = -1.0f * ty1_7_1;
					_t_7874 = ty3_9_1 + _t_7873;
					_t_7875 = _t_7874;
				
				}
		else
				{
					float _t_7876;
					float _t_7877;
					float _t_7878;
				
					_t_7876 = -1.0f * ty1_7_1;
					_t_7877 = ty3_9_1 + _t_7876;
					_t_7878 = -1.0f * _t_7877;
					_t_7875 = _t_7878;
				
				}
		
			_t_7879 = _t_7875 * _t_315;
			_t_7880 = -1.0f * ty1_7_1;
			_t_7881 = ty3_9_1 + _t_7880;
			_t_7882 = -1.0f * _t_7881;
			_t_7883 = _t_7882 < 0.0f;
			if(_t_7883)
				{
					float _t_7884;
					float _t_7885;
				
					_t_7884 = -1.0f * ty1_7_1;
					_t_7885 = ty3_9_1 + _t_7884;
					_t_7886 = _t_7885;
				
				}
		else
				{
					float _t_7887;
					float _t_7888;
					float _t_7889;
				
					_t_7887 = -1.0f * ty1_7_1;
					_t_7888 = ty3_9_1 + _t_7887;
					_t_7889 = -1.0f * _t_7888;
					_t_7886 = _t_7889;
				
				}
		
			_t_7890 = _t_7886 * _t_315;
			_t_7891 = 0.0f < _t_7890;
			if(_t_7891)
				{
				
					_t_7892 = px1_11_1;
				
				}
		else
				{
				
					_t_7892 = px0_10_1;
				
				}
		
			_t_7893 = _t_7879 * _t_7892;
			_t_7894 = -1.0f * ty1_7_1;
			_t_7895 = ty3_9_1 + _t_7894;
			_t_7896 = -1.0f * _t_7895;
			_t_7897 = _t_7896 < 0.0f;
			if(_t_7897)
				{
					float _t_7898;
					float _t_7899;
				
					_t_7898 = -1.0f * tx3_6_1;
					_t_7899 = tx1_4_1 + _t_7898;
					_t_7900 = _t_7899;
				
				}
		else
				{
					float _t_7901;
					float _t_7902;
					float _t_7903;
				
					_t_7901 = -1.0f * tx3_6_1;
					_t_7902 = tx1_4_1 + _t_7901;
					_t_7903 = -1.0f * _t_7902;
					_t_7900 = _t_7903;
				
				}
		
			_t_7904 = _t_7900 * _t_315;
			_t_7905 = -1.0f * ty1_7_1;
			_t_7906 = ty3_9_1 + _t_7905;
			_t_7907 = -1.0f * _t_7906;
			_t_7908 = _t_7907 < 0.0f;
			if(_t_7908)
				{
					float _t_7909;
					float _t_7910;
				
					_t_7909 = -1.0f * tx3_6_1;
					_t_7910 = tx1_4_1 + _t_7909;
					_t_7911 = _t_7910;
				
				}
		else
				{
					float _t_7912;
					float _t_7913;
					float _t_7914;
				
					_t_7912 = -1.0f * tx3_6_1;
					_t_7913 = tx1_4_1 + _t_7912;
					_t_7914 = -1.0f * _t_7913;
					_t_7911 = _t_7914;
				
				}
		
			_t_7915 = _t_7911 * _t_315;
			_t_7916 = 0.0f < _t_7915;
			if(_t_7916)
				{
				
					_t_7917 = py1_13_1;
				
				}
		else
				{
				
					_t_7917 = py0_12_1;
				
				}
		
			_t_7918 = _t_7904 * _t_7917;
			_t_7919 = _t_7893 + _t_7918;
			_t_7920 = _t_7465 < _t_7919;
			_t_7921 = _t_7868 && _t_7920;
			_t_7922 = _t_7816 && _t_7921;
			if(_t_7922)
				{
				
					_t_7923 = 1.0f;
				
				}
		else
				{
				
					_t_7923 = 0.0f;
				
				}
		
			_t_7924 = _t_7923 * _t_315;
			_t_7925 = _t_7924;
		
		}
else
		{
		
			_t_7925 = 0.0f;
		
		}

	_t_7466 = _t_7588 * _t_7925;

	return _t_7466;
}
__device__ float tegpixellet_block_33(float ty3_9_1,float ty1_7_1,float _t_315,float _t_7465,float tx1_4_1,float tx3_6_1,float y__3091_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1){
	float _t_7467;
	float _t_7468;
	float _t_7469;
	bool _t_7470;
	float _t_7473;
	float _t_7477;
	float _t_7478;
	float _t_7479;
	float _t_7480;
	float _t_7481;
	bool _t_7482;
	float _t_7485;
	float _t_7489;
	float _t_7490;
	float _t_7491;
	float _t_7492;
	float _t_7493;
	float _t_7494;
	float _t_7495;
	bool _t_7496;
	float _t_7499;
	float _t_7503;
	float _t_7504;
	float _t_7505;
	float _t_7506;
	bool _t_7507;
	float _t_7510;
	float _t_7514;
	float _t_7515;
	float _t_7516;
	float _t_7517;
	float _t_7518;
	bool _t_7519;
	float _t_7522;
	float _t_7526;
	float _t_7527;
	float _t_7528;
	float _t_7529;
	float _t_7530;
	float _t_7531;
	float _t_7532;
	float _t_7533;
	float _t_7534;
	float _t_7535;
	bool _t_7536;
	float _t_7539;
	float _t_7543;
	float _t_7544;
	float _t_7545;

	float _t_7466;

	_t_7467 = -1.0f * ty1_7_1;
	_t_7468 = ty3_9_1 + _t_7467;
	_t_7469 = -1.0f * _t_7468;
	_t_7470 = _t_7469 < 0.0f;
	if(_t_7470)
		{
			float _t_7471;
			float _t_7472;
		
			_t_7471 = -1.0f * ty1_7_1;
			_t_7472 = ty3_9_1 + _t_7471;
			_t_7473 = _t_7472;
		
		}
else
		{
			float _t_7474;
			float _t_7475;
			float _t_7476;
		
			_t_7474 = -1.0f * ty1_7_1;
			_t_7475 = ty3_9_1 + _t_7474;
			_t_7476 = -1.0f * _t_7475;
			_t_7473 = _t_7476;
		
		}

	_t_7477 = _t_7473 * _t_315;
	_t_7478 = _t_7477 * _t_7465;
	_t_7479 = -1.0f * ty1_7_1;
	_t_7480 = ty3_9_1 + _t_7479;
	_t_7481 = -1.0f * _t_7480;
	_t_7482 = _t_7481 < 0.0f;
	if(_t_7482)
		{
			float _t_7483;
			float _t_7484;
		
			_t_7483 = -1.0f * tx3_6_1;
			_t_7484 = tx1_4_1 + _t_7483;
			_t_7485 = _t_7484;
		
		}
else
		{
			float _t_7486;
			float _t_7487;
			float _t_7488;
		
			_t_7486 = -1.0f * tx3_6_1;
			_t_7487 = tx1_4_1 + _t_7486;
			_t_7488 = -1.0f * _t_7487;
			_t_7485 = _t_7488;
		
		}

	_t_7489 = _t_7485 * _t_315;
	_t_7490 = _t_7489 * -1.0f;
	_t_7491 = _t_7490 * y__3091_1;
	_t_7492 = _t_7478 + _t_7491;
	_t_7493 = -1.0f * ty1_7_1;
	_t_7494 = ty3_9_1 + _t_7493;
	_t_7495 = -1.0f * _t_7494;
	_t_7496 = _t_7495 < 0.0f;
	if(_t_7496)
		{
			float _t_7497;
			float _t_7498;
		
			_t_7497 = -1.0f * tx3_6_1;
			_t_7498 = tx1_4_1 + _t_7497;
			_t_7499 = _t_7498;
		
		}
else
		{
			float _t_7500;
			float _t_7501;
			float _t_7502;
		
			_t_7500 = -1.0f * tx3_6_1;
			_t_7501 = tx1_4_1 + _t_7500;
			_t_7502 = -1.0f * _t_7501;
			_t_7499 = _t_7502;
		
		}

	_t_7503 = _t_7499 * _t_315;
	_t_7504 = -1.0f * ty1_7_1;
	_t_7505 = ty3_9_1 + _t_7504;
	_t_7506 = -1.0f * _t_7505;
	_t_7507 = _t_7506 < 0.0f;
	if(_t_7507)
		{
			float _t_7508;
			float _t_7509;
		
			_t_7508 = -1.0f * tx3_6_1;
			_t_7509 = tx1_4_1 + _t_7508;
			_t_7510 = _t_7509;
		
		}
else
		{
			float _t_7511;
			float _t_7512;
			float _t_7513;
		
			_t_7511 = -1.0f * tx3_6_1;
			_t_7512 = tx1_4_1 + _t_7511;
			_t_7513 = -1.0f * _t_7512;
			_t_7510 = _t_7513;
		
		}

	_t_7514 = _t_7510 * _t_315;
	_t_7515 = _t_7503 * _t_7514;
	_t_7516 = -1.0f * ty1_7_1;
	_t_7517 = ty3_9_1 + _t_7516;
	_t_7518 = -1.0f * _t_7517;
	_t_7519 = _t_7518 < 0.0f;
	if(_t_7519)
		{
			float _t_7520;
			float _t_7521;
		
			_t_7520 = -1.0f * ty1_7_1;
			_t_7521 = ty3_9_1 + _t_7520;
			_t_7522 = _t_7521;
		
		}
else
		{
			float _t_7523;
			float _t_7524;
			float _t_7525;
		
			_t_7523 = -1.0f * ty1_7_1;
			_t_7524 = ty3_9_1 + _t_7523;
			_t_7525 = -1.0f * _t_7524;
			_t_7522 = _t_7525;
		
		}

	_t_7526 = _t_7522 * _t_315;
	_t_7527 = 1.0f + _t_7526;
	_t_7528 = 1.0f / _t_7527;
	_t_7529 = _t_7515 * _t_7528;
	_t_7530 = _t_7529 * -1.0f;
	_t_7531 = 1.0f + _t_7530;
	_t_7532 = _t_7531 * y__3091_1;
	_t_7533 = -1.0f * ty1_7_1;
	_t_7534 = ty3_9_1 + _t_7533;
	_t_7535 = -1.0f * _t_7534;
	_t_7536 = _t_7535 < 0.0f;
	if(_t_7536)
		{
			float _t_7537;
			float _t_7538;
		
			_t_7537 = -1.0f * tx3_6_1;
			_t_7538 = tx1_4_1 + _t_7537;
			_t_7539 = _t_7538;
		
		}
else
		{
			float _t_7540;
			float _t_7541;
			float _t_7542;
		
			_t_7540 = -1.0f * tx3_6_1;
			_t_7541 = tx1_4_1 + _t_7540;
			_t_7542 = -1.0f * _t_7541;
			_t_7539 = _t_7542;
		
		}

	_t_7543 = _t_7539 * _t_315;
	_t_7544 = _t_7543 * _t_7465;
	_t_7545 = _t_7532 + _t_7544;
	_t_7466 = tegpixellet_block_34(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,_t_7492,_t_7545,ty3_9_1,tx3_6_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,_t_315,y__3091_1,_t_7465);

	return _t_7466;
}
__device__ float tegpixelbody_block_24(float ty3_9_1,float ty1_7_1,float _t_315,float px0_10_1,float px1_11_1,float tx1_4_1,float tx3_6_1,float py0_12_1,float py1_13_1,float y__3091_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_7309;
	float _t_7310;
	float _t_7311;
	bool _t_7312;
	float _t_7315;
	float _t_7319;
	float _t_7320;
	float _t_7321;
	float _t_7322;
	bool _t_7323;
	float _t_7326;
	float _t_7330;
	bool _t_7331;
	float _t_7332;
	float _t_7333;
	float _t_7334;
	float _t_7335;
	float _t_7336;
	bool _t_7337;
	float _t_7340;
	float _t_7344;
	float _t_7345;
	float _t_7346;
	float _t_7347;
	bool _t_7348;
	float _t_7351;
	float _t_7355;
	bool _t_7356;
	float _t_7357;
	float _t_7358;
	float _t_7359;
	float _t_7360;
	float _t_7361;
	float _t_7362;
	bool _t_7363;
	float _t_7368;
	float _t_7374;
	float _t_7375;
	float _t_7376;
	float _t_7377;
	bool _t_7378;
	float _t_7379;
	float _t_7380;
	float _t_7381;
	bool _t_7382;
	float _t_7385;
	float _t_7389;
	float _t_7390;
	float _t_7391;
	float _t_7392;
	bool _t_7393;
	float _t_7396;
	float _t_7400;
	bool _t_7401;
	float _t_7402;
	float _t_7403;
	float _t_7404;
	float _t_7405;
	float _t_7406;
	bool _t_7407;
	float _t_7410;
	float _t_7414;
	float _t_7415;
	float _t_7416;
	float _t_7417;
	bool _t_7418;
	float _t_7421;
	float _t_7425;
	bool _t_7426;
	float _t_7427;
	float _t_7428;
	float _t_7429;
	float _t_7430;
	float _t_7431;
	float _t_7432;
	bool _t_7433;
	float _t_7438;
	float _t_7444;
	float _t_7445;
	float _t_7446;
	float _t_7447;
	bool _t_7448;
	bool _t_7449;

	float _t_7308;

	_t_7309 = -1.0f * ty1_7_1;
	_t_7310 = ty3_9_1 + _t_7309;
	_t_7311 = -1.0f * _t_7310;
	_t_7312 = _t_7311 < 0.0f;
	if(_t_7312)
		{
			float _t_7313;
			float _t_7314;
		
			_t_7313 = -1.0f * ty1_7_1;
			_t_7314 = ty3_9_1 + _t_7313;
			_t_7315 = _t_7314;
		
		}
else
		{
			float _t_7316;
			float _t_7317;
			float _t_7318;
		
			_t_7316 = -1.0f * ty1_7_1;
			_t_7317 = ty3_9_1 + _t_7316;
			_t_7318 = -1.0f * _t_7317;
			_t_7315 = _t_7318;
		
		}

	_t_7319 = _t_7315 * _t_315;
	_t_7320 = -1.0f * ty1_7_1;
	_t_7321 = ty3_9_1 + _t_7320;
	_t_7322 = -1.0f * _t_7321;
	_t_7323 = _t_7322 < 0.0f;
	if(_t_7323)
		{
			float _t_7324;
			float _t_7325;
		
			_t_7324 = -1.0f * ty1_7_1;
			_t_7325 = ty3_9_1 + _t_7324;
			_t_7326 = _t_7325;
		
		}
else
		{
			float _t_7327;
			float _t_7328;
			float _t_7329;
		
			_t_7327 = -1.0f * ty1_7_1;
			_t_7328 = ty3_9_1 + _t_7327;
			_t_7329 = -1.0f * _t_7328;
			_t_7326 = _t_7329;
		
		}

	_t_7330 = _t_7326 * _t_315;
	_t_7331 = 0.0f < _t_7330;
	if(_t_7331)
		{
		
			_t_7332 = px0_10_1;
		
		}
else
		{
		
			_t_7332 = px1_11_1;
		
		}

	_t_7333 = _t_7319 * _t_7332;
	_t_7334 = -1.0f * ty1_7_1;
	_t_7335 = ty3_9_1 + _t_7334;
	_t_7336 = -1.0f * _t_7335;
	_t_7337 = _t_7336 < 0.0f;
	if(_t_7337)
		{
			float _t_7338;
			float _t_7339;
		
			_t_7338 = -1.0f * tx3_6_1;
			_t_7339 = tx1_4_1 + _t_7338;
			_t_7340 = _t_7339;
		
		}
else
		{
			float _t_7341;
			float _t_7342;
			float _t_7343;
		
			_t_7341 = -1.0f * tx3_6_1;
			_t_7342 = tx1_4_1 + _t_7341;
			_t_7343 = -1.0f * _t_7342;
			_t_7340 = _t_7343;
		
		}

	_t_7344 = _t_7340 * _t_315;
	_t_7345 = -1.0f * ty1_7_1;
	_t_7346 = ty3_9_1 + _t_7345;
	_t_7347 = -1.0f * _t_7346;
	_t_7348 = _t_7347 < 0.0f;
	if(_t_7348)
		{
			float _t_7349;
			float _t_7350;
		
			_t_7349 = -1.0f * tx3_6_1;
			_t_7350 = tx1_4_1 + _t_7349;
			_t_7351 = _t_7350;
		
		}
else
		{
			float _t_7352;
			float _t_7353;
			float _t_7354;
		
			_t_7352 = -1.0f * tx3_6_1;
			_t_7353 = tx1_4_1 + _t_7352;
			_t_7354 = -1.0f * _t_7353;
			_t_7351 = _t_7354;
		
		}

	_t_7355 = _t_7351 * _t_315;
	_t_7356 = 0.0f < _t_7355;
	if(_t_7356)
		{
		
			_t_7357 = py0_12_1;
		
		}
else
		{
		
			_t_7357 = py1_13_1;
		
		}

	_t_7358 = _t_7344 * _t_7357;
	_t_7359 = _t_7333 + _t_7358;
	_t_7360 = -1.0f * ty1_7_1;
	_t_7361 = ty3_9_1 + _t_7360;
	_t_7362 = -1.0f * _t_7361;
	_t_7363 = _t_7362 < 0.0f;
	if(_t_7363)
		{
			float _t_7364;
			float _t_7365;
			float _t_7366;
			float _t_7367;
		
			_t_7364 = tx3_6_1 * ty1_7_1;
			_t_7365 = tx1_4_1 * ty3_9_1;
			_t_7366 = _t_7365 * -1.0f;
			_t_7367 = _t_7364 + _t_7366;
			_t_7368 = _t_7367;
		
		}
else
		{
			float _t_7369;
			float _t_7370;
			float _t_7371;
			float _t_7372;
			float _t_7373;
		
			_t_7369 = tx3_6_1 * ty1_7_1;
			_t_7370 = tx1_4_1 * ty3_9_1;
			_t_7371 = _t_7370 * -1.0f;
			_t_7372 = _t_7369 + _t_7371;
			_t_7373 = -1.0f * _t_7372;
			_t_7368 = _t_7373;
		
		}

	_t_7374 = -1.0f * _t_7368;
	_t_7375 = _t_7374 * _t_315;
	_t_7376 = _t_7375 * -1.0f;
	_t_7377 = _t_7359 + _t_7376;
	_t_7378 = _t_7377 < 0.0f;
	_t_7379 = -1.0f * ty1_7_1;
	_t_7380 = ty3_9_1 + _t_7379;
	_t_7381 = -1.0f * _t_7380;
	_t_7382 = _t_7381 < 0.0f;
	if(_t_7382)
		{
			float _t_7383;
			float _t_7384;
		
			_t_7383 = -1.0f * ty1_7_1;
			_t_7384 = ty3_9_1 + _t_7383;
			_t_7385 = _t_7384;
		
		}
else
		{
			float _t_7386;
			float _t_7387;
			float _t_7388;
		
			_t_7386 = -1.0f * ty1_7_1;
			_t_7387 = ty3_9_1 + _t_7386;
			_t_7388 = -1.0f * _t_7387;
			_t_7385 = _t_7388;
		
		}

	_t_7389 = _t_7385 * _t_315;
	_t_7390 = -1.0f * ty1_7_1;
	_t_7391 = ty3_9_1 + _t_7390;
	_t_7392 = -1.0f * _t_7391;
	_t_7393 = _t_7392 < 0.0f;
	if(_t_7393)
		{
			float _t_7394;
			float _t_7395;
		
			_t_7394 = -1.0f * ty1_7_1;
			_t_7395 = ty3_9_1 + _t_7394;
			_t_7396 = _t_7395;
		
		}
else
		{
			float _t_7397;
			float _t_7398;
			float _t_7399;
		
			_t_7397 = -1.0f * ty1_7_1;
			_t_7398 = ty3_9_1 + _t_7397;
			_t_7399 = -1.0f * _t_7398;
			_t_7396 = _t_7399;
		
		}

	_t_7400 = _t_7396 * _t_315;
	_t_7401 = 0.0f < _t_7400;
	if(_t_7401)
		{
		
			_t_7402 = px1_11_1;
		
		}
else
		{
		
			_t_7402 = px0_10_1;
		
		}

	_t_7403 = _t_7389 * _t_7402;
	_t_7404 = -1.0f * ty1_7_1;
	_t_7405 = ty3_9_1 + _t_7404;
	_t_7406 = -1.0f * _t_7405;
	_t_7407 = _t_7406 < 0.0f;
	if(_t_7407)
		{
			float _t_7408;
			float _t_7409;
		
			_t_7408 = -1.0f * tx3_6_1;
			_t_7409 = tx1_4_1 + _t_7408;
			_t_7410 = _t_7409;
		
		}
else
		{
			float _t_7411;
			float _t_7412;
			float _t_7413;
		
			_t_7411 = -1.0f * tx3_6_1;
			_t_7412 = tx1_4_1 + _t_7411;
			_t_7413 = -1.0f * _t_7412;
			_t_7410 = _t_7413;
		
		}

	_t_7414 = _t_7410 * _t_315;
	_t_7415 = -1.0f * ty1_7_1;
	_t_7416 = ty3_9_1 + _t_7415;
	_t_7417 = -1.0f * _t_7416;
	_t_7418 = _t_7417 < 0.0f;
	if(_t_7418)
		{
			float _t_7419;
			float _t_7420;
		
			_t_7419 = -1.0f * tx3_6_1;
			_t_7420 = tx1_4_1 + _t_7419;
			_t_7421 = _t_7420;
		
		}
else
		{
			float _t_7422;
			float _t_7423;
			float _t_7424;
		
			_t_7422 = -1.0f * tx3_6_1;
			_t_7423 = tx1_4_1 + _t_7422;
			_t_7424 = -1.0f * _t_7423;
			_t_7421 = _t_7424;
		
		}

	_t_7425 = _t_7421 * _t_315;
	_t_7426 = 0.0f < _t_7425;
	if(_t_7426)
		{
		
			_t_7427 = py1_13_1;
		
		}
else
		{
		
			_t_7427 = py0_12_1;
		
		}

	_t_7428 = _t_7414 * _t_7427;
	_t_7429 = _t_7403 + _t_7428;
	_t_7430 = -1.0f * ty1_7_1;
	_t_7431 = ty3_9_1 + _t_7430;
	_t_7432 = -1.0f * _t_7431;
	_t_7433 = _t_7432 < 0.0f;
	if(_t_7433)
		{
			float _t_7434;
			float _t_7435;
			float _t_7436;
			float _t_7437;
		
			_t_7434 = tx3_6_1 * ty1_7_1;
			_t_7435 = tx1_4_1 * ty3_9_1;
			_t_7436 = _t_7435 * -1.0f;
			_t_7437 = _t_7434 + _t_7436;
			_t_7438 = _t_7437;
		
		}
else
		{
			float _t_7439;
			float _t_7440;
			float _t_7441;
			float _t_7442;
			float _t_7443;
		
			_t_7439 = tx3_6_1 * ty1_7_1;
			_t_7440 = tx1_4_1 * ty3_9_1;
			_t_7441 = _t_7440 * -1.0f;
			_t_7442 = _t_7439 + _t_7441;
			_t_7443 = -1.0f * _t_7442;
			_t_7438 = _t_7443;
		
		}

	_t_7444 = -1.0f * _t_7438;
	_t_7445 = _t_7444 * _t_315;
	_t_7446 = _t_7445 * -1.0f;
	_t_7447 = _t_7429 + _t_7446;
	_t_7448 = 0.0f < _t_7447;
	_t_7449 = _t_7378 && _t_7448;
	if(_t_7449)
		{
			float _t_7450;
			float _t_7451;
			float _t_7452;
			bool _t_7453;
			float _t_7458;
			float _t_7464;
			float _t_7465;
			float _t_7466;
		
			_t_7450 = -1.0f * ty1_7_1;
			_t_7451 = ty3_9_1 + _t_7450;
			_t_7452 = -1.0f * _t_7451;
			_t_7453 = _t_7452 < 0.0f;
			if(_t_7453)
				{
					float _t_7454;
					float _t_7455;
					float _t_7456;
					float _t_7457;
				
					_t_7454 = tx3_6_1 * ty1_7_1;
					_t_7455 = tx1_4_1 * ty3_9_1;
					_t_7456 = _t_7455 * -1.0f;
					_t_7457 = _t_7454 + _t_7456;
					_t_7458 = _t_7457;
				
				}
		else
				{
					float _t_7459;
					float _t_7460;
					float _t_7461;
					float _t_7462;
					float _t_7463;
				
					_t_7459 = tx3_6_1 * ty1_7_1;
					_t_7460 = tx1_4_1 * ty3_9_1;
					_t_7461 = _t_7460 * -1.0f;
					_t_7462 = _t_7459 + _t_7461;
					_t_7463 = -1.0f * _t_7462;
					_t_7458 = _t_7463;
				
				}
		
			_t_7464 = -1.0f * _t_7458;
			_t_7465 = _t_7464 * _t_315;
			_t_7466 = tegpixellet_block_33(ty3_9_1,ty1_7_1,_t_315,_t_7465,tx1_4_1,tx3_6_1,y__3091_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1);
			_t_7308 = _t_7466;
		
		}
else
		{
		
			_t_7308 = 0.0f;
		
		}


	return _t_7308;
}
__device__ float tegpixelintegrator_24(float ty3_9_1,float pc1_15_1,float tc2_19_1,float ty2_8_1,float _t_7307,float pc0_14_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float py1_13_1,float pc2_16_1,float tx2_5_1,float px1_11_1,float tc0_17_1,float py0_12_1,float _t_315,float tc1_18_1,float px0_10_1,float _t_7198){
    float y__3091_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_7307 - _t_7198)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__3091_1 = _t_7198 + __step__ * (i + (float)(0.5));
        float _t_7308;
		_t_7308 = tegpixelbody_block_24(ty3_9_1,ty1_7_1,_t_315,px0_10_1,px1_11_1,tx1_4_1,tx3_6_1,py0_12_1,py1_13_1,y__3091_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);;
        __output__ = __output__ + _t_7308 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_8(float ty3_9_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float _t_315,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_7090;
	float _t_7091;
	float _t_7092;
	bool _t_7093;
	float _t_7096;
	float _t_7100;
	float _t_7101;
	float _t_7102;
	float _t_7103;
	float _t_7104;
	bool _t_7105;
	float _t_7108;
	float _t_7112;
	float _t_7113;
	bool _t_7114;
	float _t_7115;
	float _t_7116;
	float _t_7117;
	float _t_7118;
	float _t_7119;
	bool _t_7120;
	float _t_7123;
	float _t_7127;
	float _t_7128;
	float _t_7129;
	float _t_7130;
	bool _t_7131;
	float _t_7134;
	float _t_7138;
	float _t_7139;
	float _t_7140;
	float _t_7141;
	float _t_7142;
	bool _t_7143;
	float _t_7146;
	float _t_7150;
	float _t_7151;
	float _t_7152;
	float _t_7153;
	float _t_7154;
	float _t_7155;
	float _t_7156;
	float _t_7157;
	float _t_7158;
	bool _t_7159;
	float _t_7162;
	float _t_7166;
	float _t_7167;
	float _t_7168;
	float _t_7169;
	bool _t_7170;
	float _t_7173;
	float _t_7177;
	float _t_7178;
	float _t_7179;
	float _t_7180;
	float _t_7181;
	bool _t_7182;
	float _t_7185;
	float _t_7189;
	float _t_7190;
	float _t_7191;
	float _t_7192;
	float _t_7193;
	float _t_7194;
	bool _t_7195;
	float _t_7196;
	float _t_7197;
	float _t_7198;
	float _t_7199;
	float _t_7200;
	float _t_7201;
	bool _t_7202;
	float _t_7205;
	float _t_7209;
	float _t_7210;
	float _t_7211;
	float _t_7212;
	float _t_7213;
	bool _t_7214;
	float _t_7217;
	float _t_7221;
	float _t_7222;
	bool _t_7223;
	float _t_7224;
	float _t_7225;
	float _t_7226;
	float _t_7227;
	float _t_7228;
	bool _t_7229;
	float _t_7232;
	float _t_7236;
	float _t_7237;
	float _t_7238;
	float _t_7239;
	bool _t_7240;
	float _t_7243;
	float _t_7247;
	float _t_7248;
	float _t_7249;
	float _t_7250;
	float _t_7251;
	bool _t_7252;
	float _t_7255;
	float _t_7259;
	float _t_7260;
	float _t_7261;
	float _t_7262;
	float _t_7263;
	float _t_7264;
	float _t_7265;
	float _t_7266;
	float _t_7267;
	bool _t_7268;
	float _t_7271;
	float _t_7275;
	float _t_7276;
	float _t_7277;
	float _t_7278;
	bool _t_7279;
	float _t_7282;
	float _t_7286;
	float _t_7287;
	float _t_7288;
	float _t_7289;
	float _t_7290;
	bool _t_7291;
	float _t_7294;
	float _t_7298;
	float _t_7299;
	float _t_7300;
	float _t_7301;
	float _t_7302;
	float _t_7303;
	bool _t_7304;
	float _t_7305;
	float _t_7306;
	float _t_7307;

	float _t_316;

	_t_7090 = -1.0f * ty1_7_1;
	_t_7091 = ty3_9_1 + _t_7090;
	_t_7092 = -1.0f * _t_7091;
	_t_7093 = _t_7092 < 0.0f;
	if(_t_7093)
		{
			float _t_7094;
			float _t_7095;
		
			_t_7094 = -1.0f * tx3_6_1;
			_t_7095 = tx1_4_1 + _t_7094;
			_t_7096 = _t_7095;
		
		}
else
		{
			float _t_7097;
			float _t_7098;
			float _t_7099;
		
			_t_7097 = -1.0f * tx3_6_1;
			_t_7098 = tx1_4_1 + _t_7097;
			_t_7099 = -1.0f * _t_7098;
			_t_7096 = _t_7099;
		
		}

	_t_7100 = _t_7096 * _t_315;
	_t_7101 = _t_7100 * -1.0f;
	_t_7102 = -1.0f * ty1_7_1;
	_t_7103 = ty3_9_1 + _t_7102;
	_t_7104 = -1.0f * _t_7103;
	_t_7105 = _t_7104 < 0.0f;
	if(_t_7105)
		{
			float _t_7106;
			float _t_7107;
		
			_t_7106 = -1.0f * tx3_6_1;
			_t_7107 = tx1_4_1 + _t_7106;
			_t_7108 = _t_7107;
		
		}
else
		{
			float _t_7109;
			float _t_7110;
			float _t_7111;
		
			_t_7109 = -1.0f * tx3_6_1;
			_t_7110 = tx1_4_1 + _t_7109;
			_t_7111 = -1.0f * _t_7110;
			_t_7108 = _t_7111;
		
		}

	_t_7112 = _t_7108 * _t_315;
	_t_7113 = _t_7112 * -1.0f;
	_t_7114 = 0.0f < _t_7113;
	if(_t_7114)
		{
		
			_t_7115 = px0_10_1;
		
		}
else
		{
		
			_t_7115 = px1_11_1;
		
		}

	_t_7116 = _t_7101 * _t_7115;
	_t_7117 = -1.0f * ty1_7_1;
	_t_7118 = ty3_9_1 + _t_7117;
	_t_7119 = -1.0f * _t_7118;
	_t_7120 = _t_7119 < 0.0f;
	if(_t_7120)
		{
			float _t_7121;
			float _t_7122;
		
			_t_7121 = -1.0f * tx3_6_1;
			_t_7122 = tx1_4_1 + _t_7121;
			_t_7123 = _t_7122;
		
		}
else
		{
			float _t_7124;
			float _t_7125;
			float _t_7126;
		
			_t_7124 = -1.0f * tx3_6_1;
			_t_7125 = tx1_4_1 + _t_7124;
			_t_7126 = -1.0f * _t_7125;
			_t_7123 = _t_7126;
		
		}

	_t_7127 = _t_7123 * _t_315;
	_t_7128 = -1.0f * ty1_7_1;
	_t_7129 = ty3_9_1 + _t_7128;
	_t_7130 = -1.0f * _t_7129;
	_t_7131 = _t_7130 < 0.0f;
	if(_t_7131)
		{
			float _t_7132;
			float _t_7133;
		
			_t_7132 = -1.0f * tx3_6_1;
			_t_7133 = tx1_4_1 + _t_7132;
			_t_7134 = _t_7133;
		
		}
else
		{
			float _t_7135;
			float _t_7136;
			float _t_7137;
		
			_t_7135 = -1.0f * tx3_6_1;
			_t_7136 = tx1_4_1 + _t_7135;
			_t_7137 = -1.0f * _t_7136;
			_t_7134 = _t_7137;
		
		}

	_t_7138 = _t_7134 * _t_315;
	_t_7139 = _t_7127 * _t_7138;
	_t_7140 = -1.0f * ty1_7_1;
	_t_7141 = ty3_9_1 + _t_7140;
	_t_7142 = -1.0f * _t_7141;
	_t_7143 = _t_7142 < 0.0f;
	if(_t_7143)
		{
			float _t_7144;
			float _t_7145;
		
			_t_7144 = -1.0f * ty1_7_1;
			_t_7145 = ty3_9_1 + _t_7144;
			_t_7146 = _t_7145;
		
		}
else
		{
			float _t_7147;
			float _t_7148;
			float _t_7149;
		
			_t_7147 = -1.0f * ty1_7_1;
			_t_7148 = ty3_9_1 + _t_7147;
			_t_7149 = -1.0f * _t_7148;
			_t_7146 = _t_7149;
		
		}

	_t_7150 = _t_7146 * _t_315;
	_t_7151 = 1.0f + _t_7150;
	_t_7152 = 1.0f / _t_7151;
	_t_7153 = _t_7139 * _t_7152;
	_t_7154 = _t_7153 * -1.0f;
	_t_7155 = 1.0f + _t_7154;
	_t_7156 = -1.0f * ty1_7_1;
	_t_7157 = ty3_9_1 + _t_7156;
	_t_7158 = -1.0f * _t_7157;
	_t_7159 = _t_7158 < 0.0f;
	if(_t_7159)
		{
			float _t_7160;
			float _t_7161;
		
			_t_7160 = -1.0f * tx3_6_1;
			_t_7161 = tx1_4_1 + _t_7160;
			_t_7162 = _t_7161;
		
		}
else
		{
			float _t_7163;
			float _t_7164;
			float _t_7165;
		
			_t_7163 = -1.0f * tx3_6_1;
			_t_7164 = tx1_4_1 + _t_7163;
			_t_7165 = -1.0f * _t_7164;
			_t_7162 = _t_7165;
		
		}

	_t_7166 = _t_7162 * _t_315;
	_t_7167 = -1.0f * ty1_7_1;
	_t_7168 = ty3_9_1 + _t_7167;
	_t_7169 = -1.0f * _t_7168;
	_t_7170 = _t_7169 < 0.0f;
	if(_t_7170)
		{
			float _t_7171;
			float _t_7172;
		
			_t_7171 = -1.0f * tx3_6_1;
			_t_7172 = tx1_4_1 + _t_7171;
			_t_7173 = _t_7172;
		
		}
else
		{
			float _t_7174;
			float _t_7175;
			float _t_7176;
		
			_t_7174 = -1.0f * tx3_6_1;
			_t_7175 = tx1_4_1 + _t_7174;
			_t_7176 = -1.0f * _t_7175;
			_t_7173 = _t_7176;
		
		}

	_t_7177 = _t_7173 * _t_315;
	_t_7178 = _t_7166 * _t_7177;
	_t_7179 = -1.0f * ty1_7_1;
	_t_7180 = ty3_9_1 + _t_7179;
	_t_7181 = -1.0f * _t_7180;
	_t_7182 = _t_7181 < 0.0f;
	if(_t_7182)
		{
			float _t_7183;
			float _t_7184;
		
			_t_7183 = -1.0f * ty1_7_1;
			_t_7184 = ty3_9_1 + _t_7183;
			_t_7185 = _t_7184;
		
		}
else
		{
			float _t_7186;
			float _t_7187;
			float _t_7188;
		
			_t_7186 = -1.0f * ty1_7_1;
			_t_7187 = ty3_9_1 + _t_7186;
			_t_7188 = -1.0f * _t_7187;
			_t_7185 = _t_7188;
		
		}

	_t_7189 = _t_7185 * _t_315;
	_t_7190 = 1.0f + _t_7189;
	_t_7191 = 1.0f / _t_7190;
	_t_7192 = _t_7178 * _t_7191;
	_t_7193 = _t_7192 * -1.0f;
	_t_7194 = 1.0f + _t_7193;
	_t_7195 = 0.0f < _t_7194;
	if(_t_7195)
		{
		
			_t_7196 = py0_12_1;
		
		}
else
		{
		
			_t_7196 = py1_13_1;
		
		}

	_t_7197 = _t_7155 * _t_7196;
	_t_7198 = _t_7116 + _t_7197;
	_t_7199 = -1.0f * ty1_7_1;
	_t_7200 = ty3_9_1 + _t_7199;
	_t_7201 = -1.0f * _t_7200;
	_t_7202 = _t_7201 < 0.0f;
	if(_t_7202)
		{
			float _t_7203;
			float _t_7204;
		
			_t_7203 = -1.0f * tx3_6_1;
			_t_7204 = tx1_4_1 + _t_7203;
			_t_7205 = _t_7204;
		
		}
else
		{
			float _t_7206;
			float _t_7207;
			float _t_7208;
		
			_t_7206 = -1.0f * tx3_6_1;
			_t_7207 = tx1_4_1 + _t_7206;
			_t_7208 = -1.0f * _t_7207;
			_t_7205 = _t_7208;
		
		}

	_t_7209 = _t_7205 * _t_315;
	_t_7210 = _t_7209 * -1.0f;
	_t_7211 = -1.0f * ty1_7_1;
	_t_7212 = ty3_9_1 + _t_7211;
	_t_7213 = -1.0f * _t_7212;
	_t_7214 = _t_7213 < 0.0f;
	if(_t_7214)
		{
			float _t_7215;
			float _t_7216;
		
			_t_7215 = -1.0f * tx3_6_1;
			_t_7216 = tx1_4_1 + _t_7215;
			_t_7217 = _t_7216;
		
		}
else
		{
			float _t_7218;
			float _t_7219;
			float _t_7220;
		
			_t_7218 = -1.0f * tx3_6_1;
			_t_7219 = tx1_4_1 + _t_7218;
			_t_7220 = -1.0f * _t_7219;
			_t_7217 = _t_7220;
		
		}

	_t_7221 = _t_7217 * _t_315;
	_t_7222 = _t_7221 * -1.0f;
	_t_7223 = 0.0f < _t_7222;
	if(_t_7223)
		{
		
			_t_7224 = px1_11_1;
		
		}
else
		{
		
			_t_7224 = px0_10_1;
		
		}

	_t_7225 = _t_7210 * _t_7224;
	_t_7226 = -1.0f * ty1_7_1;
	_t_7227 = ty3_9_1 + _t_7226;
	_t_7228 = -1.0f * _t_7227;
	_t_7229 = _t_7228 < 0.0f;
	if(_t_7229)
		{
			float _t_7230;
			float _t_7231;
		
			_t_7230 = -1.0f * tx3_6_1;
			_t_7231 = tx1_4_1 + _t_7230;
			_t_7232 = _t_7231;
		
		}
else
		{
			float _t_7233;
			float _t_7234;
			float _t_7235;
		
			_t_7233 = -1.0f * tx3_6_1;
			_t_7234 = tx1_4_1 + _t_7233;
			_t_7235 = -1.0f * _t_7234;
			_t_7232 = _t_7235;
		
		}

	_t_7236 = _t_7232 * _t_315;
	_t_7237 = -1.0f * ty1_7_1;
	_t_7238 = ty3_9_1 + _t_7237;
	_t_7239 = -1.0f * _t_7238;
	_t_7240 = _t_7239 < 0.0f;
	if(_t_7240)
		{
			float _t_7241;
			float _t_7242;
		
			_t_7241 = -1.0f * tx3_6_1;
			_t_7242 = tx1_4_1 + _t_7241;
			_t_7243 = _t_7242;
		
		}
else
		{
			float _t_7244;
			float _t_7245;
			float _t_7246;
		
			_t_7244 = -1.0f * tx3_6_1;
			_t_7245 = tx1_4_1 + _t_7244;
			_t_7246 = -1.0f * _t_7245;
			_t_7243 = _t_7246;
		
		}

	_t_7247 = _t_7243 * _t_315;
	_t_7248 = _t_7236 * _t_7247;
	_t_7249 = -1.0f * ty1_7_1;
	_t_7250 = ty3_9_1 + _t_7249;
	_t_7251 = -1.0f * _t_7250;
	_t_7252 = _t_7251 < 0.0f;
	if(_t_7252)
		{
			float _t_7253;
			float _t_7254;
		
			_t_7253 = -1.0f * ty1_7_1;
			_t_7254 = ty3_9_1 + _t_7253;
			_t_7255 = _t_7254;
		
		}
else
		{
			float _t_7256;
			float _t_7257;
			float _t_7258;
		
			_t_7256 = -1.0f * ty1_7_1;
			_t_7257 = ty3_9_1 + _t_7256;
			_t_7258 = -1.0f * _t_7257;
			_t_7255 = _t_7258;
		
		}

	_t_7259 = _t_7255 * _t_315;
	_t_7260 = 1.0f + _t_7259;
	_t_7261 = 1.0f / _t_7260;
	_t_7262 = _t_7248 * _t_7261;
	_t_7263 = _t_7262 * -1.0f;
	_t_7264 = 1.0f + _t_7263;
	_t_7265 = -1.0f * ty1_7_1;
	_t_7266 = ty3_9_1 + _t_7265;
	_t_7267 = -1.0f * _t_7266;
	_t_7268 = _t_7267 < 0.0f;
	if(_t_7268)
		{
			float _t_7269;
			float _t_7270;
		
			_t_7269 = -1.0f * tx3_6_1;
			_t_7270 = tx1_4_1 + _t_7269;
			_t_7271 = _t_7270;
		
		}
else
		{
			float _t_7272;
			float _t_7273;
			float _t_7274;
		
			_t_7272 = -1.0f * tx3_6_1;
			_t_7273 = tx1_4_1 + _t_7272;
			_t_7274 = -1.0f * _t_7273;
			_t_7271 = _t_7274;
		
		}

	_t_7275 = _t_7271 * _t_315;
	_t_7276 = -1.0f * ty1_7_1;
	_t_7277 = ty3_9_1 + _t_7276;
	_t_7278 = -1.0f * _t_7277;
	_t_7279 = _t_7278 < 0.0f;
	if(_t_7279)
		{
			float _t_7280;
			float _t_7281;
		
			_t_7280 = -1.0f * tx3_6_1;
			_t_7281 = tx1_4_1 + _t_7280;
			_t_7282 = _t_7281;
		
		}
else
		{
			float _t_7283;
			float _t_7284;
			float _t_7285;
		
			_t_7283 = -1.0f * tx3_6_1;
			_t_7284 = tx1_4_1 + _t_7283;
			_t_7285 = -1.0f * _t_7284;
			_t_7282 = _t_7285;
		
		}

	_t_7286 = _t_7282 * _t_315;
	_t_7287 = _t_7275 * _t_7286;
	_t_7288 = -1.0f * ty1_7_1;
	_t_7289 = ty3_9_1 + _t_7288;
	_t_7290 = -1.0f * _t_7289;
	_t_7291 = _t_7290 < 0.0f;
	if(_t_7291)
		{
			float _t_7292;
			float _t_7293;
		
			_t_7292 = -1.0f * ty1_7_1;
			_t_7293 = ty3_9_1 + _t_7292;
			_t_7294 = _t_7293;
		
		}
else
		{
			float _t_7295;
			float _t_7296;
			float _t_7297;
		
			_t_7295 = -1.0f * ty1_7_1;
			_t_7296 = ty3_9_1 + _t_7295;
			_t_7297 = -1.0f * _t_7296;
			_t_7294 = _t_7297;
		
		}

	_t_7298 = _t_7294 * _t_315;
	_t_7299 = 1.0f + _t_7298;
	_t_7300 = 1.0f / _t_7299;
	_t_7301 = _t_7287 * _t_7300;
	_t_7302 = _t_7301 * -1.0f;
	_t_7303 = 1.0f + _t_7302;
	_t_7304 = 0.0f < _t_7303;
	if(_t_7304)
		{
		
			_t_7305 = py1_13_1;
		
		}
else
		{
		
			_t_7305 = py0_12_1;
		
		}

	_t_7306 = _t_7264 * _t_7305;
	_t_7307 = _t_7225 + _t_7306;
	_t_316 = tegpixelintegrator_24(ty3_9_1,pc1_15_1,tc2_19_1,ty2_8_1,_t_7307,pc0_14_1,ty1_7_1,tx1_4_1,tx3_6_1,py1_13_1,pc2_16_1,tx2_5_1,px1_11_1,tc0_17_1,py0_12_1,_t_315,tc1_18_1,px0_10_1,_t_7198);

	return _t_316;
}
__device__ float tegpixellet_block_36(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float _t_8328,float _t_8381,float ty3_9_1,float tx3_6_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float _t_343,float y__3165_1,float _t_8301){
	float _t_8382;
	float _t_8383;
	float _t_8384;
	float _t_8385;
	float _t_8386;
	float _t_8387;
	float _t_8388;
	float _t_8389;
	float _t_8390;
	float _t_8391;
	float _t_8392;
	float _t_8393;
	float _t_8394;
	float _t_8395;
	float _t_8396;
	float _t_8397;
	float _t_8398;
	float _t_8399;
	float _t_8400;
	float _t_8401;
	float _t_8402;
	float _t_8403;
	float _t_8404;
	bool _t_8405;
	float _t_8406;
	float _t_8407;
	float _t_8408;
	float _t_8409;
	float _t_8410;
	float _t_8411;
	float _t_8412;
	float _t_8413;
	float _t_8414;
	float _t_8415;
	float _t_8416;
	float _t_8417;
	float _t_8418;
	bool _t_8419;
	float _t_8420;
	float _t_8421;
	float _t_8422;
	float _t_8423;
	bool _t_8424;
	bool _t_8425;
	bool _t_8426;
	bool _t_8427;
	bool _t_8428;
	bool _t_8429;
	bool _t_8430;
	float _t_8760;

	float _t_8302;

	_t_8382 = -1.0f * pc0_14_1;
	_t_8383 = tc0_17_1 + _t_8382;
	_t_8384 = _t_8383 * _t_8383;
	_t_8385 = -1.0f * pc1_15_1;
	_t_8386 = tc1_18_1 + _t_8385;
	_t_8387 = _t_8386 * _t_8386;
	_t_8388 = _t_8384 + _t_8387;
	_t_8389 = -1.0f * pc2_16_1;
	_t_8390 = tc2_19_1 + _t_8389;
	_t_8391 = _t_8390 * _t_8390;
	_t_8392 = _t_8388 + _t_8391;
	_t_8393 = tx1_4_1 * ty2_8_1;
	_t_8394 = tx2_5_1 * ty1_7_1;
	_t_8395 = _t_8394 * -1.0f;
	_t_8396 = _t_8393 + _t_8395;
	_t_8397 = -1.0f * ty2_8_1;
	_t_8398 = ty1_7_1 + _t_8397;
	_t_8399 = _t_8398 * _t_8328;
	_t_8400 = _t_8396 + _t_8399;
	_t_8401 = -1.0f * tx1_4_1;
	_t_8402 = tx2_5_1 + _t_8401;
	_t_8403 = _t_8402 * _t_8381;
	_t_8404 = _t_8400 + _t_8403;
	_t_8405 = _t_8404 < 0.0f;
	if(_t_8405)
		{
		
			_t_8406 = 1.0f;
		
		}
else
		{
		
			_t_8406 = 0.0f;
		
		}

	_t_8407 = tx2_5_1 * ty3_9_1;
	_t_8408 = tx3_6_1 * ty2_8_1;
	_t_8409 = _t_8408 * -1.0f;
	_t_8410 = _t_8407 + _t_8409;
	_t_8411 = -1.0f * ty3_9_1;
	_t_8412 = ty2_8_1 + _t_8411;
	_t_8413 = _t_8412 * _t_8328;
	_t_8414 = _t_8410 + _t_8413;
	_t_8415 = -1.0f * tx2_5_1;
	_t_8416 = tx3_6_1 + _t_8415;
	_t_8417 = _t_8416 * _t_8381;
	_t_8418 = _t_8414 + _t_8417;
	_t_8419 = _t_8418 < 0.0f;
	if(_t_8419)
		{
		
			_t_8420 = 1.0f;
		
		}
else
		{
		
			_t_8420 = 0.0f;
		
		}

	_t_8421 = _t_8406 * _t_8420;
	_t_8422 = _t_8392 * _t_8421;
	_t_8423 = _t_8422 * _t_8381;
	_t_8424 = py0_12_1 < _t_8381;
	_t_8425 = _t_8381 < py1_13_1;
	_t_8426 = _t_8424 && _t_8425;
	_t_8427 = px0_10_1 < _t_8328;
	_t_8428 = _t_8328 < px1_11_1;
	_t_8429 = _t_8427 && _t_8428;
	_t_8430 = _t_8426 && _t_8429;
	if(_t_8430)
		{
			float _t_8431;
			float _t_8432;
			float _t_8433;
			bool _t_8434;
			float _t_8437;
			float _t_8441;
			float _t_8442;
			float _t_8443;
			float _t_8444;
			float _t_8445;
			bool _t_8446;
			float _t_8449;
			float _t_8453;
			float _t_8454;
			bool _t_8455;
			float _t_8456;
			float _t_8457;
			float _t_8458;
			float _t_8459;
			float _t_8460;
			bool _t_8461;
			float _t_8464;
			float _t_8468;
			float _t_8469;
			float _t_8470;
			float _t_8471;
			bool _t_8472;
			float _t_8475;
			float _t_8479;
			float _t_8480;
			float _t_8481;
			float _t_8482;
			float _t_8483;
			bool _t_8484;
			float _t_8487;
			float _t_8491;
			float _t_8492;
			float _t_8493;
			float _t_8494;
			float _t_8495;
			float _t_8496;
			float _t_8497;
			float _t_8498;
			float _t_8499;
			bool _t_8500;
			float _t_8503;
			float _t_8507;
			float _t_8508;
			float _t_8509;
			float _t_8510;
			bool _t_8511;
			float _t_8514;
			float _t_8518;
			float _t_8519;
			float _t_8520;
			float _t_8521;
			float _t_8522;
			bool _t_8523;
			float _t_8526;
			float _t_8530;
			float _t_8531;
			float _t_8532;
			float _t_8533;
			float _t_8534;
			float _t_8535;
			bool _t_8536;
			float _t_8537;
			float _t_8538;
			float _t_8539;
			bool _t_8540;
			float _t_8541;
			float _t_8542;
			float _t_8543;
			bool _t_8544;
			float _t_8547;
			float _t_8551;
			float _t_8552;
			float _t_8553;
			float _t_8554;
			float _t_8555;
			bool _t_8556;
			float _t_8559;
			float _t_8563;
			float _t_8564;
			bool _t_8565;
			float _t_8566;
			float _t_8567;
			float _t_8568;
			float _t_8569;
			float _t_8570;
			bool _t_8571;
			float _t_8574;
			float _t_8578;
			float _t_8579;
			float _t_8580;
			float _t_8581;
			bool _t_8582;
			float _t_8585;
			float _t_8589;
			float _t_8590;
			float _t_8591;
			float _t_8592;
			float _t_8593;
			bool _t_8594;
			float _t_8597;
			float _t_8601;
			float _t_8602;
			float _t_8603;
			float _t_8604;
			float _t_8605;
			float _t_8606;
			float _t_8607;
			float _t_8608;
			float _t_8609;
			bool _t_8610;
			float _t_8613;
			float _t_8617;
			float _t_8618;
			float _t_8619;
			float _t_8620;
			bool _t_8621;
			float _t_8624;
			float _t_8628;
			float _t_8629;
			float _t_8630;
			float _t_8631;
			float _t_8632;
			bool _t_8633;
			float _t_8636;
			float _t_8640;
			float _t_8641;
			float _t_8642;
			float _t_8643;
			float _t_8644;
			float _t_8645;
			bool _t_8646;
			float _t_8647;
			float _t_8648;
			float _t_8649;
			bool _t_8650;
			bool _t_8651;
			float _t_8652;
			float _t_8653;
			float _t_8654;
			bool _t_8655;
			float _t_8658;
			float _t_8662;
			float _t_8663;
			float _t_8664;
			float _t_8665;
			bool _t_8666;
			float _t_8669;
			float _t_8673;
			bool _t_8674;
			float _t_8675;
			float _t_8676;
			float _t_8677;
			float _t_8678;
			float _t_8679;
			bool _t_8680;
			float _t_8683;
			float _t_8687;
			float _t_8688;
			float _t_8689;
			float _t_8690;
			bool _t_8691;
			float _t_8694;
			float _t_8698;
			bool _t_8699;
			float _t_8700;
			float _t_8701;
			float _t_8702;
			bool _t_8703;
			float _t_8704;
			float _t_8705;
			float _t_8706;
			bool _t_8707;
			float _t_8710;
			float _t_8714;
			float _t_8715;
			float _t_8716;
			float _t_8717;
			bool _t_8718;
			float _t_8721;
			float _t_8725;
			bool _t_8726;
			float _t_8727;
			float _t_8728;
			float _t_8729;
			float _t_8730;
			float _t_8731;
			bool _t_8732;
			float _t_8735;
			float _t_8739;
			float _t_8740;
			float _t_8741;
			float _t_8742;
			bool _t_8743;
			float _t_8746;
			float _t_8750;
			bool _t_8751;
			float _t_8752;
			float _t_8753;
			float _t_8754;
			bool _t_8755;
			bool _t_8756;
			bool _t_8757;
			float _t_8758;
			float _t_8759;
		
			_t_8431 = -1.0f * ty1_7_1;
			_t_8432 = ty3_9_1 + _t_8431;
			_t_8433 = -1.0f * _t_8432;
			_t_8434 = _t_8433 < 0.0f;
			if(_t_8434)
				{
					float _t_8435;
					float _t_8436;
				
					_t_8435 = -1.0f * tx3_6_1;
					_t_8436 = tx1_4_1 + _t_8435;
					_t_8437 = _t_8436;
				
				}
		else
				{
					float _t_8438;
					float _t_8439;
					float _t_8440;
				
					_t_8438 = -1.0f * tx3_6_1;
					_t_8439 = tx1_4_1 + _t_8438;
					_t_8440 = -1.0f * _t_8439;
					_t_8437 = _t_8440;
				
				}
		
			_t_8441 = _t_8437 * _t_343;
			_t_8442 = _t_8441 * -1.0f;
			_t_8443 = -1.0f * ty1_7_1;
			_t_8444 = ty3_9_1 + _t_8443;
			_t_8445 = -1.0f * _t_8444;
			_t_8446 = _t_8445 < 0.0f;
			if(_t_8446)
				{
					float _t_8447;
					float _t_8448;
				
					_t_8447 = -1.0f * tx3_6_1;
					_t_8448 = tx1_4_1 + _t_8447;
					_t_8449 = _t_8448;
				
				}
		else
				{
					float _t_8450;
					float _t_8451;
					float _t_8452;
				
					_t_8450 = -1.0f * tx3_6_1;
					_t_8451 = tx1_4_1 + _t_8450;
					_t_8452 = -1.0f * _t_8451;
					_t_8449 = _t_8452;
				
				}
		
			_t_8453 = _t_8449 * _t_343;
			_t_8454 = _t_8453 * -1.0f;
			_t_8455 = 0.0f < _t_8454;
			if(_t_8455)
				{
				
					_t_8456 = px0_10_1;
				
				}
		else
				{
				
					_t_8456 = px1_11_1;
				
				}
		
			_t_8457 = _t_8442 * _t_8456;
			_t_8458 = -1.0f * ty1_7_1;
			_t_8459 = ty3_9_1 + _t_8458;
			_t_8460 = -1.0f * _t_8459;
			_t_8461 = _t_8460 < 0.0f;
			if(_t_8461)
				{
					float _t_8462;
					float _t_8463;
				
					_t_8462 = -1.0f * tx3_6_1;
					_t_8463 = tx1_4_1 + _t_8462;
					_t_8464 = _t_8463;
				
				}
		else
				{
					float _t_8465;
					float _t_8466;
					float _t_8467;
				
					_t_8465 = -1.0f * tx3_6_1;
					_t_8466 = tx1_4_1 + _t_8465;
					_t_8467 = -1.0f * _t_8466;
					_t_8464 = _t_8467;
				
				}
		
			_t_8468 = _t_8464 * _t_343;
			_t_8469 = -1.0f * ty1_7_1;
			_t_8470 = ty3_9_1 + _t_8469;
			_t_8471 = -1.0f * _t_8470;
			_t_8472 = _t_8471 < 0.0f;
			if(_t_8472)
				{
					float _t_8473;
					float _t_8474;
				
					_t_8473 = -1.0f * tx3_6_1;
					_t_8474 = tx1_4_1 + _t_8473;
					_t_8475 = _t_8474;
				
				}
		else
				{
					float _t_8476;
					float _t_8477;
					float _t_8478;
				
					_t_8476 = -1.0f * tx3_6_1;
					_t_8477 = tx1_4_1 + _t_8476;
					_t_8478 = -1.0f * _t_8477;
					_t_8475 = _t_8478;
				
				}
		
			_t_8479 = _t_8475 * _t_343;
			_t_8480 = _t_8468 * _t_8479;
			_t_8481 = -1.0f * ty1_7_1;
			_t_8482 = ty3_9_1 + _t_8481;
			_t_8483 = -1.0f * _t_8482;
			_t_8484 = _t_8483 < 0.0f;
			if(_t_8484)
				{
					float _t_8485;
					float _t_8486;
				
					_t_8485 = -1.0f * ty1_7_1;
					_t_8486 = ty3_9_1 + _t_8485;
					_t_8487 = _t_8486;
				
				}
		else
				{
					float _t_8488;
					float _t_8489;
					float _t_8490;
				
					_t_8488 = -1.0f * ty1_7_1;
					_t_8489 = ty3_9_1 + _t_8488;
					_t_8490 = -1.0f * _t_8489;
					_t_8487 = _t_8490;
				
				}
		
			_t_8491 = _t_8487 * _t_343;
			_t_8492 = 1.0f + _t_8491;
			_t_8493 = 1.0f / _t_8492;
			_t_8494 = _t_8480 * _t_8493;
			_t_8495 = _t_8494 * -1.0f;
			_t_8496 = 1.0f + _t_8495;
			_t_8497 = -1.0f * ty1_7_1;
			_t_8498 = ty3_9_1 + _t_8497;
			_t_8499 = -1.0f * _t_8498;
			_t_8500 = _t_8499 < 0.0f;
			if(_t_8500)
				{
					float _t_8501;
					float _t_8502;
				
					_t_8501 = -1.0f * tx3_6_1;
					_t_8502 = tx1_4_1 + _t_8501;
					_t_8503 = _t_8502;
				
				}
		else
				{
					float _t_8504;
					float _t_8505;
					float _t_8506;
				
					_t_8504 = -1.0f * tx3_6_1;
					_t_8505 = tx1_4_1 + _t_8504;
					_t_8506 = -1.0f * _t_8505;
					_t_8503 = _t_8506;
				
				}
		
			_t_8507 = _t_8503 * _t_343;
			_t_8508 = -1.0f * ty1_7_1;
			_t_8509 = ty3_9_1 + _t_8508;
			_t_8510 = -1.0f * _t_8509;
			_t_8511 = _t_8510 < 0.0f;
			if(_t_8511)
				{
					float _t_8512;
					float _t_8513;
				
					_t_8512 = -1.0f * tx3_6_1;
					_t_8513 = tx1_4_1 + _t_8512;
					_t_8514 = _t_8513;
				
				}
		else
				{
					float _t_8515;
					float _t_8516;
					float _t_8517;
				
					_t_8515 = -1.0f * tx3_6_1;
					_t_8516 = tx1_4_1 + _t_8515;
					_t_8517 = -1.0f * _t_8516;
					_t_8514 = _t_8517;
				
				}
		
			_t_8518 = _t_8514 * _t_343;
			_t_8519 = _t_8507 * _t_8518;
			_t_8520 = -1.0f * ty1_7_1;
			_t_8521 = ty3_9_1 + _t_8520;
			_t_8522 = -1.0f * _t_8521;
			_t_8523 = _t_8522 < 0.0f;
			if(_t_8523)
				{
					float _t_8524;
					float _t_8525;
				
					_t_8524 = -1.0f * ty1_7_1;
					_t_8525 = ty3_9_1 + _t_8524;
					_t_8526 = _t_8525;
				
				}
		else
				{
					float _t_8527;
					float _t_8528;
					float _t_8529;
				
					_t_8527 = -1.0f * ty1_7_1;
					_t_8528 = ty3_9_1 + _t_8527;
					_t_8529 = -1.0f * _t_8528;
					_t_8526 = _t_8529;
				
				}
		
			_t_8530 = _t_8526 * _t_343;
			_t_8531 = 1.0f + _t_8530;
			_t_8532 = 1.0f / _t_8531;
			_t_8533 = _t_8519 * _t_8532;
			_t_8534 = _t_8533 * -1.0f;
			_t_8535 = 1.0f + _t_8534;
			_t_8536 = 0.0f < _t_8535;
			if(_t_8536)
				{
				
					_t_8537 = py0_12_1;
				
				}
		else
				{
				
					_t_8537 = py1_13_1;
				
				}
		
			_t_8538 = _t_8496 * _t_8537;
			_t_8539 = _t_8457 + _t_8538;
			_t_8540 = _t_8539 < y__3165_1;
			_t_8541 = -1.0f * ty1_7_1;
			_t_8542 = ty3_9_1 + _t_8541;
			_t_8543 = -1.0f * _t_8542;
			_t_8544 = _t_8543 < 0.0f;
			if(_t_8544)
				{
					float _t_8545;
					float _t_8546;
				
					_t_8545 = -1.0f * tx3_6_1;
					_t_8546 = tx1_4_1 + _t_8545;
					_t_8547 = _t_8546;
				
				}
		else
				{
					float _t_8548;
					float _t_8549;
					float _t_8550;
				
					_t_8548 = -1.0f * tx3_6_1;
					_t_8549 = tx1_4_1 + _t_8548;
					_t_8550 = -1.0f * _t_8549;
					_t_8547 = _t_8550;
				
				}
		
			_t_8551 = _t_8547 * _t_343;
			_t_8552 = _t_8551 * -1.0f;
			_t_8553 = -1.0f * ty1_7_1;
			_t_8554 = ty3_9_1 + _t_8553;
			_t_8555 = -1.0f * _t_8554;
			_t_8556 = _t_8555 < 0.0f;
			if(_t_8556)
				{
					float _t_8557;
					float _t_8558;
				
					_t_8557 = -1.0f * tx3_6_1;
					_t_8558 = tx1_4_1 + _t_8557;
					_t_8559 = _t_8558;
				
				}
		else
				{
					float _t_8560;
					float _t_8561;
					float _t_8562;
				
					_t_8560 = -1.0f * tx3_6_1;
					_t_8561 = tx1_4_1 + _t_8560;
					_t_8562 = -1.0f * _t_8561;
					_t_8559 = _t_8562;
				
				}
		
			_t_8563 = _t_8559 * _t_343;
			_t_8564 = _t_8563 * -1.0f;
			_t_8565 = 0.0f < _t_8564;
			if(_t_8565)
				{
				
					_t_8566 = px1_11_1;
				
				}
		else
				{
				
					_t_8566 = px0_10_1;
				
				}
		
			_t_8567 = _t_8552 * _t_8566;
			_t_8568 = -1.0f * ty1_7_1;
			_t_8569 = ty3_9_1 + _t_8568;
			_t_8570 = -1.0f * _t_8569;
			_t_8571 = _t_8570 < 0.0f;
			if(_t_8571)
				{
					float _t_8572;
					float _t_8573;
				
					_t_8572 = -1.0f * tx3_6_1;
					_t_8573 = tx1_4_1 + _t_8572;
					_t_8574 = _t_8573;
				
				}
		else
				{
					float _t_8575;
					float _t_8576;
					float _t_8577;
				
					_t_8575 = -1.0f * tx3_6_1;
					_t_8576 = tx1_4_1 + _t_8575;
					_t_8577 = -1.0f * _t_8576;
					_t_8574 = _t_8577;
				
				}
		
			_t_8578 = _t_8574 * _t_343;
			_t_8579 = -1.0f * ty1_7_1;
			_t_8580 = ty3_9_1 + _t_8579;
			_t_8581 = -1.0f * _t_8580;
			_t_8582 = _t_8581 < 0.0f;
			if(_t_8582)
				{
					float _t_8583;
					float _t_8584;
				
					_t_8583 = -1.0f * tx3_6_1;
					_t_8584 = tx1_4_1 + _t_8583;
					_t_8585 = _t_8584;
				
				}
		else
				{
					float _t_8586;
					float _t_8587;
					float _t_8588;
				
					_t_8586 = -1.0f * tx3_6_1;
					_t_8587 = tx1_4_1 + _t_8586;
					_t_8588 = -1.0f * _t_8587;
					_t_8585 = _t_8588;
				
				}
		
			_t_8589 = _t_8585 * _t_343;
			_t_8590 = _t_8578 * _t_8589;
			_t_8591 = -1.0f * ty1_7_1;
			_t_8592 = ty3_9_1 + _t_8591;
			_t_8593 = -1.0f * _t_8592;
			_t_8594 = _t_8593 < 0.0f;
			if(_t_8594)
				{
					float _t_8595;
					float _t_8596;
				
					_t_8595 = -1.0f * ty1_7_1;
					_t_8596 = ty3_9_1 + _t_8595;
					_t_8597 = _t_8596;
				
				}
		else
				{
					float _t_8598;
					float _t_8599;
					float _t_8600;
				
					_t_8598 = -1.0f * ty1_7_1;
					_t_8599 = ty3_9_1 + _t_8598;
					_t_8600 = -1.0f * _t_8599;
					_t_8597 = _t_8600;
				
				}
		
			_t_8601 = _t_8597 * _t_343;
			_t_8602 = 1.0f + _t_8601;
			_t_8603 = 1.0f / _t_8602;
			_t_8604 = _t_8590 * _t_8603;
			_t_8605 = _t_8604 * -1.0f;
			_t_8606 = 1.0f + _t_8605;
			_t_8607 = -1.0f * ty1_7_1;
			_t_8608 = ty3_9_1 + _t_8607;
			_t_8609 = -1.0f * _t_8608;
			_t_8610 = _t_8609 < 0.0f;
			if(_t_8610)
				{
					float _t_8611;
					float _t_8612;
				
					_t_8611 = -1.0f * tx3_6_1;
					_t_8612 = tx1_4_1 + _t_8611;
					_t_8613 = _t_8612;
				
				}
		else
				{
					float _t_8614;
					float _t_8615;
					float _t_8616;
				
					_t_8614 = -1.0f * tx3_6_1;
					_t_8615 = tx1_4_1 + _t_8614;
					_t_8616 = -1.0f * _t_8615;
					_t_8613 = _t_8616;
				
				}
		
			_t_8617 = _t_8613 * _t_343;
			_t_8618 = -1.0f * ty1_7_1;
			_t_8619 = ty3_9_1 + _t_8618;
			_t_8620 = -1.0f * _t_8619;
			_t_8621 = _t_8620 < 0.0f;
			if(_t_8621)
				{
					float _t_8622;
					float _t_8623;
				
					_t_8622 = -1.0f * tx3_6_1;
					_t_8623 = tx1_4_1 + _t_8622;
					_t_8624 = _t_8623;
				
				}
		else
				{
					float _t_8625;
					float _t_8626;
					float _t_8627;
				
					_t_8625 = -1.0f * tx3_6_1;
					_t_8626 = tx1_4_1 + _t_8625;
					_t_8627 = -1.0f * _t_8626;
					_t_8624 = _t_8627;
				
				}
		
			_t_8628 = _t_8624 * _t_343;
			_t_8629 = _t_8617 * _t_8628;
			_t_8630 = -1.0f * ty1_7_1;
			_t_8631 = ty3_9_1 + _t_8630;
			_t_8632 = -1.0f * _t_8631;
			_t_8633 = _t_8632 < 0.0f;
			if(_t_8633)
				{
					float _t_8634;
					float _t_8635;
				
					_t_8634 = -1.0f * ty1_7_1;
					_t_8635 = ty3_9_1 + _t_8634;
					_t_8636 = _t_8635;
				
				}
		else
				{
					float _t_8637;
					float _t_8638;
					float _t_8639;
				
					_t_8637 = -1.0f * ty1_7_1;
					_t_8638 = ty3_9_1 + _t_8637;
					_t_8639 = -1.0f * _t_8638;
					_t_8636 = _t_8639;
				
				}
		
			_t_8640 = _t_8636 * _t_343;
			_t_8641 = 1.0f + _t_8640;
			_t_8642 = 1.0f / _t_8641;
			_t_8643 = _t_8629 * _t_8642;
			_t_8644 = _t_8643 * -1.0f;
			_t_8645 = 1.0f + _t_8644;
			_t_8646 = 0.0f < _t_8645;
			if(_t_8646)
				{
				
					_t_8647 = py1_13_1;
				
				}
		else
				{
				
					_t_8647 = py0_12_1;
				
				}
		
			_t_8648 = _t_8606 * _t_8647;
			_t_8649 = _t_8567 + _t_8648;
			_t_8650 = y__3165_1 < _t_8649;
			_t_8651 = _t_8540 && _t_8650;
			_t_8652 = -1.0f * ty1_7_1;
			_t_8653 = ty3_9_1 + _t_8652;
			_t_8654 = -1.0f * _t_8653;
			_t_8655 = _t_8654 < 0.0f;
			if(_t_8655)
				{
					float _t_8656;
					float _t_8657;
				
					_t_8656 = -1.0f * ty1_7_1;
					_t_8657 = ty3_9_1 + _t_8656;
					_t_8658 = _t_8657;
				
				}
		else
				{
					float _t_8659;
					float _t_8660;
					float _t_8661;
				
					_t_8659 = -1.0f * ty1_7_1;
					_t_8660 = ty3_9_1 + _t_8659;
					_t_8661 = -1.0f * _t_8660;
					_t_8658 = _t_8661;
				
				}
		
			_t_8662 = _t_8658 * _t_343;
			_t_8663 = -1.0f * ty1_7_1;
			_t_8664 = ty3_9_1 + _t_8663;
			_t_8665 = -1.0f * _t_8664;
			_t_8666 = _t_8665 < 0.0f;
			if(_t_8666)
				{
					float _t_8667;
					float _t_8668;
				
					_t_8667 = -1.0f * ty1_7_1;
					_t_8668 = ty3_9_1 + _t_8667;
					_t_8669 = _t_8668;
				
				}
		else
				{
					float _t_8670;
					float _t_8671;
					float _t_8672;
				
					_t_8670 = -1.0f * ty1_7_1;
					_t_8671 = ty3_9_1 + _t_8670;
					_t_8672 = -1.0f * _t_8671;
					_t_8669 = _t_8672;
				
				}
		
			_t_8673 = _t_8669 * _t_343;
			_t_8674 = 0.0f < _t_8673;
			if(_t_8674)
				{
				
					_t_8675 = px0_10_1;
				
				}
		else
				{
				
					_t_8675 = px1_11_1;
				
				}
		
			_t_8676 = _t_8662 * _t_8675;
			_t_8677 = -1.0f * ty1_7_1;
			_t_8678 = ty3_9_1 + _t_8677;
			_t_8679 = -1.0f * _t_8678;
			_t_8680 = _t_8679 < 0.0f;
			if(_t_8680)
				{
					float _t_8681;
					float _t_8682;
				
					_t_8681 = -1.0f * tx3_6_1;
					_t_8682 = tx1_4_1 + _t_8681;
					_t_8683 = _t_8682;
				
				}
		else
				{
					float _t_8684;
					float _t_8685;
					float _t_8686;
				
					_t_8684 = -1.0f * tx3_6_1;
					_t_8685 = tx1_4_1 + _t_8684;
					_t_8686 = -1.0f * _t_8685;
					_t_8683 = _t_8686;
				
				}
		
			_t_8687 = _t_8683 * _t_343;
			_t_8688 = -1.0f * ty1_7_1;
			_t_8689 = ty3_9_1 + _t_8688;
			_t_8690 = -1.0f * _t_8689;
			_t_8691 = _t_8690 < 0.0f;
			if(_t_8691)
				{
					float _t_8692;
					float _t_8693;
				
					_t_8692 = -1.0f * tx3_6_1;
					_t_8693 = tx1_4_1 + _t_8692;
					_t_8694 = _t_8693;
				
				}
		else
				{
					float _t_8695;
					float _t_8696;
					float _t_8697;
				
					_t_8695 = -1.0f * tx3_6_1;
					_t_8696 = tx1_4_1 + _t_8695;
					_t_8697 = -1.0f * _t_8696;
					_t_8694 = _t_8697;
				
				}
		
			_t_8698 = _t_8694 * _t_343;
			_t_8699 = 0.0f < _t_8698;
			if(_t_8699)
				{
				
					_t_8700 = py0_12_1;
				
				}
		else
				{
				
					_t_8700 = py1_13_1;
				
				}
		
			_t_8701 = _t_8687 * _t_8700;
			_t_8702 = _t_8676 + _t_8701;
			_t_8703 = _t_8702 < _t_8301;
			_t_8704 = -1.0f * ty1_7_1;
			_t_8705 = ty3_9_1 + _t_8704;
			_t_8706 = -1.0f * _t_8705;
			_t_8707 = _t_8706 < 0.0f;
			if(_t_8707)
				{
					float _t_8708;
					float _t_8709;
				
					_t_8708 = -1.0f * ty1_7_1;
					_t_8709 = ty3_9_1 + _t_8708;
					_t_8710 = _t_8709;
				
				}
		else
				{
					float _t_8711;
					float _t_8712;
					float _t_8713;
				
					_t_8711 = -1.0f * ty1_7_1;
					_t_8712 = ty3_9_1 + _t_8711;
					_t_8713 = -1.0f * _t_8712;
					_t_8710 = _t_8713;
				
				}
		
			_t_8714 = _t_8710 * _t_343;
			_t_8715 = -1.0f * ty1_7_1;
			_t_8716 = ty3_9_1 + _t_8715;
			_t_8717 = -1.0f * _t_8716;
			_t_8718 = _t_8717 < 0.0f;
			if(_t_8718)
				{
					float _t_8719;
					float _t_8720;
				
					_t_8719 = -1.0f * ty1_7_1;
					_t_8720 = ty3_9_1 + _t_8719;
					_t_8721 = _t_8720;
				
				}
		else
				{
					float _t_8722;
					float _t_8723;
					float _t_8724;
				
					_t_8722 = -1.0f * ty1_7_1;
					_t_8723 = ty3_9_1 + _t_8722;
					_t_8724 = -1.0f * _t_8723;
					_t_8721 = _t_8724;
				
				}
		
			_t_8725 = _t_8721 * _t_343;
			_t_8726 = 0.0f < _t_8725;
			if(_t_8726)
				{
				
					_t_8727 = px1_11_1;
				
				}
		else
				{
				
					_t_8727 = px0_10_1;
				
				}
		
			_t_8728 = _t_8714 * _t_8727;
			_t_8729 = -1.0f * ty1_7_1;
			_t_8730 = ty3_9_1 + _t_8729;
			_t_8731 = -1.0f * _t_8730;
			_t_8732 = _t_8731 < 0.0f;
			if(_t_8732)
				{
					float _t_8733;
					float _t_8734;
				
					_t_8733 = -1.0f * tx3_6_1;
					_t_8734 = tx1_4_1 + _t_8733;
					_t_8735 = _t_8734;
				
				}
		else
				{
					float _t_8736;
					float _t_8737;
					float _t_8738;
				
					_t_8736 = -1.0f * tx3_6_1;
					_t_8737 = tx1_4_1 + _t_8736;
					_t_8738 = -1.0f * _t_8737;
					_t_8735 = _t_8738;
				
				}
		
			_t_8739 = _t_8735 * _t_343;
			_t_8740 = -1.0f * ty1_7_1;
			_t_8741 = ty3_9_1 + _t_8740;
			_t_8742 = -1.0f * _t_8741;
			_t_8743 = _t_8742 < 0.0f;
			if(_t_8743)
				{
					float _t_8744;
					float _t_8745;
				
					_t_8744 = -1.0f * tx3_6_1;
					_t_8745 = tx1_4_1 + _t_8744;
					_t_8746 = _t_8745;
				
				}
		else
				{
					float _t_8747;
					float _t_8748;
					float _t_8749;
				
					_t_8747 = -1.0f * tx3_6_1;
					_t_8748 = tx1_4_1 + _t_8747;
					_t_8749 = -1.0f * _t_8748;
					_t_8746 = _t_8749;
				
				}
		
			_t_8750 = _t_8746 * _t_343;
			_t_8751 = 0.0f < _t_8750;
			if(_t_8751)
				{
				
					_t_8752 = py1_13_1;
				
				}
		else
				{
				
					_t_8752 = py0_12_1;
				
				}
		
			_t_8753 = _t_8739 * _t_8752;
			_t_8754 = _t_8728 + _t_8753;
			_t_8755 = _t_8301 < _t_8754;
			_t_8756 = _t_8703 && _t_8755;
			_t_8757 = _t_8651 && _t_8756;
			if(_t_8757)
				{
				
					_t_8758 = 1.0f;
				
				}
		else
				{
				
					_t_8758 = 0.0f;
				
				}
		
			_t_8759 = _t_8758 * _t_343;
			_t_8760 = _t_8759;
		
		}
else
		{
		
			_t_8760 = 0.0f;
		
		}

	_t_8302 = _t_8423 * _t_8760;

	return _t_8302;
}
__device__ float tegpixellet_block_35(float ty3_9_1,float ty1_7_1,float _t_343,float _t_8301,float tx1_4_1,float tx3_6_1,float y__3165_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1){
	float _t_8303;
	float _t_8304;
	float _t_8305;
	bool _t_8306;
	float _t_8309;
	float _t_8313;
	float _t_8314;
	float _t_8315;
	float _t_8316;
	float _t_8317;
	bool _t_8318;
	float _t_8321;
	float _t_8325;
	float _t_8326;
	float _t_8327;
	float _t_8328;
	float _t_8329;
	float _t_8330;
	float _t_8331;
	bool _t_8332;
	float _t_8335;
	float _t_8339;
	float _t_8340;
	float _t_8341;
	float _t_8342;
	bool _t_8343;
	float _t_8346;
	float _t_8350;
	float _t_8351;
	float _t_8352;
	float _t_8353;
	float _t_8354;
	bool _t_8355;
	float _t_8358;
	float _t_8362;
	float _t_8363;
	float _t_8364;
	float _t_8365;
	float _t_8366;
	float _t_8367;
	float _t_8368;
	float _t_8369;
	float _t_8370;
	float _t_8371;
	bool _t_8372;
	float _t_8375;
	float _t_8379;
	float _t_8380;
	float _t_8381;

	float _t_8302;

	_t_8303 = -1.0f * ty1_7_1;
	_t_8304 = ty3_9_1 + _t_8303;
	_t_8305 = -1.0f * _t_8304;
	_t_8306 = _t_8305 < 0.0f;
	if(_t_8306)
		{
			float _t_8307;
			float _t_8308;
		
			_t_8307 = -1.0f * ty1_7_1;
			_t_8308 = ty3_9_1 + _t_8307;
			_t_8309 = _t_8308;
		
		}
else
		{
			float _t_8310;
			float _t_8311;
			float _t_8312;
		
			_t_8310 = -1.0f * ty1_7_1;
			_t_8311 = ty3_9_1 + _t_8310;
			_t_8312 = -1.0f * _t_8311;
			_t_8309 = _t_8312;
		
		}

	_t_8313 = _t_8309 * _t_343;
	_t_8314 = _t_8313 * _t_8301;
	_t_8315 = -1.0f * ty1_7_1;
	_t_8316 = ty3_9_1 + _t_8315;
	_t_8317 = -1.0f * _t_8316;
	_t_8318 = _t_8317 < 0.0f;
	if(_t_8318)
		{
			float _t_8319;
			float _t_8320;
		
			_t_8319 = -1.0f * tx3_6_1;
			_t_8320 = tx1_4_1 + _t_8319;
			_t_8321 = _t_8320;
		
		}
else
		{
			float _t_8322;
			float _t_8323;
			float _t_8324;
		
			_t_8322 = -1.0f * tx3_6_1;
			_t_8323 = tx1_4_1 + _t_8322;
			_t_8324 = -1.0f * _t_8323;
			_t_8321 = _t_8324;
		
		}

	_t_8325 = _t_8321 * _t_343;
	_t_8326 = _t_8325 * -1.0f;
	_t_8327 = _t_8326 * y__3165_1;
	_t_8328 = _t_8314 + _t_8327;
	_t_8329 = -1.0f * ty1_7_1;
	_t_8330 = ty3_9_1 + _t_8329;
	_t_8331 = -1.0f * _t_8330;
	_t_8332 = _t_8331 < 0.0f;
	if(_t_8332)
		{
			float _t_8333;
			float _t_8334;
		
			_t_8333 = -1.0f * tx3_6_1;
			_t_8334 = tx1_4_1 + _t_8333;
			_t_8335 = _t_8334;
		
		}
else
		{
			float _t_8336;
			float _t_8337;
			float _t_8338;
		
			_t_8336 = -1.0f * tx3_6_1;
			_t_8337 = tx1_4_1 + _t_8336;
			_t_8338 = -1.0f * _t_8337;
			_t_8335 = _t_8338;
		
		}

	_t_8339 = _t_8335 * _t_343;
	_t_8340 = -1.0f * ty1_7_1;
	_t_8341 = ty3_9_1 + _t_8340;
	_t_8342 = -1.0f * _t_8341;
	_t_8343 = _t_8342 < 0.0f;
	if(_t_8343)
		{
			float _t_8344;
			float _t_8345;
		
			_t_8344 = -1.0f * tx3_6_1;
			_t_8345 = tx1_4_1 + _t_8344;
			_t_8346 = _t_8345;
		
		}
else
		{
			float _t_8347;
			float _t_8348;
			float _t_8349;
		
			_t_8347 = -1.0f * tx3_6_1;
			_t_8348 = tx1_4_1 + _t_8347;
			_t_8349 = -1.0f * _t_8348;
			_t_8346 = _t_8349;
		
		}

	_t_8350 = _t_8346 * _t_343;
	_t_8351 = _t_8339 * _t_8350;
	_t_8352 = -1.0f * ty1_7_1;
	_t_8353 = ty3_9_1 + _t_8352;
	_t_8354 = -1.0f * _t_8353;
	_t_8355 = _t_8354 < 0.0f;
	if(_t_8355)
		{
			float _t_8356;
			float _t_8357;
		
			_t_8356 = -1.0f * ty1_7_1;
			_t_8357 = ty3_9_1 + _t_8356;
			_t_8358 = _t_8357;
		
		}
else
		{
			float _t_8359;
			float _t_8360;
			float _t_8361;
		
			_t_8359 = -1.0f * ty1_7_1;
			_t_8360 = ty3_9_1 + _t_8359;
			_t_8361 = -1.0f * _t_8360;
			_t_8358 = _t_8361;
		
		}

	_t_8362 = _t_8358 * _t_343;
	_t_8363 = 1.0f + _t_8362;
	_t_8364 = 1.0f / _t_8363;
	_t_8365 = _t_8351 * _t_8364;
	_t_8366 = _t_8365 * -1.0f;
	_t_8367 = 1.0f + _t_8366;
	_t_8368 = _t_8367 * y__3165_1;
	_t_8369 = -1.0f * ty1_7_1;
	_t_8370 = ty3_9_1 + _t_8369;
	_t_8371 = -1.0f * _t_8370;
	_t_8372 = _t_8371 < 0.0f;
	if(_t_8372)
		{
			float _t_8373;
			float _t_8374;
		
			_t_8373 = -1.0f * tx3_6_1;
			_t_8374 = tx1_4_1 + _t_8373;
			_t_8375 = _t_8374;
		
		}
else
		{
			float _t_8376;
			float _t_8377;
			float _t_8378;
		
			_t_8376 = -1.0f * tx3_6_1;
			_t_8377 = tx1_4_1 + _t_8376;
			_t_8378 = -1.0f * _t_8377;
			_t_8375 = _t_8378;
		
		}

	_t_8379 = _t_8375 * _t_343;
	_t_8380 = _t_8379 * _t_8301;
	_t_8381 = _t_8368 + _t_8380;
	_t_8302 = tegpixellet_block_36(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,_t_8328,_t_8381,ty3_9_1,tx3_6_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,_t_343,y__3165_1,_t_8301);

	return _t_8302;
}
__device__ float tegpixelbody_block_25(float ty3_9_1,float ty1_7_1,float _t_343,float px0_10_1,float px1_11_1,float tx1_4_1,float tx3_6_1,float py0_12_1,float py1_13_1,float y__3165_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_8145;
	float _t_8146;
	float _t_8147;
	bool _t_8148;
	float _t_8151;
	float _t_8155;
	float _t_8156;
	float _t_8157;
	float _t_8158;
	bool _t_8159;
	float _t_8162;
	float _t_8166;
	bool _t_8167;
	float _t_8168;
	float _t_8169;
	float _t_8170;
	float _t_8171;
	float _t_8172;
	bool _t_8173;
	float _t_8176;
	float _t_8180;
	float _t_8181;
	float _t_8182;
	float _t_8183;
	bool _t_8184;
	float _t_8187;
	float _t_8191;
	bool _t_8192;
	float _t_8193;
	float _t_8194;
	float _t_8195;
	float _t_8196;
	float _t_8197;
	float _t_8198;
	bool _t_8199;
	float _t_8204;
	float _t_8210;
	float _t_8211;
	float _t_8212;
	float _t_8213;
	bool _t_8214;
	float _t_8215;
	float _t_8216;
	float _t_8217;
	bool _t_8218;
	float _t_8221;
	float _t_8225;
	float _t_8226;
	float _t_8227;
	float _t_8228;
	bool _t_8229;
	float _t_8232;
	float _t_8236;
	bool _t_8237;
	float _t_8238;
	float _t_8239;
	float _t_8240;
	float _t_8241;
	float _t_8242;
	bool _t_8243;
	float _t_8246;
	float _t_8250;
	float _t_8251;
	float _t_8252;
	float _t_8253;
	bool _t_8254;
	float _t_8257;
	float _t_8261;
	bool _t_8262;
	float _t_8263;
	float _t_8264;
	float _t_8265;
	float _t_8266;
	float _t_8267;
	float _t_8268;
	bool _t_8269;
	float _t_8274;
	float _t_8280;
	float _t_8281;
	float _t_8282;
	float _t_8283;
	bool _t_8284;
	bool _t_8285;

	float _t_8144;

	_t_8145 = -1.0f * ty1_7_1;
	_t_8146 = ty3_9_1 + _t_8145;
	_t_8147 = -1.0f * _t_8146;
	_t_8148 = _t_8147 < 0.0f;
	if(_t_8148)
		{
			float _t_8149;
			float _t_8150;
		
			_t_8149 = -1.0f * ty1_7_1;
			_t_8150 = ty3_9_1 + _t_8149;
			_t_8151 = _t_8150;
		
		}
else
		{
			float _t_8152;
			float _t_8153;
			float _t_8154;
		
			_t_8152 = -1.0f * ty1_7_1;
			_t_8153 = ty3_9_1 + _t_8152;
			_t_8154 = -1.0f * _t_8153;
			_t_8151 = _t_8154;
		
		}

	_t_8155 = _t_8151 * _t_343;
	_t_8156 = -1.0f * ty1_7_1;
	_t_8157 = ty3_9_1 + _t_8156;
	_t_8158 = -1.0f * _t_8157;
	_t_8159 = _t_8158 < 0.0f;
	if(_t_8159)
		{
			float _t_8160;
			float _t_8161;
		
			_t_8160 = -1.0f * ty1_7_1;
			_t_8161 = ty3_9_1 + _t_8160;
			_t_8162 = _t_8161;
		
		}
else
		{
			float _t_8163;
			float _t_8164;
			float _t_8165;
		
			_t_8163 = -1.0f * ty1_7_1;
			_t_8164 = ty3_9_1 + _t_8163;
			_t_8165 = -1.0f * _t_8164;
			_t_8162 = _t_8165;
		
		}

	_t_8166 = _t_8162 * _t_343;
	_t_8167 = 0.0f < _t_8166;
	if(_t_8167)
		{
		
			_t_8168 = px0_10_1;
		
		}
else
		{
		
			_t_8168 = px1_11_1;
		
		}

	_t_8169 = _t_8155 * _t_8168;
	_t_8170 = -1.0f * ty1_7_1;
	_t_8171 = ty3_9_1 + _t_8170;
	_t_8172 = -1.0f * _t_8171;
	_t_8173 = _t_8172 < 0.0f;
	if(_t_8173)
		{
			float _t_8174;
			float _t_8175;
		
			_t_8174 = -1.0f * tx3_6_1;
			_t_8175 = tx1_4_1 + _t_8174;
			_t_8176 = _t_8175;
		
		}
else
		{
			float _t_8177;
			float _t_8178;
			float _t_8179;
		
			_t_8177 = -1.0f * tx3_6_1;
			_t_8178 = tx1_4_1 + _t_8177;
			_t_8179 = -1.0f * _t_8178;
			_t_8176 = _t_8179;
		
		}

	_t_8180 = _t_8176 * _t_343;
	_t_8181 = -1.0f * ty1_7_1;
	_t_8182 = ty3_9_1 + _t_8181;
	_t_8183 = -1.0f * _t_8182;
	_t_8184 = _t_8183 < 0.0f;
	if(_t_8184)
		{
			float _t_8185;
			float _t_8186;
		
			_t_8185 = -1.0f * tx3_6_1;
			_t_8186 = tx1_4_1 + _t_8185;
			_t_8187 = _t_8186;
		
		}
else
		{
			float _t_8188;
			float _t_8189;
			float _t_8190;
		
			_t_8188 = -1.0f * tx3_6_1;
			_t_8189 = tx1_4_1 + _t_8188;
			_t_8190 = -1.0f * _t_8189;
			_t_8187 = _t_8190;
		
		}

	_t_8191 = _t_8187 * _t_343;
	_t_8192 = 0.0f < _t_8191;
	if(_t_8192)
		{
		
			_t_8193 = py0_12_1;
		
		}
else
		{
		
			_t_8193 = py1_13_1;
		
		}

	_t_8194 = _t_8180 * _t_8193;
	_t_8195 = _t_8169 + _t_8194;
	_t_8196 = -1.0f * ty1_7_1;
	_t_8197 = ty3_9_1 + _t_8196;
	_t_8198 = -1.0f * _t_8197;
	_t_8199 = _t_8198 < 0.0f;
	if(_t_8199)
		{
			float _t_8200;
			float _t_8201;
			float _t_8202;
			float _t_8203;
		
			_t_8200 = tx3_6_1 * ty1_7_1;
			_t_8201 = tx1_4_1 * ty3_9_1;
			_t_8202 = _t_8201 * -1.0f;
			_t_8203 = _t_8200 + _t_8202;
			_t_8204 = _t_8203;
		
		}
else
		{
			float _t_8205;
			float _t_8206;
			float _t_8207;
			float _t_8208;
			float _t_8209;
		
			_t_8205 = tx3_6_1 * ty1_7_1;
			_t_8206 = tx1_4_1 * ty3_9_1;
			_t_8207 = _t_8206 * -1.0f;
			_t_8208 = _t_8205 + _t_8207;
			_t_8209 = -1.0f * _t_8208;
			_t_8204 = _t_8209;
		
		}

	_t_8210 = -1.0f * _t_8204;
	_t_8211 = _t_8210 * _t_343;
	_t_8212 = _t_8211 * -1.0f;
	_t_8213 = _t_8195 + _t_8212;
	_t_8214 = _t_8213 < 0.0f;
	_t_8215 = -1.0f * ty1_7_1;
	_t_8216 = ty3_9_1 + _t_8215;
	_t_8217 = -1.0f * _t_8216;
	_t_8218 = _t_8217 < 0.0f;
	if(_t_8218)
		{
			float _t_8219;
			float _t_8220;
		
			_t_8219 = -1.0f * ty1_7_1;
			_t_8220 = ty3_9_1 + _t_8219;
			_t_8221 = _t_8220;
		
		}
else
		{
			float _t_8222;
			float _t_8223;
			float _t_8224;
		
			_t_8222 = -1.0f * ty1_7_1;
			_t_8223 = ty3_9_1 + _t_8222;
			_t_8224 = -1.0f * _t_8223;
			_t_8221 = _t_8224;
		
		}

	_t_8225 = _t_8221 * _t_343;
	_t_8226 = -1.0f * ty1_7_1;
	_t_8227 = ty3_9_1 + _t_8226;
	_t_8228 = -1.0f * _t_8227;
	_t_8229 = _t_8228 < 0.0f;
	if(_t_8229)
		{
			float _t_8230;
			float _t_8231;
		
			_t_8230 = -1.0f * ty1_7_1;
			_t_8231 = ty3_9_1 + _t_8230;
			_t_8232 = _t_8231;
		
		}
else
		{
			float _t_8233;
			float _t_8234;
			float _t_8235;
		
			_t_8233 = -1.0f * ty1_7_1;
			_t_8234 = ty3_9_1 + _t_8233;
			_t_8235 = -1.0f * _t_8234;
			_t_8232 = _t_8235;
		
		}

	_t_8236 = _t_8232 * _t_343;
	_t_8237 = 0.0f < _t_8236;
	if(_t_8237)
		{
		
			_t_8238 = px1_11_1;
		
		}
else
		{
		
			_t_8238 = px0_10_1;
		
		}

	_t_8239 = _t_8225 * _t_8238;
	_t_8240 = -1.0f * ty1_7_1;
	_t_8241 = ty3_9_1 + _t_8240;
	_t_8242 = -1.0f * _t_8241;
	_t_8243 = _t_8242 < 0.0f;
	if(_t_8243)
		{
			float _t_8244;
			float _t_8245;
		
			_t_8244 = -1.0f * tx3_6_1;
			_t_8245 = tx1_4_1 + _t_8244;
			_t_8246 = _t_8245;
		
		}
else
		{
			float _t_8247;
			float _t_8248;
			float _t_8249;
		
			_t_8247 = -1.0f * tx3_6_1;
			_t_8248 = tx1_4_1 + _t_8247;
			_t_8249 = -1.0f * _t_8248;
			_t_8246 = _t_8249;
		
		}

	_t_8250 = _t_8246 * _t_343;
	_t_8251 = -1.0f * ty1_7_1;
	_t_8252 = ty3_9_1 + _t_8251;
	_t_8253 = -1.0f * _t_8252;
	_t_8254 = _t_8253 < 0.0f;
	if(_t_8254)
		{
			float _t_8255;
			float _t_8256;
		
			_t_8255 = -1.0f * tx3_6_1;
			_t_8256 = tx1_4_1 + _t_8255;
			_t_8257 = _t_8256;
		
		}
else
		{
			float _t_8258;
			float _t_8259;
			float _t_8260;
		
			_t_8258 = -1.0f * tx3_6_1;
			_t_8259 = tx1_4_1 + _t_8258;
			_t_8260 = -1.0f * _t_8259;
			_t_8257 = _t_8260;
		
		}

	_t_8261 = _t_8257 * _t_343;
	_t_8262 = 0.0f < _t_8261;
	if(_t_8262)
		{
		
			_t_8263 = py1_13_1;
		
		}
else
		{
		
			_t_8263 = py0_12_1;
		
		}

	_t_8264 = _t_8250 * _t_8263;
	_t_8265 = _t_8239 + _t_8264;
	_t_8266 = -1.0f * ty1_7_1;
	_t_8267 = ty3_9_1 + _t_8266;
	_t_8268 = -1.0f * _t_8267;
	_t_8269 = _t_8268 < 0.0f;
	if(_t_8269)
		{
			float _t_8270;
			float _t_8271;
			float _t_8272;
			float _t_8273;
		
			_t_8270 = tx3_6_1 * ty1_7_1;
			_t_8271 = tx1_4_1 * ty3_9_1;
			_t_8272 = _t_8271 * -1.0f;
			_t_8273 = _t_8270 + _t_8272;
			_t_8274 = _t_8273;
		
		}
else
		{
			float _t_8275;
			float _t_8276;
			float _t_8277;
			float _t_8278;
			float _t_8279;
		
			_t_8275 = tx3_6_1 * ty1_7_1;
			_t_8276 = tx1_4_1 * ty3_9_1;
			_t_8277 = _t_8276 * -1.0f;
			_t_8278 = _t_8275 + _t_8277;
			_t_8279 = -1.0f * _t_8278;
			_t_8274 = _t_8279;
		
		}

	_t_8280 = -1.0f * _t_8274;
	_t_8281 = _t_8280 * _t_343;
	_t_8282 = _t_8281 * -1.0f;
	_t_8283 = _t_8265 + _t_8282;
	_t_8284 = 0.0f < _t_8283;
	_t_8285 = _t_8214 && _t_8284;
	if(_t_8285)
		{
			float _t_8286;
			float _t_8287;
			float _t_8288;
			bool _t_8289;
			float _t_8294;
			float _t_8300;
			float _t_8301;
			float _t_8302;
		
			_t_8286 = -1.0f * ty1_7_1;
			_t_8287 = ty3_9_1 + _t_8286;
			_t_8288 = -1.0f * _t_8287;
			_t_8289 = _t_8288 < 0.0f;
			if(_t_8289)
				{
					float _t_8290;
					float _t_8291;
					float _t_8292;
					float _t_8293;
				
					_t_8290 = tx3_6_1 * ty1_7_1;
					_t_8291 = tx1_4_1 * ty3_9_1;
					_t_8292 = _t_8291 * -1.0f;
					_t_8293 = _t_8290 + _t_8292;
					_t_8294 = _t_8293;
				
				}
		else
				{
					float _t_8295;
					float _t_8296;
					float _t_8297;
					float _t_8298;
					float _t_8299;
				
					_t_8295 = tx3_6_1 * ty1_7_1;
					_t_8296 = tx1_4_1 * ty3_9_1;
					_t_8297 = _t_8296 * -1.0f;
					_t_8298 = _t_8295 + _t_8297;
					_t_8299 = -1.0f * _t_8298;
					_t_8294 = _t_8299;
				
				}
		
			_t_8300 = -1.0f * _t_8294;
			_t_8301 = _t_8300 * _t_343;
			_t_8302 = tegpixellet_block_35(ty3_9_1,ty1_7_1,_t_343,_t_8301,tx1_4_1,tx3_6_1,y__3165_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1);
			_t_8144 = _t_8302;
		
		}
else
		{
		
			_t_8144 = 0.0f;
		
		}


	return _t_8144;
}
__device__ float tegpixelintegrator_25(float ty3_9_1,float pc1_15_1,float _t_8143,float tc2_19_1,float ty2_8_1,float _t_343,float pc0_14_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float py1_13_1,float pc2_16_1,float tx2_5_1,float px1_11_1,float tc0_17_1,float py0_12_1,float _t_8034,float tc1_18_1,float px0_10_1){
    float y__3165_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_8143 - _t_8034)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__3165_1 = _t_8034 + __step__ * (i + (float)(0.5));
        float _t_8144;
		_t_8144 = tegpixelbody_block_25(ty3_9_1,ty1_7_1,_t_343,px0_10_1,px1_11_1,tx1_4_1,tx3_6_1,py0_12_1,py1_13_1,y__3165_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);;
        __output__ = __output__ + _t_8144 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_9(float ty3_9_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float _t_343,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_7926;
	float _t_7927;
	float _t_7928;
	bool _t_7929;
	float _t_7932;
	float _t_7936;
	float _t_7937;
	float _t_7938;
	float _t_7939;
	float _t_7940;
	bool _t_7941;
	float _t_7944;
	float _t_7948;
	float _t_7949;
	bool _t_7950;
	float _t_7951;
	float _t_7952;
	float _t_7953;
	float _t_7954;
	float _t_7955;
	bool _t_7956;
	float _t_7959;
	float _t_7963;
	float _t_7964;
	float _t_7965;
	float _t_7966;
	bool _t_7967;
	float _t_7970;
	float _t_7974;
	float _t_7975;
	float _t_7976;
	float _t_7977;
	float _t_7978;
	bool _t_7979;
	float _t_7982;
	float _t_7986;
	float _t_7987;
	float _t_7988;
	float _t_7989;
	float _t_7990;
	float _t_7991;
	float _t_7992;
	float _t_7993;
	float _t_7994;
	bool _t_7995;
	float _t_7998;
	float _t_8002;
	float _t_8003;
	float _t_8004;
	float _t_8005;
	bool _t_8006;
	float _t_8009;
	float _t_8013;
	float _t_8014;
	float _t_8015;
	float _t_8016;
	float _t_8017;
	bool _t_8018;
	float _t_8021;
	float _t_8025;
	float _t_8026;
	float _t_8027;
	float _t_8028;
	float _t_8029;
	float _t_8030;
	bool _t_8031;
	float _t_8032;
	float _t_8033;
	float _t_8034;
	float _t_8035;
	float _t_8036;
	float _t_8037;
	bool _t_8038;
	float _t_8041;
	float _t_8045;
	float _t_8046;
	float _t_8047;
	float _t_8048;
	float _t_8049;
	bool _t_8050;
	float _t_8053;
	float _t_8057;
	float _t_8058;
	bool _t_8059;
	float _t_8060;
	float _t_8061;
	float _t_8062;
	float _t_8063;
	float _t_8064;
	bool _t_8065;
	float _t_8068;
	float _t_8072;
	float _t_8073;
	float _t_8074;
	float _t_8075;
	bool _t_8076;
	float _t_8079;
	float _t_8083;
	float _t_8084;
	float _t_8085;
	float _t_8086;
	float _t_8087;
	bool _t_8088;
	float _t_8091;
	float _t_8095;
	float _t_8096;
	float _t_8097;
	float _t_8098;
	float _t_8099;
	float _t_8100;
	float _t_8101;
	float _t_8102;
	float _t_8103;
	bool _t_8104;
	float _t_8107;
	float _t_8111;
	float _t_8112;
	float _t_8113;
	float _t_8114;
	bool _t_8115;
	float _t_8118;
	float _t_8122;
	float _t_8123;
	float _t_8124;
	float _t_8125;
	float _t_8126;
	bool _t_8127;
	float _t_8130;
	float _t_8134;
	float _t_8135;
	float _t_8136;
	float _t_8137;
	float _t_8138;
	float _t_8139;
	bool _t_8140;
	float _t_8141;
	float _t_8142;
	float _t_8143;

	float _t_344;

	_t_7926 = -1.0f * ty1_7_1;
	_t_7927 = ty3_9_1 + _t_7926;
	_t_7928 = -1.0f * _t_7927;
	_t_7929 = _t_7928 < 0.0f;
	if(_t_7929)
		{
			float _t_7930;
			float _t_7931;
		
			_t_7930 = -1.0f * tx3_6_1;
			_t_7931 = tx1_4_1 + _t_7930;
			_t_7932 = _t_7931;
		
		}
else
		{
			float _t_7933;
			float _t_7934;
			float _t_7935;
		
			_t_7933 = -1.0f * tx3_6_1;
			_t_7934 = tx1_4_1 + _t_7933;
			_t_7935 = -1.0f * _t_7934;
			_t_7932 = _t_7935;
		
		}

	_t_7936 = _t_7932 * _t_343;
	_t_7937 = _t_7936 * -1.0f;
	_t_7938 = -1.0f * ty1_7_1;
	_t_7939 = ty3_9_1 + _t_7938;
	_t_7940 = -1.0f * _t_7939;
	_t_7941 = _t_7940 < 0.0f;
	if(_t_7941)
		{
			float _t_7942;
			float _t_7943;
		
			_t_7942 = -1.0f * tx3_6_1;
			_t_7943 = tx1_4_1 + _t_7942;
			_t_7944 = _t_7943;
		
		}
else
		{
			float _t_7945;
			float _t_7946;
			float _t_7947;
		
			_t_7945 = -1.0f * tx3_6_1;
			_t_7946 = tx1_4_1 + _t_7945;
			_t_7947 = -1.0f * _t_7946;
			_t_7944 = _t_7947;
		
		}

	_t_7948 = _t_7944 * _t_343;
	_t_7949 = _t_7948 * -1.0f;
	_t_7950 = 0.0f < _t_7949;
	if(_t_7950)
		{
		
			_t_7951 = px0_10_1;
		
		}
else
		{
		
			_t_7951 = px1_11_1;
		
		}

	_t_7952 = _t_7937 * _t_7951;
	_t_7953 = -1.0f * ty1_7_1;
	_t_7954 = ty3_9_1 + _t_7953;
	_t_7955 = -1.0f * _t_7954;
	_t_7956 = _t_7955 < 0.0f;
	if(_t_7956)
		{
			float _t_7957;
			float _t_7958;
		
			_t_7957 = -1.0f * tx3_6_1;
			_t_7958 = tx1_4_1 + _t_7957;
			_t_7959 = _t_7958;
		
		}
else
		{
			float _t_7960;
			float _t_7961;
			float _t_7962;
		
			_t_7960 = -1.0f * tx3_6_1;
			_t_7961 = tx1_4_1 + _t_7960;
			_t_7962 = -1.0f * _t_7961;
			_t_7959 = _t_7962;
		
		}

	_t_7963 = _t_7959 * _t_343;
	_t_7964 = -1.0f * ty1_7_1;
	_t_7965 = ty3_9_1 + _t_7964;
	_t_7966 = -1.0f * _t_7965;
	_t_7967 = _t_7966 < 0.0f;
	if(_t_7967)
		{
			float _t_7968;
			float _t_7969;
		
			_t_7968 = -1.0f * tx3_6_1;
			_t_7969 = tx1_4_1 + _t_7968;
			_t_7970 = _t_7969;
		
		}
else
		{
			float _t_7971;
			float _t_7972;
			float _t_7973;
		
			_t_7971 = -1.0f * tx3_6_1;
			_t_7972 = tx1_4_1 + _t_7971;
			_t_7973 = -1.0f * _t_7972;
			_t_7970 = _t_7973;
		
		}

	_t_7974 = _t_7970 * _t_343;
	_t_7975 = _t_7963 * _t_7974;
	_t_7976 = -1.0f * ty1_7_1;
	_t_7977 = ty3_9_1 + _t_7976;
	_t_7978 = -1.0f * _t_7977;
	_t_7979 = _t_7978 < 0.0f;
	if(_t_7979)
		{
			float _t_7980;
			float _t_7981;
		
			_t_7980 = -1.0f * ty1_7_1;
			_t_7981 = ty3_9_1 + _t_7980;
			_t_7982 = _t_7981;
		
		}
else
		{
			float _t_7983;
			float _t_7984;
			float _t_7985;
		
			_t_7983 = -1.0f * ty1_7_1;
			_t_7984 = ty3_9_1 + _t_7983;
			_t_7985 = -1.0f * _t_7984;
			_t_7982 = _t_7985;
		
		}

	_t_7986 = _t_7982 * _t_343;
	_t_7987 = 1.0f + _t_7986;
	_t_7988 = 1.0f / _t_7987;
	_t_7989 = _t_7975 * _t_7988;
	_t_7990 = _t_7989 * -1.0f;
	_t_7991 = 1.0f + _t_7990;
	_t_7992 = -1.0f * ty1_7_1;
	_t_7993 = ty3_9_1 + _t_7992;
	_t_7994 = -1.0f * _t_7993;
	_t_7995 = _t_7994 < 0.0f;
	if(_t_7995)
		{
			float _t_7996;
			float _t_7997;
		
			_t_7996 = -1.0f * tx3_6_1;
			_t_7997 = tx1_4_1 + _t_7996;
			_t_7998 = _t_7997;
		
		}
else
		{
			float _t_7999;
			float _t_8000;
			float _t_8001;
		
			_t_7999 = -1.0f * tx3_6_1;
			_t_8000 = tx1_4_1 + _t_7999;
			_t_8001 = -1.0f * _t_8000;
			_t_7998 = _t_8001;
		
		}

	_t_8002 = _t_7998 * _t_343;
	_t_8003 = -1.0f * ty1_7_1;
	_t_8004 = ty3_9_1 + _t_8003;
	_t_8005 = -1.0f * _t_8004;
	_t_8006 = _t_8005 < 0.0f;
	if(_t_8006)
		{
			float _t_8007;
			float _t_8008;
		
			_t_8007 = -1.0f * tx3_6_1;
			_t_8008 = tx1_4_1 + _t_8007;
			_t_8009 = _t_8008;
		
		}
else
		{
			float _t_8010;
			float _t_8011;
			float _t_8012;
		
			_t_8010 = -1.0f * tx3_6_1;
			_t_8011 = tx1_4_1 + _t_8010;
			_t_8012 = -1.0f * _t_8011;
			_t_8009 = _t_8012;
		
		}

	_t_8013 = _t_8009 * _t_343;
	_t_8014 = _t_8002 * _t_8013;
	_t_8015 = -1.0f * ty1_7_1;
	_t_8016 = ty3_9_1 + _t_8015;
	_t_8017 = -1.0f * _t_8016;
	_t_8018 = _t_8017 < 0.0f;
	if(_t_8018)
		{
			float _t_8019;
			float _t_8020;
		
			_t_8019 = -1.0f * ty1_7_1;
			_t_8020 = ty3_9_1 + _t_8019;
			_t_8021 = _t_8020;
		
		}
else
		{
			float _t_8022;
			float _t_8023;
			float _t_8024;
		
			_t_8022 = -1.0f * ty1_7_1;
			_t_8023 = ty3_9_1 + _t_8022;
			_t_8024 = -1.0f * _t_8023;
			_t_8021 = _t_8024;
		
		}

	_t_8025 = _t_8021 * _t_343;
	_t_8026 = 1.0f + _t_8025;
	_t_8027 = 1.0f / _t_8026;
	_t_8028 = _t_8014 * _t_8027;
	_t_8029 = _t_8028 * -1.0f;
	_t_8030 = 1.0f + _t_8029;
	_t_8031 = 0.0f < _t_8030;
	if(_t_8031)
		{
		
			_t_8032 = py0_12_1;
		
		}
else
		{
		
			_t_8032 = py1_13_1;
		
		}

	_t_8033 = _t_7991 * _t_8032;
	_t_8034 = _t_7952 + _t_8033;
	_t_8035 = -1.0f * ty1_7_1;
	_t_8036 = ty3_9_1 + _t_8035;
	_t_8037 = -1.0f * _t_8036;
	_t_8038 = _t_8037 < 0.0f;
	if(_t_8038)
		{
			float _t_8039;
			float _t_8040;
		
			_t_8039 = -1.0f * tx3_6_1;
			_t_8040 = tx1_4_1 + _t_8039;
			_t_8041 = _t_8040;
		
		}
else
		{
			float _t_8042;
			float _t_8043;
			float _t_8044;
		
			_t_8042 = -1.0f * tx3_6_1;
			_t_8043 = tx1_4_1 + _t_8042;
			_t_8044 = -1.0f * _t_8043;
			_t_8041 = _t_8044;
		
		}

	_t_8045 = _t_8041 * _t_343;
	_t_8046 = _t_8045 * -1.0f;
	_t_8047 = -1.0f * ty1_7_1;
	_t_8048 = ty3_9_1 + _t_8047;
	_t_8049 = -1.0f * _t_8048;
	_t_8050 = _t_8049 < 0.0f;
	if(_t_8050)
		{
			float _t_8051;
			float _t_8052;
		
			_t_8051 = -1.0f * tx3_6_1;
			_t_8052 = tx1_4_1 + _t_8051;
			_t_8053 = _t_8052;
		
		}
else
		{
			float _t_8054;
			float _t_8055;
			float _t_8056;
		
			_t_8054 = -1.0f * tx3_6_1;
			_t_8055 = tx1_4_1 + _t_8054;
			_t_8056 = -1.0f * _t_8055;
			_t_8053 = _t_8056;
		
		}

	_t_8057 = _t_8053 * _t_343;
	_t_8058 = _t_8057 * -1.0f;
	_t_8059 = 0.0f < _t_8058;
	if(_t_8059)
		{
		
			_t_8060 = px1_11_1;
		
		}
else
		{
		
			_t_8060 = px0_10_1;
		
		}

	_t_8061 = _t_8046 * _t_8060;
	_t_8062 = -1.0f * ty1_7_1;
	_t_8063 = ty3_9_1 + _t_8062;
	_t_8064 = -1.0f * _t_8063;
	_t_8065 = _t_8064 < 0.0f;
	if(_t_8065)
		{
			float _t_8066;
			float _t_8067;
		
			_t_8066 = -1.0f * tx3_6_1;
			_t_8067 = tx1_4_1 + _t_8066;
			_t_8068 = _t_8067;
		
		}
else
		{
			float _t_8069;
			float _t_8070;
			float _t_8071;
		
			_t_8069 = -1.0f * tx3_6_1;
			_t_8070 = tx1_4_1 + _t_8069;
			_t_8071 = -1.0f * _t_8070;
			_t_8068 = _t_8071;
		
		}

	_t_8072 = _t_8068 * _t_343;
	_t_8073 = -1.0f * ty1_7_1;
	_t_8074 = ty3_9_1 + _t_8073;
	_t_8075 = -1.0f * _t_8074;
	_t_8076 = _t_8075 < 0.0f;
	if(_t_8076)
		{
			float _t_8077;
			float _t_8078;
		
			_t_8077 = -1.0f * tx3_6_1;
			_t_8078 = tx1_4_1 + _t_8077;
			_t_8079 = _t_8078;
		
		}
else
		{
			float _t_8080;
			float _t_8081;
			float _t_8082;
		
			_t_8080 = -1.0f * tx3_6_1;
			_t_8081 = tx1_4_1 + _t_8080;
			_t_8082 = -1.0f * _t_8081;
			_t_8079 = _t_8082;
		
		}

	_t_8083 = _t_8079 * _t_343;
	_t_8084 = _t_8072 * _t_8083;
	_t_8085 = -1.0f * ty1_7_1;
	_t_8086 = ty3_9_1 + _t_8085;
	_t_8087 = -1.0f * _t_8086;
	_t_8088 = _t_8087 < 0.0f;
	if(_t_8088)
		{
			float _t_8089;
			float _t_8090;
		
			_t_8089 = -1.0f * ty1_7_1;
			_t_8090 = ty3_9_1 + _t_8089;
			_t_8091 = _t_8090;
		
		}
else
		{
			float _t_8092;
			float _t_8093;
			float _t_8094;
		
			_t_8092 = -1.0f * ty1_7_1;
			_t_8093 = ty3_9_1 + _t_8092;
			_t_8094 = -1.0f * _t_8093;
			_t_8091 = _t_8094;
		
		}

	_t_8095 = _t_8091 * _t_343;
	_t_8096 = 1.0f + _t_8095;
	_t_8097 = 1.0f / _t_8096;
	_t_8098 = _t_8084 * _t_8097;
	_t_8099 = _t_8098 * -1.0f;
	_t_8100 = 1.0f + _t_8099;
	_t_8101 = -1.0f * ty1_7_1;
	_t_8102 = ty3_9_1 + _t_8101;
	_t_8103 = -1.0f * _t_8102;
	_t_8104 = _t_8103 < 0.0f;
	if(_t_8104)
		{
			float _t_8105;
			float _t_8106;
		
			_t_8105 = -1.0f * tx3_6_1;
			_t_8106 = tx1_4_1 + _t_8105;
			_t_8107 = _t_8106;
		
		}
else
		{
			float _t_8108;
			float _t_8109;
			float _t_8110;
		
			_t_8108 = -1.0f * tx3_6_1;
			_t_8109 = tx1_4_1 + _t_8108;
			_t_8110 = -1.0f * _t_8109;
			_t_8107 = _t_8110;
		
		}

	_t_8111 = _t_8107 * _t_343;
	_t_8112 = -1.0f * ty1_7_1;
	_t_8113 = ty3_9_1 + _t_8112;
	_t_8114 = -1.0f * _t_8113;
	_t_8115 = _t_8114 < 0.0f;
	if(_t_8115)
		{
			float _t_8116;
			float _t_8117;
		
			_t_8116 = -1.0f * tx3_6_1;
			_t_8117 = tx1_4_1 + _t_8116;
			_t_8118 = _t_8117;
		
		}
else
		{
			float _t_8119;
			float _t_8120;
			float _t_8121;
		
			_t_8119 = -1.0f * tx3_6_1;
			_t_8120 = tx1_4_1 + _t_8119;
			_t_8121 = -1.0f * _t_8120;
			_t_8118 = _t_8121;
		
		}

	_t_8122 = _t_8118 * _t_343;
	_t_8123 = _t_8111 * _t_8122;
	_t_8124 = -1.0f * ty1_7_1;
	_t_8125 = ty3_9_1 + _t_8124;
	_t_8126 = -1.0f * _t_8125;
	_t_8127 = _t_8126 < 0.0f;
	if(_t_8127)
		{
			float _t_8128;
			float _t_8129;
		
			_t_8128 = -1.0f * ty1_7_1;
			_t_8129 = ty3_9_1 + _t_8128;
			_t_8130 = _t_8129;
		
		}
else
		{
			float _t_8131;
			float _t_8132;
			float _t_8133;
		
			_t_8131 = -1.0f * ty1_7_1;
			_t_8132 = ty3_9_1 + _t_8131;
			_t_8133 = -1.0f * _t_8132;
			_t_8130 = _t_8133;
		
		}

	_t_8134 = _t_8130 * _t_343;
	_t_8135 = 1.0f + _t_8134;
	_t_8136 = 1.0f / _t_8135;
	_t_8137 = _t_8123 * _t_8136;
	_t_8138 = _t_8137 * -1.0f;
	_t_8139 = 1.0f + _t_8138;
	_t_8140 = 0.0f < _t_8139;
	if(_t_8140)
		{
		
			_t_8141 = py1_13_1;
		
		}
else
		{
		
			_t_8141 = py0_12_1;
		
		}

	_t_8142 = _t_8100 * _t_8141;
	_t_8143 = _t_8061 + _t_8142;
	_t_344 = tegpixelintegrator_25(ty3_9_1,pc1_15_1,_t_8143,tc2_19_1,ty2_8_1,_t_343,pc0_14_1,ty1_7_1,tx1_4_1,tx3_6_1,py1_13_1,pc2_16_1,tx2_5_1,px1_11_1,tc0_17_1,py0_12_1,_t_8034,tc1_18_1,px0_10_1);

	return _t_344;
}
__device__ float tegpixellet_block_38(float py0_12_1,float _t_9216,float py1_13_1,float px0_10_1,float _t_9163,float px1_11_1,float ty1_7_1,float ty2_8_1,float tx2_5_1,float tx1_4_1,float _t_371,float y__3239_1,float _t_9136,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	bool _t_9217;
	bool _t_9218;
	bool _t_9219;
	bool _t_9220;
	bool _t_9221;
	bool _t_9222;
	bool _t_9223;
	float _t_9553;
	float _t_9554;
	float _t_9555;
	float _t_9556;
	float _t_9557;
	float _t_9558;
	float _t_9559;
	float _t_9560;
	float _t_9561;
	float _t_9562;
	float _t_9563;
	float _t_9564;
	float _t_9565;
	float _t_9566;
	float _t_9567;
	float _t_9568;
	float _t_9569;
	float _t_9570;
	float _t_9571;
	float _t_9572;
	float _t_9573;
	float _t_9574;
	float _t_9575;
	float _t_9576;
	bool _t_9577;
	float _t_9578;
	float _t_9579;
	float _t_9580;
	float _t_9581;
	float _t_9582;
	float _t_9583;
	float _t_9584;
	float _t_9585;
	float _t_9586;
	float _t_9587;
	float _t_9588;
	float _t_9589;
	float _t_9590;
	float _t_9591;
	bool _t_9592;
	float _t_9593;
	float _t_9594;
	float _t_9595;
	float _t_9596;
	float _t_9597;
	float _t_9598;
	float _t_9599;
	float _t_9600;
	float _t_9601;
	float _t_9602;
	float _t_9603;
	float _t_9604;
	float _t_9605;
	float _t_9606;
	float _t_9607;
	float _t_9608;
	float _t_9609;
	float _t_9610;
	float _t_9611;
	float _t_9612;
	float _t_9613;
	float _t_9614;
	float _t_9615;
	float _t_9616;
	float _t_9617;
	float _t_9618;
	bool _t_9619;
	float _t_9620;
	float _t_9621;
	float _t_9622;
	float _t_9623;
	float _t_9624;
	float _t_9625;
	float _t_9626;
	float _t_9627;
	float _t_9628;
	float _t_9629;
	float _t_9630;
	float _t_9631;
	float _t_9632;
	float _t_9633;
	bool _t_9634;
	float _t_9635;
	float _t_9636;
	float _t_9637;
	float _t_9638;
	float _t_9639;

	float _t_9137;

	_t_9217 = py0_12_1 < _t_9216;
	_t_9218 = _t_9216 < py1_13_1;
	_t_9219 = _t_9217 && _t_9218;
	_t_9220 = px0_10_1 < _t_9163;
	_t_9221 = _t_9163 < px1_11_1;
	_t_9222 = _t_9220 && _t_9221;
	_t_9223 = _t_9219 && _t_9222;
	if(_t_9223)
		{
			float _t_9224;
			float _t_9225;
			float _t_9226;
			bool _t_9227;
			float _t_9230;
			float _t_9234;
			float _t_9235;
			float _t_9236;
			float _t_9237;
			float _t_9238;
			bool _t_9239;
			float _t_9242;
			float _t_9246;
			float _t_9247;
			bool _t_9248;
			float _t_9249;
			float _t_9250;
			float _t_9251;
			float _t_9252;
			float _t_9253;
			bool _t_9254;
			float _t_9257;
			float _t_9261;
			float _t_9262;
			float _t_9263;
			float _t_9264;
			bool _t_9265;
			float _t_9268;
			float _t_9272;
			float _t_9273;
			float _t_9274;
			float _t_9275;
			float _t_9276;
			bool _t_9277;
			float _t_9280;
			float _t_9284;
			float _t_9285;
			float _t_9286;
			float _t_9287;
			float _t_9288;
			float _t_9289;
			float _t_9290;
			float _t_9291;
			float _t_9292;
			bool _t_9293;
			float _t_9296;
			float _t_9300;
			float _t_9301;
			float _t_9302;
			float _t_9303;
			bool _t_9304;
			float _t_9307;
			float _t_9311;
			float _t_9312;
			float _t_9313;
			float _t_9314;
			float _t_9315;
			bool _t_9316;
			float _t_9319;
			float _t_9323;
			float _t_9324;
			float _t_9325;
			float _t_9326;
			float _t_9327;
			float _t_9328;
			bool _t_9329;
			float _t_9330;
			float _t_9331;
			float _t_9332;
			bool _t_9333;
			float _t_9334;
			float _t_9335;
			float _t_9336;
			bool _t_9337;
			float _t_9340;
			float _t_9344;
			float _t_9345;
			float _t_9346;
			float _t_9347;
			float _t_9348;
			bool _t_9349;
			float _t_9352;
			float _t_9356;
			float _t_9357;
			bool _t_9358;
			float _t_9359;
			float _t_9360;
			float _t_9361;
			float _t_9362;
			float _t_9363;
			bool _t_9364;
			float _t_9367;
			float _t_9371;
			float _t_9372;
			float _t_9373;
			float _t_9374;
			bool _t_9375;
			float _t_9378;
			float _t_9382;
			float _t_9383;
			float _t_9384;
			float _t_9385;
			float _t_9386;
			bool _t_9387;
			float _t_9390;
			float _t_9394;
			float _t_9395;
			float _t_9396;
			float _t_9397;
			float _t_9398;
			float _t_9399;
			float _t_9400;
			float _t_9401;
			float _t_9402;
			bool _t_9403;
			float _t_9406;
			float _t_9410;
			float _t_9411;
			float _t_9412;
			float _t_9413;
			bool _t_9414;
			float _t_9417;
			float _t_9421;
			float _t_9422;
			float _t_9423;
			float _t_9424;
			float _t_9425;
			bool _t_9426;
			float _t_9429;
			float _t_9433;
			float _t_9434;
			float _t_9435;
			float _t_9436;
			float _t_9437;
			float _t_9438;
			bool _t_9439;
			float _t_9440;
			float _t_9441;
			float _t_9442;
			bool _t_9443;
			bool _t_9444;
			float _t_9445;
			float _t_9446;
			float _t_9447;
			bool _t_9448;
			float _t_9451;
			float _t_9455;
			float _t_9456;
			float _t_9457;
			float _t_9458;
			bool _t_9459;
			float _t_9462;
			float _t_9466;
			bool _t_9467;
			float _t_9468;
			float _t_9469;
			float _t_9470;
			float _t_9471;
			float _t_9472;
			bool _t_9473;
			float _t_9476;
			float _t_9480;
			float _t_9481;
			float _t_9482;
			float _t_9483;
			bool _t_9484;
			float _t_9487;
			float _t_9491;
			bool _t_9492;
			float _t_9493;
			float _t_9494;
			float _t_9495;
			bool _t_9496;
			float _t_9497;
			float _t_9498;
			float _t_9499;
			bool _t_9500;
			float _t_9503;
			float _t_9507;
			float _t_9508;
			float _t_9509;
			float _t_9510;
			bool _t_9511;
			float _t_9514;
			float _t_9518;
			bool _t_9519;
			float _t_9520;
			float _t_9521;
			float _t_9522;
			float _t_9523;
			float _t_9524;
			bool _t_9525;
			float _t_9528;
			float _t_9532;
			float _t_9533;
			float _t_9534;
			float _t_9535;
			bool _t_9536;
			float _t_9539;
			float _t_9543;
			bool _t_9544;
			float _t_9545;
			float _t_9546;
			float _t_9547;
			bool _t_9548;
			bool _t_9549;
			bool _t_9550;
			float _t_9551;
			float _t_9552;
		
			_t_9224 = -1.0f * ty2_8_1;
			_t_9225 = ty1_7_1 + _t_9224;
			_t_9226 = -1.0f * _t_9225;
			_t_9227 = _t_9226 < 0.0f;
			if(_t_9227)
				{
					float _t_9228;
					float _t_9229;
				
					_t_9228 = -1.0f * tx1_4_1;
					_t_9229 = tx2_5_1 + _t_9228;
					_t_9230 = _t_9229;
				
				}
		else
				{
					float _t_9231;
					float _t_9232;
					float _t_9233;
				
					_t_9231 = -1.0f * tx1_4_1;
					_t_9232 = tx2_5_1 + _t_9231;
					_t_9233 = -1.0f * _t_9232;
					_t_9230 = _t_9233;
				
				}
		
			_t_9234 = _t_9230 * _t_371;
			_t_9235 = _t_9234 * -1.0f;
			_t_9236 = -1.0f * ty2_8_1;
			_t_9237 = ty1_7_1 + _t_9236;
			_t_9238 = -1.0f * _t_9237;
			_t_9239 = _t_9238 < 0.0f;
			if(_t_9239)
				{
					float _t_9240;
					float _t_9241;
				
					_t_9240 = -1.0f * tx1_4_1;
					_t_9241 = tx2_5_1 + _t_9240;
					_t_9242 = _t_9241;
				
				}
		else
				{
					float _t_9243;
					float _t_9244;
					float _t_9245;
				
					_t_9243 = -1.0f * tx1_4_1;
					_t_9244 = tx2_5_1 + _t_9243;
					_t_9245 = -1.0f * _t_9244;
					_t_9242 = _t_9245;
				
				}
		
			_t_9246 = _t_9242 * _t_371;
			_t_9247 = _t_9246 * -1.0f;
			_t_9248 = 0.0f < _t_9247;
			if(_t_9248)
				{
				
					_t_9249 = px0_10_1;
				
				}
		else
				{
				
					_t_9249 = px1_11_1;
				
				}
		
			_t_9250 = _t_9235 * _t_9249;
			_t_9251 = -1.0f * ty2_8_1;
			_t_9252 = ty1_7_1 + _t_9251;
			_t_9253 = -1.0f * _t_9252;
			_t_9254 = _t_9253 < 0.0f;
			if(_t_9254)
				{
					float _t_9255;
					float _t_9256;
				
					_t_9255 = -1.0f * tx1_4_1;
					_t_9256 = tx2_5_1 + _t_9255;
					_t_9257 = _t_9256;
				
				}
		else
				{
					float _t_9258;
					float _t_9259;
					float _t_9260;
				
					_t_9258 = -1.0f * tx1_4_1;
					_t_9259 = tx2_5_1 + _t_9258;
					_t_9260 = -1.0f * _t_9259;
					_t_9257 = _t_9260;
				
				}
		
			_t_9261 = _t_9257 * _t_371;
			_t_9262 = -1.0f * ty2_8_1;
			_t_9263 = ty1_7_1 + _t_9262;
			_t_9264 = -1.0f * _t_9263;
			_t_9265 = _t_9264 < 0.0f;
			if(_t_9265)
				{
					float _t_9266;
					float _t_9267;
				
					_t_9266 = -1.0f * tx1_4_1;
					_t_9267 = tx2_5_1 + _t_9266;
					_t_9268 = _t_9267;
				
				}
		else
				{
					float _t_9269;
					float _t_9270;
					float _t_9271;
				
					_t_9269 = -1.0f * tx1_4_1;
					_t_9270 = tx2_5_1 + _t_9269;
					_t_9271 = -1.0f * _t_9270;
					_t_9268 = _t_9271;
				
				}
		
			_t_9272 = _t_9268 * _t_371;
			_t_9273 = _t_9261 * _t_9272;
			_t_9274 = -1.0f * ty2_8_1;
			_t_9275 = ty1_7_1 + _t_9274;
			_t_9276 = -1.0f * _t_9275;
			_t_9277 = _t_9276 < 0.0f;
			if(_t_9277)
				{
					float _t_9278;
					float _t_9279;
				
					_t_9278 = -1.0f * ty2_8_1;
					_t_9279 = ty1_7_1 + _t_9278;
					_t_9280 = _t_9279;
				
				}
		else
				{
					float _t_9281;
					float _t_9282;
					float _t_9283;
				
					_t_9281 = -1.0f * ty2_8_1;
					_t_9282 = ty1_7_1 + _t_9281;
					_t_9283 = -1.0f * _t_9282;
					_t_9280 = _t_9283;
				
				}
		
			_t_9284 = _t_9280 * _t_371;
			_t_9285 = 1.0f + _t_9284;
			_t_9286 = 1.0f / _t_9285;
			_t_9287 = _t_9273 * _t_9286;
			_t_9288 = _t_9287 * -1.0f;
			_t_9289 = 1.0f + _t_9288;
			_t_9290 = -1.0f * ty2_8_1;
			_t_9291 = ty1_7_1 + _t_9290;
			_t_9292 = -1.0f * _t_9291;
			_t_9293 = _t_9292 < 0.0f;
			if(_t_9293)
				{
					float _t_9294;
					float _t_9295;
				
					_t_9294 = -1.0f * tx1_4_1;
					_t_9295 = tx2_5_1 + _t_9294;
					_t_9296 = _t_9295;
				
				}
		else
				{
					float _t_9297;
					float _t_9298;
					float _t_9299;
				
					_t_9297 = -1.0f * tx1_4_1;
					_t_9298 = tx2_5_1 + _t_9297;
					_t_9299 = -1.0f * _t_9298;
					_t_9296 = _t_9299;
				
				}
		
			_t_9300 = _t_9296 * _t_371;
			_t_9301 = -1.0f * ty2_8_1;
			_t_9302 = ty1_7_1 + _t_9301;
			_t_9303 = -1.0f * _t_9302;
			_t_9304 = _t_9303 < 0.0f;
			if(_t_9304)
				{
					float _t_9305;
					float _t_9306;
				
					_t_9305 = -1.0f * tx1_4_1;
					_t_9306 = tx2_5_1 + _t_9305;
					_t_9307 = _t_9306;
				
				}
		else
				{
					float _t_9308;
					float _t_9309;
					float _t_9310;
				
					_t_9308 = -1.0f * tx1_4_1;
					_t_9309 = tx2_5_1 + _t_9308;
					_t_9310 = -1.0f * _t_9309;
					_t_9307 = _t_9310;
				
				}
		
			_t_9311 = _t_9307 * _t_371;
			_t_9312 = _t_9300 * _t_9311;
			_t_9313 = -1.0f * ty2_8_1;
			_t_9314 = ty1_7_1 + _t_9313;
			_t_9315 = -1.0f * _t_9314;
			_t_9316 = _t_9315 < 0.0f;
			if(_t_9316)
				{
					float _t_9317;
					float _t_9318;
				
					_t_9317 = -1.0f * ty2_8_1;
					_t_9318 = ty1_7_1 + _t_9317;
					_t_9319 = _t_9318;
				
				}
		else
				{
					float _t_9320;
					float _t_9321;
					float _t_9322;
				
					_t_9320 = -1.0f * ty2_8_1;
					_t_9321 = ty1_7_1 + _t_9320;
					_t_9322 = -1.0f * _t_9321;
					_t_9319 = _t_9322;
				
				}
		
			_t_9323 = _t_9319 * _t_371;
			_t_9324 = 1.0f + _t_9323;
			_t_9325 = 1.0f / _t_9324;
			_t_9326 = _t_9312 * _t_9325;
			_t_9327 = _t_9326 * -1.0f;
			_t_9328 = 1.0f + _t_9327;
			_t_9329 = 0.0f < _t_9328;
			if(_t_9329)
				{
				
					_t_9330 = py0_12_1;
				
				}
		else
				{
				
					_t_9330 = py1_13_1;
				
				}
		
			_t_9331 = _t_9289 * _t_9330;
			_t_9332 = _t_9250 + _t_9331;
			_t_9333 = _t_9332 < y__3239_1;
			_t_9334 = -1.0f * ty2_8_1;
			_t_9335 = ty1_7_1 + _t_9334;
			_t_9336 = -1.0f * _t_9335;
			_t_9337 = _t_9336 < 0.0f;
			if(_t_9337)
				{
					float _t_9338;
					float _t_9339;
				
					_t_9338 = -1.0f * tx1_4_1;
					_t_9339 = tx2_5_1 + _t_9338;
					_t_9340 = _t_9339;
				
				}
		else
				{
					float _t_9341;
					float _t_9342;
					float _t_9343;
				
					_t_9341 = -1.0f * tx1_4_1;
					_t_9342 = tx2_5_1 + _t_9341;
					_t_9343 = -1.0f * _t_9342;
					_t_9340 = _t_9343;
				
				}
		
			_t_9344 = _t_9340 * _t_371;
			_t_9345 = _t_9344 * -1.0f;
			_t_9346 = -1.0f * ty2_8_1;
			_t_9347 = ty1_7_1 + _t_9346;
			_t_9348 = -1.0f * _t_9347;
			_t_9349 = _t_9348 < 0.0f;
			if(_t_9349)
				{
					float _t_9350;
					float _t_9351;
				
					_t_9350 = -1.0f * tx1_4_1;
					_t_9351 = tx2_5_1 + _t_9350;
					_t_9352 = _t_9351;
				
				}
		else
				{
					float _t_9353;
					float _t_9354;
					float _t_9355;
				
					_t_9353 = -1.0f * tx1_4_1;
					_t_9354 = tx2_5_1 + _t_9353;
					_t_9355 = -1.0f * _t_9354;
					_t_9352 = _t_9355;
				
				}
		
			_t_9356 = _t_9352 * _t_371;
			_t_9357 = _t_9356 * -1.0f;
			_t_9358 = 0.0f < _t_9357;
			if(_t_9358)
				{
				
					_t_9359 = px1_11_1;
				
				}
		else
				{
				
					_t_9359 = px0_10_1;
				
				}
		
			_t_9360 = _t_9345 * _t_9359;
			_t_9361 = -1.0f * ty2_8_1;
			_t_9362 = ty1_7_1 + _t_9361;
			_t_9363 = -1.0f * _t_9362;
			_t_9364 = _t_9363 < 0.0f;
			if(_t_9364)
				{
					float _t_9365;
					float _t_9366;
				
					_t_9365 = -1.0f * tx1_4_1;
					_t_9366 = tx2_5_1 + _t_9365;
					_t_9367 = _t_9366;
				
				}
		else
				{
					float _t_9368;
					float _t_9369;
					float _t_9370;
				
					_t_9368 = -1.0f * tx1_4_1;
					_t_9369 = tx2_5_1 + _t_9368;
					_t_9370 = -1.0f * _t_9369;
					_t_9367 = _t_9370;
				
				}
		
			_t_9371 = _t_9367 * _t_371;
			_t_9372 = -1.0f * ty2_8_1;
			_t_9373 = ty1_7_1 + _t_9372;
			_t_9374 = -1.0f * _t_9373;
			_t_9375 = _t_9374 < 0.0f;
			if(_t_9375)
				{
					float _t_9376;
					float _t_9377;
				
					_t_9376 = -1.0f * tx1_4_1;
					_t_9377 = tx2_5_1 + _t_9376;
					_t_9378 = _t_9377;
				
				}
		else
				{
					float _t_9379;
					float _t_9380;
					float _t_9381;
				
					_t_9379 = -1.0f * tx1_4_1;
					_t_9380 = tx2_5_1 + _t_9379;
					_t_9381 = -1.0f * _t_9380;
					_t_9378 = _t_9381;
				
				}
		
			_t_9382 = _t_9378 * _t_371;
			_t_9383 = _t_9371 * _t_9382;
			_t_9384 = -1.0f * ty2_8_1;
			_t_9385 = ty1_7_1 + _t_9384;
			_t_9386 = -1.0f * _t_9385;
			_t_9387 = _t_9386 < 0.0f;
			if(_t_9387)
				{
					float _t_9388;
					float _t_9389;
				
					_t_9388 = -1.0f * ty2_8_1;
					_t_9389 = ty1_7_1 + _t_9388;
					_t_9390 = _t_9389;
				
				}
		else
				{
					float _t_9391;
					float _t_9392;
					float _t_9393;
				
					_t_9391 = -1.0f * ty2_8_1;
					_t_9392 = ty1_7_1 + _t_9391;
					_t_9393 = -1.0f * _t_9392;
					_t_9390 = _t_9393;
				
				}
		
			_t_9394 = _t_9390 * _t_371;
			_t_9395 = 1.0f + _t_9394;
			_t_9396 = 1.0f / _t_9395;
			_t_9397 = _t_9383 * _t_9396;
			_t_9398 = _t_9397 * -1.0f;
			_t_9399 = 1.0f + _t_9398;
			_t_9400 = -1.0f * ty2_8_1;
			_t_9401 = ty1_7_1 + _t_9400;
			_t_9402 = -1.0f * _t_9401;
			_t_9403 = _t_9402 < 0.0f;
			if(_t_9403)
				{
					float _t_9404;
					float _t_9405;
				
					_t_9404 = -1.0f * tx1_4_1;
					_t_9405 = tx2_5_1 + _t_9404;
					_t_9406 = _t_9405;
				
				}
		else
				{
					float _t_9407;
					float _t_9408;
					float _t_9409;
				
					_t_9407 = -1.0f * tx1_4_1;
					_t_9408 = tx2_5_1 + _t_9407;
					_t_9409 = -1.0f * _t_9408;
					_t_9406 = _t_9409;
				
				}
		
			_t_9410 = _t_9406 * _t_371;
			_t_9411 = -1.0f * ty2_8_1;
			_t_9412 = ty1_7_1 + _t_9411;
			_t_9413 = -1.0f * _t_9412;
			_t_9414 = _t_9413 < 0.0f;
			if(_t_9414)
				{
					float _t_9415;
					float _t_9416;
				
					_t_9415 = -1.0f * tx1_4_1;
					_t_9416 = tx2_5_1 + _t_9415;
					_t_9417 = _t_9416;
				
				}
		else
				{
					float _t_9418;
					float _t_9419;
					float _t_9420;
				
					_t_9418 = -1.0f * tx1_4_1;
					_t_9419 = tx2_5_1 + _t_9418;
					_t_9420 = -1.0f * _t_9419;
					_t_9417 = _t_9420;
				
				}
		
			_t_9421 = _t_9417 * _t_371;
			_t_9422 = _t_9410 * _t_9421;
			_t_9423 = -1.0f * ty2_8_1;
			_t_9424 = ty1_7_1 + _t_9423;
			_t_9425 = -1.0f * _t_9424;
			_t_9426 = _t_9425 < 0.0f;
			if(_t_9426)
				{
					float _t_9427;
					float _t_9428;
				
					_t_9427 = -1.0f * ty2_8_1;
					_t_9428 = ty1_7_1 + _t_9427;
					_t_9429 = _t_9428;
				
				}
		else
				{
					float _t_9430;
					float _t_9431;
					float _t_9432;
				
					_t_9430 = -1.0f * ty2_8_1;
					_t_9431 = ty1_7_1 + _t_9430;
					_t_9432 = -1.0f * _t_9431;
					_t_9429 = _t_9432;
				
				}
		
			_t_9433 = _t_9429 * _t_371;
			_t_9434 = 1.0f + _t_9433;
			_t_9435 = 1.0f / _t_9434;
			_t_9436 = _t_9422 * _t_9435;
			_t_9437 = _t_9436 * -1.0f;
			_t_9438 = 1.0f + _t_9437;
			_t_9439 = 0.0f < _t_9438;
			if(_t_9439)
				{
				
					_t_9440 = py1_13_1;
				
				}
		else
				{
				
					_t_9440 = py0_12_1;
				
				}
		
			_t_9441 = _t_9399 * _t_9440;
			_t_9442 = _t_9360 + _t_9441;
			_t_9443 = y__3239_1 < _t_9442;
			_t_9444 = _t_9333 && _t_9443;
			_t_9445 = -1.0f * ty2_8_1;
			_t_9446 = ty1_7_1 + _t_9445;
			_t_9447 = -1.0f * _t_9446;
			_t_9448 = _t_9447 < 0.0f;
			if(_t_9448)
				{
					float _t_9449;
					float _t_9450;
				
					_t_9449 = -1.0f * ty2_8_1;
					_t_9450 = ty1_7_1 + _t_9449;
					_t_9451 = _t_9450;
				
				}
		else
				{
					float _t_9452;
					float _t_9453;
					float _t_9454;
				
					_t_9452 = -1.0f * ty2_8_1;
					_t_9453 = ty1_7_1 + _t_9452;
					_t_9454 = -1.0f * _t_9453;
					_t_9451 = _t_9454;
				
				}
		
			_t_9455 = _t_9451 * _t_371;
			_t_9456 = -1.0f * ty2_8_1;
			_t_9457 = ty1_7_1 + _t_9456;
			_t_9458 = -1.0f * _t_9457;
			_t_9459 = _t_9458 < 0.0f;
			if(_t_9459)
				{
					float _t_9460;
					float _t_9461;
				
					_t_9460 = -1.0f * ty2_8_1;
					_t_9461 = ty1_7_1 + _t_9460;
					_t_9462 = _t_9461;
				
				}
		else
				{
					float _t_9463;
					float _t_9464;
					float _t_9465;
				
					_t_9463 = -1.0f * ty2_8_1;
					_t_9464 = ty1_7_1 + _t_9463;
					_t_9465 = -1.0f * _t_9464;
					_t_9462 = _t_9465;
				
				}
		
			_t_9466 = _t_9462 * _t_371;
			_t_9467 = 0.0f < _t_9466;
			if(_t_9467)
				{
				
					_t_9468 = px0_10_1;
				
				}
		else
				{
				
					_t_9468 = px1_11_1;
				
				}
		
			_t_9469 = _t_9455 * _t_9468;
			_t_9470 = -1.0f * ty2_8_1;
			_t_9471 = ty1_7_1 + _t_9470;
			_t_9472 = -1.0f * _t_9471;
			_t_9473 = _t_9472 < 0.0f;
			if(_t_9473)
				{
					float _t_9474;
					float _t_9475;
				
					_t_9474 = -1.0f * tx1_4_1;
					_t_9475 = tx2_5_1 + _t_9474;
					_t_9476 = _t_9475;
				
				}
		else
				{
					float _t_9477;
					float _t_9478;
					float _t_9479;
				
					_t_9477 = -1.0f * tx1_4_1;
					_t_9478 = tx2_5_1 + _t_9477;
					_t_9479 = -1.0f * _t_9478;
					_t_9476 = _t_9479;
				
				}
		
			_t_9480 = _t_9476 * _t_371;
			_t_9481 = -1.0f * ty2_8_1;
			_t_9482 = ty1_7_1 + _t_9481;
			_t_9483 = -1.0f * _t_9482;
			_t_9484 = _t_9483 < 0.0f;
			if(_t_9484)
				{
					float _t_9485;
					float _t_9486;
				
					_t_9485 = -1.0f * tx1_4_1;
					_t_9486 = tx2_5_1 + _t_9485;
					_t_9487 = _t_9486;
				
				}
		else
				{
					float _t_9488;
					float _t_9489;
					float _t_9490;
				
					_t_9488 = -1.0f * tx1_4_1;
					_t_9489 = tx2_5_1 + _t_9488;
					_t_9490 = -1.0f * _t_9489;
					_t_9487 = _t_9490;
				
				}
		
			_t_9491 = _t_9487 * _t_371;
			_t_9492 = 0.0f < _t_9491;
			if(_t_9492)
				{
				
					_t_9493 = py0_12_1;
				
				}
		else
				{
				
					_t_9493 = py1_13_1;
				
				}
		
			_t_9494 = _t_9480 * _t_9493;
			_t_9495 = _t_9469 + _t_9494;
			_t_9496 = _t_9495 < _t_9136;
			_t_9497 = -1.0f * ty2_8_1;
			_t_9498 = ty1_7_1 + _t_9497;
			_t_9499 = -1.0f * _t_9498;
			_t_9500 = _t_9499 < 0.0f;
			if(_t_9500)
				{
					float _t_9501;
					float _t_9502;
				
					_t_9501 = -1.0f * ty2_8_1;
					_t_9502 = ty1_7_1 + _t_9501;
					_t_9503 = _t_9502;
				
				}
		else
				{
					float _t_9504;
					float _t_9505;
					float _t_9506;
				
					_t_9504 = -1.0f * ty2_8_1;
					_t_9505 = ty1_7_1 + _t_9504;
					_t_9506 = -1.0f * _t_9505;
					_t_9503 = _t_9506;
				
				}
		
			_t_9507 = _t_9503 * _t_371;
			_t_9508 = -1.0f * ty2_8_1;
			_t_9509 = ty1_7_1 + _t_9508;
			_t_9510 = -1.0f * _t_9509;
			_t_9511 = _t_9510 < 0.0f;
			if(_t_9511)
				{
					float _t_9512;
					float _t_9513;
				
					_t_9512 = -1.0f * ty2_8_1;
					_t_9513 = ty1_7_1 + _t_9512;
					_t_9514 = _t_9513;
				
				}
		else
				{
					float _t_9515;
					float _t_9516;
					float _t_9517;
				
					_t_9515 = -1.0f * ty2_8_1;
					_t_9516 = ty1_7_1 + _t_9515;
					_t_9517 = -1.0f * _t_9516;
					_t_9514 = _t_9517;
				
				}
		
			_t_9518 = _t_9514 * _t_371;
			_t_9519 = 0.0f < _t_9518;
			if(_t_9519)
				{
				
					_t_9520 = px1_11_1;
				
				}
		else
				{
				
					_t_9520 = px0_10_1;
				
				}
		
			_t_9521 = _t_9507 * _t_9520;
			_t_9522 = -1.0f * ty2_8_1;
			_t_9523 = ty1_7_1 + _t_9522;
			_t_9524 = -1.0f * _t_9523;
			_t_9525 = _t_9524 < 0.0f;
			if(_t_9525)
				{
					float _t_9526;
					float _t_9527;
				
					_t_9526 = -1.0f * tx1_4_1;
					_t_9527 = tx2_5_1 + _t_9526;
					_t_9528 = _t_9527;
				
				}
		else
				{
					float _t_9529;
					float _t_9530;
					float _t_9531;
				
					_t_9529 = -1.0f * tx1_4_1;
					_t_9530 = tx2_5_1 + _t_9529;
					_t_9531 = -1.0f * _t_9530;
					_t_9528 = _t_9531;
				
				}
		
			_t_9532 = _t_9528 * _t_371;
			_t_9533 = -1.0f * ty2_8_1;
			_t_9534 = ty1_7_1 + _t_9533;
			_t_9535 = -1.0f * _t_9534;
			_t_9536 = _t_9535 < 0.0f;
			if(_t_9536)
				{
					float _t_9537;
					float _t_9538;
				
					_t_9537 = -1.0f * tx1_4_1;
					_t_9538 = tx2_5_1 + _t_9537;
					_t_9539 = _t_9538;
				
				}
		else
				{
					float _t_9540;
					float _t_9541;
					float _t_9542;
				
					_t_9540 = -1.0f * tx1_4_1;
					_t_9541 = tx2_5_1 + _t_9540;
					_t_9542 = -1.0f * _t_9541;
					_t_9539 = _t_9542;
				
				}
		
			_t_9543 = _t_9539 * _t_371;
			_t_9544 = 0.0f < _t_9543;
			if(_t_9544)
				{
				
					_t_9545 = py1_13_1;
				
				}
		else
				{
				
					_t_9545 = py0_12_1;
				
				}
		
			_t_9546 = _t_9532 * _t_9545;
			_t_9547 = _t_9521 + _t_9546;
			_t_9548 = _t_9136 < _t_9547;
			_t_9549 = _t_9496 && _t_9548;
			_t_9550 = _t_9444 && _t_9549;
			if(_t_9550)
				{
				
					_t_9551 = 1.0f;
				
				}
		else
				{
				
					_t_9551 = 0.0f;
				
				}
		
			_t_9552 = _t_9551 * _t_371;
			_t_9553 = _t_9552;
		
		}
else
		{
		
			_t_9553 = 0.0f;
		
		}

	_t_9554 = -1.0f * pc0_14_1;
	_t_9555 = tc0_17_1 + _t_9554;
	_t_9556 = _t_9555 * _t_9555;
	_t_9557 = -1.0f * pc1_15_1;
	_t_9558 = tc1_18_1 + _t_9557;
	_t_9559 = _t_9558 * _t_9558;
	_t_9560 = _t_9556 + _t_9559;
	_t_9561 = -1.0f * pc2_16_1;
	_t_9562 = tc2_19_1 + _t_9561;
	_t_9563 = _t_9562 * _t_9562;
	_t_9564 = _t_9560 + _t_9563;
	_t_9565 = tx3_6_1 * ty1_7_1;
	_t_9566 = tx1_4_1 * ty3_9_1;
	_t_9567 = _t_9566 * -1.0f;
	_t_9568 = _t_9565 + _t_9567;
	_t_9569 = -1.0f * ty1_7_1;
	_t_9570 = ty3_9_1 + _t_9569;
	_t_9571 = _t_9570 * _t_9163;
	_t_9572 = _t_9568 + _t_9571;
	_t_9573 = -1.0f * tx3_6_1;
	_t_9574 = tx1_4_1 + _t_9573;
	_t_9575 = _t_9574 * _t_9216;
	_t_9576 = _t_9572 + _t_9575;
	_t_9577 = _t_9576 < 0.0f;
	if(_t_9577)
		{
		
			_t_9578 = 1.0f;
		
		}
else
		{
		
			_t_9578 = 0.0f;
		
		}

	_t_9579 = _t_9564 * _t_9578;
	_t_9580 = tx2_5_1 * ty3_9_1;
	_t_9581 = tx3_6_1 * ty2_8_1;
	_t_9582 = _t_9581 * -1.0f;
	_t_9583 = _t_9580 + _t_9582;
	_t_9584 = -1.0f * ty3_9_1;
	_t_9585 = ty2_8_1 + _t_9584;
	_t_9586 = _t_9585 * _t_9163;
	_t_9587 = _t_9583 + _t_9586;
	_t_9588 = -1.0f * tx2_5_1;
	_t_9589 = tx3_6_1 + _t_9588;
	_t_9590 = _t_9589 * _t_9216;
	_t_9591 = _t_9587 + _t_9590;
	_t_9592 = _t_9591 < 0.0f;
	if(_t_9592)
		{
		
			_t_9593 = 1.0f;
		
		}
else
		{
		
			_t_9593 = 0.0f;
		
		}

	_t_9594 = _t_9579 * _t_9593;
	_t_9595 = _t_9594 * tx2_5_1;
	_t_9596 = -1.0f * pc0_14_1;
	_t_9597 = tc0_17_1 + _t_9596;
	_t_9598 = _t_9597 * _t_9597;
	_t_9599 = -1.0f * pc1_15_1;
	_t_9600 = tc1_18_1 + _t_9599;
	_t_9601 = _t_9600 * _t_9600;
	_t_9602 = _t_9598 + _t_9601;
	_t_9603 = -1.0f * pc2_16_1;
	_t_9604 = tc2_19_1 + _t_9603;
	_t_9605 = _t_9604 * _t_9604;
	_t_9606 = _t_9602 + _t_9605;
	_t_9607 = tx3_6_1 * ty1_7_1;
	_t_9608 = tx1_4_1 * ty3_9_1;
	_t_9609 = _t_9608 * -1.0f;
	_t_9610 = _t_9607 + _t_9609;
	_t_9611 = -1.0f * ty1_7_1;
	_t_9612 = ty3_9_1 + _t_9611;
	_t_9613 = _t_9612 * _t_9163;
	_t_9614 = _t_9610 + _t_9613;
	_t_9615 = -1.0f * tx3_6_1;
	_t_9616 = tx1_4_1 + _t_9615;
	_t_9617 = _t_9616 * _t_9216;
	_t_9618 = _t_9614 + _t_9617;
	_t_9619 = _t_9618 < 0.0f;
	if(_t_9619)
		{
		
			_t_9620 = 1.0f;
		
		}
else
		{
		
			_t_9620 = 0.0f;
		
		}

	_t_9621 = _t_9606 * _t_9620;
	_t_9622 = tx2_5_1 * ty3_9_1;
	_t_9623 = tx3_6_1 * ty2_8_1;
	_t_9624 = _t_9623 * -1.0f;
	_t_9625 = _t_9622 + _t_9624;
	_t_9626 = -1.0f * ty3_9_1;
	_t_9627 = ty2_8_1 + _t_9626;
	_t_9628 = _t_9627 * _t_9163;
	_t_9629 = _t_9625 + _t_9628;
	_t_9630 = -1.0f * tx2_5_1;
	_t_9631 = tx3_6_1 + _t_9630;
	_t_9632 = _t_9631 * _t_9216;
	_t_9633 = _t_9629 + _t_9632;
	_t_9634 = _t_9633 < 0.0f;
	if(_t_9634)
		{
		
			_t_9635 = 1.0f;
		
		}
else
		{
		
			_t_9635 = 0.0f;
		
		}

	_t_9636 = _t_9621 * _t_9635;
	_t_9637 = _t_9636 * _t_9163;
	_t_9638 = _t_9637 * -1.0f;
	_t_9639 = _t_9595 + _t_9638;
	_t_9137 = _t_9553 * _t_9639;

	return _t_9137;
}
__device__ float tegpixellet_block_37(float ty1_7_1,float ty2_8_1,float _t_371,float _t_9136,float tx2_5_1,float tx1_4_1,float y__3239_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	float _t_9138;
	float _t_9139;
	float _t_9140;
	bool _t_9141;
	float _t_9144;
	float _t_9148;
	float _t_9149;
	float _t_9150;
	float _t_9151;
	float _t_9152;
	bool _t_9153;
	float _t_9156;
	float _t_9160;
	float _t_9161;
	float _t_9162;
	float _t_9163;
	float _t_9164;
	float _t_9165;
	float _t_9166;
	bool _t_9167;
	float _t_9170;
	float _t_9174;
	float _t_9175;
	float _t_9176;
	float _t_9177;
	bool _t_9178;
	float _t_9181;
	float _t_9185;
	float _t_9186;
	float _t_9187;
	float _t_9188;
	float _t_9189;
	bool _t_9190;
	float _t_9193;
	float _t_9197;
	float _t_9198;
	float _t_9199;
	float _t_9200;
	float _t_9201;
	float _t_9202;
	float _t_9203;
	float _t_9204;
	float _t_9205;
	float _t_9206;
	bool _t_9207;
	float _t_9210;
	float _t_9214;
	float _t_9215;
	float _t_9216;

	float _t_9137;

	_t_9138 = -1.0f * ty2_8_1;
	_t_9139 = ty1_7_1 + _t_9138;
	_t_9140 = -1.0f * _t_9139;
	_t_9141 = _t_9140 < 0.0f;
	if(_t_9141)
		{
			float _t_9142;
			float _t_9143;
		
			_t_9142 = -1.0f * ty2_8_1;
			_t_9143 = ty1_7_1 + _t_9142;
			_t_9144 = _t_9143;
		
		}
else
		{
			float _t_9145;
			float _t_9146;
			float _t_9147;
		
			_t_9145 = -1.0f * ty2_8_1;
			_t_9146 = ty1_7_1 + _t_9145;
			_t_9147 = -1.0f * _t_9146;
			_t_9144 = _t_9147;
		
		}

	_t_9148 = _t_9144 * _t_371;
	_t_9149 = _t_9148 * _t_9136;
	_t_9150 = -1.0f * ty2_8_1;
	_t_9151 = ty1_7_1 + _t_9150;
	_t_9152 = -1.0f * _t_9151;
	_t_9153 = _t_9152 < 0.0f;
	if(_t_9153)
		{
			float _t_9154;
			float _t_9155;
		
			_t_9154 = -1.0f * tx1_4_1;
			_t_9155 = tx2_5_1 + _t_9154;
			_t_9156 = _t_9155;
		
		}
else
		{
			float _t_9157;
			float _t_9158;
			float _t_9159;
		
			_t_9157 = -1.0f * tx1_4_1;
			_t_9158 = tx2_5_1 + _t_9157;
			_t_9159 = -1.0f * _t_9158;
			_t_9156 = _t_9159;
		
		}

	_t_9160 = _t_9156 * _t_371;
	_t_9161 = _t_9160 * -1.0f;
	_t_9162 = _t_9161 * y__3239_1;
	_t_9163 = _t_9149 + _t_9162;
	_t_9164 = -1.0f * ty2_8_1;
	_t_9165 = ty1_7_1 + _t_9164;
	_t_9166 = -1.0f * _t_9165;
	_t_9167 = _t_9166 < 0.0f;
	if(_t_9167)
		{
			float _t_9168;
			float _t_9169;
		
			_t_9168 = -1.0f * tx1_4_1;
			_t_9169 = tx2_5_1 + _t_9168;
			_t_9170 = _t_9169;
		
		}
else
		{
			float _t_9171;
			float _t_9172;
			float _t_9173;
		
			_t_9171 = -1.0f * tx1_4_1;
			_t_9172 = tx2_5_1 + _t_9171;
			_t_9173 = -1.0f * _t_9172;
			_t_9170 = _t_9173;
		
		}

	_t_9174 = _t_9170 * _t_371;
	_t_9175 = -1.0f * ty2_8_1;
	_t_9176 = ty1_7_1 + _t_9175;
	_t_9177 = -1.0f * _t_9176;
	_t_9178 = _t_9177 < 0.0f;
	if(_t_9178)
		{
			float _t_9179;
			float _t_9180;
		
			_t_9179 = -1.0f * tx1_4_1;
			_t_9180 = tx2_5_1 + _t_9179;
			_t_9181 = _t_9180;
		
		}
else
		{
			float _t_9182;
			float _t_9183;
			float _t_9184;
		
			_t_9182 = -1.0f * tx1_4_1;
			_t_9183 = tx2_5_1 + _t_9182;
			_t_9184 = -1.0f * _t_9183;
			_t_9181 = _t_9184;
		
		}

	_t_9185 = _t_9181 * _t_371;
	_t_9186 = _t_9174 * _t_9185;
	_t_9187 = -1.0f * ty2_8_1;
	_t_9188 = ty1_7_1 + _t_9187;
	_t_9189 = -1.0f * _t_9188;
	_t_9190 = _t_9189 < 0.0f;
	if(_t_9190)
		{
			float _t_9191;
			float _t_9192;
		
			_t_9191 = -1.0f * ty2_8_1;
			_t_9192 = ty1_7_1 + _t_9191;
			_t_9193 = _t_9192;
		
		}
else
		{
			float _t_9194;
			float _t_9195;
			float _t_9196;
		
			_t_9194 = -1.0f * ty2_8_1;
			_t_9195 = ty1_7_1 + _t_9194;
			_t_9196 = -1.0f * _t_9195;
			_t_9193 = _t_9196;
		
		}

	_t_9197 = _t_9193 * _t_371;
	_t_9198 = 1.0f + _t_9197;
	_t_9199 = 1.0f / _t_9198;
	_t_9200 = _t_9186 * _t_9199;
	_t_9201 = _t_9200 * -1.0f;
	_t_9202 = 1.0f + _t_9201;
	_t_9203 = _t_9202 * y__3239_1;
	_t_9204 = -1.0f * ty2_8_1;
	_t_9205 = ty1_7_1 + _t_9204;
	_t_9206 = -1.0f * _t_9205;
	_t_9207 = _t_9206 < 0.0f;
	if(_t_9207)
		{
			float _t_9208;
			float _t_9209;
		
			_t_9208 = -1.0f * tx1_4_1;
			_t_9209 = tx2_5_1 + _t_9208;
			_t_9210 = _t_9209;
		
		}
else
		{
			float _t_9211;
			float _t_9212;
			float _t_9213;
		
			_t_9211 = -1.0f * tx1_4_1;
			_t_9212 = tx2_5_1 + _t_9211;
			_t_9213 = -1.0f * _t_9212;
			_t_9210 = _t_9213;
		
		}

	_t_9214 = _t_9210 * _t_371;
	_t_9215 = _t_9214 * _t_9136;
	_t_9216 = _t_9203 + _t_9215;
	_t_9137 = tegpixellet_block_38(py0_12_1,_t_9216,py1_13_1,px0_10_1,_t_9163,px1_11_1,ty1_7_1,ty2_8_1,tx2_5_1,tx1_4_1,_t_371,y__3239_1,_t_9136,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);

	return _t_9137;
}
__device__ float tegpixelbody_block_26(float ty1_7_1,float ty2_8_1,float _t_371,float px0_10_1,float px1_11_1,float tx2_5_1,float tx1_4_1,float py0_12_1,float py1_13_1,float y__3239_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	float _t_8980;
	float _t_8981;
	float _t_8982;
	bool _t_8983;
	float _t_8986;
	float _t_8990;
	float _t_8991;
	float _t_8992;
	float _t_8993;
	bool _t_8994;
	float _t_8997;
	float _t_9001;
	bool _t_9002;
	float _t_9003;
	float _t_9004;
	float _t_9005;
	float _t_9006;
	float _t_9007;
	bool _t_9008;
	float _t_9011;
	float _t_9015;
	float _t_9016;
	float _t_9017;
	float _t_9018;
	bool _t_9019;
	float _t_9022;
	float _t_9026;
	bool _t_9027;
	float _t_9028;
	float _t_9029;
	float _t_9030;
	float _t_9031;
	float _t_9032;
	float _t_9033;
	bool _t_9034;
	float _t_9039;
	float _t_9045;
	float _t_9046;
	float _t_9047;
	float _t_9048;
	bool _t_9049;
	float _t_9050;
	float _t_9051;
	float _t_9052;
	bool _t_9053;
	float _t_9056;
	float _t_9060;
	float _t_9061;
	float _t_9062;
	float _t_9063;
	bool _t_9064;
	float _t_9067;
	float _t_9071;
	bool _t_9072;
	float _t_9073;
	float _t_9074;
	float _t_9075;
	float _t_9076;
	float _t_9077;
	bool _t_9078;
	float _t_9081;
	float _t_9085;
	float _t_9086;
	float _t_9087;
	float _t_9088;
	bool _t_9089;
	float _t_9092;
	float _t_9096;
	bool _t_9097;
	float _t_9098;
	float _t_9099;
	float _t_9100;
	float _t_9101;
	float _t_9102;
	float _t_9103;
	bool _t_9104;
	float _t_9109;
	float _t_9115;
	float _t_9116;
	float _t_9117;
	float _t_9118;
	bool _t_9119;
	bool _t_9120;

	float _t_8979;

	_t_8980 = -1.0f * ty2_8_1;
	_t_8981 = ty1_7_1 + _t_8980;
	_t_8982 = -1.0f * _t_8981;
	_t_8983 = _t_8982 < 0.0f;
	if(_t_8983)
		{
			float _t_8984;
			float _t_8985;
		
			_t_8984 = -1.0f * ty2_8_1;
			_t_8985 = ty1_7_1 + _t_8984;
			_t_8986 = _t_8985;
		
		}
else
		{
			float _t_8987;
			float _t_8988;
			float _t_8989;
		
			_t_8987 = -1.0f * ty2_8_1;
			_t_8988 = ty1_7_1 + _t_8987;
			_t_8989 = -1.0f * _t_8988;
			_t_8986 = _t_8989;
		
		}

	_t_8990 = _t_8986 * _t_371;
	_t_8991 = -1.0f * ty2_8_1;
	_t_8992 = ty1_7_1 + _t_8991;
	_t_8993 = -1.0f * _t_8992;
	_t_8994 = _t_8993 < 0.0f;
	if(_t_8994)
		{
			float _t_8995;
			float _t_8996;
		
			_t_8995 = -1.0f * ty2_8_1;
			_t_8996 = ty1_7_1 + _t_8995;
			_t_8997 = _t_8996;
		
		}
else
		{
			float _t_8998;
			float _t_8999;
			float _t_9000;
		
			_t_8998 = -1.0f * ty2_8_1;
			_t_8999 = ty1_7_1 + _t_8998;
			_t_9000 = -1.0f * _t_8999;
			_t_8997 = _t_9000;
		
		}

	_t_9001 = _t_8997 * _t_371;
	_t_9002 = 0.0f < _t_9001;
	if(_t_9002)
		{
		
			_t_9003 = px0_10_1;
		
		}
else
		{
		
			_t_9003 = px1_11_1;
		
		}

	_t_9004 = _t_8990 * _t_9003;
	_t_9005 = -1.0f * ty2_8_1;
	_t_9006 = ty1_7_1 + _t_9005;
	_t_9007 = -1.0f * _t_9006;
	_t_9008 = _t_9007 < 0.0f;
	if(_t_9008)
		{
			float _t_9009;
			float _t_9010;
		
			_t_9009 = -1.0f * tx1_4_1;
			_t_9010 = tx2_5_1 + _t_9009;
			_t_9011 = _t_9010;
		
		}
else
		{
			float _t_9012;
			float _t_9013;
			float _t_9014;
		
			_t_9012 = -1.0f * tx1_4_1;
			_t_9013 = tx2_5_1 + _t_9012;
			_t_9014 = -1.0f * _t_9013;
			_t_9011 = _t_9014;
		
		}

	_t_9015 = _t_9011 * _t_371;
	_t_9016 = -1.0f * ty2_8_1;
	_t_9017 = ty1_7_1 + _t_9016;
	_t_9018 = -1.0f * _t_9017;
	_t_9019 = _t_9018 < 0.0f;
	if(_t_9019)
		{
			float _t_9020;
			float _t_9021;
		
			_t_9020 = -1.0f * tx1_4_1;
			_t_9021 = tx2_5_1 + _t_9020;
			_t_9022 = _t_9021;
		
		}
else
		{
			float _t_9023;
			float _t_9024;
			float _t_9025;
		
			_t_9023 = -1.0f * tx1_4_1;
			_t_9024 = tx2_5_1 + _t_9023;
			_t_9025 = -1.0f * _t_9024;
			_t_9022 = _t_9025;
		
		}

	_t_9026 = _t_9022 * _t_371;
	_t_9027 = 0.0f < _t_9026;
	if(_t_9027)
		{
		
			_t_9028 = py0_12_1;
		
		}
else
		{
		
			_t_9028 = py1_13_1;
		
		}

	_t_9029 = _t_9015 * _t_9028;
	_t_9030 = _t_9004 + _t_9029;
	_t_9031 = -1.0f * ty2_8_1;
	_t_9032 = ty1_7_1 + _t_9031;
	_t_9033 = -1.0f * _t_9032;
	_t_9034 = _t_9033 < 0.0f;
	if(_t_9034)
		{
			float _t_9035;
			float _t_9036;
			float _t_9037;
			float _t_9038;
		
			_t_9035 = tx1_4_1 * ty2_8_1;
			_t_9036 = tx2_5_1 * ty1_7_1;
			_t_9037 = _t_9036 * -1.0f;
			_t_9038 = _t_9035 + _t_9037;
			_t_9039 = _t_9038;
		
		}
else
		{
			float _t_9040;
			float _t_9041;
			float _t_9042;
			float _t_9043;
			float _t_9044;
		
			_t_9040 = tx1_4_1 * ty2_8_1;
			_t_9041 = tx2_5_1 * ty1_7_1;
			_t_9042 = _t_9041 * -1.0f;
			_t_9043 = _t_9040 + _t_9042;
			_t_9044 = -1.0f * _t_9043;
			_t_9039 = _t_9044;
		
		}

	_t_9045 = -1.0f * _t_9039;
	_t_9046 = _t_9045 * _t_371;
	_t_9047 = _t_9046 * -1.0f;
	_t_9048 = _t_9030 + _t_9047;
	_t_9049 = _t_9048 < 0.0f;
	_t_9050 = -1.0f * ty2_8_1;
	_t_9051 = ty1_7_1 + _t_9050;
	_t_9052 = -1.0f * _t_9051;
	_t_9053 = _t_9052 < 0.0f;
	if(_t_9053)
		{
			float _t_9054;
			float _t_9055;
		
			_t_9054 = -1.0f * ty2_8_1;
			_t_9055 = ty1_7_1 + _t_9054;
			_t_9056 = _t_9055;
		
		}
else
		{
			float _t_9057;
			float _t_9058;
			float _t_9059;
		
			_t_9057 = -1.0f * ty2_8_1;
			_t_9058 = ty1_7_1 + _t_9057;
			_t_9059 = -1.0f * _t_9058;
			_t_9056 = _t_9059;
		
		}

	_t_9060 = _t_9056 * _t_371;
	_t_9061 = -1.0f * ty2_8_1;
	_t_9062 = ty1_7_1 + _t_9061;
	_t_9063 = -1.0f * _t_9062;
	_t_9064 = _t_9063 < 0.0f;
	if(_t_9064)
		{
			float _t_9065;
			float _t_9066;
		
			_t_9065 = -1.0f * ty2_8_1;
			_t_9066 = ty1_7_1 + _t_9065;
			_t_9067 = _t_9066;
		
		}
else
		{
			float _t_9068;
			float _t_9069;
			float _t_9070;
		
			_t_9068 = -1.0f * ty2_8_1;
			_t_9069 = ty1_7_1 + _t_9068;
			_t_9070 = -1.0f * _t_9069;
			_t_9067 = _t_9070;
		
		}

	_t_9071 = _t_9067 * _t_371;
	_t_9072 = 0.0f < _t_9071;
	if(_t_9072)
		{
		
			_t_9073 = px1_11_1;
		
		}
else
		{
		
			_t_9073 = px0_10_1;
		
		}

	_t_9074 = _t_9060 * _t_9073;
	_t_9075 = -1.0f * ty2_8_1;
	_t_9076 = ty1_7_1 + _t_9075;
	_t_9077 = -1.0f * _t_9076;
	_t_9078 = _t_9077 < 0.0f;
	if(_t_9078)
		{
			float _t_9079;
			float _t_9080;
		
			_t_9079 = -1.0f * tx1_4_1;
			_t_9080 = tx2_5_1 + _t_9079;
			_t_9081 = _t_9080;
		
		}
else
		{
			float _t_9082;
			float _t_9083;
			float _t_9084;
		
			_t_9082 = -1.0f * tx1_4_1;
			_t_9083 = tx2_5_1 + _t_9082;
			_t_9084 = -1.0f * _t_9083;
			_t_9081 = _t_9084;
		
		}

	_t_9085 = _t_9081 * _t_371;
	_t_9086 = -1.0f * ty2_8_1;
	_t_9087 = ty1_7_1 + _t_9086;
	_t_9088 = -1.0f * _t_9087;
	_t_9089 = _t_9088 < 0.0f;
	if(_t_9089)
		{
			float _t_9090;
			float _t_9091;
		
			_t_9090 = -1.0f * tx1_4_1;
			_t_9091 = tx2_5_1 + _t_9090;
			_t_9092 = _t_9091;
		
		}
else
		{
			float _t_9093;
			float _t_9094;
			float _t_9095;
		
			_t_9093 = -1.0f * tx1_4_1;
			_t_9094 = tx2_5_1 + _t_9093;
			_t_9095 = -1.0f * _t_9094;
			_t_9092 = _t_9095;
		
		}

	_t_9096 = _t_9092 * _t_371;
	_t_9097 = 0.0f < _t_9096;
	if(_t_9097)
		{
		
			_t_9098 = py1_13_1;
		
		}
else
		{
		
			_t_9098 = py0_12_1;
		
		}

	_t_9099 = _t_9085 * _t_9098;
	_t_9100 = _t_9074 + _t_9099;
	_t_9101 = -1.0f * ty2_8_1;
	_t_9102 = ty1_7_1 + _t_9101;
	_t_9103 = -1.0f * _t_9102;
	_t_9104 = _t_9103 < 0.0f;
	if(_t_9104)
		{
			float _t_9105;
			float _t_9106;
			float _t_9107;
			float _t_9108;
		
			_t_9105 = tx1_4_1 * ty2_8_1;
			_t_9106 = tx2_5_1 * ty1_7_1;
			_t_9107 = _t_9106 * -1.0f;
			_t_9108 = _t_9105 + _t_9107;
			_t_9109 = _t_9108;
		
		}
else
		{
			float _t_9110;
			float _t_9111;
			float _t_9112;
			float _t_9113;
			float _t_9114;
		
			_t_9110 = tx1_4_1 * ty2_8_1;
			_t_9111 = tx2_5_1 * ty1_7_1;
			_t_9112 = _t_9111 * -1.0f;
			_t_9113 = _t_9110 + _t_9112;
			_t_9114 = -1.0f * _t_9113;
			_t_9109 = _t_9114;
		
		}

	_t_9115 = -1.0f * _t_9109;
	_t_9116 = _t_9115 * _t_371;
	_t_9117 = _t_9116 * -1.0f;
	_t_9118 = _t_9100 + _t_9117;
	_t_9119 = 0.0f < _t_9118;
	_t_9120 = _t_9049 && _t_9119;
	if(_t_9120)
		{
			float _t_9121;
			float _t_9122;
			float _t_9123;
			bool _t_9124;
			float _t_9129;
			float _t_9135;
			float _t_9136;
			float _t_9137;
		
			_t_9121 = -1.0f * ty2_8_1;
			_t_9122 = ty1_7_1 + _t_9121;
			_t_9123 = -1.0f * _t_9122;
			_t_9124 = _t_9123 < 0.0f;
			if(_t_9124)
				{
					float _t_9125;
					float _t_9126;
					float _t_9127;
					float _t_9128;
				
					_t_9125 = tx1_4_1 * ty2_8_1;
					_t_9126 = tx2_5_1 * ty1_7_1;
					_t_9127 = _t_9126 * -1.0f;
					_t_9128 = _t_9125 + _t_9127;
					_t_9129 = _t_9128;
				
				}
		else
				{
					float _t_9130;
					float _t_9131;
					float _t_9132;
					float _t_9133;
					float _t_9134;
				
					_t_9130 = tx1_4_1 * ty2_8_1;
					_t_9131 = tx2_5_1 * ty1_7_1;
					_t_9132 = _t_9131 * -1.0f;
					_t_9133 = _t_9130 + _t_9132;
					_t_9134 = -1.0f * _t_9133;
					_t_9129 = _t_9134;
				
				}
		
			_t_9135 = -1.0f * _t_9129;
			_t_9136 = _t_9135 * _t_371;
			_t_9137 = tegpixellet_block_37(ty1_7_1,ty2_8_1,_t_371,_t_9136,tx2_5_1,tx1_4_1,y__3239_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);
			_t_8979 = _t_9137;
		
		}
else
		{
		
			_t_8979 = 0.0f;
		
		}


	return _t_8979;
}
__device__ float tegpixelintegrator_26(float _t_371,float pc1_15_1,float ty3_9_1,float tc2_19_1,float _t_8978,float ty2_8_1,float pc0_14_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float tx2_5_1,float py1_13_1,float pc2_16_1,float px1_11_1,float tc0_17_1,float py0_12_1,float _t_8869,float tc1_18_1,float px0_10_1){
    float y__3239_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_8978 - _t_8869)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__3239_1 = _t_8869 + __step__ * (i + (float)(0.5));
        float _t_8979;
		_t_8979 = tegpixelbody_block_26(ty1_7_1,ty2_8_1,_t_371,px0_10_1,px1_11_1,tx2_5_1,tx1_4_1,py0_12_1,py1_13_1,y__3239_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);;
        __output__ = __output__ + _t_8979 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_10(float ty1_7_1,float ty2_8_1,float tx2_5_1,float tx1_4_1,float _t_371,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	float _t_8761;
	float _t_8762;
	float _t_8763;
	bool _t_8764;
	float _t_8767;
	float _t_8771;
	float _t_8772;
	float _t_8773;
	float _t_8774;
	float _t_8775;
	bool _t_8776;
	float _t_8779;
	float _t_8783;
	float _t_8784;
	bool _t_8785;
	float _t_8786;
	float _t_8787;
	float _t_8788;
	float _t_8789;
	float _t_8790;
	bool _t_8791;
	float _t_8794;
	float _t_8798;
	float _t_8799;
	float _t_8800;
	float _t_8801;
	bool _t_8802;
	float _t_8805;
	float _t_8809;
	float _t_8810;
	float _t_8811;
	float _t_8812;
	float _t_8813;
	bool _t_8814;
	float _t_8817;
	float _t_8821;
	float _t_8822;
	float _t_8823;
	float _t_8824;
	float _t_8825;
	float _t_8826;
	float _t_8827;
	float _t_8828;
	float _t_8829;
	bool _t_8830;
	float _t_8833;
	float _t_8837;
	float _t_8838;
	float _t_8839;
	float _t_8840;
	bool _t_8841;
	float _t_8844;
	float _t_8848;
	float _t_8849;
	float _t_8850;
	float _t_8851;
	float _t_8852;
	bool _t_8853;
	float _t_8856;
	float _t_8860;
	float _t_8861;
	float _t_8862;
	float _t_8863;
	float _t_8864;
	float _t_8865;
	bool _t_8866;
	float _t_8867;
	float _t_8868;
	float _t_8869;
	float _t_8870;
	float _t_8871;
	float _t_8872;
	bool _t_8873;
	float _t_8876;
	float _t_8880;
	float _t_8881;
	float _t_8882;
	float _t_8883;
	float _t_8884;
	bool _t_8885;
	float _t_8888;
	float _t_8892;
	float _t_8893;
	bool _t_8894;
	float _t_8895;
	float _t_8896;
	float _t_8897;
	float _t_8898;
	float _t_8899;
	bool _t_8900;
	float _t_8903;
	float _t_8907;
	float _t_8908;
	float _t_8909;
	float _t_8910;
	bool _t_8911;
	float _t_8914;
	float _t_8918;
	float _t_8919;
	float _t_8920;
	float _t_8921;
	float _t_8922;
	bool _t_8923;
	float _t_8926;
	float _t_8930;
	float _t_8931;
	float _t_8932;
	float _t_8933;
	float _t_8934;
	float _t_8935;
	float _t_8936;
	float _t_8937;
	float _t_8938;
	bool _t_8939;
	float _t_8942;
	float _t_8946;
	float _t_8947;
	float _t_8948;
	float _t_8949;
	bool _t_8950;
	float _t_8953;
	float _t_8957;
	float _t_8958;
	float _t_8959;
	float _t_8960;
	float _t_8961;
	bool _t_8962;
	float _t_8965;
	float _t_8969;
	float _t_8970;
	float _t_8971;
	float _t_8972;
	float _t_8973;
	float _t_8974;
	bool _t_8975;
	float _t_8976;
	float _t_8977;
	float _t_8978;

	float _t_372;

	_t_8761 = -1.0f * ty2_8_1;
	_t_8762 = ty1_7_1 + _t_8761;
	_t_8763 = -1.0f * _t_8762;
	_t_8764 = _t_8763 < 0.0f;
	if(_t_8764)
		{
			float _t_8765;
			float _t_8766;
		
			_t_8765 = -1.0f * tx1_4_1;
			_t_8766 = tx2_5_1 + _t_8765;
			_t_8767 = _t_8766;
		
		}
else
		{
			float _t_8768;
			float _t_8769;
			float _t_8770;
		
			_t_8768 = -1.0f * tx1_4_1;
			_t_8769 = tx2_5_1 + _t_8768;
			_t_8770 = -1.0f * _t_8769;
			_t_8767 = _t_8770;
		
		}

	_t_8771 = _t_8767 * _t_371;
	_t_8772 = _t_8771 * -1.0f;
	_t_8773 = -1.0f * ty2_8_1;
	_t_8774 = ty1_7_1 + _t_8773;
	_t_8775 = -1.0f * _t_8774;
	_t_8776 = _t_8775 < 0.0f;
	if(_t_8776)
		{
			float _t_8777;
			float _t_8778;
		
			_t_8777 = -1.0f * tx1_4_1;
			_t_8778 = tx2_5_1 + _t_8777;
			_t_8779 = _t_8778;
		
		}
else
		{
			float _t_8780;
			float _t_8781;
			float _t_8782;
		
			_t_8780 = -1.0f * tx1_4_1;
			_t_8781 = tx2_5_1 + _t_8780;
			_t_8782 = -1.0f * _t_8781;
			_t_8779 = _t_8782;
		
		}

	_t_8783 = _t_8779 * _t_371;
	_t_8784 = _t_8783 * -1.0f;
	_t_8785 = 0.0f < _t_8784;
	if(_t_8785)
		{
		
			_t_8786 = px0_10_1;
		
		}
else
		{
		
			_t_8786 = px1_11_1;
		
		}

	_t_8787 = _t_8772 * _t_8786;
	_t_8788 = -1.0f * ty2_8_1;
	_t_8789 = ty1_7_1 + _t_8788;
	_t_8790 = -1.0f * _t_8789;
	_t_8791 = _t_8790 < 0.0f;
	if(_t_8791)
		{
			float _t_8792;
			float _t_8793;
		
			_t_8792 = -1.0f * tx1_4_1;
			_t_8793 = tx2_5_1 + _t_8792;
			_t_8794 = _t_8793;
		
		}
else
		{
			float _t_8795;
			float _t_8796;
			float _t_8797;
		
			_t_8795 = -1.0f * tx1_4_1;
			_t_8796 = tx2_5_1 + _t_8795;
			_t_8797 = -1.0f * _t_8796;
			_t_8794 = _t_8797;
		
		}

	_t_8798 = _t_8794 * _t_371;
	_t_8799 = -1.0f * ty2_8_1;
	_t_8800 = ty1_7_1 + _t_8799;
	_t_8801 = -1.0f * _t_8800;
	_t_8802 = _t_8801 < 0.0f;
	if(_t_8802)
		{
			float _t_8803;
			float _t_8804;
		
			_t_8803 = -1.0f * tx1_4_1;
			_t_8804 = tx2_5_1 + _t_8803;
			_t_8805 = _t_8804;
		
		}
else
		{
			float _t_8806;
			float _t_8807;
			float _t_8808;
		
			_t_8806 = -1.0f * tx1_4_1;
			_t_8807 = tx2_5_1 + _t_8806;
			_t_8808 = -1.0f * _t_8807;
			_t_8805 = _t_8808;
		
		}

	_t_8809 = _t_8805 * _t_371;
	_t_8810 = _t_8798 * _t_8809;
	_t_8811 = -1.0f * ty2_8_1;
	_t_8812 = ty1_7_1 + _t_8811;
	_t_8813 = -1.0f * _t_8812;
	_t_8814 = _t_8813 < 0.0f;
	if(_t_8814)
		{
			float _t_8815;
			float _t_8816;
		
			_t_8815 = -1.0f * ty2_8_1;
			_t_8816 = ty1_7_1 + _t_8815;
			_t_8817 = _t_8816;
		
		}
else
		{
			float _t_8818;
			float _t_8819;
			float _t_8820;
		
			_t_8818 = -1.0f * ty2_8_1;
			_t_8819 = ty1_7_1 + _t_8818;
			_t_8820 = -1.0f * _t_8819;
			_t_8817 = _t_8820;
		
		}

	_t_8821 = _t_8817 * _t_371;
	_t_8822 = 1.0f + _t_8821;
	_t_8823 = 1.0f / _t_8822;
	_t_8824 = _t_8810 * _t_8823;
	_t_8825 = _t_8824 * -1.0f;
	_t_8826 = 1.0f + _t_8825;
	_t_8827 = -1.0f * ty2_8_1;
	_t_8828 = ty1_7_1 + _t_8827;
	_t_8829 = -1.0f * _t_8828;
	_t_8830 = _t_8829 < 0.0f;
	if(_t_8830)
		{
			float _t_8831;
			float _t_8832;
		
			_t_8831 = -1.0f * tx1_4_1;
			_t_8832 = tx2_5_1 + _t_8831;
			_t_8833 = _t_8832;
		
		}
else
		{
			float _t_8834;
			float _t_8835;
			float _t_8836;
		
			_t_8834 = -1.0f * tx1_4_1;
			_t_8835 = tx2_5_1 + _t_8834;
			_t_8836 = -1.0f * _t_8835;
			_t_8833 = _t_8836;
		
		}

	_t_8837 = _t_8833 * _t_371;
	_t_8838 = -1.0f * ty2_8_1;
	_t_8839 = ty1_7_1 + _t_8838;
	_t_8840 = -1.0f * _t_8839;
	_t_8841 = _t_8840 < 0.0f;
	if(_t_8841)
		{
			float _t_8842;
			float _t_8843;
		
			_t_8842 = -1.0f * tx1_4_1;
			_t_8843 = tx2_5_1 + _t_8842;
			_t_8844 = _t_8843;
		
		}
else
		{
			float _t_8845;
			float _t_8846;
			float _t_8847;
		
			_t_8845 = -1.0f * tx1_4_1;
			_t_8846 = tx2_5_1 + _t_8845;
			_t_8847 = -1.0f * _t_8846;
			_t_8844 = _t_8847;
		
		}

	_t_8848 = _t_8844 * _t_371;
	_t_8849 = _t_8837 * _t_8848;
	_t_8850 = -1.0f * ty2_8_1;
	_t_8851 = ty1_7_1 + _t_8850;
	_t_8852 = -1.0f * _t_8851;
	_t_8853 = _t_8852 < 0.0f;
	if(_t_8853)
		{
			float _t_8854;
			float _t_8855;
		
			_t_8854 = -1.0f * ty2_8_1;
			_t_8855 = ty1_7_1 + _t_8854;
			_t_8856 = _t_8855;
		
		}
else
		{
			float _t_8857;
			float _t_8858;
			float _t_8859;
		
			_t_8857 = -1.0f * ty2_8_1;
			_t_8858 = ty1_7_1 + _t_8857;
			_t_8859 = -1.0f * _t_8858;
			_t_8856 = _t_8859;
		
		}

	_t_8860 = _t_8856 * _t_371;
	_t_8861 = 1.0f + _t_8860;
	_t_8862 = 1.0f / _t_8861;
	_t_8863 = _t_8849 * _t_8862;
	_t_8864 = _t_8863 * -1.0f;
	_t_8865 = 1.0f + _t_8864;
	_t_8866 = 0.0f < _t_8865;
	if(_t_8866)
		{
		
			_t_8867 = py0_12_1;
		
		}
else
		{
		
			_t_8867 = py1_13_1;
		
		}

	_t_8868 = _t_8826 * _t_8867;
	_t_8869 = _t_8787 + _t_8868;
	_t_8870 = -1.0f * ty2_8_1;
	_t_8871 = ty1_7_1 + _t_8870;
	_t_8872 = -1.0f * _t_8871;
	_t_8873 = _t_8872 < 0.0f;
	if(_t_8873)
		{
			float _t_8874;
			float _t_8875;
		
			_t_8874 = -1.0f * tx1_4_1;
			_t_8875 = tx2_5_1 + _t_8874;
			_t_8876 = _t_8875;
		
		}
else
		{
			float _t_8877;
			float _t_8878;
			float _t_8879;
		
			_t_8877 = -1.0f * tx1_4_1;
			_t_8878 = tx2_5_1 + _t_8877;
			_t_8879 = -1.0f * _t_8878;
			_t_8876 = _t_8879;
		
		}

	_t_8880 = _t_8876 * _t_371;
	_t_8881 = _t_8880 * -1.0f;
	_t_8882 = -1.0f * ty2_8_1;
	_t_8883 = ty1_7_1 + _t_8882;
	_t_8884 = -1.0f * _t_8883;
	_t_8885 = _t_8884 < 0.0f;
	if(_t_8885)
		{
			float _t_8886;
			float _t_8887;
		
			_t_8886 = -1.0f * tx1_4_1;
			_t_8887 = tx2_5_1 + _t_8886;
			_t_8888 = _t_8887;
		
		}
else
		{
			float _t_8889;
			float _t_8890;
			float _t_8891;
		
			_t_8889 = -1.0f * tx1_4_1;
			_t_8890 = tx2_5_1 + _t_8889;
			_t_8891 = -1.0f * _t_8890;
			_t_8888 = _t_8891;
		
		}

	_t_8892 = _t_8888 * _t_371;
	_t_8893 = _t_8892 * -1.0f;
	_t_8894 = 0.0f < _t_8893;
	if(_t_8894)
		{
		
			_t_8895 = px1_11_1;
		
		}
else
		{
		
			_t_8895 = px0_10_1;
		
		}

	_t_8896 = _t_8881 * _t_8895;
	_t_8897 = -1.0f * ty2_8_1;
	_t_8898 = ty1_7_1 + _t_8897;
	_t_8899 = -1.0f * _t_8898;
	_t_8900 = _t_8899 < 0.0f;
	if(_t_8900)
		{
			float _t_8901;
			float _t_8902;
		
			_t_8901 = -1.0f * tx1_4_1;
			_t_8902 = tx2_5_1 + _t_8901;
			_t_8903 = _t_8902;
		
		}
else
		{
			float _t_8904;
			float _t_8905;
			float _t_8906;
		
			_t_8904 = -1.0f * tx1_4_1;
			_t_8905 = tx2_5_1 + _t_8904;
			_t_8906 = -1.0f * _t_8905;
			_t_8903 = _t_8906;
		
		}

	_t_8907 = _t_8903 * _t_371;
	_t_8908 = -1.0f * ty2_8_1;
	_t_8909 = ty1_7_1 + _t_8908;
	_t_8910 = -1.0f * _t_8909;
	_t_8911 = _t_8910 < 0.0f;
	if(_t_8911)
		{
			float _t_8912;
			float _t_8913;
		
			_t_8912 = -1.0f * tx1_4_1;
			_t_8913 = tx2_5_1 + _t_8912;
			_t_8914 = _t_8913;
		
		}
else
		{
			float _t_8915;
			float _t_8916;
			float _t_8917;
		
			_t_8915 = -1.0f * tx1_4_1;
			_t_8916 = tx2_5_1 + _t_8915;
			_t_8917 = -1.0f * _t_8916;
			_t_8914 = _t_8917;
		
		}

	_t_8918 = _t_8914 * _t_371;
	_t_8919 = _t_8907 * _t_8918;
	_t_8920 = -1.0f * ty2_8_1;
	_t_8921 = ty1_7_1 + _t_8920;
	_t_8922 = -1.0f * _t_8921;
	_t_8923 = _t_8922 < 0.0f;
	if(_t_8923)
		{
			float _t_8924;
			float _t_8925;
		
			_t_8924 = -1.0f * ty2_8_1;
			_t_8925 = ty1_7_1 + _t_8924;
			_t_8926 = _t_8925;
		
		}
else
		{
			float _t_8927;
			float _t_8928;
			float _t_8929;
		
			_t_8927 = -1.0f * ty2_8_1;
			_t_8928 = ty1_7_1 + _t_8927;
			_t_8929 = -1.0f * _t_8928;
			_t_8926 = _t_8929;
		
		}

	_t_8930 = _t_8926 * _t_371;
	_t_8931 = 1.0f + _t_8930;
	_t_8932 = 1.0f / _t_8931;
	_t_8933 = _t_8919 * _t_8932;
	_t_8934 = _t_8933 * -1.0f;
	_t_8935 = 1.0f + _t_8934;
	_t_8936 = -1.0f * ty2_8_1;
	_t_8937 = ty1_7_1 + _t_8936;
	_t_8938 = -1.0f * _t_8937;
	_t_8939 = _t_8938 < 0.0f;
	if(_t_8939)
		{
			float _t_8940;
			float _t_8941;
		
			_t_8940 = -1.0f * tx1_4_1;
			_t_8941 = tx2_5_1 + _t_8940;
			_t_8942 = _t_8941;
		
		}
else
		{
			float _t_8943;
			float _t_8944;
			float _t_8945;
		
			_t_8943 = -1.0f * tx1_4_1;
			_t_8944 = tx2_5_1 + _t_8943;
			_t_8945 = -1.0f * _t_8944;
			_t_8942 = _t_8945;
		
		}

	_t_8946 = _t_8942 * _t_371;
	_t_8947 = -1.0f * ty2_8_1;
	_t_8948 = ty1_7_1 + _t_8947;
	_t_8949 = -1.0f * _t_8948;
	_t_8950 = _t_8949 < 0.0f;
	if(_t_8950)
		{
			float _t_8951;
			float _t_8952;
		
			_t_8951 = -1.0f * tx1_4_1;
			_t_8952 = tx2_5_1 + _t_8951;
			_t_8953 = _t_8952;
		
		}
else
		{
			float _t_8954;
			float _t_8955;
			float _t_8956;
		
			_t_8954 = -1.0f * tx1_4_1;
			_t_8955 = tx2_5_1 + _t_8954;
			_t_8956 = -1.0f * _t_8955;
			_t_8953 = _t_8956;
		
		}

	_t_8957 = _t_8953 * _t_371;
	_t_8958 = _t_8946 * _t_8957;
	_t_8959 = -1.0f * ty2_8_1;
	_t_8960 = ty1_7_1 + _t_8959;
	_t_8961 = -1.0f * _t_8960;
	_t_8962 = _t_8961 < 0.0f;
	if(_t_8962)
		{
			float _t_8963;
			float _t_8964;
		
			_t_8963 = -1.0f * ty2_8_1;
			_t_8964 = ty1_7_1 + _t_8963;
			_t_8965 = _t_8964;
		
		}
else
		{
			float _t_8966;
			float _t_8967;
			float _t_8968;
		
			_t_8966 = -1.0f * ty2_8_1;
			_t_8967 = ty1_7_1 + _t_8966;
			_t_8968 = -1.0f * _t_8967;
			_t_8965 = _t_8968;
		
		}

	_t_8969 = _t_8965 * _t_371;
	_t_8970 = 1.0f + _t_8969;
	_t_8971 = 1.0f / _t_8970;
	_t_8972 = _t_8958 * _t_8971;
	_t_8973 = _t_8972 * -1.0f;
	_t_8974 = 1.0f + _t_8973;
	_t_8975 = 0.0f < _t_8974;
	if(_t_8975)
		{
		
			_t_8976 = py1_13_1;
		
		}
else
		{
		
			_t_8976 = py0_12_1;
		
		}

	_t_8977 = _t_8935 * _t_8976;
	_t_8978 = _t_8896 + _t_8977;
	_t_372 = tegpixelintegrator_26(_t_371,pc1_15_1,ty3_9_1,tc2_19_1,_t_8978,ty2_8_1,pc0_14_1,ty1_7_1,tx1_4_1,tx3_6_1,tx2_5_1,py1_13_1,pc2_16_1,px1_11_1,tc0_17_1,py0_12_1,_t_8869,tc1_18_1,px0_10_1);

	return _t_372;
}
__device__ float tegpixellet_block_40(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float _t_10042,float _t_10095,float ty3_9_1,float tx3_6_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float _t_399,float y__3313_1,float _t_10015){
	float _t_10096;
	float _t_10097;
	float _t_10098;
	float _t_10099;
	float _t_10100;
	float _t_10101;
	float _t_10102;
	float _t_10103;
	float _t_10104;
	float _t_10105;
	float _t_10106;
	float _t_10107;
	float _t_10108;
	float _t_10109;
	float _t_10110;
	float _t_10111;
	float _t_10112;
	float _t_10113;
	float _t_10114;
	float _t_10115;
	float _t_10116;
	float _t_10117;
	float _t_10118;
	bool _t_10119;
	float _t_10120;
	float _t_10121;
	float _t_10122;
	float _t_10123;
	float _t_10124;
	float _t_10125;
	float _t_10126;
	float _t_10127;
	float _t_10128;
	float _t_10129;
	float _t_10130;
	float _t_10131;
	float _t_10132;
	bool _t_10133;
	float _t_10134;
	float _t_10135;
	float _t_10136;
	float _t_10137;
	float _t_10138;
	bool _t_10139;
	bool _t_10140;
	bool _t_10141;
	bool _t_10142;
	bool _t_10143;
	bool _t_10144;
	bool _t_10145;
	float _t_10475;

	float _t_10016;

	_t_10096 = -1.0f * pc0_14_1;
	_t_10097 = tc0_17_1 + _t_10096;
	_t_10098 = _t_10097 * _t_10097;
	_t_10099 = -1.0f * pc1_15_1;
	_t_10100 = tc1_18_1 + _t_10099;
	_t_10101 = _t_10100 * _t_10100;
	_t_10102 = _t_10098 + _t_10101;
	_t_10103 = -1.0f * pc2_16_1;
	_t_10104 = tc2_19_1 + _t_10103;
	_t_10105 = _t_10104 * _t_10104;
	_t_10106 = _t_10102 + _t_10105;
	_t_10107 = tx1_4_1 * ty2_8_1;
	_t_10108 = tx2_5_1 * ty1_7_1;
	_t_10109 = _t_10108 * -1.0f;
	_t_10110 = _t_10107 + _t_10109;
	_t_10111 = -1.0f * ty2_8_1;
	_t_10112 = ty1_7_1 + _t_10111;
	_t_10113 = _t_10112 * _t_10042;
	_t_10114 = _t_10110 + _t_10113;
	_t_10115 = -1.0f * tx1_4_1;
	_t_10116 = tx2_5_1 + _t_10115;
	_t_10117 = _t_10116 * _t_10095;
	_t_10118 = _t_10114 + _t_10117;
	_t_10119 = _t_10118 < 0.0f;
	if(_t_10119)
		{
		
			_t_10120 = 1.0f;
		
		}
else
		{
		
			_t_10120 = 0.0f;
		
		}

	_t_10121 = tx2_5_1 * ty3_9_1;
	_t_10122 = tx3_6_1 * ty2_8_1;
	_t_10123 = _t_10122 * -1.0f;
	_t_10124 = _t_10121 + _t_10123;
	_t_10125 = -1.0f * ty3_9_1;
	_t_10126 = ty2_8_1 + _t_10125;
	_t_10127 = _t_10126 * _t_10042;
	_t_10128 = _t_10124 + _t_10127;
	_t_10129 = -1.0f * tx2_5_1;
	_t_10130 = tx3_6_1 + _t_10129;
	_t_10131 = _t_10130 * _t_10095;
	_t_10132 = _t_10128 + _t_10131;
	_t_10133 = _t_10132 < 0.0f;
	if(_t_10133)
		{
		
			_t_10134 = 1.0f;
		
		}
else
		{
		
			_t_10134 = 0.0f;
		
		}

	_t_10135 = _t_10120 * _t_10134;
	_t_10136 = _t_10106 * _t_10135;
	_t_10137 = _t_10136 * tx3_6_1;
	_t_10138 = _t_10137 * -1.0f;
	_t_10139 = py0_12_1 < _t_10095;
	_t_10140 = _t_10095 < py1_13_1;
	_t_10141 = _t_10139 && _t_10140;
	_t_10142 = px0_10_1 < _t_10042;
	_t_10143 = _t_10042 < px1_11_1;
	_t_10144 = _t_10142 && _t_10143;
	_t_10145 = _t_10141 && _t_10144;
	if(_t_10145)
		{
			float _t_10146;
			float _t_10147;
			float _t_10148;
			bool _t_10149;
			float _t_10152;
			float _t_10156;
			float _t_10157;
			float _t_10158;
			float _t_10159;
			float _t_10160;
			bool _t_10161;
			float _t_10164;
			float _t_10168;
			float _t_10169;
			bool _t_10170;
			float _t_10171;
			float _t_10172;
			float _t_10173;
			float _t_10174;
			float _t_10175;
			bool _t_10176;
			float _t_10179;
			float _t_10183;
			float _t_10184;
			float _t_10185;
			float _t_10186;
			bool _t_10187;
			float _t_10190;
			float _t_10194;
			float _t_10195;
			float _t_10196;
			float _t_10197;
			float _t_10198;
			bool _t_10199;
			float _t_10202;
			float _t_10206;
			float _t_10207;
			float _t_10208;
			float _t_10209;
			float _t_10210;
			float _t_10211;
			float _t_10212;
			float _t_10213;
			float _t_10214;
			bool _t_10215;
			float _t_10218;
			float _t_10222;
			float _t_10223;
			float _t_10224;
			float _t_10225;
			bool _t_10226;
			float _t_10229;
			float _t_10233;
			float _t_10234;
			float _t_10235;
			float _t_10236;
			float _t_10237;
			bool _t_10238;
			float _t_10241;
			float _t_10245;
			float _t_10246;
			float _t_10247;
			float _t_10248;
			float _t_10249;
			float _t_10250;
			bool _t_10251;
			float _t_10252;
			float _t_10253;
			float _t_10254;
			bool _t_10255;
			float _t_10256;
			float _t_10257;
			float _t_10258;
			bool _t_10259;
			float _t_10262;
			float _t_10266;
			float _t_10267;
			float _t_10268;
			float _t_10269;
			float _t_10270;
			bool _t_10271;
			float _t_10274;
			float _t_10278;
			float _t_10279;
			bool _t_10280;
			float _t_10281;
			float _t_10282;
			float _t_10283;
			float _t_10284;
			float _t_10285;
			bool _t_10286;
			float _t_10289;
			float _t_10293;
			float _t_10294;
			float _t_10295;
			float _t_10296;
			bool _t_10297;
			float _t_10300;
			float _t_10304;
			float _t_10305;
			float _t_10306;
			float _t_10307;
			float _t_10308;
			bool _t_10309;
			float _t_10312;
			float _t_10316;
			float _t_10317;
			float _t_10318;
			float _t_10319;
			float _t_10320;
			float _t_10321;
			float _t_10322;
			float _t_10323;
			float _t_10324;
			bool _t_10325;
			float _t_10328;
			float _t_10332;
			float _t_10333;
			float _t_10334;
			float _t_10335;
			bool _t_10336;
			float _t_10339;
			float _t_10343;
			float _t_10344;
			float _t_10345;
			float _t_10346;
			float _t_10347;
			bool _t_10348;
			float _t_10351;
			float _t_10355;
			float _t_10356;
			float _t_10357;
			float _t_10358;
			float _t_10359;
			float _t_10360;
			bool _t_10361;
			float _t_10362;
			float _t_10363;
			float _t_10364;
			bool _t_10365;
			bool _t_10366;
			float _t_10367;
			float _t_10368;
			float _t_10369;
			bool _t_10370;
			float _t_10373;
			float _t_10377;
			float _t_10378;
			float _t_10379;
			float _t_10380;
			bool _t_10381;
			float _t_10384;
			float _t_10388;
			bool _t_10389;
			float _t_10390;
			float _t_10391;
			float _t_10392;
			float _t_10393;
			float _t_10394;
			bool _t_10395;
			float _t_10398;
			float _t_10402;
			float _t_10403;
			float _t_10404;
			float _t_10405;
			bool _t_10406;
			float _t_10409;
			float _t_10413;
			bool _t_10414;
			float _t_10415;
			float _t_10416;
			float _t_10417;
			bool _t_10418;
			float _t_10419;
			float _t_10420;
			float _t_10421;
			bool _t_10422;
			float _t_10425;
			float _t_10429;
			float _t_10430;
			float _t_10431;
			float _t_10432;
			bool _t_10433;
			float _t_10436;
			float _t_10440;
			bool _t_10441;
			float _t_10442;
			float _t_10443;
			float _t_10444;
			float _t_10445;
			float _t_10446;
			bool _t_10447;
			float _t_10450;
			float _t_10454;
			float _t_10455;
			float _t_10456;
			float _t_10457;
			bool _t_10458;
			float _t_10461;
			float _t_10465;
			bool _t_10466;
			float _t_10467;
			float _t_10468;
			float _t_10469;
			bool _t_10470;
			bool _t_10471;
			bool _t_10472;
			float _t_10473;
			float _t_10474;
		
			_t_10146 = -1.0f * ty1_7_1;
			_t_10147 = ty3_9_1 + _t_10146;
			_t_10148 = -1.0f * _t_10147;
			_t_10149 = _t_10148 < 0.0f;
			if(_t_10149)
				{
					float _t_10150;
					float _t_10151;
				
					_t_10150 = -1.0f * tx3_6_1;
					_t_10151 = tx1_4_1 + _t_10150;
					_t_10152 = _t_10151;
				
				}
		else
				{
					float _t_10153;
					float _t_10154;
					float _t_10155;
				
					_t_10153 = -1.0f * tx3_6_1;
					_t_10154 = tx1_4_1 + _t_10153;
					_t_10155 = -1.0f * _t_10154;
					_t_10152 = _t_10155;
				
				}
		
			_t_10156 = _t_10152 * _t_399;
			_t_10157 = _t_10156 * -1.0f;
			_t_10158 = -1.0f * ty1_7_1;
			_t_10159 = ty3_9_1 + _t_10158;
			_t_10160 = -1.0f * _t_10159;
			_t_10161 = _t_10160 < 0.0f;
			if(_t_10161)
				{
					float _t_10162;
					float _t_10163;
				
					_t_10162 = -1.0f * tx3_6_1;
					_t_10163 = tx1_4_1 + _t_10162;
					_t_10164 = _t_10163;
				
				}
		else
				{
					float _t_10165;
					float _t_10166;
					float _t_10167;
				
					_t_10165 = -1.0f * tx3_6_1;
					_t_10166 = tx1_4_1 + _t_10165;
					_t_10167 = -1.0f * _t_10166;
					_t_10164 = _t_10167;
				
				}
		
			_t_10168 = _t_10164 * _t_399;
			_t_10169 = _t_10168 * -1.0f;
			_t_10170 = 0.0f < _t_10169;
			if(_t_10170)
				{
				
					_t_10171 = px0_10_1;
				
				}
		else
				{
				
					_t_10171 = px1_11_1;
				
				}
		
			_t_10172 = _t_10157 * _t_10171;
			_t_10173 = -1.0f * ty1_7_1;
			_t_10174 = ty3_9_1 + _t_10173;
			_t_10175 = -1.0f * _t_10174;
			_t_10176 = _t_10175 < 0.0f;
			if(_t_10176)
				{
					float _t_10177;
					float _t_10178;
				
					_t_10177 = -1.0f * tx3_6_1;
					_t_10178 = tx1_4_1 + _t_10177;
					_t_10179 = _t_10178;
				
				}
		else
				{
					float _t_10180;
					float _t_10181;
					float _t_10182;
				
					_t_10180 = -1.0f * tx3_6_1;
					_t_10181 = tx1_4_1 + _t_10180;
					_t_10182 = -1.0f * _t_10181;
					_t_10179 = _t_10182;
				
				}
		
			_t_10183 = _t_10179 * _t_399;
			_t_10184 = -1.0f * ty1_7_1;
			_t_10185 = ty3_9_1 + _t_10184;
			_t_10186 = -1.0f * _t_10185;
			_t_10187 = _t_10186 < 0.0f;
			if(_t_10187)
				{
					float _t_10188;
					float _t_10189;
				
					_t_10188 = -1.0f * tx3_6_1;
					_t_10189 = tx1_4_1 + _t_10188;
					_t_10190 = _t_10189;
				
				}
		else
				{
					float _t_10191;
					float _t_10192;
					float _t_10193;
				
					_t_10191 = -1.0f * tx3_6_1;
					_t_10192 = tx1_4_1 + _t_10191;
					_t_10193 = -1.0f * _t_10192;
					_t_10190 = _t_10193;
				
				}
		
			_t_10194 = _t_10190 * _t_399;
			_t_10195 = _t_10183 * _t_10194;
			_t_10196 = -1.0f * ty1_7_1;
			_t_10197 = ty3_9_1 + _t_10196;
			_t_10198 = -1.0f * _t_10197;
			_t_10199 = _t_10198 < 0.0f;
			if(_t_10199)
				{
					float _t_10200;
					float _t_10201;
				
					_t_10200 = -1.0f * ty1_7_1;
					_t_10201 = ty3_9_1 + _t_10200;
					_t_10202 = _t_10201;
				
				}
		else
				{
					float _t_10203;
					float _t_10204;
					float _t_10205;
				
					_t_10203 = -1.0f * ty1_7_1;
					_t_10204 = ty3_9_1 + _t_10203;
					_t_10205 = -1.0f * _t_10204;
					_t_10202 = _t_10205;
				
				}
		
			_t_10206 = _t_10202 * _t_399;
			_t_10207 = 1.0f + _t_10206;
			_t_10208 = 1.0f / _t_10207;
			_t_10209 = _t_10195 * _t_10208;
			_t_10210 = _t_10209 * -1.0f;
			_t_10211 = 1.0f + _t_10210;
			_t_10212 = -1.0f * ty1_7_1;
			_t_10213 = ty3_9_1 + _t_10212;
			_t_10214 = -1.0f * _t_10213;
			_t_10215 = _t_10214 < 0.0f;
			if(_t_10215)
				{
					float _t_10216;
					float _t_10217;
				
					_t_10216 = -1.0f * tx3_6_1;
					_t_10217 = tx1_4_1 + _t_10216;
					_t_10218 = _t_10217;
				
				}
		else
				{
					float _t_10219;
					float _t_10220;
					float _t_10221;
				
					_t_10219 = -1.0f * tx3_6_1;
					_t_10220 = tx1_4_1 + _t_10219;
					_t_10221 = -1.0f * _t_10220;
					_t_10218 = _t_10221;
				
				}
		
			_t_10222 = _t_10218 * _t_399;
			_t_10223 = -1.0f * ty1_7_1;
			_t_10224 = ty3_9_1 + _t_10223;
			_t_10225 = -1.0f * _t_10224;
			_t_10226 = _t_10225 < 0.0f;
			if(_t_10226)
				{
					float _t_10227;
					float _t_10228;
				
					_t_10227 = -1.0f * tx3_6_1;
					_t_10228 = tx1_4_1 + _t_10227;
					_t_10229 = _t_10228;
				
				}
		else
				{
					float _t_10230;
					float _t_10231;
					float _t_10232;
				
					_t_10230 = -1.0f * tx3_6_1;
					_t_10231 = tx1_4_1 + _t_10230;
					_t_10232 = -1.0f * _t_10231;
					_t_10229 = _t_10232;
				
				}
		
			_t_10233 = _t_10229 * _t_399;
			_t_10234 = _t_10222 * _t_10233;
			_t_10235 = -1.0f * ty1_7_1;
			_t_10236 = ty3_9_1 + _t_10235;
			_t_10237 = -1.0f * _t_10236;
			_t_10238 = _t_10237 < 0.0f;
			if(_t_10238)
				{
					float _t_10239;
					float _t_10240;
				
					_t_10239 = -1.0f * ty1_7_1;
					_t_10240 = ty3_9_1 + _t_10239;
					_t_10241 = _t_10240;
				
				}
		else
				{
					float _t_10242;
					float _t_10243;
					float _t_10244;
				
					_t_10242 = -1.0f * ty1_7_1;
					_t_10243 = ty3_9_1 + _t_10242;
					_t_10244 = -1.0f * _t_10243;
					_t_10241 = _t_10244;
				
				}
		
			_t_10245 = _t_10241 * _t_399;
			_t_10246 = 1.0f + _t_10245;
			_t_10247 = 1.0f / _t_10246;
			_t_10248 = _t_10234 * _t_10247;
			_t_10249 = _t_10248 * -1.0f;
			_t_10250 = 1.0f + _t_10249;
			_t_10251 = 0.0f < _t_10250;
			if(_t_10251)
				{
				
					_t_10252 = py0_12_1;
				
				}
		else
				{
				
					_t_10252 = py1_13_1;
				
				}
		
			_t_10253 = _t_10211 * _t_10252;
			_t_10254 = _t_10172 + _t_10253;
			_t_10255 = _t_10254 < y__3313_1;
			_t_10256 = -1.0f * ty1_7_1;
			_t_10257 = ty3_9_1 + _t_10256;
			_t_10258 = -1.0f * _t_10257;
			_t_10259 = _t_10258 < 0.0f;
			if(_t_10259)
				{
					float _t_10260;
					float _t_10261;
				
					_t_10260 = -1.0f * tx3_6_1;
					_t_10261 = tx1_4_1 + _t_10260;
					_t_10262 = _t_10261;
				
				}
		else
				{
					float _t_10263;
					float _t_10264;
					float _t_10265;
				
					_t_10263 = -1.0f * tx3_6_1;
					_t_10264 = tx1_4_1 + _t_10263;
					_t_10265 = -1.0f * _t_10264;
					_t_10262 = _t_10265;
				
				}
		
			_t_10266 = _t_10262 * _t_399;
			_t_10267 = _t_10266 * -1.0f;
			_t_10268 = -1.0f * ty1_7_1;
			_t_10269 = ty3_9_1 + _t_10268;
			_t_10270 = -1.0f * _t_10269;
			_t_10271 = _t_10270 < 0.0f;
			if(_t_10271)
				{
					float _t_10272;
					float _t_10273;
				
					_t_10272 = -1.0f * tx3_6_1;
					_t_10273 = tx1_4_1 + _t_10272;
					_t_10274 = _t_10273;
				
				}
		else
				{
					float _t_10275;
					float _t_10276;
					float _t_10277;
				
					_t_10275 = -1.0f * tx3_6_1;
					_t_10276 = tx1_4_1 + _t_10275;
					_t_10277 = -1.0f * _t_10276;
					_t_10274 = _t_10277;
				
				}
		
			_t_10278 = _t_10274 * _t_399;
			_t_10279 = _t_10278 * -1.0f;
			_t_10280 = 0.0f < _t_10279;
			if(_t_10280)
				{
				
					_t_10281 = px1_11_1;
				
				}
		else
				{
				
					_t_10281 = px0_10_1;
				
				}
		
			_t_10282 = _t_10267 * _t_10281;
			_t_10283 = -1.0f * ty1_7_1;
			_t_10284 = ty3_9_1 + _t_10283;
			_t_10285 = -1.0f * _t_10284;
			_t_10286 = _t_10285 < 0.0f;
			if(_t_10286)
				{
					float _t_10287;
					float _t_10288;
				
					_t_10287 = -1.0f * tx3_6_1;
					_t_10288 = tx1_4_1 + _t_10287;
					_t_10289 = _t_10288;
				
				}
		else
				{
					float _t_10290;
					float _t_10291;
					float _t_10292;
				
					_t_10290 = -1.0f * tx3_6_1;
					_t_10291 = tx1_4_1 + _t_10290;
					_t_10292 = -1.0f * _t_10291;
					_t_10289 = _t_10292;
				
				}
		
			_t_10293 = _t_10289 * _t_399;
			_t_10294 = -1.0f * ty1_7_1;
			_t_10295 = ty3_9_1 + _t_10294;
			_t_10296 = -1.0f * _t_10295;
			_t_10297 = _t_10296 < 0.0f;
			if(_t_10297)
				{
					float _t_10298;
					float _t_10299;
				
					_t_10298 = -1.0f * tx3_6_1;
					_t_10299 = tx1_4_1 + _t_10298;
					_t_10300 = _t_10299;
				
				}
		else
				{
					float _t_10301;
					float _t_10302;
					float _t_10303;
				
					_t_10301 = -1.0f * tx3_6_1;
					_t_10302 = tx1_4_1 + _t_10301;
					_t_10303 = -1.0f * _t_10302;
					_t_10300 = _t_10303;
				
				}
		
			_t_10304 = _t_10300 * _t_399;
			_t_10305 = _t_10293 * _t_10304;
			_t_10306 = -1.0f * ty1_7_1;
			_t_10307 = ty3_9_1 + _t_10306;
			_t_10308 = -1.0f * _t_10307;
			_t_10309 = _t_10308 < 0.0f;
			if(_t_10309)
				{
					float _t_10310;
					float _t_10311;
				
					_t_10310 = -1.0f * ty1_7_1;
					_t_10311 = ty3_9_1 + _t_10310;
					_t_10312 = _t_10311;
				
				}
		else
				{
					float _t_10313;
					float _t_10314;
					float _t_10315;
				
					_t_10313 = -1.0f * ty1_7_1;
					_t_10314 = ty3_9_1 + _t_10313;
					_t_10315 = -1.0f * _t_10314;
					_t_10312 = _t_10315;
				
				}
		
			_t_10316 = _t_10312 * _t_399;
			_t_10317 = 1.0f + _t_10316;
			_t_10318 = 1.0f / _t_10317;
			_t_10319 = _t_10305 * _t_10318;
			_t_10320 = _t_10319 * -1.0f;
			_t_10321 = 1.0f + _t_10320;
			_t_10322 = -1.0f * ty1_7_1;
			_t_10323 = ty3_9_1 + _t_10322;
			_t_10324 = -1.0f * _t_10323;
			_t_10325 = _t_10324 < 0.0f;
			if(_t_10325)
				{
					float _t_10326;
					float _t_10327;
				
					_t_10326 = -1.0f * tx3_6_1;
					_t_10327 = tx1_4_1 + _t_10326;
					_t_10328 = _t_10327;
				
				}
		else
				{
					float _t_10329;
					float _t_10330;
					float _t_10331;
				
					_t_10329 = -1.0f * tx3_6_1;
					_t_10330 = tx1_4_1 + _t_10329;
					_t_10331 = -1.0f * _t_10330;
					_t_10328 = _t_10331;
				
				}
		
			_t_10332 = _t_10328 * _t_399;
			_t_10333 = -1.0f * ty1_7_1;
			_t_10334 = ty3_9_1 + _t_10333;
			_t_10335 = -1.0f * _t_10334;
			_t_10336 = _t_10335 < 0.0f;
			if(_t_10336)
				{
					float _t_10337;
					float _t_10338;
				
					_t_10337 = -1.0f * tx3_6_1;
					_t_10338 = tx1_4_1 + _t_10337;
					_t_10339 = _t_10338;
				
				}
		else
				{
					float _t_10340;
					float _t_10341;
					float _t_10342;
				
					_t_10340 = -1.0f * tx3_6_1;
					_t_10341 = tx1_4_1 + _t_10340;
					_t_10342 = -1.0f * _t_10341;
					_t_10339 = _t_10342;
				
				}
		
			_t_10343 = _t_10339 * _t_399;
			_t_10344 = _t_10332 * _t_10343;
			_t_10345 = -1.0f * ty1_7_1;
			_t_10346 = ty3_9_1 + _t_10345;
			_t_10347 = -1.0f * _t_10346;
			_t_10348 = _t_10347 < 0.0f;
			if(_t_10348)
				{
					float _t_10349;
					float _t_10350;
				
					_t_10349 = -1.0f * ty1_7_1;
					_t_10350 = ty3_9_1 + _t_10349;
					_t_10351 = _t_10350;
				
				}
		else
				{
					float _t_10352;
					float _t_10353;
					float _t_10354;
				
					_t_10352 = -1.0f * ty1_7_1;
					_t_10353 = ty3_9_1 + _t_10352;
					_t_10354 = -1.0f * _t_10353;
					_t_10351 = _t_10354;
				
				}
		
			_t_10355 = _t_10351 * _t_399;
			_t_10356 = 1.0f + _t_10355;
			_t_10357 = 1.0f / _t_10356;
			_t_10358 = _t_10344 * _t_10357;
			_t_10359 = _t_10358 * -1.0f;
			_t_10360 = 1.0f + _t_10359;
			_t_10361 = 0.0f < _t_10360;
			if(_t_10361)
				{
				
					_t_10362 = py1_13_1;
				
				}
		else
				{
				
					_t_10362 = py0_12_1;
				
				}
		
			_t_10363 = _t_10321 * _t_10362;
			_t_10364 = _t_10282 + _t_10363;
			_t_10365 = y__3313_1 < _t_10364;
			_t_10366 = _t_10255 && _t_10365;
			_t_10367 = -1.0f * ty1_7_1;
			_t_10368 = ty3_9_1 + _t_10367;
			_t_10369 = -1.0f * _t_10368;
			_t_10370 = _t_10369 < 0.0f;
			if(_t_10370)
				{
					float _t_10371;
					float _t_10372;
				
					_t_10371 = -1.0f * ty1_7_1;
					_t_10372 = ty3_9_1 + _t_10371;
					_t_10373 = _t_10372;
				
				}
		else
				{
					float _t_10374;
					float _t_10375;
					float _t_10376;
				
					_t_10374 = -1.0f * ty1_7_1;
					_t_10375 = ty3_9_1 + _t_10374;
					_t_10376 = -1.0f * _t_10375;
					_t_10373 = _t_10376;
				
				}
		
			_t_10377 = _t_10373 * _t_399;
			_t_10378 = -1.0f * ty1_7_1;
			_t_10379 = ty3_9_1 + _t_10378;
			_t_10380 = -1.0f * _t_10379;
			_t_10381 = _t_10380 < 0.0f;
			if(_t_10381)
				{
					float _t_10382;
					float _t_10383;
				
					_t_10382 = -1.0f * ty1_7_1;
					_t_10383 = ty3_9_1 + _t_10382;
					_t_10384 = _t_10383;
				
				}
		else
				{
					float _t_10385;
					float _t_10386;
					float _t_10387;
				
					_t_10385 = -1.0f * ty1_7_1;
					_t_10386 = ty3_9_1 + _t_10385;
					_t_10387 = -1.0f * _t_10386;
					_t_10384 = _t_10387;
				
				}
		
			_t_10388 = _t_10384 * _t_399;
			_t_10389 = 0.0f < _t_10388;
			if(_t_10389)
				{
				
					_t_10390 = px0_10_1;
				
				}
		else
				{
				
					_t_10390 = px1_11_1;
				
				}
		
			_t_10391 = _t_10377 * _t_10390;
			_t_10392 = -1.0f * ty1_7_1;
			_t_10393 = ty3_9_1 + _t_10392;
			_t_10394 = -1.0f * _t_10393;
			_t_10395 = _t_10394 < 0.0f;
			if(_t_10395)
				{
					float _t_10396;
					float _t_10397;
				
					_t_10396 = -1.0f * tx3_6_1;
					_t_10397 = tx1_4_1 + _t_10396;
					_t_10398 = _t_10397;
				
				}
		else
				{
					float _t_10399;
					float _t_10400;
					float _t_10401;
				
					_t_10399 = -1.0f * tx3_6_1;
					_t_10400 = tx1_4_1 + _t_10399;
					_t_10401 = -1.0f * _t_10400;
					_t_10398 = _t_10401;
				
				}
		
			_t_10402 = _t_10398 * _t_399;
			_t_10403 = -1.0f * ty1_7_1;
			_t_10404 = ty3_9_1 + _t_10403;
			_t_10405 = -1.0f * _t_10404;
			_t_10406 = _t_10405 < 0.0f;
			if(_t_10406)
				{
					float _t_10407;
					float _t_10408;
				
					_t_10407 = -1.0f * tx3_6_1;
					_t_10408 = tx1_4_1 + _t_10407;
					_t_10409 = _t_10408;
				
				}
		else
				{
					float _t_10410;
					float _t_10411;
					float _t_10412;
				
					_t_10410 = -1.0f * tx3_6_1;
					_t_10411 = tx1_4_1 + _t_10410;
					_t_10412 = -1.0f * _t_10411;
					_t_10409 = _t_10412;
				
				}
		
			_t_10413 = _t_10409 * _t_399;
			_t_10414 = 0.0f < _t_10413;
			if(_t_10414)
				{
				
					_t_10415 = py0_12_1;
				
				}
		else
				{
				
					_t_10415 = py1_13_1;
				
				}
		
			_t_10416 = _t_10402 * _t_10415;
			_t_10417 = _t_10391 + _t_10416;
			_t_10418 = _t_10417 < _t_10015;
			_t_10419 = -1.0f * ty1_7_1;
			_t_10420 = ty3_9_1 + _t_10419;
			_t_10421 = -1.0f * _t_10420;
			_t_10422 = _t_10421 < 0.0f;
			if(_t_10422)
				{
					float _t_10423;
					float _t_10424;
				
					_t_10423 = -1.0f * ty1_7_1;
					_t_10424 = ty3_9_1 + _t_10423;
					_t_10425 = _t_10424;
				
				}
		else
				{
					float _t_10426;
					float _t_10427;
					float _t_10428;
				
					_t_10426 = -1.0f * ty1_7_1;
					_t_10427 = ty3_9_1 + _t_10426;
					_t_10428 = -1.0f * _t_10427;
					_t_10425 = _t_10428;
				
				}
		
			_t_10429 = _t_10425 * _t_399;
			_t_10430 = -1.0f * ty1_7_1;
			_t_10431 = ty3_9_1 + _t_10430;
			_t_10432 = -1.0f * _t_10431;
			_t_10433 = _t_10432 < 0.0f;
			if(_t_10433)
				{
					float _t_10434;
					float _t_10435;
				
					_t_10434 = -1.0f * ty1_7_1;
					_t_10435 = ty3_9_1 + _t_10434;
					_t_10436 = _t_10435;
				
				}
		else
				{
					float _t_10437;
					float _t_10438;
					float _t_10439;
				
					_t_10437 = -1.0f * ty1_7_1;
					_t_10438 = ty3_9_1 + _t_10437;
					_t_10439 = -1.0f * _t_10438;
					_t_10436 = _t_10439;
				
				}
		
			_t_10440 = _t_10436 * _t_399;
			_t_10441 = 0.0f < _t_10440;
			if(_t_10441)
				{
				
					_t_10442 = px1_11_1;
				
				}
		else
				{
				
					_t_10442 = px0_10_1;
				
				}
		
			_t_10443 = _t_10429 * _t_10442;
			_t_10444 = -1.0f * ty1_7_1;
			_t_10445 = ty3_9_1 + _t_10444;
			_t_10446 = -1.0f * _t_10445;
			_t_10447 = _t_10446 < 0.0f;
			if(_t_10447)
				{
					float _t_10448;
					float _t_10449;
				
					_t_10448 = -1.0f * tx3_6_1;
					_t_10449 = tx1_4_1 + _t_10448;
					_t_10450 = _t_10449;
				
				}
		else
				{
					float _t_10451;
					float _t_10452;
					float _t_10453;
				
					_t_10451 = -1.0f * tx3_6_1;
					_t_10452 = tx1_4_1 + _t_10451;
					_t_10453 = -1.0f * _t_10452;
					_t_10450 = _t_10453;
				
				}
		
			_t_10454 = _t_10450 * _t_399;
			_t_10455 = -1.0f * ty1_7_1;
			_t_10456 = ty3_9_1 + _t_10455;
			_t_10457 = -1.0f * _t_10456;
			_t_10458 = _t_10457 < 0.0f;
			if(_t_10458)
				{
					float _t_10459;
					float _t_10460;
				
					_t_10459 = -1.0f * tx3_6_1;
					_t_10460 = tx1_4_1 + _t_10459;
					_t_10461 = _t_10460;
				
				}
		else
				{
					float _t_10462;
					float _t_10463;
					float _t_10464;
				
					_t_10462 = -1.0f * tx3_6_1;
					_t_10463 = tx1_4_1 + _t_10462;
					_t_10464 = -1.0f * _t_10463;
					_t_10461 = _t_10464;
				
				}
		
			_t_10465 = _t_10461 * _t_399;
			_t_10466 = 0.0f < _t_10465;
			if(_t_10466)
				{
				
					_t_10467 = py1_13_1;
				
				}
		else
				{
				
					_t_10467 = py0_12_1;
				
				}
		
			_t_10468 = _t_10454 * _t_10467;
			_t_10469 = _t_10443 + _t_10468;
			_t_10470 = _t_10015 < _t_10469;
			_t_10471 = _t_10418 && _t_10470;
			_t_10472 = _t_10366 && _t_10471;
			if(_t_10472)
				{
				
					_t_10473 = 1.0f;
				
				}
		else
				{
				
					_t_10473 = 0.0f;
				
				}
		
			_t_10474 = _t_10473 * _t_399;
			_t_10475 = _t_10474;
		
		}
else
		{
		
			_t_10475 = 0.0f;
		
		}

	_t_10016 = _t_10138 * _t_10475;

	return _t_10016;
}
__device__ float tegpixellet_block_39(float ty3_9_1,float ty1_7_1,float _t_399,float _t_10015,float tx1_4_1,float tx3_6_1,float y__3313_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1){
	float _t_10017;
	float _t_10018;
	float _t_10019;
	bool _t_10020;
	float _t_10023;
	float _t_10027;
	float _t_10028;
	float _t_10029;
	float _t_10030;
	float _t_10031;
	bool _t_10032;
	float _t_10035;
	float _t_10039;
	float _t_10040;
	float _t_10041;
	float _t_10042;
	float _t_10043;
	float _t_10044;
	float _t_10045;
	bool _t_10046;
	float _t_10049;
	float _t_10053;
	float _t_10054;
	float _t_10055;
	float _t_10056;
	bool _t_10057;
	float _t_10060;
	float _t_10064;
	float _t_10065;
	float _t_10066;
	float _t_10067;
	float _t_10068;
	bool _t_10069;
	float _t_10072;
	float _t_10076;
	float _t_10077;
	float _t_10078;
	float _t_10079;
	float _t_10080;
	float _t_10081;
	float _t_10082;
	float _t_10083;
	float _t_10084;
	float _t_10085;
	bool _t_10086;
	float _t_10089;
	float _t_10093;
	float _t_10094;
	float _t_10095;

	float _t_10016;

	_t_10017 = -1.0f * ty1_7_1;
	_t_10018 = ty3_9_1 + _t_10017;
	_t_10019 = -1.0f * _t_10018;
	_t_10020 = _t_10019 < 0.0f;
	if(_t_10020)
		{
			float _t_10021;
			float _t_10022;
		
			_t_10021 = -1.0f * ty1_7_1;
			_t_10022 = ty3_9_1 + _t_10021;
			_t_10023 = _t_10022;
		
		}
else
		{
			float _t_10024;
			float _t_10025;
			float _t_10026;
		
			_t_10024 = -1.0f * ty1_7_1;
			_t_10025 = ty3_9_1 + _t_10024;
			_t_10026 = -1.0f * _t_10025;
			_t_10023 = _t_10026;
		
		}

	_t_10027 = _t_10023 * _t_399;
	_t_10028 = _t_10027 * _t_10015;
	_t_10029 = -1.0f * ty1_7_1;
	_t_10030 = ty3_9_1 + _t_10029;
	_t_10031 = -1.0f * _t_10030;
	_t_10032 = _t_10031 < 0.0f;
	if(_t_10032)
		{
			float _t_10033;
			float _t_10034;
		
			_t_10033 = -1.0f * tx3_6_1;
			_t_10034 = tx1_4_1 + _t_10033;
			_t_10035 = _t_10034;
		
		}
else
		{
			float _t_10036;
			float _t_10037;
			float _t_10038;
		
			_t_10036 = -1.0f * tx3_6_1;
			_t_10037 = tx1_4_1 + _t_10036;
			_t_10038 = -1.0f * _t_10037;
			_t_10035 = _t_10038;
		
		}

	_t_10039 = _t_10035 * _t_399;
	_t_10040 = _t_10039 * -1.0f;
	_t_10041 = _t_10040 * y__3313_1;
	_t_10042 = _t_10028 + _t_10041;
	_t_10043 = -1.0f * ty1_7_1;
	_t_10044 = ty3_9_1 + _t_10043;
	_t_10045 = -1.0f * _t_10044;
	_t_10046 = _t_10045 < 0.0f;
	if(_t_10046)
		{
			float _t_10047;
			float _t_10048;
		
			_t_10047 = -1.0f * tx3_6_1;
			_t_10048 = tx1_4_1 + _t_10047;
			_t_10049 = _t_10048;
		
		}
else
		{
			float _t_10050;
			float _t_10051;
			float _t_10052;
		
			_t_10050 = -1.0f * tx3_6_1;
			_t_10051 = tx1_4_1 + _t_10050;
			_t_10052 = -1.0f * _t_10051;
			_t_10049 = _t_10052;
		
		}

	_t_10053 = _t_10049 * _t_399;
	_t_10054 = -1.0f * ty1_7_1;
	_t_10055 = ty3_9_1 + _t_10054;
	_t_10056 = -1.0f * _t_10055;
	_t_10057 = _t_10056 < 0.0f;
	if(_t_10057)
		{
			float _t_10058;
			float _t_10059;
		
			_t_10058 = -1.0f * tx3_6_1;
			_t_10059 = tx1_4_1 + _t_10058;
			_t_10060 = _t_10059;
		
		}
else
		{
			float _t_10061;
			float _t_10062;
			float _t_10063;
		
			_t_10061 = -1.0f * tx3_6_1;
			_t_10062 = tx1_4_1 + _t_10061;
			_t_10063 = -1.0f * _t_10062;
			_t_10060 = _t_10063;
		
		}

	_t_10064 = _t_10060 * _t_399;
	_t_10065 = _t_10053 * _t_10064;
	_t_10066 = -1.0f * ty1_7_1;
	_t_10067 = ty3_9_1 + _t_10066;
	_t_10068 = -1.0f * _t_10067;
	_t_10069 = _t_10068 < 0.0f;
	if(_t_10069)
		{
			float _t_10070;
			float _t_10071;
		
			_t_10070 = -1.0f * ty1_7_1;
			_t_10071 = ty3_9_1 + _t_10070;
			_t_10072 = _t_10071;
		
		}
else
		{
			float _t_10073;
			float _t_10074;
			float _t_10075;
		
			_t_10073 = -1.0f * ty1_7_1;
			_t_10074 = ty3_9_1 + _t_10073;
			_t_10075 = -1.0f * _t_10074;
			_t_10072 = _t_10075;
		
		}

	_t_10076 = _t_10072 * _t_399;
	_t_10077 = 1.0f + _t_10076;
	_t_10078 = 1.0f / _t_10077;
	_t_10079 = _t_10065 * _t_10078;
	_t_10080 = _t_10079 * -1.0f;
	_t_10081 = 1.0f + _t_10080;
	_t_10082 = _t_10081 * y__3313_1;
	_t_10083 = -1.0f * ty1_7_1;
	_t_10084 = ty3_9_1 + _t_10083;
	_t_10085 = -1.0f * _t_10084;
	_t_10086 = _t_10085 < 0.0f;
	if(_t_10086)
		{
			float _t_10087;
			float _t_10088;
		
			_t_10087 = -1.0f * tx3_6_1;
			_t_10088 = tx1_4_1 + _t_10087;
			_t_10089 = _t_10088;
		
		}
else
		{
			float _t_10090;
			float _t_10091;
			float _t_10092;
		
			_t_10090 = -1.0f * tx3_6_1;
			_t_10091 = tx1_4_1 + _t_10090;
			_t_10092 = -1.0f * _t_10091;
			_t_10089 = _t_10092;
		
		}

	_t_10093 = _t_10089 * _t_399;
	_t_10094 = _t_10093 * _t_10015;
	_t_10095 = _t_10082 + _t_10094;
	_t_10016 = tegpixellet_block_40(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,_t_10042,_t_10095,ty3_9_1,tx3_6_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,_t_399,y__3313_1,_t_10015);

	return _t_10016;
}
__device__ float tegpixelbody_block_27(float ty3_9_1,float ty1_7_1,float _t_399,float px0_10_1,float px1_11_1,float tx1_4_1,float tx3_6_1,float py0_12_1,float py1_13_1,float y__3313_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_9859;
	float _t_9860;
	float _t_9861;
	bool _t_9862;
	float _t_9865;
	float _t_9869;
	float _t_9870;
	float _t_9871;
	float _t_9872;
	bool _t_9873;
	float _t_9876;
	float _t_9880;
	bool _t_9881;
	float _t_9882;
	float _t_9883;
	float _t_9884;
	float _t_9885;
	float _t_9886;
	bool _t_9887;
	float _t_9890;
	float _t_9894;
	float _t_9895;
	float _t_9896;
	float _t_9897;
	bool _t_9898;
	float _t_9901;
	float _t_9905;
	bool _t_9906;
	float _t_9907;
	float _t_9908;
	float _t_9909;
	float _t_9910;
	float _t_9911;
	float _t_9912;
	bool _t_9913;
	float _t_9918;
	float _t_9924;
	float _t_9925;
	float _t_9926;
	float _t_9927;
	bool _t_9928;
	float _t_9929;
	float _t_9930;
	float _t_9931;
	bool _t_9932;
	float _t_9935;
	float _t_9939;
	float _t_9940;
	float _t_9941;
	float _t_9942;
	bool _t_9943;
	float _t_9946;
	float _t_9950;
	bool _t_9951;
	float _t_9952;
	float _t_9953;
	float _t_9954;
	float _t_9955;
	float _t_9956;
	bool _t_9957;
	float _t_9960;
	float _t_9964;
	float _t_9965;
	float _t_9966;
	float _t_9967;
	bool _t_9968;
	float _t_9971;
	float _t_9975;
	bool _t_9976;
	float _t_9977;
	float _t_9978;
	float _t_9979;
	float _t_9980;
	float _t_9981;
	float _t_9982;
	bool _t_9983;
	float _t_9988;
	float _t_9994;
	float _t_9995;
	float _t_9996;
	float _t_9997;
	bool _t_9998;
	bool _t_9999;

	float _t_9858;

	_t_9859 = -1.0f * ty1_7_1;
	_t_9860 = ty3_9_1 + _t_9859;
	_t_9861 = -1.0f * _t_9860;
	_t_9862 = _t_9861 < 0.0f;
	if(_t_9862)
		{
			float _t_9863;
			float _t_9864;
		
			_t_9863 = -1.0f * ty1_7_1;
			_t_9864 = ty3_9_1 + _t_9863;
			_t_9865 = _t_9864;
		
		}
else
		{
			float _t_9866;
			float _t_9867;
			float _t_9868;
		
			_t_9866 = -1.0f * ty1_7_1;
			_t_9867 = ty3_9_1 + _t_9866;
			_t_9868 = -1.0f * _t_9867;
			_t_9865 = _t_9868;
		
		}

	_t_9869 = _t_9865 * _t_399;
	_t_9870 = -1.0f * ty1_7_1;
	_t_9871 = ty3_9_1 + _t_9870;
	_t_9872 = -1.0f * _t_9871;
	_t_9873 = _t_9872 < 0.0f;
	if(_t_9873)
		{
			float _t_9874;
			float _t_9875;
		
			_t_9874 = -1.0f * ty1_7_1;
			_t_9875 = ty3_9_1 + _t_9874;
			_t_9876 = _t_9875;
		
		}
else
		{
			float _t_9877;
			float _t_9878;
			float _t_9879;
		
			_t_9877 = -1.0f * ty1_7_1;
			_t_9878 = ty3_9_1 + _t_9877;
			_t_9879 = -1.0f * _t_9878;
			_t_9876 = _t_9879;
		
		}

	_t_9880 = _t_9876 * _t_399;
	_t_9881 = 0.0f < _t_9880;
	if(_t_9881)
		{
		
			_t_9882 = px0_10_1;
		
		}
else
		{
		
			_t_9882 = px1_11_1;
		
		}

	_t_9883 = _t_9869 * _t_9882;
	_t_9884 = -1.0f * ty1_7_1;
	_t_9885 = ty3_9_1 + _t_9884;
	_t_9886 = -1.0f * _t_9885;
	_t_9887 = _t_9886 < 0.0f;
	if(_t_9887)
		{
			float _t_9888;
			float _t_9889;
		
			_t_9888 = -1.0f * tx3_6_1;
			_t_9889 = tx1_4_1 + _t_9888;
			_t_9890 = _t_9889;
		
		}
else
		{
			float _t_9891;
			float _t_9892;
			float _t_9893;
		
			_t_9891 = -1.0f * tx3_6_1;
			_t_9892 = tx1_4_1 + _t_9891;
			_t_9893 = -1.0f * _t_9892;
			_t_9890 = _t_9893;
		
		}

	_t_9894 = _t_9890 * _t_399;
	_t_9895 = -1.0f * ty1_7_1;
	_t_9896 = ty3_9_1 + _t_9895;
	_t_9897 = -1.0f * _t_9896;
	_t_9898 = _t_9897 < 0.0f;
	if(_t_9898)
		{
			float _t_9899;
			float _t_9900;
		
			_t_9899 = -1.0f * tx3_6_1;
			_t_9900 = tx1_4_1 + _t_9899;
			_t_9901 = _t_9900;
		
		}
else
		{
			float _t_9902;
			float _t_9903;
			float _t_9904;
		
			_t_9902 = -1.0f * tx3_6_1;
			_t_9903 = tx1_4_1 + _t_9902;
			_t_9904 = -1.0f * _t_9903;
			_t_9901 = _t_9904;
		
		}

	_t_9905 = _t_9901 * _t_399;
	_t_9906 = 0.0f < _t_9905;
	if(_t_9906)
		{
		
			_t_9907 = py0_12_1;
		
		}
else
		{
		
			_t_9907 = py1_13_1;
		
		}

	_t_9908 = _t_9894 * _t_9907;
	_t_9909 = _t_9883 + _t_9908;
	_t_9910 = -1.0f * ty1_7_1;
	_t_9911 = ty3_9_1 + _t_9910;
	_t_9912 = -1.0f * _t_9911;
	_t_9913 = _t_9912 < 0.0f;
	if(_t_9913)
		{
			float _t_9914;
			float _t_9915;
			float _t_9916;
			float _t_9917;
		
			_t_9914 = tx3_6_1 * ty1_7_1;
			_t_9915 = tx1_4_1 * ty3_9_1;
			_t_9916 = _t_9915 * -1.0f;
			_t_9917 = _t_9914 + _t_9916;
			_t_9918 = _t_9917;
		
		}
else
		{
			float _t_9919;
			float _t_9920;
			float _t_9921;
			float _t_9922;
			float _t_9923;
		
			_t_9919 = tx3_6_1 * ty1_7_1;
			_t_9920 = tx1_4_1 * ty3_9_1;
			_t_9921 = _t_9920 * -1.0f;
			_t_9922 = _t_9919 + _t_9921;
			_t_9923 = -1.0f * _t_9922;
			_t_9918 = _t_9923;
		
		}

	_t_9924 = -1.0f * _t_9918;
	_t_9925 = _t_9924 * _t_399;
	_t_9926 = _t_9925 * -1.0f;
	_t_9927 = _t_9909 + _t_9926;
	_t_9928 = _t_9927 < 0.0f;
	_t_9929 = -1.0f * ty1_7_1;
	_t_9930 = ty3_9_1 + _t_9929;
	_t_9931 = -1.0f * _t_9930;
	_t_9932 = _t_9931 < 0.0f;
	if(_t_9932)
		{
			float _t_9933;
			float _t_9934;
		
			_t_9933 = -1.0f * ty1_7_1;
			_t_9934 = ty3_9_1 + _t_9933;
			_t_9935 = _t_9934;
		
		}
else
		{
			float _t_9936;
			float _t_9937;
			float _t_9938;
		
			_t_9936 = -1.0f * ty1_7_1;
			_t_9937 = ty3_9_1 + _t_9936;
			_t_9938 = -1.0f * _t_9937;
			_t_9935 = _t_9938;
		
		}

	_t_9939 = _t_9935 * _t_399;
	_t_9940 = -1.0f * ty1_7_1;
	_t_9941 = ty3_9_1 + _t_9940;
	_t_9942 = -1.0f * _t_9941;
	_t_9943 = _t_9942 < 0.0f;
	if(_t_9943)
		{
			float _t_9944;
			float _t_9945;
		
			_t_9944 = -1.0f * ty1_7_1;
			_t_9945 = ty3_9_1 + _t_9944;
			_t_9946 = _t_9945;
		
		}
else
		{
			float _t_9947;
			float _t_9948;
			float _t_9949;
		
			_t_9947 = -1.0f * ty1_7_1;
			_t_9948 = ty3_9_1 + _t_9947;
			_t_9949 = -1.0f * _t_9948;
			_t_9946 = _t_9949;
		
		}

	_t_9950 = _t_9946 * _t_399;
	_t_9951 = 0.0f < _t_9950;
	if(_t_9951)
		{
		
			_t_9952 = px1_11_1;
		
		}
else
		{
		
			_t_9952 = px0_10_1;
		
		}

	_t_9953 = _t_9939 * _t_9952;
	_t_9954 = -1.0f * ty1_7_1;
	_t_9955 = ty3_9_1 + _t_9954;
	_t_9956 = -1.0f * _t_9955;
	_t_9957 = _t_9956 < 0.0f;
	if(_t_9957)
		{
			float _t_9958;
			float _t_9959;
		
			_t_9958 = -1.0f * tx3_6_1;
			_t_9959 = tx1_4_1 + _t_9958;
			_t_9960 = _t_9959;
		
		}
else
		{
			float _t_9961;
			float _t_9962;
			float _t_9963;
		
			_t_9961 = -1.0f * tx3_6_1;
			_t_9962 = tx1_4_1 + _t_9961;
			_t_9963 = -1.0f * _t_9962;
			_t_9960 = _t_9963;
		
		}

	_t_9964 = _t_9960 * _t_399;
	_t_9965 = -1.0f * ty1_7_1;
	_t_9966 = ty3_9_1 + _t_9965;
	_t_9967 = -1.0f * _t_9966;
	_t_9968 = _t_9967 < 0.0f;
	if(_t_9968)
		{
			float _t_9969;
			float _t_9970;
		
			_t_9969 = -1.0f * tx3_6_1;
			_t_9970 = tx1_4_1 + _t_9969;
			_t_9971 = _t_9970;
		
		}
else
		{
			float _t_9972;
			float _t_9973;
			float _t_9974;
		
			_t_9972 = -1.0f * tx3_6_1;
			_t_9973 = tx1_4_1 + _t_9972;
			_t_9974 = -1.0f * _t_9973;
			_t_9971 = _t_9974;
		
		}

	_t_9975 = _t_9971 * _t_399;
	_t_9976 = 0.0f < _t_9975;
	if(_t_9976)
		{
		
			_t_9977 = py1_13_1;
		
		}
else
		{
		
			_t_9977 = py0_12_1;
		
		}

	_t_9978 = _t_9964 * _t_9977;
	_t_9979 = _t_9953 + _t_9978;
	_t_9980 = -1.0f * ty1_7_1;
	_t_9981 = ty3_9_1 + _t_9980;
	_t_9982 = -1.0f * _t_9981;
	_t_9983 = _t_9982 < 0.0f;
	if(_t_9983)
		{
			float _t_9984;
			float _t_9985;
			float _t_9986;
			float _t_9987;
		
			_t_9984 = tx3_6_1 * ty1_7_1;
			_t_9985 = tx1_4_1 * ty3_9_1;
			_t_9986 = _t_9985 * -1.0f;
			_t_9987 = _t_9984 + _t_9986;
			_t_9988 = _t_9987;
		
		}
else
		{
			float _t_9989;
			float _t_9990;
			float _t_9991;
			float _t_9992;
			float _t_9993;
		
			_t_9989 = tx3_6_1 * ty1_7_1;
			_t_9990 = tx1_4_1 * ty3_9_1;
			_t_9991 = _t_9990 * -1.0f;
			_t_9992 = _t_9989 + _t_9991;
			_t_9993 = -1.0f * _t_9992;
			_t_9988 = _t_9993;
		
		}

	_t_9994 = -1.0f * _t_9988;
	_t_9995 = _t_9994 * _t_399;
	_t_9996 = _t_9995 * -1.0f;
	_t_9997 = _t_9979 + _t_9996;
	_t_9998 = 0.0f < _t_9997;
	_t_9999 = _t_9928 && _t_9998;
	if(_t_9999)
		{
			float _t_10000;
			float _t_10001;
			float _t_10002;
			bool _t_10003;
			float _t_10008;
			float _t_10014;
			float _t_10015;
			float _t_10016;
		
			_t_10000 = -1.0f * ty1_7_1;
			_t_10001 = ty3_9_1 + _t_10000;
			_t_10002 = -1.0f * _t_10001;
			_t_10003 = _t_10002 < 0.0f;
			if(_t_10003)
				{
					float _t_10004;
					float _t_10005;
					float _t_10006;
					float _t_10007;
				
					_t_10004 = tx3_6_1 * ty1_7_1;
					_t_10005 = tx1_4_1 * ty3_9_1;
					_t_10006 = _t_10005 * -1.0f;
					_t_10007 = _t_10004 + _t_10006;
					_t_10008 = _t_10007;
				
				}
		else
				{
					float _t_10009;
					float _t_10010;
					float _t_10011;
					float _t_10012;
					float _t_10013;
				
					_t_10009 = tx3_6_1 * ty1_7_1;
					_t_10010 = tx1_4_1 * ty3_9_1;
					_t_10011 = _t_10010 * -1.0f;
					_t_10012 = _t_10009 + _t_10011;
					_t_10013 = -1.0f * _t_10012;
					_t_10008 = _t_10013;
				
				}
		
			_t_10014 = -1.0f * _t_10008;
			_t_10015 = _t_10014 * _t_399;
			_t_10016 = tegpixellet_block_39(ty3_9_1,ty1_7_1,_t_399,_t_10015,tx1_4_1,tx3_6_1,y__3313_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1);
			_t_9858 = _t_10016;
		
		}
else
		{
		
			_t_9858 = 0.0f;
		
		}


	return _t_9858;
}
__device__ float tegpixelintegrator_27(float ty3_9_1,float pc1_15_1,float _t_9748,float _t_399,float tc2_19_1,float ty2_8_1,float pc0_14_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float py1_13_1,float pc2_16_1,float tx2_5_1,float _t_9857,float px1_11_1,float tc0_17_1,float py0_12_1,float tc1_18_1,float px0_10_1){
    float y__3313_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_9857 - _t_9748)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__3313_1 = _t_9748 + __step__ * (i + (float)(0.5));
        float _t_9858;
		_t_9858 = tegpixelbody_block_27(ty3_9_1,ty1_7_1,_t_399,px0_10_1,px1_11_1,tx1_4_1,tx3_6_1,py0_12_1,py1_13_1,y__3313_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);;
        __output__ = __output__ + _t_9858 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_11(float ty3_9_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float _t_399,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_9640;
	float _t_9641;
	float _t_9642;
	bool _t_9643;
	float _t_9646;
	float _t_9650;
	float _t_9651;
	float _t_9652;
	float _t_9653;
	float _t_9654;
	bool _t_9655;
	float _t_9658;
	float _t_9662;
	float _t_9663;
	bool _t_9664;
	float _t_9665;
	float _t_9666;
	float _t_9667;
	float _t_9668;
	float _t_9669;
	bool _t_9670;
	float _t_9673;
	float _t_9677;
	float _t_9678;
	float _t_9679;
	float _t_9680;
	bool _t_9681;
	float _t_9684;
	float _t_9688;
	float _t_9689;
	float _t_9690;
	float _t_9691;
	float _t_9692;
	bool _t_9693;
	float _t_9696;
	float _t_9700;
	float _t_9701;
	float _t_9702;
	float _t_9703;
	float _t_9704;
	float _t_9705;
	float _t_9706;
	float _t_9707;
	float _t_9708;
	bool _t_9709;
	float _t_9712;
	float _t_9716;
	float _t_9717;
	float _t_9718;
	float _t_9719;
	bool _t_9720;
	float _t_9723;
	float _t_9727;
	float _t_9728;
	float _t_9729;
	float _t_9730;
	float _t_9731;
	bool _t_9732;
	float _t_9735;
	float _t_9739;
	float _t_9740;
	float _t_9741;
	float _t_9742;
	float _t_9743;
	float _t_9744;
	bool _t_9745;
	float _t_9746;
	float _t_9747;
	float _t_9748;
	float _t_9749;
	float _t_9750;
	float _t_9751;
	bool _t_9752;
	float _t_9755;
	float _t_9759;
	float _t_9760;
	float _t_9761;
	float _t_9762;
	float _t_9763;
	bool _t_9764;
	float _t_9767;
	float _t_9771;
	float _t_9772;
	bool _t_9773;
	float _t_9774;
	float _t_9775;
	float _t_9776;
	float _t_9777;
	float _t_9778;
	bool _t_9779;
	float _t_9782;
	float _t_9786;
	float _t_9787;
	float _t_9788;
	float _t_9789;
	bool _t_9790;
	float _t_9793;
	float _t_9797;
	float _t_9798;
	float _t_9799;
	float _t_9800;
	float _t_9801;
	bool _t_9802;
	float _t_9805;
	float _t_9809;
	float _t_9810;
	float _t_9811;
	float _t_9812;
	float _t_9813;
	float _t_9814;
	float _t_9815;
	float _t_9816;
	float _t_9817;
	bool _t_9818;
	float _t_9821;
	float _t_9825;
	float _t_9826;
	float _t_9827;
	float _t_9828;
	bool _t_9829;
	float _t_9832;
	float _t_9836;
	float _t_9837;
	float _t_9838;
	float _t_9839;
	float _t_9840;
	bool _t_9841;
	float _t_9844;
	float _t_9848;
	float _t_9849;
	float _t_9850;
	float _t_9851;
	float _t_9852;
	float _t_9853;
	bool _t_9854;
	float _t_9855;
	float _t_9856;
	float _t_9857;

	float _t_400;

	_t_9640 = -1.0f * ty1_7_1;
	_t_9641 = ty3_9_1 + _t_9640;
	_t_9642 = -1.0f * _t_9641;
	_t_9643 = _t_9642 < 0.0f;
	if(_t_9643)
		{
			float _t_9644;
			float _t_9645;
		
			_t_9644 = -1.0f * tx3_6_1;
			_t_9645 = tx1_4_1 + _t_9644;
			_t_9646 = _t_9645;
		
		}
else
		{
			float _t_9647;
			float _t_9648;
			float _t_9649;
		
			_t_9647 = -1.0f * tx3_6_1;
			_t_9648 = tx1_4_1 + _t_9647;
			_t_9649 = -1.0f * _t_9648;
			_t_9646 = _t_9649;
		
		}

	_t_9650 = _t_9646 * _t_399;
	_t_9651 = _t_9650 * -1.0f;
	_t_9652 = -1.0f * ty1_7_1;
	_t_9653 = ty3_9_1 + _t_9652;
	_t_9654 = -1.0f * _t_9653;
	_t_9655 = _t_9654 < 0.0f;
	if(_t_9655)
		{
			float _t_9656;
			float _t_9657;
		
			_t_9656 = -1.0f * tx3_6_1;
			_t_9657 = tx1_4_1 + _t_9656;
			_t_9658 = _t_9657;
		
		}
else
		{
			float _t_9659;
			float _t_9660;
			float _t_9661;
		
			_t_9659 = -1.0f * tx3_6_1;
			_t_9660 = tx1_4_1 + _t_9659;
			_t_9661 = -1.0f * _t_9660;
			_t_9658 = _t_9661;
		
		}

	_t_9662 = _t_9658 * _t_399;
	_t_9663 = _t_9662 * -1.0f;
	_t_9664 = 0.0f < _t_9663;
	if(_t_9664)
		{
		
			_t_9665 = px0_10_1;
		
		}
else
		{
		
			_t_9665 = px1_11_1;
		
		}

	_t_9666 = _t_9651 * _t_9665;
	_t_9667 = -1.0f * ty1_7_1;
	_t_9668 = ty3_9_1 + _t_9667;
	_t_9669 = -1.0f * _t_9668;
	_t_9670 = _t_9669 < 0.0f;
	if(_t_9670)
		{
			float _t_9671;
			float _t_9672;
		
			_t_9671 = -1.0f * tx3_6_1;
			_t_9672 = tx1_4_1 + _t_9671;
			_t_9673 = _t_9672;
		
		}
else
		{
			float _t_9674;
			float _t_9675;
			float _t_9676;
		
			_t_9674 = -1.0f * tx3_6_1;
			_t_9675 = tx1_4_1 + _t_9674;
			_t_9676 = -1.0f * _t_9675;
			_t_9673 = _t_9676;
		
		}

	_t_9677 = _t_9673 * _t_399;
	_t_9678 = -1.0f * ty1_7_1;
	_t_9679 = ty3_9_1 + _t_9678;
	_t_9680 = -1.0f * _t_9679;
	_t_9681 = _t_9680 < 0.0f;
	if(_t_9681)
		{
			float _t_9682;
			float _t_9683;
		
			_t_9682 = -1.0f * tx3_6_1;
			_t_9683 = tx1_4_1 + _t_9682;
			_t_9684 = _t_9683;
		
		}
else
		{
			float _t_9685;
			float _t_9686;
			float _t_9687;
		
			_t_9685 = -1.0f * tx3_6_1;
			_t_9686 = tx1_4_1 + _t_9685;
			_t_9687 = -1.0f * _t_9686;
			_t_9684 = _t_9687;
		
		}

	_t_9688 = _t_9684 * _t_399;
	_t_9689 = _t_9677 * _t_9688;
	_t_9690 = -1.0f * ty1_7_1;
	_t_9691 = ty3_9_1 + _t_9690;
	_t_9692 = -1.0f * _t_9691;
	_t_9693 = _t_9692 < 0.0f;
	if(_t_9693)
		{
			float _t_9694;
			float _t_9695;
		
			_t_9694 = -1.0f * ty1_7_1;
			_t_9695 = ty3_9_1 + _t_9694;
			_t_9696 = _t_9695;
		
		}
else
		{
			float _t_9697;
			float _t_9698;
			float _t_9699;
		
			_t_9697 = -1.0f * ty1_7_1;
			_t_9698 = ty3_9_1 + _t_9697;
			_t_9699 = -1.0f * _t_9698;
			_t_9696 = _t_9699;
		
		}

	_t_9700 = _t_9696 * _t_399;
	_t_9701 = 1.0f + _t_9700;
	_t_9702 = 1.0f / _t_9701;
	_t_9703 = _t_9689 * _t_9702;
	_t_9704 = _t_9703 * -1.0f;
	_t_9705 = 1.0f + _t_9704;
	_t_9706 = -1.0f * ty1_7_1;
	_t_9707 = ty3_9_1 + _t_9706;
	_t_9708 = -1.0f * _t_9707;
	_t_9709 = _t_9708 < 0.0f;
	if(_t_9709)
		{
			float _t_9710;
			float _t_9711;
		
			_t_9710 = -1.0f * tx3_6_1;
			_t_9711 = tx1_4_1 + _t_9710;
			_t_9712 = _t_9711;
		
		}
else
		{
			float _t_9713;
			float _t_9714;
			float _t_9715;
		
			_t_9713 = -1.0f * tx3_6_1;
			_t_9714 = tx1_4_1 + _t_9713;
			_t_9715 = -1.0f * _t_9714;
			_t_9712 = _t_9715;
		
		}

	_t_9716 = _t_9712 * _t_399;
	_t_9717 = -1.0f * ty1_7_1;
	_t_9718 = ty3_9_1 + _t_9717;
	_t_9719 = -1.0f * _t_9718;
	_t_9720 = _t_9719 < 0.0f;
	if(_t_9720)
		{
			float _t_9721;
			float _t_9722;
		
			_t_9721 = -1.0f * tx3_6_1;
			_t_9722 = tx1_4_1 + _t_9721;
			_t_9723 = _t_9722;
		
		}
else
		{
			float _t_9724;
			float _t_9725;
			float _t_9726;
		
			_t_9724 = -1.0f * tx3_6_1;
			_t_9725 = tx1_4_1 + _t_9724;
			_t_9726 = -1.0f * _t_9725;
			_t_9723 = _t_9726;
		
		}

	_t_9727 = _t_9723 * _t_399;
	_t_9728 = _t_9716 * _t_9727;
	_t_9729 = -1.0f * ty1_7_1;
	_t_9730 = ty3_9_1 + _t_9729;
	_t_9731 = -1.0f * _t_9730;
	_t_9732 = _t_9731 < 0.0f;
	if(_t_9732)
		{
			float _t_9733;
			float _t_9734;
		
			_t_9733 = -1.0f * ty1_7_1;
			_t_9734 = ty3_9_1 + _t_9733;
			_t_9735 = _t_9734;
		
		}
else
		{
			float _t_9736;
			float _t_9737;
			float _t_9738;
		
			_t_9736 = -1.0f * ty1_7_1;
			_t_9737 = ty3_9_1 + _t_9736;
			_t_9738 = -1.0f * _t_9737;
			_t_9735 = _t_9738;
		
		}

	_t_9739 = _t_9735 * _t_399;
	_t_9740 = 1.0f + _t_9739;
	_t_9741 = 1.0f / _t_9740;
	_t_9742 = _t_9728 * _t_9741;
	_t_9743 = _t_9742 * -1.0f;
	_t_9744 = 1.0f + _t_9743;
	_t_9745 = 0.0f < _t_9744;
	if(_t_9745)
		{
		
			_t_9746 = py0_12_1;
		
		}
else
		{
		
			_t_9746 = py1_13_1;
		
		}

	_t_9747 = _t_9705 * _t_9746;
	_t_9748 = _t_9666 + _t_9747;
	_t_9749 = -1.0f * ty1_7_1;
	_t_9750 = ty3_9_1 + _t_9749;
	_t_9751 = -1.0f * _t_9750;
	_t_9752 = _t_9751 < 0.0f;
	if(_t_9752)
		{
			float _t_9753;
			float _t_9754;
		
			_t_9753 = -1.0f * tx3_6_1;
			_t_9754 = tx1_4_1 + _t_9753;
			_t_9755 = _t_9754;
		
		}
else
		{
			float _t_9756;
			float _t_9757;
			float _t_9758;
		
			_t_9756 = -1.0f * tx3_6_1;
			_t_9757 = tx1_4_1 + _t_9756;
			_t_9758 = -1.0f * _t_9757;
			_t_9755 = _t_9758;
		
		}

	_t_9759 = _t_9755 * _t_399;
	_t_9760 = _t_9759 * -1.0f;
	_t_9761 = -1.0f * ty1_7_1;
	_t_9762 = ty3_9_1 + _t_9761;
	_t_9763 = -1.0f * _t_9762;
	_t_9764 = _t_9763 < 0.0f;
	if(_t_9764)
		{
			float _t_9765;
			float _t_9766;
		
			_t_9765 = -1.0f * tx3_6_1;
			_t_9766 = tx1_4_1 + _t_9765;
			_t_9767 = _t_9766;
		
		}
else
		{
			float _t_9768;
			float _t_9769;
			float _t_9770;
		
			_t_9768 = -1.0f * tx3_6_1;
			_t_9769 = tx1_4_1 + _t_9768;
			_t_9770 = -1.0f * _t_9769;
			_t_9767 = _t_9770;
		
		}

	_t_9771 = _t_9767 * _t_399;
	_t_9772 = _t_9771 * -1.0f;
	_t_9773 = 0.0f < _t_9772;
	if(_t_9773)
		{
		
			_t_9774 = px1_11_1;
		
		}
else
		{
		
			_t_9774 = px0_10_1;
		
		}

	_t_9775 = _t_9760 * _t_9774;
	_t_9776 = -1.0f * ty1_7_1;
	_t_9777 = ty3_9_1 + _t_9776;
	_t_9778 = -1.0f * _t_9777;
	_t_9779 = _t_9778 < 0.0f;
	if(_t_9779)
		{
			float _t_9780;
			float _t_9781;
		
			_t_9780 = -1.0f * tx3_6_1;
			_t_9781 = tx1_4_1 + _t_9780;
			_t_9782 = _t_9781;
		
		}
else
		{
			float _t_9783;
			float _t_9784;
			float _t_9785;
		
			_t_9783 = -1.0f * tx3_6_1;
			_t_9784 = tx1_4_1 + _t_9783;
			_t_9785 = -1.0f * _t_9784;
			_t_9782 = _t_9785;
		
		}

	_t_9786 = _t_9782 * _t_399;
	_t_9787 = -1.0f * ty1_7_1;
	_t_9788 = ty3_9_1 + _t_9787;
	_t_9789 = -1.0f * _t_9788;
	_t_9790 = _t_9789 < 0.0f;
	if(_t_9790)
		{
			float _t_9791;
			float _t_9792;
		
			_t_9791 = -1.0f * tx3_6_1;
			_t_9792 = tx1_4_1 + _t_9791;
			_t_9793 = _t_9792;
		
		}
else
		{
			float _t_9794;
			float _t_9795;
			float _t_9796;
		
			_t_9794 = -1.0f * tx3_6_1;
			_t_9795 = tx1_4_1 + _t_9794;
			_t_9796 = -1.0f * _t_9795;
			_t_9793 = _t_9796;
		
		}

	_t_9797 = _t_9793 * _t_399;
	_t_9798 = _t_9786 * _t_9797;
	_t_9799 = -1.0f * ty1_7_1;
	_t_9800 = ty3_9_1 + _t_9799;
	_t_9801 = -1.0f * _t_9800;
	_t_9802 = _t_9801 < 0.0f;
	if(_t_9802)
		{
			float _t_9803;
			float _t_9804;
		
			_t_9803 = -1.0f * ty1_7_1;
			_t_9804 = ty3_9_1 + _t_9803;
			_t_9805 = _t_9804;
		
		}
else
		{
			float _t_9806;
			float _t_9807;
			float _t_9808;
		
			_t_9806 = -1.0f * ty1_7_1;
			_t_9807 = ty3_9_1 + _t_9806;
			_t_9808 = -1.0f * _t_9807;
			_t_9805 = _t_9808;
		
		}

	_t_9809 = _t_9805 * _t_399;
	_t_9810 = 1.0f + _t_9809;
	_t_9811 = 1.0f / _t_9810;
	_t_9812 = _t_9798 * _t_9811;
	_t_9813 = _t_9812 * -1.0f;
	_t_9814 = 1.0f + _t_9813;
	_t_9815 = -1.0f * ty1_7_1;
	_t_9816 = ty3_9_1 + _t_9815;
	_t_9817 = -1.0f * _t_9816;
	_t_9818 = _t_9817 < 0.0f;
	if(_t_9818)
		{
			float _t_9819;
			float _t_9820;
		
			_t_9819 = -1.0f * tx3_6_1;
			_t_9820 = tx1_4_1 + _t_9819;
			_t_9821 = _t_9820;
		
		}
else
		{
			float _t_9822;
			float _t_9823;
			float _t_9824;
		
			_t_9822 = -1.0f * tx3_6_1;
			_t_9823 = tx1_4_1 + _t_9822;
			_t_9824 = -1.0f * _t_9823;
			_t_9821 = _t_9824;
		
		}

	_t_9825 = _t_9821 * _t_399;
	_t_9826 = -1.0f * ty1_7_1;
	_t_9827 = ty3_9_1 + _t_9826;
	_t_9828 = -1.0f * _t_9827;
	_t_9829 = _t_9828 < 0.0f;
	if(_t_9829)
		{
			float _t_9830;
			float _t_9831;
		
			_t_9830 = -1.0f * tx3_6_1;
			_t_9831 = tx1_4_1 + _t_9830;
			_t_9832 = _t_9831;
		
		}
else
		{
			float _t_9833;
			float _t_9834;
			float _t_9835;
		
			_t_9833 = -1.0f * tx3_6_1;
			_t_9834 = tx1_4_1 + _t_9833;
			_t_9835 = -1.0f * _t_9834;
			_t_9832 = _t_9835;
		
		}

	_t_9836 = _t_9832 * _t_399;
	_t_9837 = _t_9825 * _t_9836;
	_t_9838 = -1.0f * ty1_7_1;
	_t_9839 = ty3_9_1 + _t_9838;
	_t_9840 = -1.0f * _t_9839;
	_t_9841 = _t_9840 < 0.0f;
	if(_t_9841)
		{
			float _t_9842;
			float _t_9843;
		
			_t_9842 = -1.0f * ty1_7_1;
			_t_9843 = ty3_9_1 + _t_9842;
			_t_9844 = _t_9843;
		
		}
else
		{
			float _t_9845;
			float _t_9846;
			float _t_9847;
		
			_t_9845 = -1.0f * ty1_7_1;
			_t_9846 = ty3_9_1 + _t_9845;
			_t_9847 = -1.0f * _t_9846;
			_t_9844 = _t_9847;
		
		}

	_t_9848 = _t_9844 * _t_399;
	_t_9849 = 1.0f + _t_9848;
	_t_9850 = 1.0f / _t_9849;
	_t_9851 = _t_9837 * _t_9850;
	_t_9852 = _t_9851 * -1.0f;
	_t_9853 = 1.0f + _t_9852;
	_t_9854 = 0.0f < _t_9853;
	if(_t_9854)
		{
		
			_t_9855 = py1_13_1;
		
		}
else
		{
		
			_t_9855 = py0_12_1;
		
		}

	_t_9856 = _t_9814 * _t_9855;
	_t_9857 = _t_9775 + _t_9856;
	_t_400 = tegpixelintegrator_27(ty3_9_1,pc1_15_1,_t_9748,_t_399,tc2_19_1,ty2_8_1,pc0_14_1,ty1_7_1,tx1_4_1,tx3_6_1,py1_13_1,pc2_16_1,tx2_5_1,_t_9857,px1_11_1,tc0_17_1,py0_12_1,tc1_18_1,px0_10_1);

	return _t_400;
}
__device__ float tegpixellet_block_42(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float _t_10878,float _t_10931,float ty3_9_1,float tx3_6_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float _t_427,float y__3387_1,float _t_10851){
	float _t_10932;
	float _t_10933;
	float _t_10934;
	float _t_10935;
	float _t_10936;
	float _t_10937;
	float _t_10938;
	float _t_10939;
	float _t_10940;
	float _t_10941;
	float _t_10942;
	float _t_10943;
	float _t_10944;
	float _t_10945;
	float _t_10946;
	float _t_10947;
	float _t_10948;
	float _t_10949;
	float _t_10950;
	float _t_10951;
	float _t_10952;
	float _t_10953;
	float _t_10954;
	bool _t_10955;
	float _t_10956;
	float _t_10957;
	float _t_10958;
	float _t_10959;
	float _t_10960;
	float _t_10961;
	float _t_10962;
	float _t_10963;
	float _t_10964;
	float _t_10965;
	float _t_10966;
	float _t_10967;
	float _t_10968;
	bool _t_10969;
	float _t_10970;
	float _t_10971;
	float _t_10972;
	float _t_10973;
	bool _t_10974;
	bool _t_10975;
	bool _t_10976;
	bool _t_10977;
	bool _t_10978;
	bool _t_10979;
	bool _t_10980;
	float _t_11310;

	float _t_10852;

	_t_10932 = -1.0f * pc0_14_1;
	_t_10933 = tc0_17_1 + _t_10932;
	_t_10934 = _t_10933 * _t_10933;
	_t_10935 = -1.0f * pc1_15_1;
	_t_10936 = tc1_18_1 + _t_10935;
	_t_10937 = _t_10936 * _t_10936;
	_t_10938 = _t_10934 + _t_10937;
	_t_10939 = -1.0f * pc2_16_1;
	_t_10940 = tc2_19_1 + _t_10939;
	_t_10941 = _t_10940 * _t_10940;
	_t_10942 = _t_10938 + _t_10941;
	_t_10943 = tx1_4_1 * ty2_8_1;
	_t_10944 = tx2_5_1 * ty1_7_1;
	_t_10945 = _t_10944 * -1.0f;
	_t_10946 = _t_10943 + _t_10945;
	_t_10947 = -1.0f * ty2_8_1;
	_t_10948 = ty1_7_1 + _t_10947;
	_t_10949 = _t_10948 * _t_10878;
	_t_10950 = _t_10946 + _t_10949;
	_t_10951 = -1.0f * tx1_4_1;
	_t_10952 = tx2_5_1 + _t_10951;
	_t_10953 = _t_10952 * _t_10931;
	_t_10954 = _t_10950 + _t_10953;
	_t_10955 = _t_10954 < 0.0f;
	if(_t_10955)
		{
		
			_t_10956 = 1.0f;
		
		}
else
		{
		
			_t_10956 = 0.0f;
		
		}

	_t_10957 = tx2_5_1 * ty3_9_1;
	_t_10958 = tx3_6_1 * ty2_8_1;
	_t_10959 = _t_10958 * -1.0f;
	_t_10960 = _t_10957 + _t_10959;
	_t_10961 = -1.0f * ty3_9_1;
	_t_10962 = ty2_8_1 + _t_10961;
	_t_10963 = _t_10962 * _t_10878;
	_t_10964 = _t_10960 + _t_10963;
	_t_10965 = -1.0f * tx2_5_1;
	_t_10966 = tx3_6_1 + _t_10965;
	_t_10967 = _t_10966 * _t_10931;
	_t_10968 = _t_10964 + _t_10967;
	_t_10969 = _t_10968 < 0.0f;
	if(_t_10969)
		{
		
			_t_10970 = 1.0f;
		
		}
else
		{
		
			_t_10970 = 0.0f;
		
		}

	_t_10971 = _t_10956 * _t_10970;
	_t_10972 = _t_10942 * _t_10971;
	_t_10973 = _t_10972 * _t_10878;
	_t_10974 = py0_12_1 < _t_10931;
	_t_10975 = _t_10931 < py1_13_1;
	_t_10976 = _t_10974 && _t_10975;
	_t_10977 = px0_10_1 < _t_10878;
	_t_10978 = _t_10878 < px1_11_1;
	_t_10979 = _t_10977 && _t_10978;
	_t_10980 = _t_10976 && _t_10979;
	if(_t_10980)
		{
			float _t_10981;
			float _t_10982;
			float _t_10983;
			bool _t_10984;
			float _t_10987;
			float _t_10991;
			float _t_10992;
			float _t_10993;
			float _t_10994;
			float _t_10995;
			bool _t_10996;
			float _t_10999;
			float _t_11003;
			float _t_11004;
			bool _t_11005;
			float _t_11006;
			float _t_11007;
			float _t_11008;
			float _t_11009;
			float _t_11010;
			bool _t_11011;
			float _t_11014;
			float _t_11018;
			float _t_11019;
			float _t_11020;
			float _t_11021;
			bool _t_11022;
			float _t_11025;
			float _t_11029;
			float _t_11030;
			float _t_11031;
			float _t_11032;
			float _t_11033;
			bool _t_11034;
			float _t_11037;
			float _t_11041;
			float _t_11042;
			float _t_11043;
			float _t_11044;
			float _t_11045;
			float _t_11046;
			float _t_11047;
			float _t_11048;
			float _t_11049;
			bool _t_11050;
			float _t_11053;
			float _t_11057;
			float _t_11058;
			float _t_11059;
			float _t_11060;
			bool _t_11061;
			float _t_11064;
			float _t_11068;
			float _t_11069;
			float _t_11070;
			float _t_11071;
			float _t_11072;
			bool _t_11073;
			float _t_11076;
			float _t_11080;
			float _t_11081;
			float _t_11082;
			float _t_11083;
			float _t_11084;
			float _t_11085;
			bool _t_11086;
			float _t_11087;
			float _t_11088;
			float _t_11089;
			bool _t_11090;
			float _t_11091;
			float _t_11092;
			float _t_11093;
			bool _t_11094;
			float _t_11097;
			float _t_11101;
			float _t_11102;
			float _t_11103;
			float _t_11104;
			float _t_11105;
			bool _t_11106;
			float _t_11109;
			float _t_11113;
			float _t_11114;
			bool _t_11115;
			float _t_11116;
			float _t_11117;
			float _t_11118;
			float _t_11119;
			float _t_11120;
			bool _t_11121;
			float _t_11124;
			float _t_11128;
			float _t_11129;
			float _t_11130;
			float _t_11131;
			bool _t_11132;
			float _t_11135;
			float _t_11139;
			float _t_11140;
			float _t_11141;
			float _t_11142;
			float _t_11143;
			bool _t_11144;
			float _t_11147;
			float _t_11151;
			float _t_11152;
			float _t_11153;
			float _t_11154;
			float _t_11155;
			float _t_11156;
			float _t_11157;
			float _t_11158;
			float _t_11159;
			bool _t_11160;
			float _t_11163;
			float _t_11167;
			float _t_11168;
			float _t_11169;
			float _t_11170;
			bool _t_11171;
			float _t_11174;
			float _t_11178;
			float _t_11179;
			float _t_11180;
			float _t_11181;
			float _t_11182;
			bool _t_11183;
			float _t_11186;
			float _t_11190;
			float _t_11191;
			float _t_11192;
			float _t_11193;
			float _t_11194;
			float _t_11195;
			bool _t_11196;
			float _t_11197;
			float _t_11198;
			float _t_11199;
			bool _t_11200;
			bool _t_11201;
			float _t_11202;
			float _t_11203;
			float _t_11204;
			bool _t_11205;
			float _t_11208;
			float _t_11212;
			float _t_11213;
			float _t_11214;
			float _t_11215;
			bool _t_11216;
			float _t_11219;
			float _t_11223;
			bool _t_11224;
			float _t_11225;
			float _t_11226;
			float _t_11227;
			float _t_11228;
			float _t_11229;
			bool _t_11230;
			float _t_11233;
			float _t_11237;
			float _t_11238;
			float _t_11239;
			float _t_11240;
			bool _t_11241;
			float _t_11244;
			float _t_11248;
			bool _t_11249;
			float _t_11250;
			float _t_11251;
			float _t_11252;
			bool _t_11253;
			float _t_11254;
			float _t_11255;
			float _t_11256;
			bool _t_11257;
			float _t_11260;
			float _t_11264;
			float _t_11265;
			float _t_11266;
			float _t_11267;
			bool _t_11268;
			float _t_11271;
			float _t_11275;
			bool _t_11276;
			float _t_11277;
			float _t_11278;
			float _t_11279;
			float _t_11280;
			float _t_11281;
			bool _t_11282;
			float _t_11285;
			float _t_11289;
			float _t_11290;
			float _t_11291;
			float _t_11292;
			bool _t_11293;
			float _t_11296;
			float _t_11300;
			bool _t_11301;
			float _t_11302;
			float _t_11303;
			float _t_11304;
			bool _t_11305;
			bool _t_11306;
			bool _t_11307;
			float _t_11308;
			float _t_11309;
		
			_t_10981 = -1.0f * ty1_7_1;
			_t_10982 = ty3_9_1 + _t_10981;
			_t_10983 = -1.0f * _t_10982;
			_t_10984 = _t_10983 < 0.0f;
			if(_t_10984)
				{
					float _t_10985;
					float _t_10986;
				
					_t_10985 = -1.0f * tx3_6_1;
					_t_10986 = tx1_4_1 + _t_10985;
					_t_10987 = _t_10986;
				
				}
		else
				{
					float _t_10988;
					float _t_10989;
					float _t_10990;
				
					_t_10988 = -1.0f * tx3_6_1;
					_t_10989 = tx1_4_1 + _t_10988;
					_t_10990 = -1.0f * _t_10989;
					_t_10987 = _t_10990;
				
				}
		
			_t_10991 = _t_10987 * _t_427;
			_t_10992 = _t_10991 * -1.0f;
			_t_10993 = -1.0f * ty1_7_1;
			_t_10994 = ty3_9_1 + _t_10993;
			_t_10995 = -1.0f * _t_10994;
			_t_10996 = _t_10995 < 0.0f;
			if(_t_10996)
				{
					float _t_10997;
					float _t_10998;
				
					_t_10997 = -1.0f * tx3_6_1;
					_t_10998 = tx1_4_1 + _t_10997;
					_t_10999 = _t_10998;
				
				}
		else
				{
					float _t_11000;
					float _t_11001;
					float _t_11002;
				
					_t_11000 = -1.0f * tx3_6_1;
					_t_11001 = tx1_4_1 + _t_11000;
					_t_11002 = -1.0f * _t_11001;
					_t_10999 = _t_11002;
				
				}
		
			_t_11003 = _t_10999 * _t_427;
			_t_11004 = _t_11003 * -1.0f;
			_t_11005 = 0.0f < _t_11004;
			if(_t_11005)
				{
				
					_t_11006 = px0_10_1;
				
				}
		else
				{
				
					_t_11006 = px1_11_1;
				
				}
		
			_t_11007 = _t_10992 * _t_11006;
			_t_11008 = -1.0f * ty1_7_1;
			_t_11009 = ty3_9_1 + _t_11008;
			_t_11010 = -1.0f * _t_11009;
			_t_11011 = _t_11010 < 0.0f;
			if(_t_11011)
				{
					float _t_11012;
					float _t_11013;
				
					_t_11012 = -1.0f * tx3_6_1;
					_t_11013 = tx1_4_1 + _t_11012;
					_t_11014 = _t_11013;
				
				}
		else
				{
					float _t_11015;
					float _t_11016;
					float _t_11017;
				
					_t_11015 = -1.0f * tx3_6_1;
					_t_11016 = tx1_4_1 + _t_11015;
					_t_11017 = -1.0f * _t_11016;
					_t_11014 = _t_11017;
				
				}
		
			_t_11018 = _t_11014 * _t_427;
			_t_11019 = -1.0f * ty1_7_1;
			_t_11020 = ty3_9_1 + _t_11019;
			_t_11021 = -1.0f * _t_11020;
			_t_11022 = _t_11021 < 0.0f;
			if(_t_11022)
				{
					float _t_11023;
					float _t_11024;
				
					_t_11023 = -1.0f * tx3_6_1;
					_t_11024 = tx1_4_1 + _t_11023;
					_t_11025 = _t_11024;
				
				}
		else
				{
					float _t_11026;
					float _t_11027;
					float _t_11028;
				
					_t_11026 = -1.0f * tx3_6_1;
					_t_11027 = tx1_4_1 + _t_11026;
					_t_11028 = -1.0f * _t_11027;
					_t_11025 = _t_11028;
				
				}
		
			_t_11029 = _t_11025 * _t_427;
			_t_11030 = _t_11018 * _t_11029;
			_t_11031 = -1.0f * ty1_7_1;
			_t_11032 = ty3_9_1 + _t_11031;
			_t_11033 = -1.0f * _t_11032;
			_t_11034 = _t_11033 < 0.0f;
			if(_t_11034)
				{
					float _t_11035;
					float _t_11036;
				
					_t_11035 = -1.0f * ty1_7_1;
					_t_11036 = ty3_9_1 + _t_11035;
					_t_11037 = _t_11036;
				
				}
		else
				{
					float _t_11038;
					float _t_11039;
					float _t_11040;
				
					_t_11038 = -1.0f * ty1_7_1;
					_t_11039 = ty3_9_1 + _t_11038;
					_t_11040 = -1.0f * _t_11039;
					_t_11037 = _t_11040;
				
				}
		
			_t_11041 = _t_11037 * _t_427;
			_t_11042 = 1.0f + _t_11041;
			_t_11043 = 1.0f / _t_11042;
			_t_11044 = _t_11030 * _t_11043;
			_t_11045 = _t_11044 * -1.0f;
			_t_11046 = 1.0f + _t_11045;
			_t_11047 = -1.0f * ty1_7_1;
			_t_11048 = ty3_9_1 + _t_11047;
			_t_11049 = -1.0f * _t_11048;
			_t_11050 = _t_11049 < 0.0f;
			if(_t_11050)
				{
					float _t_11051;
					float _t_11052;
				
					_t_11051 = -1.0f * tx3_6_1;
					_t_11052 = tx1_4_1 + _t_11051;
					_t_11053 = _t_11052;
				
				}
		else
				{
					float _t_11054;
					float _t_11055;
					float _t_11056;
				
					_t_11054 = -1.0f * tx3_6_1;
					_t_11055 = tx1_4_1 + _t_11054;
					_t_11056 = -1.0f * _t_11055;
					_t_11053 = _t_11056;
				
				}
		
			_t_11057 = _t_11053 * _t_427;
			_t_11058 = -1.0f * ty1_7_1;
			_t_11059 = ty3_9_1 + _t_11058;
			_t_11060 = -1.0f * _t_11059;
			_t_11061 = _t_11060 < 0.0f;
			if(_t_11061)
				{
					float _t_11062;
					float _t_11063;
				
					_t_11062 = -1.0f * tx3_6_1;
					_t_11063 = tx1_4_1 + _t_11062;
					_t_11064 = _t_11063;
				
				}
		else
				{
					float _t_11065;
					float _t_11066;
					float _t_11067;
				
					_t_11065 = -1.0f * tx3_6_1;
					_t_11066 = tx1_4_1 + _t_11065;
					_t_11067 = -1.0f * _t_11066;
					_t_11064 = _t_11067;
				
				}
		
			_t_11068 = _t_11064 * _t_427;
			_t_11069 = _t_11057 * _t_11068;
			_t_11070 = -1.0f * ty1_7_1;
			_t_11071 = ty3_9_1 + _t_11070;
			_t_11072 = -1.0f * _t_11071;
			_t_11073 = _t_11072 < 0.0f;
			if(_t_11073)
				{
					float _t_11074;
					float _t_11075;
				
					_t_11074 = -1.0f * ty1_7_1;
					_t_11075 = ty3_9_1 + _t_11074;
					_t_11076 = _t_11075;
				
				}
		else
				{
					float _t_11077;
					float _t_11078;
					float _t_11079;
				
					_t_11077 = -1.0f * ty1_7_1;
					_t_11078 = ty3_9_1 + _t_11077;
					_t_11079 = -1.0f * _t_11078;
					_t_11076 = _t_11079;
				
				}
		
			_t_11080 = _t_11076 * _t_427;
			_t_11081 = 1.0f + _t_11080;
			_t_11082 = 1.0f / _t_11081;
			_t_11083 = _t_11069 * _t_11082;
			_t_11084 = _t_11083 * -1.0f;
			_t_11085 = 1.0f + _t_11084;
			_t_11086 = 0.0f < _t_11085;
			if(_t_11086)
				{
				
					_t_11087 = py0_12_1;
				
				}
		else
				{
				
					_t_11087 = py1_13_1;
				
				}
		
			_t_11088 = _t_11046 * _t_11087;
			_t_11089 = _t_11007 + _t_11088;
			_t_11090 = _t_11089 < y__3387_1;
			_t_11091 = -1.0f * ty1_7_1;
			_t_11092 = ty3_9_1 + _t_11091;
			_t_11093 = -1.0f * _t_11092;
			_t_11094 = _t_11093 < 0.0f;
			if(_t_11094)
				{
					float _t_11095;
					float _t_11096;
				
					_t_11095 = -1.0f * tx3_6_1;
					_t_11096 = tx1_4_1 + _t_11095;
					_t_11097 = _t_11096;
				
				}
		else
				{
					float _t_11098;
					float _t_11099;
					float _t_11100;
				
					_t_11098 = -1.0f * tx3_6_1;
					_t_11099 = tx1_4_1 + _t_11098;
					_t_11100 = -1.0f * _t_11099;
					_t_11097 = _t_11100;
				
				}
		
			_t_11101 = _t_11097 * _t_427;
			_t_11102 = _t_11101 * -1.0f;
			_t_11103 = -1.0f * ty1_7_1;
			_t_11104 = ty3_9_1 + _t_11103;
			_t_11105 = -1.0f * _t_11104;
			_t_11106 = _t_11105 < 0.0f;
			if(_t_11106)
				{
					float _t_11107;
					float _t_11108;
				
					_t_11107 = -1.0f * tx3_6_1;
					_t_11108 = tx1_4_1 + _t_11107;
					_t_11109 = _t_11108;
				
				}
		else
				{
					float _t_11110;
					float _t_11111;
					float _t_11112;
				
					_t_11110 = -1.0f * tx3_6_1;
					_t_11111 = tx1_4_1 + _t_11110;
					_t_11112 = -1.0f * _t_11111;
					_t_11109 = _t_11112;
				
				}
		
			_t_11113 = _t_11109 * _t_427;
			_t_11114 = _t_11113 * -1.0f;
			_t_11115 = 0.0f < _t_11114;
			if(_t_11115)
				{
				
					_t_11116 = px1_11_1;
				
				}
		else
				{
				
					_t_11116 = px0_10_1;
				
				}
		
			_t_11117 = _t_11102 * _t_11116;
			_t_11118 = -1.0f * ty1_7_1;
			_t_11119 = ty3_9_1 + _t_11118;
			_t_11120 = -1.0f * _t_11119;
			_t_11121 = _t_11120 < 0.0f;
			if(_t_11121)
				{
					float _t_11122;
					float _t_11123;
				
					_t_11122 = -1.0f * tx3_6_1;
					_t_11123 = tx1_4_1 + _t_11122;
					_t_11124 = _t_11123;
				
				}
		else
				{
					float _t_11125;
					float _t_11126;
					float _t_11127;
				
					_t_11125 = -1.0f * tx3_6_1;
					_t_11126 = tx1_4_1 + _t_11125;
					_t_11127 = -1.0f * _t_11126;
					_t_11124 = _t_11127;
				
				}
		
			_t_11128 = _t_11124 * _t_427;
			_t_11129 = -1.0f * ty1_7_1;
			_t_11130 = ty3_9_1 + _t_11129;
			_t_11131 = -1.0f * _t_11130;
			_t_11132 = _t_11131 < 0.0f;
			if(_t_11132)
				{
					float _t_11133;
					float _t_11134;
				
					_t_11133 = -1.0f * tx3_6_1;
					_t_11134 = tx1_4_1 + _t_11133;
					_t_11135 = _t_11134;
				
				}
		else
				{
					float _t_11136;
					float _t_11137;
					float _t_11138;
				
					_t_11136 = -1.0f * tx3_6_1;
					_t_11137 = tx1_4_1 + _t_11136;
					_t_11138 = -1.0f * _t_11137;
					_t_11135 = _t_11138;
				
				}
		
			_t_11139 = _t_11135 * _t_427;
			_t_11140 = _t_11128 * _t_11139;
			_t_11141 = -1.0f * ty1_7_1;
			_t_11142 = ty3_9_1 + _t_11141;
			_t_11143 = -1.0f * _t_11142;
			_t_11144 = _t_11143 < 0.0f;
			if(_t_11144)
				{
					float _t_11145;
					float _t_11146;
				
					_t_11145 = -1.0f * ty1_7_1;
					_t_11146 = ty3_9_1 + _t_11145;
					_t_11147 = _t_11146;
				
				}
		else
				{
					float _t_11148;
					float _t_11149;
					float _t_11150;
				
					_t_11148 = -1.0f * ty1_7_1;
					_t_11149 = ty3_9_1 + _t_11148;
					_t_11150 = -1.0f * _t_11149;
					_t_11147 = _t_11150;
				
				}
		
			_t_11151 = _t_11147 * _t_427;
			_t_11152 = 1.0f + _t_11151;
			_t_11153 = 1.0f / _t_11152;
			_t_11154 = _t_11140 * _t_11153;
			_t_11155 = _t_11154 * -1.0f;
			_t_11156 = 1.0f + _t_11155;
			_t_11157 = -1.0f * ty1_7_1;
			_t_11158 = ty3_9_1 + _t_11157;
			_t_11159 = -1.0f * _t_11158;
			_t_11160 = _t_11159 < 0.0f;
			if(_t_11160)
				{
					float _t_11161;
					float _t_11162;
				
					_t_11161 = -1.0f * tx3_6_1;
					_t_11162 = tx1_4_1 + _t_11161;
					_t_11163 = _t_11162;
				
				}
		else
				{
					float _t_11164;
					float _t_11165;
					float _t_11166;
				
					_t_11164 = -1.0f * tx3_6_1;
					_t_11165 = tx1_4_1 + _t_11164;
					_t_11166 = -1.0f * _t_11165;
					_t_11163 = _t_11166;
				
				}
		
			_t_11167 = _t_11163 * _t_427;
			_t_11168 = -1.0f * ty1_7_1;
			_t_11169 = ty3_9_1 + _t_11168;
			_t_11170 = -1.0f * _t_11169;
			_t_11171 = _t_11170 < 0.0f;
			if(_t_11171)
				{
					float _t_11172;
					float _t_11173;
				
					_t_11172 = -1.0f * tx3_6_1;
					_t_11173 = tx1_4_1 + _t_11172;
					_t_11174 = _t_11173;
				
				}
		else
				{
					float _t_11175;
					float _t_11176;
					float _t_11177;
				
					_t_11175 = -1.0f * tx3_6_1;
					_t_11176 = tx1_4_1 + _t_11175;
					_t_11177 = -1.0f * _t_11176;
					_t_11174 = _t_11177;
				
				}
		
			_t_11178 = _t_11174 * _t_427;
			_t_11179 = _t_11167 * _t_11178;
			_t_11180 = -1.0f * ty1_7_1;
			_t_11181 = ty3_9_1 + _t_11180;
			_t_11182 = -1.0f * _t_11181;
			_t_11183 = _t_11182 < 0.0f;
			if(_t_11183)
				{
					float _t_11184;
					float _t_11185;
				
					_t_11184 = -1.0f * ty1_7_1;
					_t_11185 = ty3_9_1 + _t_11184;
					_t_11186 = _t_11185;
				
				}
		else
				{
					float _t_11187;
					float _t_11188;
					float _t_11189;
				
					_t_11187 = -1.0f * ty1_7_1;
					_t_11188 = ty3_9_1 + _t_11187;
					_t_11189 = -1.0f * _t_11188;
					_t_11186 = _t_11189;
				
				}
		
			_t_11190 = _t_11186 * _t_427;
			_t_11191 = 1.0f + _t_11190;
			_t_11192 = 1.0f / _t_11191;
			_t_11193 = _t_11179 * _t_11192;
			_t_11194 = _t_11193 * -1.0f;
			_t_11195 = 1.0f + _t_11194;
			_t_11196 = 0.0f < _t_11195;
			if(_t_11196)
				{
				
					_t_11197 = py1_13_1;
				
				}
		else
				{
				
					_t_11197 = py0_12_1;
				
				}
		
			_t_11198 = _t_11156 * _t_11197;
			_t_11199 = _t_11117 + _t_11198;
			_t_11200 = y__3387_1 < _t_11199;
			_t_11201 = _t_11090 && _t_11200;
			_t_11202 = -1.0f * ty1_7_1;
			_t_11203 = ty3_9_1 + _t_11202;
			_t_11204 = -1.0f * _t_11203;
			_t_11205 = _t_11204 < 0.0f;
			if(_t_11205)
				{
					float _t_11206;
					float _t_11207;
				
					_t_11206 = -1.0f * ty1_7_1;
					_t_11207 = ty3_9_1 + _t_11206;
					_t_11208 = _t_11207;
				
				}
		else
				{
					float _t_11209;
					float _t_11210;
					float _t_11211;
				
					_t_11209 = -1.0f * ty1_7_1;
					_t_11210 = ty3_9_1 + _t_11209;
					_t_11211 = -1.0f * _t_11210;
					_t_11208 = _t_11211;
				
				}
		
			_t_11212 = _t_11208 * _t_427;
			_t_11213 = -1.0f * ty1_7_1;
			_t_11214 = ty3_9_1 + _t_11213;
			_t_11215 = -1.0f * _t_11214;
			_t_11216 = _t_11215 < 0.0f;
			if(_t_11216)
				{
					float _t_11217;
					float _t_11218;
				
					_t_11217 = -1.0f * ty1_7_1;
					_t_11218 = ty3_9_1 + _t_11217;
					_t_11219 = _t_11218;
				
				}
		else
				{
					float _t_11220;
					float _t_11221;
					float _t_11222;
				
					_t_11220 = -1.0f * ty1_7_1;
					_t_11221 = ty3_9_1 + _t_11220;
					_t_11222 = -1.0f * _t_11221;
					_t_11219 = _t_11222;
				
				}
		
			_t_11223 = _t_11219 * _t_427;
			_t_11224 = 0.0f < _t_11223;
			if(_t_11224)
				{
				
					_t_11225 = px0_10_1;
				
				}
		else
				{
				
					_t_11225 = px1_11_1;
				
				}
		
			_t_11226 = _t_11212 * _t_11225;
			_t_11227 = -1.0f * ty1_7_1;
			_t_11228 = ty3_9_1 + _t_11227;
			_t_11229 = -1.0f * _t_11228;
			_t_11230 = _t_11229 < 0.0f;
			if(_t_11230)
				{
					float _t_11231;
					float _t_11232;
				
					_t_11231 = -1.0f * tx3_6_1;
					_t_11232 = tx1_4_1 + _t_11231;
					_t_11233 = _t_11232;
				
				}
		else
				{
					float _t_11234;
					float _t_11235;
					float _t_11236;
				
					_t_11234 = -1.0f * tx3_6_1;
					_t_11235 = tx1_4_1 + _t_11234;
					_t_11236 = -1.0f * _t_11235;
					_t_11233 = _t_11236;
				
				}
		
			_t_11237 = _t_11233 * _t_427;
			_t_11238 = -1.0f * ty1_7_1;
			_t_11239 = ty3_9_1 + _t_11238;
			_t_11240 = -1.0f * _t_11239;
			_t_11241 = _t_11240 < 0.0f;
			if(_t_11241)
				{
					float _t_11242;
					float _t_11243;
				
					_t_11242 = -1.0f * tx3_6_1;
					_t_11243 = tx1_4_1 + _t_11242;
					_t_11244 = _t_11243;
				
				}
		else
				{
					float _t_11245;
					float _t_11246;
					float _t_11247;
				
					_t_11245 = -1.0f * tx3_6_1;
					_t_11246 = tx1_4_1 + _t_11245;
					_t_11247 = -1.0f * _t_11246;
					_t_11244 = _t_11247;
				
				}
		
			_t_11248 = _t_11244 * _t_427;
			_t_11249 = 0.0f < _t_11248;
			if(_t_11249)
				{
				
					_t_11250 = py0_12_1;
				
				}
		else
				{
				
					_t_11250 = py1_13_1;
				
				}
		
			_t_11251 = _t_11237 * _t_11250;
			_t_11252 = _t_11226 + _t_11251;
			_t_11253 = _t_11252 < _t_10851;
			_t_11254 = -1.0f * ty1_7_1;
			_t_11255 = ty3_9_1 + _t_11254;
			_t_11256 = -1.0f * _t_11255;
			_t_11257 = _t_11256 < 0.0f;
			if(_t_11257)
				{
					float _t_11258;
					float _t_11259;
				
					_t_11258 = -1.0f * ty1_7_1;
					_t_11259 = ty3_9_1 + _t_11258;
					_t_11260 = _t_11259;
				
				}
		else
				{
					float _t_11261;
					float _t_11262;
					float _t_11263;
				
					_t_11261 = -1.0f * ty1_7_1;
					_t_11262 = ty3_9_1 + _t_11261;
					_t_11263 = -1.0f * _t_11262;
					_t_11260 = _t_11263;
				
				}
		
			_t_11264 = _t_11260 * _t_427;
			_t_11265 = -1.0f * ty1_7_1;
			_t_11266 = ty3_9_1 + _t_11265;
			_t_11267 = -1.0f * _t_11266;
			_t_11268 = _t_11267 < 0.0f;
			if(_t_11268)
				{
					float _t_11269;
					float _t_11270;
				
					_t_11269 = -1.0f * ty1_7_1;
					_t_11270 = ty3_9_1 + _t_11269;
					_t_11271 = _t_11270;
				
				}
		else
				{
					float _t_11272;
					float _t_11273;
					float _t_11274;
				
					_t_11272 = -1.0f * ty1_7_1;
					_t_11273 = ty3_9_1 + _t_11272;
					_t_11274 = -1.0f * _t_11273;
					_t_11271 = _t_11274;
				
				}
		
			_t_11275 = _t_11271 * _t_427;
			_t_11276 = 0.0f < _t_11275;
			if(_t_11276)
				{
				
					_t_11277 = px1_11_1;
				
				}
		else
				{
				
					_t_11277 = px0_10_1;
				
				}
		
			_t_11278 = _t_11264 * _t_11277;
			_t_11279 = -1.0f * ty1_7_1;
			_t_11280 = ty3_9_1 + _t_11279;
			_t_11281 = -1.0f * _t_11280;
			_t_11282 = _t_11281 < 0.0f;
			if(_t_11282)
				{
					float _t_11283;
					float _t_11284;
				
					_t_11283 = -1.0f * tx3_6_1;
					_t_11284 = tx1_4_1 + _t_11283;
					_t_11285 = _t_11284;
				
				}
		else
				{
					float _t_11286;
					float _t_11287;
					float _t_11288;
				
					_t_11286 = -1.0f * tx3_6_1;
					_t_11287 = tx1_4_1 + _t_11286;
					_t_11288 = -1.0f * _t_11287;
					_t_11285 = _t_11288;
				
				}
		
			_t_11289 = _t_11285 * _t_427;
			_t_11290 = -1.0f * ty1_7_1;
			_t_11291 = ty3_9_1 + _t_11290;
			_t_11292 = -1.0f * _t_11291;
			_t_11293 = _t_11292 < 0.0f;
			if(_t_11293)
				{
					float _t_11294;
					float _t_11295;
				
					_t_11294 = -1.0f * tx3_6_1;
					_t_11295 = tx1_4_1 + _t_11294;
					_t_11296 = _t_11295;
				
				}
		else
				{
					float _t_11297;
					float _t_11298;
					float _t_11299;
				
					_t_11297 = -1.0f * tx3_6_1;
					_t_11298 = tx1_4_1 + _t_11297;
					_t_11299 = -1.0f * _t_11298;
					_t_11296 = _t_11299;
				
				}
		
			_t_11300 = _t_11296 * _t_427;
			_t_11301 = 0.0f < _t_11300;
			if(_t_11301)
				{
				
					_t_11302 = py1_13_1;
				
				}
		else
				{
				
					_t_11302 = py0_12_1;
				
				}
		
			_t_11303 = _t_11289 * _t_11302;
			_t_11304 = _t_11278 + _t_11303;
			_t_11305 = _t_10851 < _t_11304;
			_t_11306 = _t_11253 && _t_11305;
			_t_11307 = _t_11201 && _t_11306;
			if(_t_11307)
				{
				
					_t_11308 = 1.0f;
				
				}
		else
				{
				
					_t_11308 = 0.0f;
				
				}
		
			_t_11309 = _t_11308 * _t_427;
			_t_11310 = _t_11309;
		
		}
else
		{
		
			_t_11310 = 0.0f;
		
		}

	_t_10852 = _t_10973 * _t_11310;

	return _t_10852;
}
__device__ float tegpixellet_block_41(float ty3_9_1,float ty1_7_1,float _t_427,float _t_10851,float tx1_4_1,float tx3_6_1,float y__3387_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1){
	float _t_10853;
	float _t_10854;
	float _t_10855;
	bool _t_10856;
	float _t_10859;
	float _t_10863;
	float _t_10864;
	float _t_10865;
	float _t_10866;
	float _t_10867;
	bool _t_10868;
	float _t_10871;
	float _t_10875;
	float _t_10876;
	float _t_10877;
	float _t_10878;
	float _t_10879;
	float _t_10880;
	float _t_10881;
	bool _t_10882;
	float _t_10885;
	float _t_10889;
	float _t_10890;
	float _t_10891;
	float _t_10892;
	bool _t_10893;
	float _t_10896;
	float _t_10900;
	float _t_10901;
	float _t_10902;
	float _t_10903;
	float _t_10904;
	bool _t_10905;
	float _t_10908;
	float _t_10912;
	float _t_10913;
	float _t_10914;
	float _t_10915;
	float _t_10916;
	float _t_10917;
	float _t_10918;
	float _t_10919;
	float _t_10920;
	float _t_10921;
	bool _t_10922;
	float _t_10925;
	float _t_10929;
	float _t_10930;
	float _t_10931;

	float _t_10852;

	_t_10853 = -1.0f * ty1_7_1;
	_t_10854 = ty3_9_1 + _t_10853;
	_t_10855 = -1.0f * _t_10854;
	_t_10856 = _t_10855 < 0.0f;
	if(_t_10856)
		{
			float _t_10857;
			float _t_10858;
		
			_t_10857 = -1.0f * ty1_7_1;
			_t_10858 = ty3_9_1 + _t_10857;
			_t_10859 = _t_10858;
		
		}
else
		{
			float _t_10860;
			float _t_10861;
			float _t_10862;
		
			_t_10860 = -1.0f * ty1_7_1;
			_t_10861 = ty3_9_1 + _t_10860;
			_t_10862 = -1.0f * _t_10861;
			_t_10859 = _t_10862;
		
		}

	_t_10863 = _t_10859 * _t_427;
	_t_10864 = _t_10863 * _t_10851;
	_t_10865 = -1.0f * ty1_7_1;
	_t_10866 = ty3_9_1 + _t_10865;
	_t_10867 = -1.0f * _t_10866;
	_t_10868 = _t_10867 < 0.0f;
	if(_t_10868)
		{
			float _t_10869;
			float _t_10870;
		
			_t_10869 = -1.0f * tx3_6_1;
			_t_10870 = tx1_4_1 + _t_10869;
			_t_10871 = _t_10870;
		
		}
else
		{
			float _t_10872;
			float _t_10873;
			float _t_10874;
		
			_t_10872 = -1.0f * tx3_6_1;
			_t_10873 = tx1_4_1 + _t_10872;
			_t_10874 = -1.0f * _t_10873;
			_t_10871 = _t_10874;
		
		}

	_t_10875 = _t_10871 * _t_427;
	_t_10876 = _t_10875 * -1.0f;
	_t_10877 = _t_10876 * y__3387_1;
	_t_10878 = _t_10864 + _t_10877;
	_t_10879 = -1.0f * ty1_7_1;
	_t_10880 = ty3_9_1 + _t_10879;
	_t_10881 = -1.0f * _t_10880;
	_t_10882 = _t_10881 < 0.0f;
	if(_t_10882)
		{
			float _t_10883;
			float _t_10884;
		
			_t_10883 = -1.0f * tx3_6_1;
			_t_10884 = tx1_4_1 + _t_10883;
			_t_10885 = _t_10884;
		
		}
else
		{
			float _t_10886;
			float _t_10887;
			float _t_10888;
		
			_t_10886 = -1.0f * tx3_6_1;
			_t_10887 = tx1_4_1 + _t_10886;
			_t_10888 = -1.0f * _t_10887;
			_t_10885 = _t_10888;
		
		}

	_t_10889 = _t_10885 * _t_427;
	_t_10890 = -1.0f * ty1_7_1;
	_t_10891 = ty3_9_1 + _t_10890;
	_t_10892 = -1.0f * _t_10891;
	_t_10893 = _t_10892 < 0.0f;
	if(_t_10893)
		{
			float _t_10894;
			float _t_10895;
		
			_t_10894 = -1.0f * tx3_6_1;
			_t_10895 = tx1_4_1 + _t_10894;
			_t_10896 = _t_10895;
		
		}
else
		{
			float _t_10897;
			float _t_10898;
			float _t_10899;
		
			_t_10897 = -1.0f * tx3_6_1;
			_t_10898 = tx1_4_1 + _t_10897;
			_t_10899 = -1.0f * _t_10898;
			_t_10896 = _t_10899;
		
		}

	_t_10900 = _t_10896 * _t_427;
	_t_10901 = _t_10889 * _t_10900;
	_t_10902 = -1.0f * ty1_7_1;
	_t_10903 = ty3_9_1 + _t_10902;
	_t_10904 = -1.0f * _t_10903;
	_t_10905 = _t_10904 < 0.0f;
	if(_t_10905)
		{
			float _t_10906;
			float _t_10907;
		
			_t_10906 = -1.0f * ty1_7_1;
			_t_10907 = ty3_9_1 + _t_10906;
			_t_10908 = _t_10907;
		
		}
else
		{
			float _t_10909;
			float _t_10910;
			float _t_10911;
		
			_t_10909 = -1.0f * ty1_7_1;
			_t_10910 = ty3_9_1 + _t_10909;
			_t_10911 = -1.0f * _t_10910;
			_t_10908 = _t_10911;
		
		}

	_t_10912 = _t_10908 * _t_427;
	_t_10913 = 1.0f + _t_10912;
	_t_10914 = 1.0f / _t_10913;
	_t_10915 = _t_10901 * _t_10914;
	_t_10916 = _t_10915 * -1.0f;
	_t_10917 = 1.0f + _t_10916;
	_t_10918 = _t_10917 * y__3387_1;
	_t_10919 = -1.0f * ty1_7_1;
	_t_10920 = ty3_9_1 + _t_10919;
	_t_10921 = -1.0f * _t_10920;
	_t_10922 = _t_10921 < 0.0f;
	if(_t_10922)
		{
			float _t_10923;
			float _t_10924;
		
			_t_10923 = -1.0f * tx3_6_1;
			_t_10924 = tx1_4_1 + _t_10923;
			_t_10925 = _t_10924;
		
		}
else
		{
			float _t_10926;
			float _t_10927;
			float _t_10928;
		
			_t_10926 = -1.0f * tx3_6_1;
			_t_10927 = tx1_4_1 + _t_10926;
			_t_10928 = -1.0f * _t_10927;
			_t_10925 = _t_10928;
		
		}

	_t_10929 = _t_10925 * _t_427;
	_t_10930 = _t_10929 * _t_10851;
	_t_10931 = _t_10918 + _t_10930;
	_t_10852 = tegpixellet_block_42(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,_t_10878,_t_10931,ty3_9_1,tx3_6_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,_t_427,y__3387_1,_t_10851);

	return _t_10852;
}
__device__ float tegpixelbody_block_28(float ty3_9_1,float ty1_7_1,float _t_427,float px0_10_1,float px1_11_1,float tx1_4_1,float tx3_6_1,float py0_12_1,float py1_13_1,float y__3387_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_10695;
	float _t_10696;
	float _t_10697;
	bool _t_10698;
	float _t_10701;
	float _t_10705;
	float _t_10706;
	float _t_10707;
	float _t_10708;
	bool _t_10709;
	float _t_10712;
	float _t_10716;
	bool _t_10717;
	float _t_10718;
	float _t_10719;
	float _t_10720;
	float _t_10721;
	float _t_10722;
	bool _t_10723;
	float _t_10726;
	float _t_10730;
	float _t_10731;
	float _t_10732;
	float _t_10733;
	bool _t_10734;
	float _t_10737;
	float _t_10741;
	bool _t_10742;
	float _t_10743;
	float _t_10744;
	float _t_10745;
	float _t_10746;
	float _t_10747;
	float _t_10748;
	bool _t_10749;
	float _t_10754;
	float _t_10760;
	float _t_10761;
	float _t_10762;
	float _t_10763;
	bool _t_10764;
	float _t_10765;
	float _t_10766;
	float _t_10767;
	bool _t_10768;
	float _t_10771;
	float _t_10775;
	float _t_10776;
	float _t_10777;
	float _t_10778;
	bool _t_10779;
	float _t_10782;
	float _t_10786;
	bool _t_10787;
	float _t_10788;
	float _t_10789;
	float _t_10790;
	float _t_10791;
	float _t_10792;
	bool _t_10793;
	float _t_10796;
	float _t_10800;
	float _t_10801;
	float _t_10802;
	float _t_10803;
	bool _t_10804;
	float _t_10807;
	float _t_10811;
	bool _t_10812;
	float _t_10813;
	float _t_10814;
	float _t_10815;
	float _t_10816;
	float _t_10817;
	float _t_10818;
	bool _t_10819;
	float _t_10824;
	float _t_10830;
	float _t_10831;
	float _t_10832;
	float _t_10833;
	bool _t_10834;
	bool _t_10835;

	float _t_10694;

	_t_10695 = -1.0f * ty1_7_1;
	_t_10696 = ty3_9_1 + _t_10695;
	_t_10697 = -1.0f * _t_10696;
	_t_10698 = _t_10697 < 0.0f;
	if(_t_10698)
		{
			float _t_10699;
			float _t_10700;
		
			_t_10699 = -1.0f * ty1_7_1;
			_t_10700 = ty3_9_1 + _t_10699;
			_t_10701 = _t_10700;
		
		}
else
		{
			float _t_10702;
			float _t_10703;
			float _t_10704;
		
			_t_10702 = -1.0f * ty1_7_1;
			_t_10703 = ty3_9_1 + _t_10702;
			_t_10704 = -1.0f * _t_10703;
			_t_10701 = _t_10704;
		
		}

	_t_10705 = _t_10701 * _t_427;
	_t_10706 = -1.0f * ty1_7_1;
	_t_10707 = ty3_9_1 + _t_10706;
	_t_10708 = -1.0f * _t_10707;
	_t_10709 = _t_10708 < 0.0f;
	if(_t_10709)
		{
			float _t_10710;
			float _t_10711;
		
			_t_10710 = -1.0f * ty1_7_1;
			_t_10711 = ty3_9_1 + _t_10710;
			_t_10712 = _t_10711;
		
		}
else
		{
			float _t_10713;
			float _t_10714;
			float _t_10715;
		
			_t_10713 = -1.0f * ty1_7_1;
			_t_10714 = ty3_9_1 + _t_10713;
			_t_10715 = -1.0f * _t_10714;
			_t_10712 = _t_10715;
		
		}

	_t_10716 = _t_10712 * _t_427;
	_t_10717 = 0.0f < _t_10716;
	if(_t_10717)
		{
		
			_t_10718 = px0_10_1;
		
		}
else
		{
		
			_t_10718 = px1_11_1;
		
		}

	_t_10719 = _t_10705 * _t_10718;
	_t_10720 = -1.0f * ty1_7_1;
	_t_10721 = ty3_9_1 + _t_10720;
	_t_10722 = -1.0f * _t_10721;
	_t_10723 = _t_10722 < 0.0f;
	if(_t_10723)
		{
			float _t_10724;
			float _t_10725;
		
			_t_10724 = -1.0f * tx3_6_1;
			_t_10725 = tx1_4_1 + _t_10724;
			_t_10726 = _t_10725;
		
		}
else
		{
			float _t_10727;
			float _t_10728;
			float _t_10729;
		
			_t_10727 = -1.0f * tx3_6_1;
			_t_10728 = tx1_4_1 + _t_10727;
			_t_10729 = -1.0f * _t_10728;
			_t_10726 = _t_10729;
		
		}

	_t_10730 = _t_10726 * _t_427;
	_t_10731 = -1.0f * ty1_7_1;
	_t_10732 = ty3_9_1 + _t_10731;
	_t_10733 = -1.0f * _t_10732;
	_t_10734 = _t_10733 < 0.0f;
	if(_t_10734)
		{
			float _t_10735;
			float _t_10736;
		
			_t_10735 = -1.0f * tx3_6_1;
			_t_10736 = tx1_4_1 + _t_10735;
			_t_10737 = _t_10736;
		
		}
else
		{
			float _t_10738;
			float _t_10739;
			float _t_10740;
		
			_t_10738 = -1.0f * tx3_6_1;
			_t_10739 = tx1_4_1 + _t_10738;
			_t_10740 = -1.0f * _t_10739;
			_t_10737 = _t_10740;
		
		}

	_t_10741 = _t_10737 * _t_427;
	_t_10742 = 0.0f < _t_10741;
	if(_t_10742)
		{
		
			_t_10743 = py0_12_1;
		
		}
else
		{
		
			_t_10743 = py1_13_1;
		
		}

	_t_10744 = _t_10730 * _t_10743;
	_t_10745 = _t_10719 + _t_10744;
	_t_10746 = -1.0f * ty1_7_1;
	_t_10747 = ty3_9_1 + _t_10746;
	_t_10748 = -1.0f * _t_10747;
	_t_10749 = _t_10748 < 0.0f;
	if(_t_10749)
		{
			float _t_10750;
			float _t_10751;
			float _t_10752;
			float _t_10753;
		
			_t_10750 = tx3_6_1 * ty1_7_1;
			_t_10751 = tx1_4_1 * ty3_9_1;
			_t_10752 = _t_10751 * -1.0f;
			_t_10753 = _t_10750 + _t_10752;
			_t_10754 = _t_10753;
		
		}
else
		{
			float _t_10755;
			float _t_10756;
			float _t_10757;
			float _t_10758;
			float _t_10759;
		
			_t_10755 = tx3_6_1 * ty1_7_1;
			_t_10756 = tx1_4_1 * ty3_9_1;
			_t_10757 = _t_10756 * -1.0f;
			_t_10758 = _t_10755 + _t_10757;
			_t_10759 = -1.0f * _t_10758;
			_t_10754 = _t_10759;
		
		}

	_t_10760 = -1.0f * _t_10754;
	_t_10761 = _t_10760 * _t_427;
	_t_10762 = _t_10761 * -1.0f;
	_t_10763 = _t_10745 + _t_10762;
	_t_10764 = _t_10763 < 0.0f;
	_t_10765 = -1.0f * ty1_7_1;
	_t_10766 = ty3_9_1 + _t_10765;
	_t_10767 = -1.0f * _t_10766;
	_t_10768 = _t_10767 < 0.0f;
	if(_t_10768)
		{
			float _t_10769;
			float _t_10770;
		
			_t_10769 = -1.0f * ty1_7_1;
			_t_10770 = ty3_9_1 + _t_10769;
			_t_10771 = _t_10770;
		
		}
else
		{
			float _t_10772;
			float _t_10773;
			float _t_10774;
		
			_t_10772 = -1.0f * ty1_7_1;
			_t_10773 = ty3_9_1 + _t_10772;
			_t_10774 = -1.0f * _t_10773;
			_t_10771 = _t_10774;
		
		}

	_t_10775 = _t_10771 * _t_427;
	_t_10776 = -1.0f * ty1_7_1;
	_t_10777 = ty3_9_1 + _t_10776;
	_t_10778 = -1.0f * _t_10777;
	_t_10779 = _t_10778 < 0.0f;
	if(_t_10779)
		{
			float _t_10780;
			float _t_10781;
		
			_t_10780 = -1.0f * ty1_7_1;
			_t_10781 = ty3_9_1 + _t_10780;
			_t_10782 = _t_10781;
		
		}
else
		{
			float _t_10783;
			float _t_10784;
			float _t_10785;
		
			_t_10783 = -1.0f * ty1_7_1;
			_t_10784 = ty3_9_1 + _t_10783;
			_t_10785 = -1.0f * _t_10784;
			_t_10782 = _t_10785;
		
		}

	_t_10786 = _t_10782 * _t_427;
	_t_10787 = 0.0f < _t_10786;
	if(_t_10787)
		{
		
			_t_10788 = px1_11_1;
		
		}
else
		{
		
			_t_10788 = px0_10_1;
		
		}

	_t_10789 = _t_10775 * _t_10788;
	_t_10790 = -1.0f * ty1_7_1;
	_t_10791 = ty3_9_1 + _t_10790;
	_t_10792 = -1.0f * _t_10791;
	_t_10793 = _t_10792 < 0.0f;
	if(_t_10793)
		{
			float _t_10794;
			float _t_10795;
		
			_t_10794 = -1.0f * tx3_6_1;
			_t_10795 = tx1_4_1 + _t_10794;
			_t_10796 = _t_10795;
		
		}
else
		{
			float _t_10797;
			float _t_10798;
			float _t_10799;
		
			_t_10797 = -1.0f * tx3_6_1;
			_t_10798 = tx1_4_1 + _t_10797;
			_t_10799 = -1.0f * _t_10798;
			_t_10796 = _t_10799;
		
		}

	_t_10800 = _t_10796 * _t_427;
	_t_10801 = -1.0f * ty1_7_1;
	_t_10802 = ty3_9_1 + _t_10801;
	_t_10803 = -1.0f * _t_10802;
	_t_10804 = _t_10803 < 0.0f;
	if(_t_10804)
		{
			float _t_10805;
			float _t_10806;
		
			_t_10805 = -1.0f * tx3_6_1;
			_t_10806 = tx1_4_1 + _t_10805;
			_t_10807 = _t_10806;
		
		}
else
		{
			float _t_10808;
			float _t_10809;
			float _t_10810;
		
			_t_10808 = -1.0f * tx3_6_1;
			_t_10809 = tx1_4_1 + _t_10808;
			_t_10810 = -1.0f * _t_10809;
			_t_10807 = _t_10810;
		
		}

	_t_10811 = _t_10807 * _t_427;
	_t_10812 = 0.0f < _t_10811;
	if(_t_10812)
		{
		
			_t_10813 = py1_13_1;
		
		}
else
		{
		
			_t_10813 = py0_12_1;
		
		}

	_t_10814 = _t_10800 * _t_10813;
	_t_10815 = _t_10789 + _t_10814;
	_t_10816 = -1.0f * ty1_7_1;
	_t_10817 = ty3_9_1 + _t_10816;
	_t_10818 = -1.0f * _t_10817;
	_t_10819 = _t_10818 < 0.0f;
	if(_t_10819)
		{
			float _t_10820;
			float _t_10821;
			float _t_10822;
			float _t_10823;
		
			_t_10820 = tx3_6_1 * ty1_7_1;
			_t_10821 = tx1_4_1 * ty3_9_1;
			_t_10822 = _t_10821 * -1.0f;
			_t_10823 = _t_10820 + _t_10822;
			_t_10824 = _t_10823;
		
		}
else
		{
			float _t_10825;
			float _t_10826;
			float _t_10827;
			float _t_10828;
			float _t_10829;
		
			_t_10825 = tx3_6_1 * ty1_7_1;
			_t_10826 = tx1_4_1 * ty3_9_1;
			_t_10827 = _t_10826 * -1.0f;
			_t_10828 = _t_10825 + _t_10827;
			_t_10829 = -1.0f * _t_10828;
			_t_10824 = _t_10829;
		
		}

	_t_10830 = -1.0f * _t_10824;
	_t_10831 = _t_10830 * _t_427;
	_t_10832 = _t_10831 * -1.0f;
	_t_10833 = _t_10815 + _t_10832;
	_t_10834 = 0.0f < _t_10833;
	_t_10835 = _t_10764 && _t_10834;
	if(_t_10835)
		{
			float _t_10836;
			float _t_10837;
			float _t_10838;
			bool _t_10839;
			float _t_10844;
			float _t_10850;
			float _t_10851;
			float _t_10852;
		
			_t_10836 = -1.0f * ty1_7_1;
			_t_10837 = ty3_9_1 + _t_10836;
			_t_10838 = -1.0f * _t_10837;
			_t_10839 = _t_10838 < 0.0f;
			if(_t_10839)
				{
					float _t_10840;
					float _t_10841;
					float _t_10842;
					float _t_10843;
				
					_t_10840 = tx3_6_1 * ty1_7_1;
					_t_10841 = tx1_4_1 * ty3_9_1;
					_t_10842 = _t_10841 * -1.0f;
					_t_10843 = _t_10840 + _t_10842;
					_t_10844 = _t_10843;
				
				}
		else
				{
					float _t_10845;
					float _t_10846;
					float _t_10847;
					float _t_10848;
					float _t_10849;
				
					_t_10845 = tx3_6_1 * ty1_7_1;
					_t_10846 = tx1_4_1 * ty3_9_1;
					_t_10847 = _t_10846 * -1.0f;
					_t_10848 = _t_10845 + _t_10847;
					_t_10849 = -1.0f * _t_10848;
					_t_10844 = _t_10849;
				
				}
		
			_t_10850 = -1.0f * _t_10844;
			_t_10851 = _t_10850 * _t_427;
			_t_10852 = tegpixellet_block_41(ty3_9_1,ty1_7_1,_t_427,_t_10851,tx1_4_1,tx3_6_1,y__3387_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1);
			_t_10694 = _t_10852;
		
		}
else
		{
		
			_t_10694 = 0.0f;
		
		}


	return _t_10694;
}
__device__ float tegpixelintegrator_28(float ty3_9_1,float pc1_15_1,float tc2_19_1,float ty2_8_1,float _t_10693,float pc0_14_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float py1_13_1,float pc2_16_1,float tx2_5_1,float px1_11_1,float tc0_17_1,float _t_427,float py0_12_1,float tc1_18_1,float px0_10_1,float _t_10584){
    float y__3387_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_10693 - _t_10584)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__3387_1 = _t_10584 + __step__ * (i + (float)(0.5));
        float _t_10694;
		_t_10694 = tegpixelbody_block_28(ty3_9_1,ty1_7_1,_t_427,px0_10_1,px1_11_1,tx1_4_1,tx3_6_1,py0_12_1,py1_13_1,y__3387_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);;
        __output__ = __output__ + _t_10694 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_12(float ty3_9_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float _t_427,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_10476;
	float _t_10477;
	float _t_10478;
	bool _t_10479;
	float _t_10482;
	float _t_10486;
	float _t_10487;
	float _t_10488;
	float _t_10489;
	float _t_10490;
	bool _t_10491;
	float _t_10494;
	float _t_10498;
	float _t_10499;
	bool _t_10500;
	float _t_10501;
	float _t_10502;
	float _t_10503;
	float _t_10504;
	float _t_10505;
	bool _t_10506;
	float _t_10509;
	float _t_10513;
	float _t_10514;
	float _t_10515;
	float _t_10516;
	bool _t_10517;
	float _t_10520;
	float _t_10524;
	float _t_10525;
	float _t_10526;
	float _t_10527;
	float _t_10528;
	bool _t_10529;
	float _t_10532;
	float _t_10536;
	float _t_10537;
	float _t_10538;
	float _t_10539;
	float _t_10540;
	float _t_10541;
	float _t_10542;
	float _t_10543;
	float _t_10544;
	bool _t_10545;
	float _t_10548;
	float _t_10552;
	float _t_10553;
	float _t_10554;
	float _t_10555;
	bool _t_10556;
	float _t_10559;
	float _t_10563;
	float _t_10564;
	float _t_10565;
	float _t_10566;
	float _t_10567;
	bool _t_10568;
	float _t_10571;
	float _t_10575;
	float _t_10576;
	float _t_10577;
	float _t_10578;
	float _t_10579;
	float _t_10580;
	bool _t_10581;
	float _t_10582;
	float _t_10583;
	float _t_10584;
	float _t_10585;
	float _t_10586;
	float _t_10587;
	bool _t_10588;
	float _t_10591;
	float _t_10595;
	float _t_10596;
	float _t_10597;
	float _t_10598;
	float _t_10599;
	bool _t_10600;
	float _t_10603;
	float _t_10607;
	float _t_10608;
	bool _t_10609;
	float _t_10610;
	float _t_10611;
	float _t_10612;
	float _t_10613;
	float _t_10614;
	bool _t_10615;
	float _t_10618;
	float _t_10622;
	float _t_10623;
	float _t_10624;
	float _t_10625;
	bool _t_10626;
	float _t_10629;
	float _t_10633;
	float _t_10634;
	float _t_10635;
	float _t_10636;
	float _t_10637;
	bool _t_10638;
	float _t_10641;
	float _t_10645;
	float _t_10646;
	float _t_10647;
	float _t_10648;
	float _t_10649;
	float _t_10650;
	float _t_10651;
	float _t_10652;
	float _t_10653;
	bool _t_10654;
	float _t_10657;
	float _t_10661;
	float _t_10662;
	float _t_10663;
	float _t_10664;
	bool _t_10665;
	float _t_10668;
	float _t_10672;
	float _t_10673;
	float _t_10674;
	float _t_10675;
	float _t_10676;
	bool _t_10677;
	float _t_10680;
	float _t_10684;
	float _t_10685;
	float _t_10686;
	float _t_10687;
	float _t_10688;
	float _t_10689;
	bool _t_10690;
	float _t_10691;
	float _t_10692;
	float _t_10693;

	float _t_428;

	_t_10476 = -1.0f * ty1_7_1;
	_t_10477 = ty3_9_1 + _t_10476;
	_t_10478 = -1.0f * _t_10477;
	_t_10479 = _t_10478 < 0.0f;
	if(_t_10479)
		{
			float _t_10480;
			float _t_10481;
		
			_t_10480 = -1.0f * tx3_6_1;
			_t_10481 = tx1_4_1 + _t_10480;
			_t_10482 = _t_10481;
		
		}
else
		{
			float _t_10483;
			float _t_10484;
			float _t_10485;
		
			_t_10483 = -1.0f * tx3_6_1;
			_t_10484 = tx1_4_1 + _t_10483;
			_t_10485 = -1.0f * _t_10484;
			_t_10482 = _t_10485;
		
		}

	_t_10486 = _t_10482 * _t_427;
	_t_10487 = _t_10486 * -1.0f;
	_t_10488 = -1.0f * ty1_7_1;
	_t_10489 = ty3_9_1 + _t_10488;
	_t_10490 = -1.0f * _t_10489;
	_t_10491 = _t_10490 < 0.0f;
	if(_t_10491)
		{
			float _t_10492;
			float _t_10493;
		
			_t_10492 = -1.0f * tx3_6_1;
			_t_10493 = tx1_4_1 + _t_10492;
			_t_10494 = _t_10493;
		
		}
else
		{
			float _t_10495;
			float _t_10496;
			float _t_10497;
		
			_t_10495 = -1.0f * tx3_6_1;
			_t_10496 = tx1_4_1 + _t_10495;
			_t_10497 = -1.0f * _t_10496;
			_t_10494 = _t_10497;
		
		}

	_t_10498 = _t_10494 * _t_427;
	_t_10499 = _t_10498 * -1.0f;
	_t_10500 = 0.0f < _t_10499;
	if(_t_10500)
		{
		
			_t_10501 = px0_10_1;
		
		}
else
		{
		
			_t_10501 = px1_11_1;
		
		}

	_t_10502 = _t_10487 * _t_10501;
	_t_10503 = -1.0f * ty1_7_1;
	_t_10504 = ty3_9_1 + _t_10503;
	_t_10505 = -1.0f * _t_10504;
	_t_10506 = _t_10505 < 0.0f;
	if(_t_10506)
		{
			float _t_10507;
			float _t_10508;
		
			_t_10507 = -1.0f * tx3_6_1;
			_t_10508 = tx1_4_1 + _t_10507;
			_t_10509 = _t_10508;
		
		}
else
		{
			float _t_10510;
			float _t_10511;
			float _t_10512;
		
			_t_10510 = -1.0f * tx3_6_1;
			_t_10511 = tx1_4_1 + _t_10510;
			_t_10512 = -1.0f * _t_10511;
			_t_10509 = _t_10512;
		
		}

	_t_10513 = _t_10509 * _t_427;
	_t_10514 = -1.0f * ty1_7_1;
	_t_10515 = ty3_9_1 + _t_10514;
	_t_10516 = -1.0f * _t_10515;
	_t_10517 = _t_10516 < 0.0f;
	if(_t_10517)
		{
			float _t_10518;
			float _t_10519;
		
			_t_10518 = -1.0f * tx3_6_1;
			_t_10519 = tx1_4_1 + _t_10518;
			_t_10520 = _t_10519;
		
		}
else
		{
			float _t_10521;
			float _t_10522;
			float _t_10523;
		
			_t_10521 = -1.0f * tx3_6_1;
			_t_10522 = tx1_4_1 + _t_10521;
			_t_10523 = -1.0f * _t_10522;
			_t_10520 = _t_10523;
		
		}

	_t_10524 = _t_10520 * _t_427;
	_t_10525 = _t_10513 * _t_10524;
	_t_10526 = -1.0f * ty1_7_1;
	_t_10527 = ty3_9_1 + _t_10526;
	_t_10528 = -1.0f * _t_10527;
	_t_10529 = _t_10528 < 0.0f;
	if(_t_10529)
		{
			float _t_10530;
			float _t_10531;
		
			_t_10530 = -1.0f * ty1_7_1;
			_t_10531 = ty3_9_1 + _t_10530;
			_t_10532 = _t_10531;
		
		}
else
		{
			float _t_10533;
			float _t_10534;
			float _t_10535;
		
			_t_10533 = -1.0f * ty1_7_1;
			_t_10534 = ty3_9_1 + _t_10533;
			_t_10535 = -1.0f * _t_10534;
			_t_10532 = _t_10535;
		
		}

	_t_10536 = _t_10532 * _t_427;
	_t_10537 = 1.0f + _t_10536;
	_t_10538 = 1.0f / _t_10537;
	_t_10539 = _t_10525 * _t_10538;
	_t_10540 = _t_10539 * -1.0f;
	_t_10541 = 1.0f + _t_10540;
	_t_10542 = -1.0f * ty1_7_1;
	_t_10543 = ty3_9_1 + _t_10542;
	_t_10544 = -1.0f * _t_10543;
	_t_10545 = _t_10544 < 0.0f;
	if(_t_10545)
		{
			float _t_10546;
			float _t_10547;
		
			_t_10546 = -1.0f * tx3_6_1;
			_t_10547 = tx1_4_1 + _t_10546;
			_t_10548 = _t_10547;
		
		}
else
		{
			float _t_10549;
			float _t_10550;
			float _t_10551;
		
			_t_10549 = -1.0f * tx3_6_1;
			_t_10550 = tx1_4_1 + _t_10549;
			_t_10551 = -1.0f * _t_10550;
			_t_10548 = _t_10551;
		
		}

	_t_10552 = _t_10548 * _t_427;
	_t_10553 = -1.0f * ty1_7_1;
	_t_10554 = ty3_9_1 + _t_10553;
	_t_10555 = -1.0f * _t_10554;
	_t_10556 = _t_10555 < 0.0f;
	if(_t_10556)
		{
			float _t_10557;
			float _t_10558;
		
			_t_10557 = -1.0f * tx3_6_1;
			_t_10558 = tx1_4_1 + _t_10557;
			_t_10559 = _t_10558;
		
		}
else
		{
			float _t_10560;
			float _t_10561;
			float _t_10562;
		
			_t_10560 = -1.0f * tx3_6_1;
			_t_10561 = tx1_4_1 + _t_10560;
			_t_10562 = -1.0f * _t_10561;
			_t_10559 = _t_10562;
		
		}

	_t_10563 = _t_10559 * _t_427;
	_t_10564 = _t_10552 * _t_10563;
	_t_10565 = -1.0f * ty1_7_1;
	_t_10566 = ty3_9_1 + _t_10565;
	_t_10567 = -1.0f * _t_10566;
	_t_10568 = _t_10567 < 0.0f;
	if(_t_10568)
		{
			float _t_10569;
			float _t_10570;
		
			_t_10569 = -1.0f * ty1_7_1;
			_t_10570 = ty3_9_1 + _t_10569;
			_t_10571 = _t_10570;
		
		}
else
		{
			float _t_10572;
			float _t_10573;
			float _t_10574;
		
			_t_10572 = -1.0f * ty1_7_1;
			_t_10573 = ty3_9_1 + _t_10572;
			_t_10574 = -1.0f * _t_10573;
			_t_10571 = _t_10574;
		
		}

	_t_10575 = _t_10571 * _t_427;
	_t_10576 = 1.0f + _t_10575;
	_t_10577 = 1.0f / _t_10576;
	_t_10578 = _t_10564 * _t_10577;
	_t_10579 = _t_10578 * -1.0f;
	_t_10580 = 1.0f + _t_10579;
	_t_10581 = 0.0f < _t_10580;
	if(_t_10581)
		{
		
			_t_10582 = py0_12_1;
		
		}
else
		{
		
			_t_10582 = py1_13_1;
		
		}

	_t_10583 = _t_10541 * _t_10582;
	_t_10584 = _t_10502 + _t_10583;
	_t_10585 = -1.0f * ty1_7_1;
	_t_10586 = ty3_9_1 + _t_10585;
	_t_10587 = -1.0f * _t_10586;
	_t_10588 = _t_10587 < 0.0f;
	if(_t_10588)
		{
			float _t_10589;
			float _t_10590;
		
			_t_10589 = -1.0f * tx3_6_1;
			_t_10590 = tx1_4_1 + _t_10589;
			_t_10591 = _t_10590;
		
		}
else
		{
			float _t_10592;
			float _t_10593;
			float _t_10594;
		
			_t_10592 = -1.0f * tx3_6_1;
			_t_10593 = tx1_4_1 + _t_10592;
			_t_10594 = -1.0f * _t_10593;
			_t_10591 = _t_10594;
		
		}

	_t_10595 = _t_10591 * _t_427;
	_t_10596 = _t_10595 * -1.0f;
	_t_10597 = -1.0f * ty1_7_1;
	_t_10598 = ty3_9_1 + _t_10597;
	_t_10599 = -1.0f * _t_10598;
	_t_10600 = _t_10599 < 0.0f;
	if(_t_10600)
		{
			float _t_10601;
			float _t_10602;
		
			_t_10601 = -1.0f * tx3_6_1;
			_t_10602 = tx1_4_1 + _t_10601;
			_t_10603 = _t_10602;
		
		}
else
		{
			float _t_10604;
			float _t_10605;
			float _t_10606;
		
			_t_10604 = -1.0f * tx3_6_1;
			_t_10605 = tx1_4_1 + _t_10604;
			_t_10606 = -1.0f * _t_10605;
			_t_10603 = _t_10606;
		
		}

	_t_10607 = _t_10603 * _t_427;
	_t_10608 = _t_10607 * -1.0f;
	_t_10609 = 0.0f < _t_10608;
	if(_t_10609)
		{
		
			_t_10610 = px1_11_1;
		
		}
else
		{
		
			_t_10610 = px0_10_1;
		
		}

	_t_10611 = _t_10596 * _t_10610;
	_t_10612 = -1.0f * ty1_7_1;
	_t_10613 = ty3_9_1 + _t_10612;
	_t_10614 = -1.0f * _t_10613;
	_t_10615 = _t_10614 < 0.0f;
	if(_t_10615)
		{
			float _t_10616;
			float _t_10617;
		
			_t_10616 = -1.0f * tx3_6_1;
			_t_10617 = tx1_4_1 + _t_10616;
			_t_10618 = _t_10617;
		
		}
else
		{
			float _t_10619;
			float _t_10620;
			float _t_10621;
		
			_t_10619 = -1.0f * tx3_6_1;
			_t_10620 = tx1_4_1 + _t_10619;
			_t_10621 = -1.0f * _t_10620;
			_t_10618 = _t_10621;
		
		}

	_t_10622 = _t_10618 * _t_427;
	_t_10623 = -1.0f * ty1_7_1;
	_t_10624 = ty3_9_1 + _t_10623;
	_t_10625 = -1.0f * _t_10624;
	_t_10626 = _t_10625 < 0.0f;
	if(_t_10626)
		{
			float _t_10627;
			float _t_10628;
		
			_t_10627 = -1.0f * tx3_6_1;
			_t_10628 = tx1_4_1 + _t_10627;
			_t_10629 = _t_10628;
		
		}
else
		{
			float _t_10630;
			float _t_10631;
			float _t_10632;
		
			_t_10630 = -1.0f * tx3_6_1;
			_t_10631 = tx1_4_1 + _t_10630;
			_t_10632 = -1.0f * _t_10631;
			_t_10629 = _t_10632;
		
		}

	_t_10633 = _t_10629 * _t_427;
	_t_10634 = _t_10622 * _t_10633;
	_t_10635 = -1.0f * ty1_7_1;
	_t_10636 = ty3_9_1 + _t_10635;
	_t_10637 = -1.0f * _t_10636;
	_t_10638 = _t_10637 < 0.0f;
	if(_t_10638)
		{
			float _t_10639;
			float _t_10640;
		
			_t_10639 = -1.0f * ty1_7_1;
			_t_10640 = ty3_9_1 + _t_10639;
			_t_10641 = _t_10640;
		
		}
else
		{
			float _t_10642;
			float _t_10643;
			float _t_10644;
		
			_t_10642 = -1.0f * ty1_7_1;
			_t_10643 = ty3_9_1 + _t_10642;
			_t_10644 = -1.0f * _t_10643;
			_t_10641 = _t_10644;
		
		}

	_t_10645 = _t_10641 * _t_427;
	_t_10646 = 1.0f + _t_10645;
	_t_10647 = 1.0f / _t_10646;
	_t_10648 = _t_10634 * _t_10647;
	_t_10649 = _t_10648 * -1.0f;
	_t_10650 = 1.0f + _t_10649;
	_t_10651 = -1.0f * ty1_7_1;
	_t_10652 = ty3_9_1 + _t_10651;
	_t_10653 = -1.0f * _t_10652;
	_t_10654 = _t_10653 < 0.0f;
	if(_t_10654)
		{
			float _t_10655;
			float _t_10656;
		
			_t_10655 = -1.0f * tx3_6_1;
			_t_10656 = tx1_4_1 + _t_10655;
			_t_10657 = _t_10656;
		
		}
else
		{
			float _t_10658;
			float _t_10659;
			float _t_10660;
		
			_t_10658 = -1.0f * tx3_6_1;
			_t_10659 = tx1_4_1 + _t_10658;
			_t_10660 = -1.0f * _t_10659;
			_t_10657 = _t_10660;
		
		}

	_t_10661 = _t_10657 * _t_427;
	_t_10662 = -1.0f * ty1_7_1;
	_t_10663 = ty3_9_1 + _t_10662;
	_t_10664 = -1.0f * _t_10663;
	_t_10665 = _t_10664 < 0.0f;
	if(_t_10665)
		{
			float _t_10666;
			float _t_10667;
		
			_t_10666 = -1.0f * tx3_6_1;
			_t_10667 = tx1_4_1 + _t_10666;
			_t_10668 = _t_10667;
		
		}
else
		{
			float _t_10669;
			float _t_10670;
			float _t_10671;
		
			_t_10669 = -1.0f * tx3_6_1;
			_t_10670 = tx1_4_1 + _t_10669;
			_t_10671 = -1.0f * _t_10670;
			_t_10668 = _t_10671;
		
		}

	_t_10672 = _t_10668 * _t_427;
	_t_10673 = _t_10661 * _t_10672;
	_t_10674 = -1.0f * ty1_7_1;
	_t_10675 = ty3_9_1 + _t_10674;
	_t_10676 = -1.0f * _t_10675;
	_t_10677 = _t_10676 < 0.0f;
	if(_t_10677)
		{
			float _t_10678;
			float _t_10679;
		
			_t_10678 = -1.0f * ty1_7_1;
			_t_10679 = ty3_9_1 + _t_10678;
			_t_10680 = _t_10679;
		
		}
else
		{
			float _t_10681;
			float _t_10682;
			float _t_10683;
		
			_t_10681 = -1.0f * ty1_7_1;
			_t_10682 = ty3_9_1 + _t_10681;
			_t_10683 = -1.0f * _t_10682;
			_t_10680 = _t_10683;
		
		}

	_t_10684 = _t_10680 * _t_427;
	_t_10685 = 1.0f + _t_10684;
	_t_10686 = 1.0f / _t_10685;
	_t_10687 = _t_10673 * _t_10686;
	_t_10688 = _t_10687 * -1.0f;
	_t_10689 = 1.0f + _t_10688;
	_t_10690 = 0.0f < _t_10689;
	if(_t_10690)
		{
		
			_t_10691 = py1_13_1;
		
		}
else
		{
		
			_t_10691 = py0_12_1;
		
		}

	_t_10692 = _t_10650 * _t_10691;
	_t_10693 = _t_10611 + _t_10692;
	_t_428 = tegpixelintegrator_28(ty3_9_1,pc1_15_1,tc2_19_1,ty2_8_1,_t_10693,pc0_14_1,ty1_7_1,tx1_4_1,tx3_6_1,py1_13_1,pc2_16_1,tx2_5_1,px1_11_1,tc0_17_1,_t_427,py0_12_1,tc1_18_1,px0_10_1,_t_10584);

	return _t_428;
}
__device__ float tegpixellet_block_44(float py0_12_1,float _t_11766,float py1_13_1,float px0_10_1,float _t_11713,float px1_11_1,float ty1_7_1,float ty2_8_1,float tx2_5_1,float tx1_4_1,float _t_455,float y__3461_1,float _t_11686,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	bool _t_11767;
	bool _t_11768;
	bool _t_11769;
	bool _t_11770;
	bool _t_11771;
	bool _t_11772;
	bool _t_11773;
	float _t_12103;
	float _t_12104;
	float _t_12105;
	float _t_12106;
	float _t_12107;
	float _t_12108;
	float _t_12109;
	float _t_12110;
	float _t_12111;
	float _t_12112;
	float _t_12113;
	float _t_12114;
	float _t_12115;
	float _t_12116;
	float _t_12117;
	float _t_12118;
	float _t_12119;
	float _t_12120;
	float _t_12121;
	float _t_12122;
	float _t_12123;
	float _t_12124;
	float _t_12125;
	float _t_12126;
	bool _t_12127;
	float _t_12128;
	float _t_12129;
	float _t_12130;
	float _t_12131;
	float _t_12132;
	float _t_12133;
	float _t_12134;
	float _t_12135;
	float _t_12136;
	float _t_12137;
	float _t_12138;
	float _t_12139;
	float _t_12140;
	float _t_12141;
	bool _t_12142;
	float _t_12143;
	float _t_12144;
	float _t_12145;
	float _t_12146;
	float _t_12147;
	float _t_12148;
	float _t_12149;
	float _t_12150;
	float _t_12151;
	float _t_12152;
	float _t_12153;
	float _t_12154;
	float _t_12155;
	float _t_12156;
	float _t_12157;
	float _t_12158;
	float _t_12159;
	float _t_12160;
	float _t_12161;
	float _t_12162;
	float _t_12163;
	float _t_12164;
	float _t_12165;
	float _t_12166;
	float _t_12167;
	float _t_12168;
	float _t_12169;
	bool _t_12170;
	float _t_12171;
	float _t_12172;
	float _t_12173;
	float _t_12174;
	float _t_12175;
	float _t_12176;
	float _t_12177;
	float _t_12178;
	float _t_12179;
	float _t_12180;
	float _t_12181;
	float _t_12182;
	float _t_12183;
	float _t_12184;
	bool _t_12185;
	float _t_12186;
	float _t_12187;
	float _t_12188;
	float _t_12189;

	float _t_11687;

	_t_11767 = py0_12_1 < _t_11766;
	_t_11768 = _t_11766 < py1_13_1;
	_t_11769 = _t_11767 && _t_11768;
	_t_11770 = px0_10_1 < _t_11713;
	_t_11771 = _t_11713 < px1_11_1;
	_t_11772 = _t_11770 && _t_11771;
	_t_11773 = _t_11769 && _t_11772;
	if(_t_11773)
		{
			float _t_11774;
			float _t_11775;
			float _t_11776;
			bool _t_11777;
			float _t_11780;
			float _t_11784;
			float _t_11785;
			float _t_11786;
			float _t_11787;
			float _t_11788;
			bool _t_11789;
			float _t_11792;
			float _t_11796;
			float _t_11797;
			bool _t_11798;
			float _t_11799;
			float _t_11800;
			float _t_11801;
			float _t_11802;
			float _t_11803;
			bool _t_11804;
			float _t_11807;
			float _t_11811;
			float _t_11812;
			float _t_11813;
			float _t_11814;
			bool _t_11815;
			float _t_11818;
			float _t_11822;
			float _t_11823;
			float _t_11824;
			float _t_11825;
			float _t_11826;
			bool _t_11827;
			float _t_11830;
			float _t_11834;
			float _t_11835;
			float _t_11836;
			float _t_11837;
			float _t_11838;
			float _t_11839;
			float _t_11840;
			float _t_11841;
			float _t_11842;
			bool _t_11843;
			float _t_11846;
			float _t_11850;
			float _t_11851;
			float _t_11852;
			float _t_11853;
			bool _t_11854;
			float _t_11857;
			float _t_11861;
			float _t_11862;
			float _t_11863;
			float _t_11864;
			float _t_11865;
			bool _t_11866;
			float _t_11869;
			float _t_11873;
			float _t_11874;
			float _t_11875;
			float _t_11876;
			float _t_11877;
			float _t_11878;
			bool _t_11879;
			float _t_11880;
			float _t_11881;
			float _t_11882;
			bool _t_11883;
			float _t_11884;
			float _t_11885;
			float _t_11886;
			bool _t_11887;
			float _t_11890;
			float _t_11894;
			float _t_11895;
			float _t_11896;
			float _t_11897;
			float _t_11898;
			bool _t_11899;
			float _t_11902;
			float _t_11906;
			float _t_11907;
			bool _t_11908;
			float _t_11909;
			float _t_11910;
			float _t_11911;
			float _t_11912;
			float _t_11913;
			bool _t_11914;
			float _t_11917;
			float _t_11921;
			float _t_11922;
			float _t_11923;
			float _t_11924;
			bool _t_11925;
			float _t_11928;
			float _t_11932;
			float _t_11933;
			float _t_11934;
			float _t_11935;
			float _t_11936;
			bool _t_11937;
			float _t_11940;
			float _t_11944;
			float _t_11945;
			float _t_11946;
			float _t_11947;
			float _t_11948;
			float _t_11949;
			float _t_11950;
			float _t_11951;
			float _t_11952;
			bool _t_11953;
			float _t_11956;
			float _t_11960;
			float _t_11961;
			float _t_11962;
			float _t_11963;
			bool _t_11964;
			float _t_11967;
			float _t_11971;
			float _t_11972;
			float _t_11973;
			float _t_11974;
			float _t_11975;
			bool _t_11976;
			float _t_11979;
			float _t_11983;
			float _t_11984;
			float _t_11985;
			float _t_11986;
			float _t_11987;
			float _t_11988;
			bool _t_11989;
			float _t_11990;
			float _t_11991;
			float _t_11992;
			bool _t_11993;
			bool _t_11994;
			float _t_11995;
			float _t_11996;
			float _t_11997;
			bool _t_11998;
			float _t_12001;
			float _t_12005;
			float _t_12006;
			float _t_12007;
			float _t_12008;
			bool _t_12009;
			float _t_12012;
			float _t_12016;
			bool _t_12017;
			float _t_12018;
			float _t_12019;
			float _t_12020;
			float _t_12021;
			float _t_12022;
			bool _t_12023;
			float _t_12026;
			float _t_12030;
			float _t_12031;
			float _t_12032;
			float _t_12033;
			bool _t_12034;
			float _t_12037;
			float _t_12041;
			bool _t_12042;
			float _t_12043;
			float _t_12044;
			float _t_12045;
			bool _t_12046;
			float _t_12047;
			float _t_12048;
			float _t_12049;
			bool _t_12050;
			float _t_12053;
			float _t_12057;
			float _t_12058;
			float _t_12059;
			float _t_12060;
			bool _t_12061;
			float _t_12064;
			float _t_12068;
			bool _t_12069;
			float _t_12070;
			float _t_12071;
			float _t_12072;
			float _t_12073;
			float _t_12074;
			bool _t_12075;
			float _t_12078;
			float _t_12082;
			float _t_12083;
			float _t_12084;
			float _t_12085;
			bool _t_12086;
			float _t_12089;
			float _t_12093;
			bool _t_12094;
			float _t_12095;
			float _t_12096;
			float _t_12097;
			bool _t_12098;
			bool _t_12099;
			bool _t_12100;
			float _t_12101;
			float _t_12102;
		
			_t_11774 = -1.0f * ty2_8_1;
			_t_11775 = ty1_7_1 + _t_11774;
			_t_11776 = -1.0f * _t_11775;
			_t_11777 = _t_11776 < 0.0f;
			if(_t_11777)
				{
					float _t_11778;
					float _t_11779;
				
					_t_11778 = -1.0f * tx1_4_1;
					_t_11779 = tx2_5_1 + _t_11778;
					_t_11780 = _t_11779;
				
				}
		else
				{
					float _t_11781;
					float _t_11782;
					float _t_11783;
				
					_t_11781 = -1.0f * tx1_4_1;
					_t_11782 = tx2_5_1 + _t_11781;
					_t_11783 = -1.0f * _t_11782;
					_t_11780 = _t_11783;
				
				}
		
			_t_11784 = _t_11780 * _t_455;
			_t_11785 = _t_11784 * -1.0f;
			_t_11786 = -1.0f * ty2_8_1;
			_t_11787 = ty1_7_1 + _t_11786;
			_t_11788 = -1.0f * _t_11787;
			_t_11789 = _t_11788 < 0.0f;
			if(_t_11789)
				{
					float _t_11790;
					float _t_11791;
				
					_t_11790 = -1.0f * tx1_4_1;
					_t_11791 = tx2_5_1 + _t_11790;
					_t_11792 = _t_11791;
				
				}
		else
				{
					float _t_11793;
					float _t_11794;
					float _t_11795;
				
					_t_11793 = -1.0f * tx1_4_1;
					_t_11794 = tx2_5_1 + _t_11793;
					_t_11795 = -1.0f * _t_11794;
					_t_11792 = _t_11795;
				
				}
		
			_t_11796 = _t_11792 * _t_455;
			_t_11797 = _t_11796 * -1.0f;
			_t_11798 = 0.0f < _t_11797;
			if(_t_11798)
				{
				
					_t_11799 = px0_10_1;
				
				}
		else
				{
				
					_t_11799 = px1_11_1;
				
				}
		
			_t_11800 = _t_11785 * _t_11799;
			_t_11801 = -1.0f * ty2_8_1;
			_t_11802 = ty1_7_1 + _t_11801;
			_t_11803 = -1.0f * _t_11802;
			_t_11804 = _t_11803 < 0.0f;
			if(_t_11804)
				{
					float _t_11805;
					float _t_11806;
				
					_t_11805 = -1.0f * tx1_4_1;
					_t_11806 = tx2_5_1 + _t_11805;
					_t_11807 = _t_11806;
				
				}
		else
				{
					float _t_11808;
					float _t_11809;
					float _t_11810;
				
					_t_11808 = -1.0f * tx1_4_1;
					_t_11809 = tx2_5_1 + _t_11808;
					_t_11810 = -1.0f * _t_11809;
					_t_11807 = _t_11810;
				
				}
		
			_t_11811 = _t_11807 * _t_455;
			_t_11812 = -1.0f * ty2_8_1;
			_t_11813 = ty1_7_1 + _t_11812;
			_t_11814 = -1.0f * _t_11813;
			_t_11815 = _t_11814 < 0.0f;
			if(_t_11815)
				{
					float _t_11816;
					float _t_11817;
				
					_t_11816 = -1.0f * tx1_4_1;
					_t_11817 = tx2_5_1 + _t_11816;
					_t_11818 = _t_11817;
				
				}
		else
				{
					float _t_11819;
					float _t_11820;
					float _t_11821;
				
					_t_11819 = -1.0f * tx1_4_1;
					_t_11820 = tx2_5_1 + _t_11819;
					_t_11821 = -1.0f * _t_11820;
					_t_11818 = _t_11821;
				
				}
		
			_t_11822 = _t_11818 * _t_455;
			_t_11823 = _t_11811 * _t_11822;
			_t_11824 = -1.0f * ty2_8_1;
			_t_11825 = ty1_7_1 + _t_11824;
			_t_11826 = -1.0f * _t_11825;
			_t_11827 = _t_11826 < 0.0f;
			if(_t_11827)
				{
					float _t_11828;
					float _t_11829;
				
					_t_11828 = -1.0f * ty2_8_1;
					_t_11829 = ty1_7_1 + _t_11828;
					_t_11830 = _t_11829;
				
				}
		else
				{
					float _t_11831;
					float _t_11832;
					float _t_11833;
				
					_t_11831 = -1.0f * ty2_8_1;
					_t_11832 = ty1_7_1 + _t_11831;
					_t_11833 = -1.0f * _t_11832;
					_t_11830 = _t_11833;
				
				}
		
			_t_11834 = _t_11830 * _t_455;
			_t_11835 = 1.0f + _t_11834;
			_t_11836 = 1.0f / _t_11835;
			_t_11837 = _t_11823 * _t_11836;
			_t_11838 = _t_11837 * -1.0f;
			_t_11839 = 1.0f + _t_11838;
			_t_11840 = -1.0f * ty2_8_1;
			_t_11841 = ty1_7_1 + _t_11840;
			_t_11842 = -1.0f * _t_11841;
			_t_11843 = _t_11842 < 0.0f;
			if(_t_11843)
				{
					float _t_11844;
					float _t_11845;
				
					_t_11844 = -1.0f * tx1_4_1;
					_t_11845 = tx2_5_1 + _t_11844;
					_t_11846 = _t_11845;
				
				}
		else
				{
					float _t_11847;
					float _t_11848;
					float _t_11849;
				
					_t_11847 = -1.0f * tx1_4_1;
					_t_11848 = tx2_5_1 + _t_11847;
					_t_11849 = -1.0f * _t_11848;
					_t_11846 = _t_11849;
				
				}
		
			_t_11850 = _t_11846 * _t_455;
			_t_11851 = -1.0f * ty2_8_1;
			_t_11852 = ty1_7_1 + _t_11851;
			_t_11853 = -1.0f * _t_11852;
			_t_11854 = _t_11853 < 0.0f;
			if(_t_11854)
				{
					float _t_11855;
					float _t_11856;
				
					_t_11855 = -1.0f * tx1_4_1;
					_t_11856 = tx2_5_1 + _t_11855;
					_t_11857 = _t_11856;
				
				}
		else
				{
					float _t_11858;
					float _t_11859;
					float _t_11860;
				
					_t_11858 = -1.0f * tx1_4_1;
					_t_11859 = tx2_5_1 + _t_11858;
					_t_11860 = -1.0f * _t_11859;
					_t_11857 = _t_11860;
				
				}
		
			_t_11861 = _t_11857 * _t_455;
			_t_11862 = _t_11850 * _t_11861;
			_t_11863 = -1.0f * ty2_8_1;
			_t_11864 = ty1_7_1 + _t_11863;
			_t_11865 = -1.0f * _t_11864;
			_t_11866 = _t_11865 < 0.0f;
			if(_t_11866)
				{
					float _t_11867;
					float _t_11868;
				
					_t_11867 = -1.0f * ty2_8_1;
					_t_11868 = ty1_7_1 + _t_11867;
					_t_11869 = _t_11868;
				
				}
		else
				{
					float _t_11870;
					float _t_11871;
					float _t_11872;
				
					_t_11870 = -1.0f * ty2_8_1;
					_t_11871 = ty1_7_1 + _t_11870;
					_t_11872 = -1.0f * _t_11871;
					_t_11869 = _t_11872;
				
				}
		
			_t_11873 = _t_11869 * _t_455;
			_t_11874 = 1.0f + _t_11873;
			_t_11875 = 1.0f / _t_11874;
			_t_11876 = _t_11862 * _t_11875;
			_t_11877 = _t_11876 * -1.0f;
			_t_11878 = 1.0f + _t_11877;
			_t_11879 = 0.0f < _t_11878;
			if(_t_11879)
				{
				
					_t_11880 = py0_12_1;
				
				}
		else
				{
				
					_t_11880 = py1_13_1;
				
				}
		
			_t_11881 = _t_11839 * _t_11880;
			_t_11882 = _t_11800 + _t_11881;
			_t_11883 = _t_11882 < y__3461_1;
			_t_11884 = -1.0f * ty2_8_1;
			_t_11885 = ty1_7_1 + _t_11884;
			_t_11886 = -1.0f * _t_11885;
			_t_11887 = _t_11886 < 0.0f;
			if(_t_11887)
				{
					float _t_11888;
					float _t_11889;
				
					_t_11888 = -1.0f * tx1_4_1;
					_t_11889 = tx2_5_1 + _t_11888;
					_t_11890 = _t_11889;
				
				}
		else
				{
					float _t_11891;
					float _t_11892;
					float _t_11893;
				
					_t_11891 = -1.0f * tx1_4_1;
					_t_11892 = tx2_5_1 + _t_11891;
					_t_11893 = -1.0f * _t_11892;
					_t_11890 = _t_11893;
				
				}
		
			_t_11894 = _t_11890 * _t_455;
			_t_11895 = _t_11894 * -1.0f;
			_t_11896 = -1.0f * ty2_8_1;
			_t_11897 = ty1_7_1 + _t_11896;
			_t_11898 = -1.0f * _t_11897;
			_t_11899 = _t_11898 < 0.0f;
			if(_t_11899)
				{
					float _t_11900;
					float _t_11901;
				
					_t_11900 = -1.0f * tx1_4_1;
					_t_11901 = tx2_5_1 + _t_11900;
					_t_11902 = _t_11901;
				
				}
		else
				{
					float _t_11903;
					float _t_11904;
					float _t_11905;
				
					_t_11903 = -1.0f * tx1_4_1;
					_t_11904 = tx2_5_1 + _t_11903;
					_t_11905 = -1.0f * _t_11904;
					_t_11902 = _t_11905;
				
				}
		
			_t_11906 = _t_11902 * _t_455;
			_t_11907 = _t_11906 * -1.0f;
			_t_11908 = 0.0f < _t_11907;
			if(_t_11908)
				{
				
					_t_11909 = px1_11_1;
				
				}
		else
				{
				
					_t_11909 = px0_10_1;
				
				}
		
			_t_11910 = _t_11895 * _t_11909;
			_t_11911 = -1.0f * ty2_8_1;
			_t_11912 = ty1_7_1 + _t_11911;
			_t_11913 = -1.0f * _t_11912;
			_t_11914 = _t_11913 < 0.0f;
			if(_t_11914)
				{
					float _t_11915;
					float _t_11916;
				
					_t_11915 = -1.0f * tx1_4_1;
					_t_11916 = tx2_5_1 + _t_11915;
					_t_11917 = _t_11916;
				
				}
		else
				{
					float _t_11918;
					float _t_11919;
					float _t_11920;
				
					_t_11918 = -1.0f * tx1_4_1;
					_t_11919 = tx2_5_1 + _t_11918;
					_t_11920 = -1.0f * _t_11919;
					_t_11917 = _t_11920;
				
				}
		
			_t_11921 = _t_11917 * _t_455;
			_t_11922 = -1.0f * ty2_8_1;
			_t_11923 = ty1_7_1 + _t_11922;
			_t_11924 = -1.0f * _t_11923;
			_t_11925 = _t_11924 < 0.0f;
			if(_t_11925)
				{
					float _t_11926;
					float _t_11927;
				
					_t_11926 = -1.0f * tx1_4_1;
					_t_11927 = tx2_5_1 + _t_11926;
					_t_11928 = _t_11927;
				
				}
		else
				{
					float _t_11929;
					float _t_11930;
					float _t_11931;
				
					_t_11929 = -1.0f * tx1_4_1;
					_t_11930 = tx2_5_1 + _t_11929;
					_t_11931 = -1.0f * _t_11930;
					_t_11928 = _t_11931;
				
				}
		
			_t_11932 = _t_11928 * _t_455;
			_t_11933 = _t_11921 * _t_11932;
			_t_11934 = -1.0f * ty2_8_1;
			_t_11935 = ty1_7_1 + _t_11934;
			_t_11936 = -1.0f * _t_11935;
			_t_11937 = _t_11936 < 0.0f;
			if(_t_11937)
				{
					float _t_11938;
					float _t_11939;
				
					_t_11938 = -1.0f * ty2_8_1;
					_t_11939 = ty1_7_1 + _t_11938;
					_t_11940 = _t_11939;
				
				}
		else
				{
					float _t_11941;
					float _t_11942;
					float _t_11943;
				
					_t_11941 = -1.0f * ty2_8_1;
					_t_11942 = ty1_7_1 + _t_11941;
					_t_11943 = -1.0f * _t_11942;
					_t_11940 = _t_11943;
				
				}
		
			_t_11944 = _t_11940 * _t_455;
			_t_11945 = 1.0f + _t_11944;
			_t_11946 = 1.0f / _t_11945;
			_t_11947 = _t_11933 * _t_11946;
			_t_11948 = _t_11947 * -1.0f;
			_t_11949 = 1.0f + _t_11948;
			_t_11950 = -1.0f * ty2_8_1;
			_t_11951 = ty1_7_1 + _t_11950;
			_t_11952 = -1.0f * _t_11951;
			_t_11953 = _t_11952 < 0.0f;
			if(_t_11953)
				{
					float _t_11954;
					float _t_11955;
				
					_t_11954 = -1.0f * tx1_4_1;
					_t_11955 = tx2_5_1 + _t_11954;
					_t_11956 = _t_11955;
				
				}
		else
				{
					float _t_11957;
					float _t_11958;
					float _t_11959;
				
					_t_11957 = -1.0f * tx1_4_1;
					_t_11958 = tx2_5_1 + _t_11957;
					_t_11959 = -1.0f * _t_11958;
					_t_11956 = _t_11959;
				
				}
		
			_t_11960 = _t_11956 * _t_455;
			_t_11961 = -1.0f * ty2_8_1;
			_t_11962 = ty1_7_1 + _t_11961;
			_t_11963 = -1.0f * _t_11962;
			_t_11964 = _t_11963 < 0.0f;
			if(_t_11964)
				{
					float _t_11965;
					float _t_11966;
				
					_t_11965 = -1.0f * tx1_4_1;
					_t_11966 = tx2_5_1 + _t_11965;
					_t_11967 = _t_11966;
				
				}
		else
				{
					float _t_11968;
					float _t_11969;
					float _t_11970;
				
					_t_11968 = -1.0f * tx1_4_1;
					_t_11969 = tx2_5_1 + _t_11968;
					_t_11970 = -1.0f * _t_11969;
					_t_11967 = _t_11970;
				
				}
		
			_t_11971 = _t_11967 * _t_455;
			_t_11972 = _t_11960 * _t_11971;
			_t_11973 = -1.0f * ty2_8_1;
			_t_11974 = ty1_7_1 + _t_11973;
			_t_11975 = -1.0f * _t_11974;
			_t_11976 = _t_11975 < 0.0f;
			if(_t_11976)
				{
					float _t_11977;
					float _t_11978;
				
					_t_11977 = -1.0f * ty2_8_1;
					_t_11978 = ty1_7_1 + _t_11977;
					_t_11979 = _t_11978;
				
				}
		else
				{
					float _t_11980;
					float _t_11981;
					float _t_11982;
				
					_t_11980 = -1.0f * ty2_8_1;
					_t_11981 = ty1_7_1 + _t_11980;
					_t_11982 = -1.0f * _t_11981;
					_t_11979 = _t_11982;
				
				}
		
			_t_11983 = _t_11979 * _t_455;
			_t_11984 = 1.0f + _t_11983;
			_t_11985 = 1.0f / _t_11984;
			_t_11986 = _t_11972 * _t_11985;
			_t_11987 = _t_11986 * -1.0f;
			_t_11988 = 1.0f + _t_11987;
			_t_11989 = 0.0f < _t_11988;
			if(_t_11989)
				{
				
					_t_11990 = py1_13_1;
				
				}
		else
				{
				
					_t_11990 = py0_12_1;
				
				}
		
			_t_11991 = _t_11949 * _t_11990;
			_t_11992 = _t_11910 + _t_11991;
			_t_11993 = y__3461_1 < _t_11992;
			_t_11994 = _t_11883 && _t_11993;
			_t_11995 = -1.0f * ty2_8_1;
			_t_11996 = ty1_7_1 + _t_11995;
			_t_11997 = -1.0f * _t_11996;
			_t_11998 = _t_11997 < 0.0f;
			if(_t_11998)
				{
					float _t_11999;
					float _t_12000;
				
					_t_11999 = -1.0f * ty2_8_1;
					_t_12000 = ty1_7_1 + _t_11999;
					_t_12001 = _t_12000;
				
				}
		else
				{
					float _t_12002;
					float _t_12003;
					float _t_12004;
				
					_t_12002 = -1.0f * ty2_8_1;
					_t_12003 = ty1_7_1 + _t_12002;
					_t_12004 = -1.0f * _t_12003;
					_t_12001 = _t_12004;
				
				}
		
			_t_12005 = _t_12001 * _t_455;
			_t_12006 = -1.0f * ty2_8_1;
			_t_12007 = ty1_7_1 + _t_12006;
			_t_12008 = -1.0f * _t_12007;
			_t_12009 = _t_12008 < 0.0f;
			if(_t_12009)
				{
					float _t_12010;
					float _t_12011;
				
					_t_12010 = -1.0f * ty2_8_1;
					_t_12011 = ty1_7_1 + _t_12010;
					_t_12012 = _t_12011;
				
				}
		else
				{
					float _t_12013;
					float _t_12014;
					float _t_12015;
				
					_t_12013 = -1.0f * ty2_8_1;
					_t_12014 = ty1_7_1 + _t_12013;
					_t_12015 = -1.0f * _t_12014;
					_t_12012 = _t_12015;
				
				}
		
			_t_12016 = _t_12012 * _t_455;
			_t_12017 = 0.0f < _t_12016;
			if(_t_12017)
				{
				
					_t_12018 = px0_10_1;
				
				}
		else
				{
				
					_t_12018 = px1_11_1;
				
				}
		
			_t_12019 = _t_12005 * _t_12018;
			_t_12020 = -1.0f * ty2_8_1;
			_t_12021 = ty1_7_1 + _t_12020;
			_t_12022 = -1.0f * _t_12021;
			_t_12023 = _t_12022 < 0.0f;
			if(_t_12023)
				{
					float _t_12024;
					float _t_12025;
				
					_t_12024 = -1.0f * tx1_4_1;
					_t_12025 = tx2_5_1 + _t_12024;
					_t_12026 = _t_12025;
				
				}
		else
				{
					float _t_12027;
					float _t_12028;
					float _t_12029;
				
					_t_12027 = -1.0f * tx1_4_1;
					_t_12028 = tx2_5_1 + _t_12027;
					_t_12029 = -1.0f * _t_12028;
					_t_12026 = _t_12029;
				
				}
		
			_t_12030 = _t_12026 * _t_455;
			_t_12031 = -1.0f * ty2_8_1;
			_t_12032 = ty1_7_1 + _t_12031;
			_t_12033 = -1.0f * _t_12032;
			_t_12034 = _t_12033 < 0.0f;
			if(_t_12034)
				{
					float _t_12035;
					float _t_12036;
				
					_t_12035 = -1.0f * tx1_4_1;
					_t_12036 = tx2_5_1 + _t_12035;
					_t_12037 = _t_12036;
				
				}
		else
				{
					float _t_12038;
					float _t_12039;
					float _t_12040;
				
					_t_12038 = -1.0f * tx1_4_1;
					_t_12039 = tx2_5_1 + _t_12038;
					_t_12040 = -1.0f * _t_12039;
					_t_12037 = _t_12040;
				
				}
		
			_t_12041 = _t_12037 * _t_455;
			_t_12042 = 0.0f < _t_12041;
			if(_t_12042)
				{
				
					_t_12043 = py0_12_1;
				
				}
		else
				{
				
					_t_12043 = py1_13_1;
				
				}
		
			_t_12044 = _t_12030 * _t_12043;
			_t_12045 = _t_12019 + _t_12044;
			_t_12046 = _t_12045 < _t_11686;
			_t_12047 = -1.0f * ty2_8_1;
			_t_12048 = ty1_7_1 + _t_12047;
			_t_12049 = -1.0f * _t_12048;
			_t_12050 = _t_12049 < 0.0f;
			if(_t_12050)
				{
					float _t_12051;
					float _t_12052;
				
					_t_12051 = -1.0f * ty2_8_1;
					_t_12052 = ty1_7_1 + _t_12051;
					_t_12053 = _t_12052;
				
				}
		else
				{
					float _t_12054;
					float _t_12055;
					float _t_12056;
				
					_t_12054 = -1.0f * ty2_8_1;
					_t_12055 = ty1_7_1 + _t_12054;
					_t_12056 = -1.0f * _t_12055;
					_t_12053 = _t_12056;
				
				}
		
			_t_12057 = _t_12053 * _t_455;
			_t_12058 = -1.0f * ty2_8_1;
			_t_12059 = ty1_7_1 + _t_12058;
			_t_12060 = -1.0f * _t_12059;
			_t_12061 = _t_12060 < 0.0f;
			if(_t_12061)
				{
					float _t_12062;
					float _t_12063;
				
					_t_12062 = -1.0f * ty2_8_1;
					_t_12063 = ty1_7_1 + _t_12062;
					_t_12064 = _t_12063;
				
				}
		else
				{
					float _t_12065;
					float _t_12066;
					float _t_12067;
				
					_t_12065 = -1.0f * ty2_8_1;
					_t_12066 = ty1_7_1 + _t_12065;
					_t_12067 = -1.0f * _t_12066;
					_t_12064 = _t_12067;
				
				}
		
			_t_12068 = _t_12064 * _t_455;
			_t_12069 = 0.0f < _t_12068;
			if(_t_12069)
				{
				
					_t_12070 = px1_11_1;
				
				}
		else
				{
				
					_t_12070 = px0_10_1;
				
				}
		
			_t_12071 = _t_12057 * _t_12070;
			_t_12072 = -1.0f * ty2_8_1;
			_t_12073 = ty1_7_1 + _t_12072;
			_t_12074 = -1.0f * _t_12073;
			_t_12075 = _t_12074 < 0.0f;
			if(_t_12075)
				{
					float _t_12076;
					float _t_12077;
				
					_t_12076 = -1.0f * tx1_4_1;
					_t_12077 = tx2_5_1 + _t_12076;
					_t_12078 = _t_12077;
				
				}
		else
				{
					float _t_12079;
					float _t_12080;
					float _t_12081;
				
					_t_12079 = -1.0f * tx1_4_1;
					_t_12080 = tx2_5_1 + _t_12079;
					_t_12081 = -1.0f * _t_12080;
					_t_12078 = _t_12081;
				
				}
		
			_t_12082 = _t_12078 * _t_455;
			_t_12083 = -1.0f * ty2_8_1;
			_t_12084 = ty1_7_1 + _t_12083;
			_t_12085 = -1.0f * _t_12084;
			_t_12086 = _t_12085 < 0.0f;
			if(_t_12086)
				{
					float _t_12087;
					float _t_12088;
				
					_t_12087 = -1.0f * tx1_4_1;
					_t_12088 = tx2_5_1 + _t_12087;
					_t_12089 = _t_12088;
				
				}
		else
				{
					float _t_12090;
					float _t_12091;
					float _t_12092;
				
					_t_12090 = -1.0f * tx1_4_1;
					_t_12091 = tx2_5_1 + _t_12090;
					_t_12092 = -1.0f * _t_12091;
					_t_12089 = _t_12092;
				
				}
		
			_t_12093 = _t_12089 * _t_455;
			_t_12094 = 0.0f < _t_12093;
			if(_t_12094)
				{
				
					_t_12095 = py1_13_1;
				
				}
		else
				{
				
					_t_12095 = py0_12_1;
				
				}
		
			_t_12096 = _t_12082 * _t_12095;
			_t_12097 = _t_12071 + _t_12096;
			_t_12098 = _t_11686 < _t_12097;
			_t_12099 = _t_12046 && _t_12098;
			_t_12100 = _t_11994 && _t_12099;
			if(_t_12100)
				{
				
					_t_12101 = 1.0f;
				
				}
		else
				{
				
					_t_12101 = 0.0f;
				
				}
		
			_t_12102 = _t_12101 * _t_455;
			_t_12103 = _t_12102;
		
		}
else
		{
		
			_t_12103 = 0.0f;
		
		}

	_t_12104 = -1.0f * pc0_14_1;
	_t_12105 = tc0_17_1 + _t_12104;
	_t_12106 = _t_12105 * _t_12105;
	_t_12107 = -1.0f * pc1_15_1;
	_t_12108 = tc1_18_1 + _t_12107;
	_t_12109 = _t_12108 * _t_12108;
	_t_12110 = _t_12106 + _t_12109;
	_t_12111 = -1.0f * pc2_16_1;
	_t_12112 = tc2_19_1 + _t_12111;
	_t_12113 = _t_12112 * _t_12112;
	_t_12114 = _t_12110 + _t_12113;
	_t_12115 = tx3_6_1 * ty1_7_1;
	_t_12116 = tx1_4_1 * ty3_9_1;
	_t_12117 = _t_12116 * -1.0f;
	_t_12118 = _t_12115 + _t_12117;
	_t_12119 = -1.0f * ty1_7_1;
	_t_12120 = ty3_9_1 + _t_12119;
	_t_12121 = _t_12120 * _t_11713;
	_t_12122 = _t_12118 + _t_12121;
	_t_12123 = -1.0f * tx3_6_1;
	_t_12124 = tx1_4_1 + _t_12123;
	_t_12125 = _t_12124 * _t_11766;
	_t_12126 = _t_12122 + _t_12125;
	_t_12127 = _t_12126 < 0.0f;
	if(_t_12127)
		{
		
			_t_12128 = 1.0f;
		
		}
else
		{
		
			_t_12128 = 0.0f;
		
		}

	_t_12129 = _t_12114 * _t_12128;
	_t_12130 = tx2_5_1 * ty3_9_1;
	_t_12131 = tx3_6_1 * ty2_8_1;
	_t_12132 = _t_12131 * -1.0f;
	_t_12133 = _t_12130 + _t_12132;
	_t_12134 = -1.0f * ty3_9_1;
	_t_12135 = ty2_8_1 + _t_12134;
	_t_12136 = _t_12135 * _t_11713;
	_t_12137 = _t_12133 + _t_12136;
	_t_12138 = -1.0f * tx2_5_1;
	_t_12139 = tx3_6_1 + _t_12138;
	_t_12140 = _t_12139 * _t_11766;
	_t_12141 = _t_12137 + _t_12140;
	_t_12142 = _t_12141 < 0.0f;
	if(_t_12142)
		{
		
			_t_12143 = 1.0f;
		
		}
else
		{
		
			_t_12143 = 0.0f;
		
		}

	_t_12144 = _t_12129 * _t_12143;
	_t_12145 = _t_12144 * tx1_4_1;
	_t_12146 = _t_12145 * -1.0f;
	_t_12147 = -1.0f * pc0_14_1;
	_t_12148 = tc0_17_1 + _t_12147;
	_t_12149 = _t_12148 * _t_12148;
	_t_12150 = -1.0f * pc1_15_1;
	_t_12151 = tc1_18_1 + _t_12150;
	_t_12152 = _t_12151 * _t_12151;
	_t_12153 = _t_12149 + _t_12152;
	_t_12154 = -1.0f * pc2_16_1;
	_t_12155 = tc2_19_1 + _t_12154;
	_t_12156 = _t_12155 * _t_12155;
	_t_12157 = _t_12153 + _t_12156;
	_t_12158 = tx3_6_1 * ty1_7_1;
	_t_12159 = tx1_4_1 * ty3_9_1;
	_t_12160 = _t_12159 * -1.0f;
	_t_12161 = _t_12158 + _t_12160;
	_t_12162 = -1.0f * ty1_7_1;
	_t_12163 = ty3_9_1 + _t_12162;
	_t_12164 = _t_12163 * _t_11713;
	_t_12165 = _t_12161 + _t_12164;
	_t_12166 = -1.0f * tx3_6_1;
	_t_12167 = tx1_4_1 + _t_12166;
	_t_12168 = _t_12167 * _t_11766;
	_t_12169 = _t_12165 + _t_12168;
	_t_12170 = _t_12169 < 0.0f;
	if(_t_12170)
		{
		
			_t_12171 = 1.0f;
		
		}
else
		{
		
			_t_12171 = 0.0f;
		
		}

	_t_12172 = _t_12157 * _t_12171;
	_t_12173 = tx2_5_1 * ty3_9_1;
	_t_12174 = tx3_6_1 * ty2_8_1;
	_t_12175 = _t_12174 * -1.0f;
	_t_12176 = _t_12173 + _t_12175;
	_t_12177 = -1.0f * ty3_9_1;
	_t_12178 = ty2_8_1 + _t_12177;
	_t_12179 = _t_12178 * _t_11713;
	_t_12180 = _t_12176 + _t_12179;
	_t_12181 = -1.0f * tx2_5_1;
	_t_12182 = tx3_6_1 + _t_12181;
	_t_12183 = _t_12182 * _t_11766;
	_t_12184 = _t_12180 + _t_12183;
	_t_12185 = _t_12184 < 0.0f;
	if(_t_12185)
		{
		
			_t_12186 = 1.0f;
		
		}
else
		{
		
			_t_12186 = 0.0f;
		
		}

	_t_12187 = _t_12172 * _t_12186;
	_t_12188 = _t_12187 * _t_11713;
	_t_12189 = _t_12146 + _t_12188;
	_t_11687 = _t_12103 * _t_12189;

	return _t_11687;
}
__device__ float tegpixellet_block_43(float ty1_7_1,float ty2_8_1,float _t_455,float _t_11686,float tx2_5_1,float tx1_4_1,float y__3461_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	float _t_11688;
	float _t_11689;
	float _t_11690;
	bool _t_11691;
	float _t_11694;
	float _t_11698;
	float _t_11699;
	float _t_11700;
	float _t_11701;
	float _t_11702;
	bool _t_11703;
	float _t_11706;
	float _t_11710;
	float _t_11711;
	float _t_11712;
	float _t_11713;
	float _t_11714;
	float _t_11715;
	float _t_11716;
	bool _t_11717;
	float _t_11720;
	float _t_11724;
	float _t_11725;
	float _t_11726;
	float _t_11727;
	bool _t_11728;
	float _t_11731;
	float _t_11735;
	float _t_11736;
	float _t_11737;
	float _t_11738;
	float _t_11739;
	bool _t_11740;
	float _t_11743;
	float _t_11747;
	float _t_11748;
	float _t_11749;
	float _t_11750;
	float _t_11751;
	float _t_11752;
	float _t_11753;
	float _t_11754;
	float _t_11755;
	float _t_11756;
	bool _t_11757;
	float _t_11760;
	float _t_11764;
	float _t_11765;
	float _t_11766;

	float _t_11687;

	_t_11688 = -1.0f * ty2_8_1;
	_t_11689 = ty1_7_1 + _t_11688;
	_t_11690 = -1.0f * _t_11689;
	_t_11691 = _t_11690 < 0.0f;
	if(_t_11691)
		{
			float _t_11692;
			float _t_11693;
		
			_t_11692 = -1.0f * ty2_8_1;
			_t_11693 = ty1_7_1 + _t_11692;
			_t_11694 = _t_11693;
		
		}
else
		{
			float _t_11695;
			float _t_11696;
			float _t_11697;
		
			_t_11695 = -1.0f * ty2_8_1;
			_t_11696 = ty1_7_1 + _t_11695;
			_t_11697 = -1.0f * _t_11696;
			_t_11694 = _t_11697;
		
		}

	_t_11698 = _t_11694 * _t_455;
	_t_11699 = _t_11698 * _t_11686;
	_t_11700 = -1.0f * ty2_8_1;
	_t_11701 = ty1_7_1 + _t_11700;
	_t_11702 = -1.0f * _t_11701;
	_t_11703 = _t_11702 < 0.0f;
	if(_t_11703)
		{
			float _t_11704;
			float _t_11705;
		
			_t_11704 = -1.0f * tx1_4_1;
			_t_11705 = tx2_5_1 + _t_11704;
			_t_11706 = _t_11705;
		
		}
else
		{
			float _t_11707;
			float _t_11708;
			float _t_11709;
		
			_t_11707 = -1.0f * tx1_4_1;
			_t_11708 = tx2_5_1 + _t_11707;
			_t_11709 = -1.0f * _t_11708;
			_t_11706 = _t_11709;
		
		}

	_t_11710 = _t_11706 * _t_455;
	_t_11711 = _t_11710 * -1.0f;
	_t_11712 = _t_11711 * y__3461_1;
	_t_11713 = _t_11699 + _t_11712;
	_t_11714 = -1.0f * ty2_8_1;
	_t_11715 = ty1_7_1 + _t_11714;
	_t_11716 = -1.0f * _t_11715;
	_t_11717 = _t_11716 < 0.0f;
	if(_t_11717)
		{
			float _t_11718;
			float _t_11719;
		
			_t_11718 = -1.0f * tx1_4_1;
			_t_11719 = tx2_5_1 + _t_11718;
			_t_11720 = _t_11719;
		
		}
else
		{
			float _t_11721;
			float _t_11722;
			float _t_11723;
		
			_t_11721 = -1.0f * tx1_4_1;
			_t_11722 = tx2_5_1 + _t_11721;
			_t_11723 = -1.0f * _t_11722;
			_t_11720 = _t_11723;
		
		}

	_t_11724 = _t_11720 * _t_455;
	_t_11725 = -1.0f * ty2_8_1;
	_t_11726 = ty1_7_1 + _t_11725;
	_t_11727 = -1.0f * _t_11726;
	_t_11728 = _t_11727 < 0.0f;
	if(_t_11728)
		{
			float _t_11729;
			float _t_11730;
		
			_t_11729 = -1.0f * tx1_4_1;
			_t_11730 = tx2_5_1 + _t_11729;
			_t_11731 = _t_11730;
		
		}
else
		{
			float _t_11732;
			float _t_11733;
			float _t_11734;
		
			_t_11732 = -1.0f * tx1_4_1;
			_t_11733 = tx2_5_1 + _t_11732;
			_t_11734 = -1.0f * _t_11733;
			_t_11731 = _t_11734;
		
		}

	_t_11735 = _t_11731 * _t_455;
	_t_11736 = _t_11724 * _t_11735;
	_t_11737 = -1.0f * ty2_8_1;
	_t_11738 = ty1_7_1 + _t_11737;
	_t_11739 = -1.0f * _t_11738;
	_t_11740 = _t_11739 < 0.0f;
	if(_t_11740)
		{
			float _t_11741;
			float _t_11742;
		
			_t_11741 = -1.0f * ty2_8_1;
			_t_11742 = ty1_7_1 + _t_11741;
			_t_11743 = _t_11742;
		
		}
else
		{
			float _t_11744;
			float _t_11745;
			float _t_11746;
		
			_t_11744 = -1.0f * ty2_8_1;
			_t_11745 = ty1_7_1 + _t_11744;
			_t_11746 = -1.0f * _t_11745;
			_t_11743 = _t_11746;
		
		}

	_t_11747 = _t_11743 * _t_455;
	_t_11748 = 1.0f + _t_11747;
	_t_11749 = 1.0f / _t_11748;
	_t_11750 = _t_11736 * _t_11749;
	_t_11751 = _t_11750 * -1.0f;
	_t_11752 = 1.0f + _t_11751;
	_t_11753 = _t_11752 * y__3461_1;
	_t_11754 = -1.0f * ty2_8_1;
	_t_11755 = ty1_7_1 + _t_11754;
	_t_11756 = -1.0f * _t_11755;
	_t_11757 = _t_11756 < 0.0f;
	if(_t_11757)
		{
			float _t_11758;
			float _t_11759;
		
			_t_11758 = -1.0f * tx1_4_1;
			_t_11759 = tx2_5_1 + _t_11758;
			_t_11760 = _t_11759;
		
		}
else
		{
			float _t_11761;
			float _t_11762;
			float _t_11763;
		
			_t_11761 = -1.0f * tx1_4_1;
			_t_11762 = tx2_5_1 + _t_11761;
			_t_11763 = -1.0f * _t_11762;
			_t_11760 = _t_11763;
		
		}

	_t_11764 = _t_11760 * _t_455;
	_t_11765 = _t_11764 * _t_11686;
	_t_11766 = _t_11753 + _t_11765;
	_t_11687 = tegpixellet_block_44(py0_12_1,_t_11766,py1_13_1,px0_10_1,_t_11713,px1_11_1,ty1_7_1,ty2_8_1,tx2_5_1,tx1_4_1,_t_455,y__3461_1,_t_11686,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);

	return _t_11687;
}
__device__ float tegpixelbody_block_29(float ty1_7_1,float ty2_8_1,float _t_455,float px0_10_1,float px1_11_1,float tx2_5_1,float tx1_4_1,float py0_12_1,float py1_13_1,float y__3461_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	float _t_11530;
	float _t_11531;
	float _t_11532;
	bool _t_11533;
	float _t_11536;
	float _t_11540;
	float _t_11541;
	float _t_11542;
	float _t_11543;
	bool _t_11544;
	float _t_11547;
	float _t_11551;
	bool _t_11552;
	float _t_11553;
	float _t_11554;
	float _t_11555;
	float _t_11556;
	float _t_11557;
	bool _t_11558;
	float _t_11561;
	float _t_11565;
	float _t_11566;
	float _t_11567;
	float _t_11568;
	bool _t_11569;
	float _t_11572;
	float _t_11576;
	bool _t_11577;
	float _t_11578;
	float _t_11579;
	float _t_11580;
	float _t_11581;
	float _t_11582;
	float _t_11583;
	bool _t_11584;
	float _t_11589;
	float _t_11595;
	float _t_11596;
	float _t_11597;
	float _t_11598;
	bool _t_11599;
	float _t_11600;
	float _t_11601;
	float _t_11602;
	bool _t_11603;
	float _t_11606;
	float _t_11610;
	float _t_11611;
	float _t_11612;
	float _t_11613;
	bool _t_11614;
	float _t_11617;
	float _t_11621;
	bool _t_11622;
	float _t_11623;
	float _t_11624;
	float _t_11625;
	float _t_11626;
	float _t_11627;
	bool _t_11628;
	float _t_11631;
	float _t_11635;
	float _t_11636;
	float _t_11637;
	float _t_11638;
	bool _t_11639;
	float _t_11642;
	float _t_11646;
	bool _t_11647;
	float _t_11648;
	float _t_11649;
	float _t_11650;
	float _t_11651;
	float _t_11652;
	float _t_11653;
	bool _t_11654;
	float _t_11659;
	float _t_11665;
	float _t_11666;
	float _t_11667;
	float _t_11668;
	bool _t_11669;
	bool _t_11670;

	float _t_11529;

	_t_11530 = -1.0f * ty2_8_1;
	_t_11531 = ty1_7_1 + _t_11530;
	_t_11532 = -1.0f * _t_11531;
	_t_11533 = _t_11532 < 0.0f;
	if(_t_11533)
		{
			float _t_11534;
			float _t_11535;
		
			_t_11534 = -1.0f * ty2_8_1;
			_t_11535 = ty1_7_1 + _t_11534;
			_t_11536 = _t_11535;
		
		}
else
		{
			float _t_11537;
			float _t_11538;
			float _t_11539;
		
			_t_11537 = -1.0f * ty2_8_1;
			_t_11538 = ty1_7_1 + _t_11537;
			_t_11539 = -1.0f * _t_11538;
			_t_11536 = _t_11539;
		
		}

	_t_11540 = _t_11536 * _t_455;
	_t_11541 = -1.0f * ty2_8_1;
	_t_11542 = ty1_7_1 + _t_11541;
	_t_11543 = -1.0f * _t_11542;
	_t_11544 = _t_11543 < 0.0f;
	if(_t_11544)
		{
			float _t_11545;
			float _t_11546;
		
			_t_11545 = -1.0f * ty2_8_1;
			_t_11546 = ty1_7_1 + _t_11545;
			_t_11547 = _t_11546;
		
		}
else
		{
			float _t_11548;
			float _t_11549;
			float _t_11550;
		
			_t_11548 = -1.0f * ty2_8_1;
			_t_11549 = ty1_7_1 + _t_11548;
			_t_11550 = -1.0f * _t_11549;
			_t_11547 = _t_11550;
		
		}

	_t_11551 = _t_11547 * _t_455;
	_t_11552 = 0.0f < _t_11551;
	if(_t_11552)
		{
		
			_t_11553 = px0_10_1;
		
		}
else
		{
		
			_t_11553 = px1_11_1;
		
		}

	_t_11554 = _t_11540 * _t_11553;
	_t_11555 = -1.0f * ty2_8_1;
	_t_11556 = ty1_7_1 + _t_11555;
	_t_11557 = -1.0f * _t_11556;
	_t_11558 = _t_11557 < 0.0f;
	if(_t_11558)
		{
			float _t_11559;
			float _t_11560;
		
			_t_11559 = -1.0f * tx1_4_1;
			_t_11560 = tx2_5_1 + _t_11559;
			_t_11561 = _t_11560;
		
		}
else
		{
			float _t_11562;
			float _t_11563;
			float _t_11564;
		
			_t_11562 = -1.0f * tx1_4_1;
			_t_11563 = tx2_5_1 + _t_11562;
			_t_11564 = -1.0f * _t_11563;
			_t_11561 = _t_11564;
		
		}

	_t_11565 = _t_11561 * _t_455;
	_t_11566 = -1.0f * ty2_8_1;
	_t_11567 = ty1_7_1 + _t_11566;
	_t_11568 = -1.0f * _t_11567;
	_t_11569 = _t_11568 < 0.0f;
	if(_t_11569)
		{
			float _t_11570;
			float _t_11571;
		
			_t_11570 = -1.0f * tx1_4_1;
			_t_11571 = tx2_5_1 + _t_11570;
			_t_11572 = _t_11571;
		
		}
else
		{
			float _t_11573;
			float _t_11574;
			float _t_11575;
		
			_t_11573 = -1.0f * tx1_4_1;
			_t_11574 = tx2_5_1 + _t_11573;
			_t_11575 = -1.0f * _t_11574;
			_t_11572 = _t_11575;
		
		}

	_t_11576 = _t_11572 * _t_455;
	_t_11577 = 0.0f < _t_11576;
	if(_t_11577)
		{
		
			_t_11578 = py0_12_1;
		
		}
else
		{
		
			_t_11578 = py1_13_1;
		
		}

	_t_11579 = _t_11565 * _t_11578;
	_t_11580 = _t_11554 + _t_11579;
	_t_11581 = -1.0f * ty2_8_1;
	_t_11582 = ty1_7_1 + _t_11581;
	_t_11583 = -1.0f * _t_11582;
	_t_11584 = _t_11583 < 0.0f;
	if(_t_11584)
		{
			float _t_11585;
			float _t_11586;
			float _t_11587;
			float _t_11588;
		
			_t_11585 = tx1_4_1 * ty2_8_1;
			_t_11586 = tx2_5_1 * ty1_7_1;
			_t_11587 = _t_11586 * -1.0f;
			_t_11588 = _t_11585 + _t_11587;
			_t_11589 = _t_11588;
		
		}
else
		{
			float _t_11590;
			float _t_11591;
			float _t_11592;
			float _t_11593;
			float _t_11594;
		
			_t_11590 = tx1_4_1 * ty2_8_1;
			_t_11591 = tx2_5_1 * ty1_7_1;
			_t_11592 = _t_11591 * -1.0f;
			_t_11593 = _t_11590 + _t_11592;
			_t_11594 = -1.0f * _t_11593;
			_t_11589 = _t_11594;
		
		}

	_t_11595 = -1.0f * _t_11589;
	_t_11596 = _t_11595 * _t_455;
	_t_11597 = _t_11596 * -1.0f;
	_t_11598 = _t_11580 + _t_11597;
	_t_11599 = _t_11598 < 0.0f;
	_t_11600 = -1.0f * ty2_8_1;
	_t_11601 = ty1_7_1 + _t_11600;
	_t_11602 = -1.0f * _t_11601;
	_t_11603 = _t_11602 < 0.0f;
	if(_t_11603)
		{
			float _t_11604;
			float _t_11605;
		
			_t_11604 = -1.0f * ty2_8_1;
			_t_11605 = ty1_7_1 + _t_11604;
			_t_11606 = _t_11605;
		
		}
else
		{
			float _t_11607;
			float _t_11608;
			float _t_11609;
		
			_t_11607 = -1.0f * ty2_8_1;
			_t_11608 = ty1_7_1 + _t_11607;
			_t_11609 = -1.0f * _t_11608;
			_t_11606 = _t_11609;
		
		}

	_t_11610 = _t_11606 * _t_455;
	_t_11611 = -1.0f * ty2_8_1;
	_t_11612 = ty1_7_1 + _t_11611;
	_t_11613 = -1.0f * _t_11612;
	_t_11614 = _t_11613 < 0.0f;
	if(_t_11614)
		{
			float _t_11615;
			float _t_11616;
		
			_t_11615 = -1.0f * ty2_8_1;
			_t_11616 = ty1_7_1 + _t_11615;
			_t_11617 = _t_11616;
		
		}
else
		{
			float _t_11618;
			float _t_11619;
			float _t_11620;
		
			_t_11618 = -1.0f * ty2_8_1;
			_t_11619 = ty1_7_1 + _t_11618;
			_t_11620 = -1.0f * _t_11619;
			_t_11617 = _t_11620;
		
		}

	_t_11621 = _t_11617 * _t_455;
	_t_11622 = 0.0f < _t_11621;
	if(_t_11622)
		{
		
			_t_11623 = px1_11_1;
		
		}
else
		{
		
			_t_11623 = px0_10_1;
		
		}

	_t_11624 = _t_11610 * _t_11623;
	_t_11625 = -1.0f * ty2_8_1;
	_t_11626 = ty1_7_1 + _t_11625;
	_t_11627 = -1.0f * _t_11626;
	_t_11628 = _t_11627 < 0.0f;
	if(_t_11628)
		{
			float _t_11629;
			float _t_11630;
		
			_t_11629 = -1.0f * tx1_4_1;
			_t_11630 = tx2_5_1 + _t_11629;
			_t_11631 = _t_11630;
		
		}
else
		{
			float _t_11632;
			float _t_11633;
			float _t_11634;
		
			_t_11632 = -1.0f * tx1_4_1;
			_t_11633 = tx2_5_1 + _t_11632;
			_t_11634 = -1.0f * _t_11633;
			_t_11631 = _t_11634;
		
		}

	_t_11635 = _t_11631 * _t_455;
	_t_11636 = -1.0f * ty2_8_1;
	_t_11637 = ty1_7_1 + _t_11636;
	_t_11638 = -1.0f * _t_11637;
	_t_11639 = _t_11638 < 0.0f;
	if(_t_11639)
		{
			float _t_11640;
			float _t_11641;
		
			_t_11640 = -1.0f * tx1_4_1;
			_t_11641 = tx2_5_1 + _t_11640;
			_t_11642 = _t_11641;
		
		}
else
		{
			float _t_11643;
			float _t_11644;
			float _t_11645;
		
			_t_11643 = -1.0f * tx1_4_1;
			_t_11644 = tx2_5_1 + _t_11643;
			_t_11645 = -1.0f * _t_11644;
			_t_11642 = _t_11645;
		
		}

	_t_11646 = _t_11642 * _t_455;
	_t_11647 = 0.0f < _t_11646;
	if(_t_11647)
		{
		
			_t_11648 = py1_13_1;
		
		}
else
		{
		
			_t_11648 = py0_12_1;
		
		}

	_t_11649 = _t_11635 * _t_11648;
	_t_11650 = _t_11624 + _t_11649;
	_t_11651 = -1.0f * ty2_8_1;
	_t_11652 = ty1_7_1 + _t_11651;
	_t_11653 = -1.0f * _t_11652;
	_t_11654 = _t_11653 < 0.0f;
	if(_t_11654)
		{
			float _t_11655;
			float _t_11656;
			float _t_11657;
			float _t_11658;
		
			_t_11655 = tx1_4_1 * ty2_8_1;
			_t_11656 = tx2_5_1 * ty1_7_1;
			_t_11657 = _t_11656 * -1.0f;
			_t_11658 = _t_11655 + _t_11657;
			_t_11659 = _t_11658;
		
		}
else
		{
			float _t_11660;
			float _t_11661;
			float _t_11662;
			float _t_11663;
			float _t_11664;
		
			_t_11660 = tx1_4_1 * ty2_8_1;
			_t_11661 = tx2_5_1 * ty1_7_1;
			_t_11662 = _t_11661 * -1.0f;
			_t_11663 = _t_11660 + _t_11662;
			_t_11664 = -1.0f * _t_11663;
			_t_11659 = _t_11664;
		
		}

	_t_11665 = -1.0f * _t_11659;
	_t_11666 = _t_11665 * _t_455;
	_t_11667 = _t_11666 * -1.0f;
	_t_11668 = _t_11650 + _t_11667;
	_t_11669 = 0.0f < _t_11668;
	_t_11670 = _t_11599 && _t_11669;
	if(_t_11670)
		{
			float _t_11671;
			float _t_11672;
			float _t_11673;
			bool _t_11674;
			float _t_11679;
			float _t_11685;
			float _t_11686;
			float _t_11687;
		
			_t_11671 = -1.0f * ty2_8_1;
			_t_11672 = ty1_7_1 + _t_11671;
			_t_11673 = -1.0f * _t_11672;
			_t_11674 = _t_11673 < 0.0f;
			if(_t_11674)
				{
					float _t_11675;
					float _t_11676;
					float _t_11677;
					float _t_11678;
				
					_t_11675 = tx1_4_1 * ty2_8_1;
					_t_11676 = tx2_5_1 * ty1_7_1;
					_t_11677 = _t_11676 * -1.0f;
					_t_11678 = _t_11675 + _t_11677;
					_t_11679 = _t_11678;
				
				}
		else
				{
					float _t_11680;
					float _t_11681;
					float _t_11682;
					float _t_11683;
					float _t_11684;
				
					_t_11680 = tx1_4_1 * ty2_8_1;
					_t_11681 = tx2_5_1 * ty1_7_1;
					_t_11682 = _t_11681 * -1.0f;
					_t_11683 = _t_11680 + _t_11682;
					_t_11684 = -1.0f * _t_11683;
					_t_11679 = _t_11684;
				
				}
		
			_t_11685 = -1.0f * _t_11679;
			_t_11686 = _t_11685 * _t_455;
			_t_11687 = tegpixellet_block_43(ty1_7_1,ty2_8_1,_t_455,_t_11686,tx2_5_1,tx1_4_1,y__3461_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);
			_t_11529 = _t_11687;
		
		}
else
		{
		
			_t_11529 = 0.0f;
		
		}


	return _t_11529;
}
__device__ float tegpixelintegrator_29(float pc1_15_1,float ty3_9_1,float _t_11528,float tc2_19_1,float _t_455,float ty2_8_1,float pc0_14_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float tx2_5_1,float py1_13_1,float pc2_16_1,float px1_11_1,float tc0_17_1,float _t_11419,float py0_12_1,float tc1_18_1,float px0_10_1){
    float y__3461_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_11528 - _t_11419)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__3461_1 = _t_11419 + __step__ * (i + (float)(0.5));
        float _t_11529;
		_t_11529 = tegpixelbody_block_29(ty1_7_1,ty2_8_1,_t_455,px0_10_1,px1_11_1,tx2_5_1,tx1_4_1,py0_12_1,py1_13_1,y__3461_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);;
        __output__ = __output__ + _t_11529 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_13(float ty1_7_1,float ty2_8_1,float tx2_5_1,float tx1_4_1,float _t_455,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty3_9_1){
	float _t_11311;
	float _t_11312;
	float _t_11313;
	bool _t_11314;
	float _t_11317;
	float _t_11321;
	float _t_11322;
	float _t_11323;
	float _t_11324;
	float _t_11325;
	bool _t_11326;
	float _t_11329;
	float _t_11333;
	float _t_11334;
	bool _t_11335;
	float _t_11336;
	float _t_11337;
	float _t_11338;
	float _t_11339;
	float _t_11340;
	bool _t_11341;
	float _t_11344;
	float _t_11348;
	float _t_11349;
	float _t_11350;
	float _t_11351;
	bool _t_11352;
	float _t_11355;
	float _t_11359;
	float _t_11360;
	float _t_11361;
	float _t_11362;
	float _t_11363;
	bool _t_11364;
	float _t_11367;
	float _t_11371;
	float _t_11372;
	float _t_11373;
	float _t_11374;
	float _t_11375;
	float _t_11376;
	float _t_11377;
	float _t_11378;
	float _t_11379;
	bool _t_11380;
	float _t_11383;
	float _t_11387;
	float _t_11388;
	float _t_11389;
	float _t_11390;
	bool _t_11391;
	float _t_11394;
	float _t_11398;
	float _t_11399;
	float _t_11400;
	float _t_11401;
	float _t_11402;
	bool _t_11403;
	float _t_11406;
	float _t_11410;
	float _t_11411;
	float _t_11412;
	float _t_11413;
	float _t_11414;
	float _t_11415;
	bool _t_11416;
	float _t_11417;
	float _t_11418;
	float _t_11419;
	float _t_11420;
	float _t_11421;
	float _t_11422;
	bool _t_11423;
	float _t_11426;
	float _t_11430;
	float _t_11431;
	float _t_11432;
	float _t_11433;
	float _t_11434;
	bool _t_11435;
	float _t_11438;
	float _t_11442;
	float _t_11443;
	bool _t_11444;
	float _t_11445;
	float _t_11446;
	float _t_11447;
	float _t_11448;
	float _t_11449;
	bool _t_11450;
	float _t_11453;
	float _t_11457;
	float _t_11458;
	float _t_11459;
	float _t_11460;
	bool _t_11461;
	float _t_11464;
	float _t_11468;
	float _t_11469;
	float _t_11470;
	float _t_11471;
	float _t_11472;
	bool _t_11473;
	float _t_11476;
	float _t_11480;
	float _t_11481;
	float _t_11482;
	float _t_11483;
	float _t_11484;
	float _t_11485;
	float _t_11486;
	float _t_11487;
	float _t_11488;
	bool _t_11489;
	float _t_11492;
	float _t_11496;
	float _t_11497;
	float _t_11498;
	float _t_11499;
	bool _t_11500;
	float _t_11503;
	float _t_11507;
	float _t_11508;
	float _t_11509;
	float _t_11510;
	float _t_11511;
	bool _t_11512;
	float _t_11515;
	float _t_11519;
	float _t_11520;
	float _t_11521;
	float _t_11522;
	float _t_11523;
	float _t_11524;
	bool _t_11525;
	float _t_11526;
	float _t_11527;
	float _t_11528;

	float _t_456;

	_t_11311 = -1.0f * ty2_8_1;
	_t_11312 = ty1_7_1 + _t_11311;
	_t_11313 = -1.0f * _t_11312;
	_t_11314 = _t_11313 < 0.0f;
	if(_t_11314)
		{
			float _t_11315;
			float _t_11316;
		
			_t_11315 = -1.0f * tx1_4_1;
			_t_11316 = tx2_5_1 + _t_11315;
			_t_11317 = _t_11316;
		
		}
else
		{
			float _t_11318;
			float _t_11319;
			float _t_11320;
		
			_t_11318 = -1.0f * tx1_4_1;
			_t_11319 = tx2_5_1 + _t_11318;
			_t_11320 = -1.0f * _t_11319;
			_t_11317 = _t_11320;
		
		}

	_t_11321 = _t_11317 * _t_455;
	_t_11322 = _t_11321 * -1.0f;
	_t_11323 = -1.0f * ty2_8_1;
	_t_11324 = ty1_7_1 + _t_11323;
	_t_11325 = -1.0f * _t_11324;
	_t_11326 = _t_11325 < 0.0f;
	if(_t_11326)
		{
			float _t_11327;
			float _t_11328;
		
			_t_11327 = -1.0f * tx1_4_1;
			_t_11328 = tx2_5_1 + _t_11327;
			_t_11329 = _t_11328;
		
		}
else
		{
			float _t_11330;
			float _t_11331;
			float _t_11332;
		
			_t_11330 = -1.0f * tx1_4_1;
			_t_11331 = tx2_5_1 + _t_11330;
			_t_11332 = -1.0f * _t_11331;
			_t_11329 = _t_11332;
		
		}

	_t_11333 = _t_11329 * _t_455;
	_t_11334 = _t_11333 * -1.0f;
	_t_11335 = 0.0f < _t_11334;
	if(_t_11335)
		{
		
			_t_11336 = px0_10_1;
		
		}
else
		{
		
			_t_11336 = px1_11_1;
		
		}

	_t_11337 = _t_11322 * _t_11336;
	_t_11338 = -1.0f * ty2_8_1;
	_t_11339 = ty1_7_1 + _t_11338;
	_t_11340 = -1.0f * _t_11339;
	_t_11341 = _t_11340 < 0.0f;
	if(_t_11341)
		{
			float _t_11342;
			float _t_11343;
		
			_t_11342 = -1.0f * tx1_4_1;
			_t_11343 = tx2_5_1 + _t_11342;
			_t_11344 = _t_11343;
		
		}
else
		{
			float _t_11345;
			float _t_11346;
			float _t_11347;
		
			_t_11345 = -1.0f * tx1_4_1;
			_t_11346 = tx2_5_1 + _t_11345;
			_t_11347 = -1.0f * _t_11346;
			_t_11344 = _t_11347;
		
		}

	_t_11348 = _t_11344 * _t_455;
	_t_11349 = -1.0f * ty2_8_1;
	_t_11350 = ty1_7_1 + _t_11349;
	_t_11351 = -1.0f * _t_11350;
	_t_11352 = _t_11351 < 0.0f;
	if(_t_11352)
		{
			float _t_11353;
			float _t_11354;
		
			_t_11353 = -1.0f * tx1_4_1;
			_t_11354 = tx2_5_1 + _t_11353;
			_t_11355 = _t_11354;
		
		}
else
		{
			float _t_11356;
			float _t_11357;
			float _t_11358;
		
			_t_11356 = -1.0f * tx1_4_1;
			_t_11357 = tx2_5_1 + _t_11356;
			_t_11358 = -1.0f * _t_11357;
			_t_11355 = _t_11358;
		
		}

	_t_11359 = _t_11355 * _t_455;
	_t_11360 = _t_11348 * _t_11359;
	_t_11361 = -1.0f * ty2_8_1;
	_t_11362 = ty1_7_1 + _t_11361;
	_t_11363 = -1.0f * _t_11362;
	_t_11364 = _t_11363 < 0.0f;
	if(_t_11364)
		{
			float _t_11365;
			float _t_11366;
		
			_t_11365 = -1.0f * ty2_8_1;
			_t_11366 = ty1_7_1 + _t_11365;
			_t_11367 = _t_11366;
		
		}
else
		{
			float _t_11368;
			float _t_11369;
			float _t_11370;
		
			_t_11368 = -1.0f * ty2_8_1;
			_t_11369 = ty1_7_1 + _t_11368;
			_t_11370 = -1.0f * _t_11369;
			_t_11367 = _t_11370;
		
		}

	_t_11371 = _t_11367 * _t_455;
	_t_11372 = 1.0f + _t_11371;
	_t_11373 = 1.0f / _t_11372;
	_t_11374 = _t_11360 * _t_11373;
	_t_11375 = _t_11374 * -1.0f;
	_t_11376 = 1.0f + _t_11375;
	_t_11377 = -1.0f * ty2_8_1;
	_t_11378 = ty1_7_1 + _t_11377;
	_t_11379 = -1.0f * _t_11378;
	_t_11380 = _t_11379 < 0.0f;
	if(_t_11380)
		{
			float _t_11381;
			float _t_11382;
		
			_t_11381 = -1.0f * tx1_4_1;
			_t_11382 = tx2_5_1 + _t_11381;
			_t_11383 = _t_11382;
		
		}
else
		{
			float _t_11384;
			float _t_11385;
			float _t_11386;
		
			_t_11384 = -1.0f * tx1_4_1;
			_t_11385 = tx2_5_1 + _t_11384;
			_t_11386 = -1.0f * _t_11385;
			_t_11383 = _t_11386;
		
		}

	_t_11387 = _t_11383 * _t_455;
	_t_11388 = -1.0f * ty2_8_1;
	_t_11389 = ty1_7_1 + _t_11388;
	_t_11390 = -1.0f * _t_11389;
	_t_11391 = _t_11390 < 0.0f;
	if(_t_11391)
		{
			float _t_11392;
			float _t_11393;
		
			_t_11392 = -1.0f * tx1_4_1;
			_t_11393 = tx2_5_1 + _t_11392;
			_t_11394 = _t_11393;
		
		}
else
		{
			float _t_11395;
			float _t_11396;
			float _t_11397;
		
			_t_11395 = -1.0f * tx1_4_1;
			_t_11396 = tx2_5_1 + _t_11395;
			_t_11397 = -1.0f * _t_11396;
			_t_11394 = _t_11397;
		
		}

	_t_11398 = _t_11394 * _t_455;
	_t_11399 = _t_11387 * _t_11398;
	_t_11400 = -1.0f * ty2_8_1;
	_t_11401 = ty1_7_1 + _t_11400;
	_t_11402 = -1.0f * _t_11401;
	_t_11403 = _t_11402 < 0.0f;
	if(_t_11403)
		{
			float _t_11404;
			float _t_11405;
		
			_t_11404 = -1.0f * ty2_8_1;
			_t_11405 = ty1_7_1 + _t_11404;
			_t_11406 = _t_11405;
		
		}
else
		{
			float _t_11407;
			float _t_11408;
			float _t_11409;
		
			_t_11407 = -1.0f * ty2_8_1;
			_t_11408 = ty1_7_1 + _t_11407;
			_t_11409 = -1.0f * _t_11408;
			_t_11406 = _t_11409;
		
		}

	_t_11410 = _t_11406 * _t_455;
	_t_11411 = 1.0f + _t_11410;
	_t_11412 = 1.0f / _t_11411;
	_t_11413 = _t_11399 * _t_11412;
	_t_11414 = _t_11413 * -1.0f;
	_t_11415 = 1.0f + _t_11414;
	_t_11416 = 0.0f < _t_11415;
	if(_t_11416)
		{
		
			_t_11417 = py0_12_1;
		
		}
else
		{
		
			_t_11417 = py1_13_1;
		
		}

	_t_11418 = _t_11376 * _t_11417;
	_t_11419 = _t_11337 + _t_11418;
	_t_11420 = -1.0f * ty2_8_1;
	_t_11421 = ty1_7_1 + _t_11420;
	_t_11422 = -1.0f * _t_11421;
	_t_11423 = _t_11422 < 0.0f;
	if(_t_11423)
		{
			float _t_11424;
			float _t_11425;
		
			_t_11424 = -1.0f * tx1_4_1;
			_t_11425 = tx2_5_1 + _t_11424;
			_t_11426 = _t_11425;
		
		}
else
		{
			float _t_11427;
			float _t_11428;
			float _t_11429;
		
			_t_11427 = -1.0f * tx1_4_1;
			_t_11428 = tx2_5_1 + _t_11427;
			_t_11429 = -1.0f * _t_11428;
			_t_11426 = _t_11429;
		
		}

	_t_11430 = _t_11426 * _t_455;
	_t_11431 = _t_11430 * -1.0f;
	_t_11432 = -1.0f * ty2_8_1;
	_t_11433 = ty1_7_1 + _t_11432;
	_t_11434 = -1.0f * _t_11433;
	_t_11435 = _t_11434 < 0.0f;
	if(_t_11435)
		{
			float _t_11436;
			float _t_11437;
		
			_t_11436 = -1.0f * tx1_4_1;
			_t_11437 = tx2_5_1 + _t_11436;
			_t_11438 = _t_11437;
		
		}
else
		{
			float _t_11439;
			float _t_11440;
			float _t_11441;
		
			_t_11439 = -1.0f * tx1_4_1;
			_t_11440 = tx2_5_1 + _t_11439;
			_t_11441 = -1.0f * _t_11440;
			_t_11438 = _t_11441;
		
		}

	_t_11442 = _t_11438 * _t_455;
	_t_11443 = _t_11442 * -1.0f;
	_t_11444 = 0.0f < _t_11443;
	if(_t_11444)
		{
		
			_t_11445 = px1_11_1;
		
		}
else
		{
		
			_t_11445 = px0_10_1;
		
		}

	_t_11446 = _t_11431 * _t_11445;
	_t_11447 = -1.0f * ty2_8_1;
	_t_11448 = ty1_7_1 + _t_11447;
	_t_11449 = -1.0f * _t_11448;
	_t_11450 = _t_11449 < 0.0f;
	if(_t_11450)
		{
			float _t_11451;
			float _t_11452;
		
			_t_11451 = -1.0f * tx1_4_1;
			_t_11452 = tx2_5_1 + _t_11451;
			_t_11453 = _t_11452;
		
		}
else
		{
			float _t_11454;
			float _t_11455;
			float _t_11456;
		
			_t_11454 = -1.0f * tx1_4_1;
			_t_11455 = tx2_5_1 + _t_11454;
			_t_11456 = -1.0f * _t_11455;
			_t_11453 = _t_11456;
		
		}

	_t_11457 = _t_11453 * _t_455;
	_t_11458 = -1.0f * ty2_8_1;
	_t_11459 = ty1_7_1 + _t_11458;
	_t_11460 = -1.0f * _t_11459;
	_t_11461 = _t_11460 < 0.0f;
	if(_t_11461)
		{
			float _t_11462;
			float _t_11463;
		
			_t_11462 = -1.0f * tx1_4_1;
			_t_11463 = tx2_5_1 + _t_11462;
			_t_11464 = _t_11463;
		
		}
else
		{
			float _t_11465;
			float _t_11466;
			float _t_11467;
		
			_t_11465 = -1.0f * tx1_4_1;
			_t_11466 = tx2_5_1 + _t_11465;
			_t_11467 = -1.0f * _t_11466;
			_t_11464 = _t_11467;
		
		}

	_t_11468 = _t_11464 * _t_455;
	_t_11469 = _t_11457 * _t_11468;
	_t_11470 = -1.0f * ty2_8_1;
	_t_11471 = ty1_7_1 + _t_11470;
	_t_11472 = -1.0f * _t_11471;
	_t_11473 = _t_11472 < 0.0f;
	if(_t_11473)
		{
			float _t_11474;
			float _t_11475;
		
			_t_11474 = -1.0f * ty2_8_1;
			_t_11475 = ty1_7_1 + _t_11474;
			_t_11476 = _t_11475;
		
		}
else
		{
			float _t_11477;
			float _t_11478;
			float _t_11479;
		
			_t_11477 = -1.0f * ty2_8_1;
			_t_11478 = ty1_7_1 + _t_11477;
			_t_11479 = -1.0f * _t_11478;
			_t_11476 = _t_11479;
		
		}

	_t_11480 = _t_11476 * _t_455;
	_t_11481 = 1.0f + _t_11480;
	_t_11482 = 1.0f / _t_11481;
	_t_11483 = _t_11469 * _t_11482;
	_t_11484 = _t_11483 * -1.0f;
	_t_11485 = 1.0f + _t_11484;
	_t_11486 = -1.0f * ty2_8_1;
	_t_11487 = ty1_7_1 + _t_11486;
	_t_11488 = -1.0f * _t_11487;
	_t_11489 = _t_11488 < 0.0f;
	if(_t_11489)
		{
			float _t_11490;
			float _t_11491;
		
			_t_11490 = -1.0f * tx1_4_1;
			_t_11491 = tx2_5_1 + _t_11490;
			_t_11492 = _t_11491;
		
		}
else
		{
			float _t_11493;
			float _t_11494;
			float _t_11495;
		
			_t_11493 = -1.0f * tx1_4_1;
			_t_11494 = tx2_5_1 + _t_11493;
			_t_11495 = -1.0f * _t_11494;
			_t_11492 = _t_11495;
		
		}

	_t_11496 = _t_11492 * _t_455;
	_t_11497 = -1.0f * ty2_8_1;
	_t_11498 = ty1_7_1 + _t_11497;
	_t_11499 = -1.0f * _t_11498;
	_t_11500 = _t_11499 < 0.0f;
	if(_t_11500)
		{
			float _t_11501;
			float _t_11502;
		
			_t_11501 = -1.0f * tx1_4_1;
			_t_11502 = tx2_5_1 + _t_11501;
			_t_11503 = _t_11502;
		
		}
else
		{
			float _t_11504;
			float _t_11505;
			float _t_11506;
		
			_t_11504 = -1.0f * tx1_4_1;
			_t_11505 = tx2_5_1 + _t_11504;
			_t_11506 = -1.0f * _t_11505;
			_t_11503 = _t_11506;
		
		}

	_t_11507 = _t_11503 * _t_455;
	_t_11508 = _t_11496 * _t_11507;
	_t_11509 = -1.0f * ty2_8_1;
	_t_11510 = ty1_7_1 + _t_11509;
	_t_11511 = -1.0f * _t_11510;
	_t_11512 = _t_11511 < 0.0f;
	if(_t_11512)
		{
			float _t_11513;
			float _t_11514;
		
			_t_11513 = -1.0f * ty2_8_1;
			_t_11514 = ty1_7_1 + _t_11513;
			_t_11515 = _t_11514;
		
		}
else
		{
			float _t_11516;
			float _t_11517;
			float _t_11518;
		
			_t_11516 = -1.0f * ty2_8_1;
			_t_11517 = ty1_7_1 + _t_11516;
			_t_11518 = -1.0f * _t_11517;
			_t_11515 = _t_11518;
		
		}

	_t_11519 = _t_11515 * _t_455;
	_t_11520 = 1.0f + _t_11519;
	_t_11521 = 1.0f / _t_11520;
	_t_11522 = _t_11508 * _t_11521;
	_t_11523 = _t_11522 * -1.0f;
	_t_11524 = 1.0f + _t_11523;
	_t_11525 = 0.0f < _t_11524;
	if(_t_11525)
		{
		
			_t_11526 = py1_13_1;
		
		}
else
		{
		
			_t_11526 = py0_12_1;
		
		}

	_t_11527 = _t_11485 * _t_11526;
	_t_11528 = _t_11446 + _t_11527;
	_t_456 = tegpixelintegrator_29(pc1_15_1,ty3_9_1,_t_11528,tc2_19_1,_t_455,ty2_8_1,pc0_14_1,ty1_7_1,tx1_4_1,tx3_6_1,tx2_5_1,py1_13_1,pc2_16_1,px1_11_1,tc0_17_1,_t_11419,py0_12_1,tc1_18_1,px0_10_1);

	return _t_456;
}
__device__ float tegpixellet_block_46(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty1_7_1,float tx1_4_1,float ty3_9_1,float _t_12592,float _t_12645,float ty2_8_1,float tx2_5_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float _t_483,float y__3535_1,float _t_12565){
	float _t_12646;
	float _t_12647;
	float _t_12648;
	float _t_12649;
	float _t_12650;
	float _t_12651;
	float _t_12652;
	float _t_12653;
	float _t_12654;
	float _t_12655;
	float _t_12656;
	float _t_12657;
	float _t_12658;
	float _t_12659;
	float _t_12660;
	float _t_12661;
	float _t_12662;
	float _t_12663;
	float _t_12664;
	float _t_12665;
	float _t_12666;
	float _t_12667;
	float _t_12668;
	bool _t_12669;
	float _t_12670;
	float _t_12671;
	float _t_12672;
	float _t_12673;
	float _t_12674;
	float _t_12675;
	float _t_12676;
	float _t_12677;
	float _t_12678;
	float _t_12679;
	float _t_12680;
	float _t_12681;
	float _t_12682;
	float _t_12683;
	bool _t_12684;
	float _t_12685;
	float _t_12686;
	float _t_12687;
	bool _t_12688;
	bool _t_12689;
	bool _t_12690;
	bool _t_12691;
	bool _t_12692;
	bool _t_12693;
	bool _t_12694;
	float _t_13024;

	float _t_12566;

	_t_12646 = -1.0f * pc0_14_1;
	_t_12647 = tc0_17_1 + _t_12646;
	_t_12648 = _t_12647 * _t_12647;
	_t_12649 = -1.0f * pc1_15_1;
	_t_12650 = tc1_18_1 + _t_12649;
	_t_12651 = _t_12650 * _t_12650;
	_t_12652 = _t_12648 + _t_12651;
	_t_12653 = -1.0f * pc2_16_1;
	_t_12654 = tc2_19_1 + _t_12653;
	_t_12655 = _t_12654 * _t_12654;
	_t_12656 = _t_12652 + _t_12655;
	_t_12657 = tx3_6_1 * ty1_7_1;
	_t_12658 = tx1_4_1 * ty3_9_1;
	_t_12659 = _t_12658 * -1.0f;
	_t_12660 = _t_12657 + _t_12659;
	_t_12661 = -1.0f * ty1_7_1;
	_t_12662 = ty3_9_1 + _t_12661;
	_t_12663 = _t_12662 * _t_12592;
	_t_12664 = _t_12660 + _t_12663;
	_t_12665 = -1.0f * tx3_6_1;
	_t_12666 = tx1_4_1 + _t_12665;
	_t_12667 = _t_12666 * _t_12645;
	_t_12668 = _t_12664 + _t_12667;
	_t_12669 = _t_12668 < 0.0f;
	if(_t_12669)
		{
		
			_t_12670 = 1.0f;
		
		}
else
		{
		
			_t_12670 = 0.0f;
		
		}

	_t_12671 = _t_12656 * _t_12670;
	_t_12672 = tx1_4_1 * ty2_8_1;
	_t_12673 = tx2_5_1 * ty1_7_1;
	_t_12674 = _t_12673 * -1.0f;
	_t_12675 = _t_12672 + _t_12674;
	_t_12676 = -1.0f * ty2_8_1;
	_t_12677 = ty1_7_1 + _t_12676;
	_t_12678 = _t_12677 * _t_12592;
	_t_12679 = _t_12675 + _t_12678;
	_t_12680 = -1.0f * tx1_4_1;
	_t_12681 = tx2_5_1 + _t_12680;
	_t_12682 = _t_12681 * _t_12645;
	_t_12683 = _t_12679 + _t_12682;
	_t_12684 = _t_12683 < 0.0f;
	if(_t_12684)
		{
		
			_t_12685 = 1.0f;
		
		}
else
		{
		
			_t_12685 = 0.0f;
		
		}

	_t_12686 = _t_12671 * _t_12685;
	_t_12687 = _t_12686 * tx3_6_1;
	_t_12688 = py0_12_1 < _t_12645;
	_t_12689 = _t_12645 < py1_13_1;
	_t_12690 = _t_12688 && _t_12689;
	_t_12691 = px0_10_1 < _t_12592;
	_t_12692 = _t_12592 < px1_11_1;
	_t_12693 = _t_12691 && _t_12692;
	_t_12694 = _t_12690 && _t_12693;
	if(_t_12694)
		{
			float _t_12695;
			float _t_12696;
			float _t_12697;
			bool _t_12698;
			float _t_12701;
			float _t_12705;
			float _t_12706;
			float _t_12707;
			float _t_12708;
			float _t_12709;
			bool _t_12710;
			float _t_12713;
			float _t_12717;
			float _t_12718;
			bool _t_12719;
			float _t_12720;
			float _t_12721;
			float _t_12722;
			float _t_12723;
			float _t_12724;
			bool _t_12725;
			float _t_12728;
			float _t_12732;
			float _t_12733;
			float _t_12734;
			float _t_12735;
			bool _t_12736;
			float _t_12739;
			float _t_12743;
			float _t_12744;
			float _t_12745;
			float _t_12746;
			float _t_12747;
			bool _t_12748;
			float _t_12751;
			float _t_12755;
			float _t_12756;
			float _t_12757;
			float _t_12758;
			float _t_12759;
			float _t_12760;
			float _t_12761;
			float _t_12762;
			float _t_12763;
			bool _t_12764;
			float _t_12767;
			float _t_12771;
			float _t_12772;
			float _t_12773;
			float _t_12774;
			bool _t_12775;
			float _t_12778;
			float _t_12782;
			float _t_12783;
			float _t_12784;
			float _t_12785;
			float _t_12786;
			bool _t_12787;
			float _t_12790;
			float _t_12794;
			float _t_12795;
			float _t_12796;
			float _t_12797;
			float _t_12798;
			float _t_12799;
			bool _t_12800;
			float _t_12801;
			float _t_12802;
			float _t_12803;
			bool _t_12804;
			float _t_12805;
			float _t_12806;
			float _t_12807;
			bool _t_12808;
			float _t_12811;
			float _t_12815;
			float _t_12816;
			float _t_12817;
			float _t_12818;
			float _t_12819;
			bool _t_12820;
			float _t_12823;
			float _t_12827;
			float _t_12828;
			bool _t_12829;
			float _t_12830;
			float _t_12831;
			float _t_12832;
			float _t_12833;
			float _t_12834;
			bool _t_12835;
			float _t_12838;
			float _t_12842;
			float _t_12843;
			float _t_12844;
			float _t_12845;
			bool _t_12846;
			float _t_12849;
			float _t_12853;
			float _t_12854;
			float _t_12855;
			float _t_12856;
			float _t_12857;
			bool _t_12858;
			float _t_12861;
			float _t_12865;
			float _t_12866;
			float _t_12867;
			float _t_12868;
			float _t_12869;
			float _t_12870;
			float _t_12871;
			float _t_12872;
			float _t_12873;
			bool _t_12874;
			float _t_12877;
			float _t_12881;
			float _t_12882;
			float _t_12883;
			float _t_12884;
			bool _t_12885;
			float _t_12888;
			float _t_12892;
			float _t_12893;
			float _t_12894;
			float _t_12895;
			float _t_12896;
			bool _t_12897;
			float _t_12900;
			float _t_12904;
			float _t_12905;
			float _t_12906;
			float _t_12907;
			float _t_12908;
			float _t_12909;
			bool _t_12910;
			float _t_12911;
			float _t_12912;
			float _t_12913;
			bool _t_12914;
			bool _t_12915;
			float _t_12916;
			float _t_12917;
			float _t_12918;
			bool _t_12919;
			float _t_12922;
			float _t_12926;
			float _t_12927;
			float _t_12928;
			float _t_12929;
			bool _t_12930;
			float _t_12933;
			float _t_12937;
			bool _t_12938;
			float _t_12939;
			float _t_12940;
			float _t_12941;
			float _t_12942;
			float _t_12943;
			bool _t_12944;
			float _t_12947;
			float _t_12951;
			float _t_12952;
			float _t_12953;
			float _t_12954;
			bool _t_12955;
			float _t_12958;
			float _t_12962;
			bool _t_12963;
			float _t_12964;
			float _t_12965;
			float _t_12966;
			bool _t_12967;
			float _t_12968;
			float _t_12969;
			float _t_12970;
			bool _t_12971;
			float _t_12974;
			float _t_12978;
			float _t_12979;
			float _t_12980;
			float _t_12981;
			bool _t_12982;
			float _t_12985;
			float _t_12989;
			bool _t_12990;
			float _t_12991;
			float _t_12992;
			float _t_12993;
			float _t_12994;
			float _t_12995;
			bool _t_12996;
			float _t_12999;
			float _t_13003;
			float _t_13004;
			float _t_13005;
			float _t_13006;
			bool _t_13007;
			float _t_13010;
			float _t_13014;
			bool _t_13015;
			float _t_13016;
			float _t_13017;
			float _t_13018;
			bool _t_13019;
			bool _t_13020;
			bool _t_13021;
			float _t_13022;
			float _t_13023;
		
			_t_12695 = -1.0f * ty3_9_1;
			_t_12696 = ty2_8_1 + _t_12695;
			_t_12697 = -1.0f * _t_12696;
			_t_12698 = _t_12697 < 0.0f;
			if(_t_12698)
				{
					float _t_12699;
					float _t_12700;
				
					_t_12699 = -1.0f * tx2_5_1;
					_t_12700 = tx3_6_1 + _t_12699;
					_t_12701 = _t_12700;
				
				}
		else
				{
					float _t_12702;
					float _t_12703;
					float _t_12704;
				
					_t_12702 = -1.0f * tx2_5_1;
					_t_12703 = tx3_6_1 + _t_12702;
					_t_12704 = -1.0f * _t_12703;
					_t_12701 = _t_12704;
				
				}
		
			_t_12705 = _t_12701 * _t_483;
			_t_12706 = _t_12705 * -1.0f;
			_t_12707 = -1.0f * ty3_9_1;
			_t_12708 = ty2_8_1 + _t_12707;
			_t_12709 = -1.0f * _t_12708;
			_t_12710 = _t_12709 < 0.0f;
			if(_t_12710)
				{
					float _t_12711;
					float _t_12712;
				
					_t_12711 = -1.0f * tx2_5_1;
					_t_12712 = tx3_6_1 + _t_12711;
					_t_12713 = _t_12712;
				
				}
		else
				{
					float _t_12714;
					float _t_12715;
					float _t_12716;
				
					_t_12714 = -1.0f * tx2_5_1;
					_t_12715 = tx3_6_1 + _t_12714;
					_t_12716 = -1.0f * _t_12715;
					_t_12713 = _t_12716;
				
				}
		
			_t_12717 = _t_12713 * _t_483;
			_t_12718 = _t_12717 * -1.0f;
			_t_12719 = 0.0f < _t_12718;
			if(_t_12719)
				{
				
					_t_12720 = px0_10_1;
				
				}
		else
				{
				
					_t_12720 = px1_11_1;
				
				}
		
			_t_12721 = _t_12706 * _t_12720;
			_t_12722 = -1.0f * ty3_9_1;
			_t_12723 = ty2_8_1 + _t_12722;
			_t_12724 = -1.0f * _t_12723;
			_t_12725 = _t_12724 < 0.0f;
			if(_t_12725)
				{
					float _t_12726;
					float _t_12727;
				
					_t_12726 = -1.0f * tx2_5_1;
					_t_12727 = tx3_6_1 + _t_12726;
					_t_12728 = _t_12727;
				
				}
		else
				{
					float _t_12729;
					float _t_12730;
					float _t_12731;
				
					_t_12729 = -1.0f * tx2_5_1;
					_t_12730 = tx3_6_1 + _t_12729;
					_t_12731 = -1.0f * _t_12730;
					_t_12728 = _t_12731;
				
				}
		
			_t_12732 = _t_12728 * _t_483;
			_t_12733 = -1.0f * ty3_9_1;
			_t_12734 = ty2_8_1 + _t_12733;
			_t_12735 = -1.0f * _t_12734;
			_t_12736 = _t_12735 < 0.0f;
			if(_t_12736)
				{
					float _t_12737;
					float _t_12738;
				
					_t_12737 = -1.0f * tx2_5_1;
					_t_12738 = tx3_6_1 + _t_12737;
					_t_12739 = _t_12738;
				
				}
		else
				{
					float _t_12740;
					float _t_12741;
					float _t_12742;
				
					_t_12740 = -1.0f * tx2_5_1;
					_t_12741 = tx3_6_1 + _t_12740;
					_t_12742 = -1.0f * _t_12741;
					_t_12739 = _t_12742;
				
				}
		
			_t_12743 = _t_12739 * _t_483;
			_t_12744 = _t_12732 * _t_12743;
			_t_12745 = -1.0f * ty3_9_1;
			_t_12746 = ty2_8_1 + _t_12745;
			_t_12747 = -1.0f * _t_12746;
			_t_12748 = _t_12747 < 0.0f;
			if(_t_12748)
				{
					float _t_12749;
					float _t_12750;
				
					_t_12749 = -1.0f * ty3_9_1;
					_t_12750 = ty2_8_1 + _t_12749;
					_t_12751 = _t_12750;
				
				}
		else
				{
					float _t_12752;
					float _t_12753;
					float _t_12754;
				
					_t_12752 = -1.0f * ty3_9_1;
					_t_12753 = ty2_8_1 + _t_12752;
					_t_12754 = -1.0f * _t_12753;
					_t_12751 = _t_12754;
				
				}
		
			_t_12755 = _t_12751 * _t_483;
			_t_12756 = 1.0f + _t_12755;
			_t_12757 = 1.0f / _t_12756;
			_t_12758 = _t_12744 * _t_12757;
			_t_12759 = _t_12758 * -1.0f;
			_t_12760 = 1.0f + _t_12759;
			_t_12761 = -1.0f * ty3_9_1;
			_t_12762 = ty2_8_1 + _t_12761;
			_t_12763 = -1.0f * _t_12762;
			_t_12764 = _t_12763 < 0.0f;
			if(_t_12764)
				{
					float _t_12765;
					float _t_12766;
				
					_t_12765 = -1.0f * tx2_5_1;
					_t_12766 = tx3_6_1 + _t_12765;
					_t_12767 = _t_12766;
				
				}
		else
				{
					float _t_12768;
					float _t_12769;
					float _t_12770;
				
					_t_12768 = -1.0f * tx2_5_1;
					_t_12769 = tx3_6_1 + _t_12768;
					_t_12770 = -1.0f * _t_12769;
					_t_12767 = _t_12770;
				
				}
		
			_t_12771 = _t_12767 * _t_483;
			_t_12772 = -1.0f * ty3_9_1;
			_t_12773 = ty2_8_1 + _t_12772;
			_t_12774 = -1.0f * _t_12773;
			_t_12775 = _t_12774 < 0.0f;
			if(_t_12775)
				{
					float _t_12776;
					float _t_12777;
				
					_t_12776 = -1.0f * tx2_5_1;
					_t_12777 = tx3_6_1 + _t_12776;
					_t_12778 = _t_12777;
				
				}
		else
				{
					float _t_12779;
					float _t_12780;
					float _t_12781;
				
					_t_12779 = -1.0f * tx2_5_1;
					_t_12780 = tx3_6_1 + _t_12779;
					_t_12781 = -1.0f * _t_12780;
					_t_12778 = _t_12781;
				
				}
		
			_t_12782 = _t_12778 * _t_483;
			_t_12783 = _t_12771 * _t_12782;
			_t_12784 = -1.0f * ty3_9_1;
			_t_12785 = ty2_8_1 + _t_12784;
			_t_12786 = -1.0f * _t_12785;
			_t_12787 = _t_12786 < 0.0f;
			if(_t_12787)
				{
					float _t_12788;
					float _t_12789;
				
					_t_12788 = -1.0f * ty3_9_1;
					_t_12789 = ty2_8_1 + _t_12788;
					_t_12790 = _t_12789;
				
				}
		else
				{
					float _t_12791;
					float _t_12792;
					float _t_12793;
				
					_t_12791 = -1.0f * ty3_9_1;
					_t_12792 = ty2_8_1 + _t_12791;
					_t_12793 = -1.0f * _t_12792;
					_t_12790 = _t_12793;
				
				}
		
			_t_12794 = _t_12790 * _t_483;
			_t_12795 = 1.0f + _t_12794;
			_t_12796 = 1.0f / _t_12795;
			_t_12797 = _t_12783 * _t_12796;
			_t_12798 = _t_12797 * -1.0f;
			_t_12799 = 1.0f + _t_12798;
			_t_12800 = 0.0f < _t_12799;
			if(_t_12800)
				{
				
					_t_12801 = py0_12_1;
				
				}
		else
				{
				
					_t_12801 = py1_13_1;
				
				}
		
			_t_12802 = _t_12760 * _t_12801;
			_t_12803 = _t_12721 + _t_12802;
			_t_12804 = _t_12803 < y__3535_1;
			_t_12805 = -1.0f * ty3_9_1;
			_t_12806 = ty2_8_1 + _t_12805;
			_t_12807 = -1.0f * _t_12806;
			_t_12808 = _t_12807 < 0.0f;
			if(_t_12808)
				{
					float _t_12809;
					float _t_12810;
				
					_t_12809 = -1.0f * tx2_5_1;
					_t_12810 = tx3_6_1 + _t_12809;
					_t_12811 = _t_12810;
				
				}
		else
				{
					float _t_12812;
					float _t_12813;
					float _t_12814;
				
					_t_12812 = -1.0f * tx2_5_1;
					_t_12813 = tx3_6_1 + _t_12812;
					_t_12814 = -1.0f * _t_12813;
					_t_12811 = _t_12814;
				
				}
		
			_t_12815 = _t_12811 * _t_483;
			_t_12816 = _t_12815 * -1.0f;
			_t_12817 = -1.0f * ty3_9_1;
			_t_12818 = ty2_8_1 + _t_12817;
			_t_12819 = -1.0f * _t_12818;
			_t_12820 = _t_12819 < 0.0f;
			if(_t_12820)
				{
					float _t_12821;
					float _t_12822;
				
					_t_12821 = -1.0f * tx2_5_1;
					_t_12822 = tx3_6_1 + _t_12821;
					_t_12823 = _t_12822;
				
				}
		else
				{
					float _t_12824;
					float _t_12825;
					float _t_12826;
				
					_t_12824 = -1.0f * tx2_5_1;
					_t_12825 = tx3_6_1 + _t_12824;
					_t_12826 = -1.0f * _t_12825;
					_t_12823 = _t_12826;
				
				}
		
			_t_12827 = _t_12823 * _t_483;
			_t_12828 = _t_12827 * -1.0f;
			_t_12829 = 0.0f < _t_12828;
			if(_t_12829)
				{
				
					_t_12830 = px1_11_1;
				
				}
		else
				{
				
					_t_12830 = px0_10_1;
				
				}
		
			_t_12831 = _t_12816 * _t_12830;
			_t_12832 = -1.0f * ty3_9_1;
			_t_12833 = ty2_8_1 + _t_12832;
			_t_12834 = -1.0f * _t_12833;
			_t_12835 = _t_12834 < 0.0f;
			if(_t_12835)
				{
					float _t_12836;
					float _t_12837;
				
					_t_12836 = -1.0f * tx2_5_1;
					_t_12837 = tx3_6_1 + _t_12836;
					_t_12838 = _t_12837;
				
				}
		else
				{
					float _t_12839;
					float _t_12840;
					float _t_12841;
				
					_t_12839 = -1.0f * tx2_5_1;
					_t_12840 = tx3_6_1 + _t_12839;
					_t_12841 = -1.0f * _t_12840;
					_t_12838 = _t_12841;
				
				}
		
			_t_12842 = _t_12838 * _t_483;
			_t_12843 = -1.0f * ty3_9_1;
			_t_12844 = ty2_8_1 + _t_12843;
			_t_12845 = -1.0f * _t_12844;
			_t_12846 = _t_12845 < 0.0f;
			if(_t_12846)
				{
					float _t_12847;
					float _t_12848;
				
					_t_12847 = -1.0f * tx2_5_1;
					_t_12848 = tx3_6_1 + _t_12847;
					_t_12849 = _t_12848;
				
				}
		else
				{
					float _t_12850;
					float _t_12851;
					float _t_12852;
				
					_t_12850 = -1.0f * tx2_5_1;
					_t_12851 = tx3_6_1 + _t_12850;
					_t_12852 = -1.0f * _t_12851;
					_t_12849 = _t_12852;
				
				}
		
			_t_12853 = _t_12849 * _t_483;
			_t_12854 = _t_12842 * _t_12853;
			_t_12855 = -1.0f * ty3_9_1;
			_t_12856 = ty2_8_1 + _t_12855;
			_t_12857 = -1.0f * _t_12856;
			_t_12858 = _t_12857 < 0.0f;
			if(_t_12858)
				{
					float _t_12859;
					float _t_12860;
				
					_t_12859 = -1.0f * ty3_9_1;
					_t_12860 = ty2_8_1 + _t_12859;
					_t_12861 = _t_12860;
				
				}
		else
				{
					float _t_12862;
					float _t_12863;
					float _t_12864;
				
					_t_12862 = -1.0f * ty3_9_1;
					_t_12863 = ty2_8_1 + _t_12862;
					_t_12864 = -1.0f * _t_12863;
					_t_12861 = _t_12864;
				
				}
		
			_t_12865 = _t_12861 * _t_483;
			_t_12866 = 1.0f + _t_12865;
			_t_12867 = 1.0f / _t_12866;
			_t_12868 = _t_12854 * _t_12867;
			_t_12869 = _t_12868 * -1.0f;
			_t_12870 = 1.0f + _t_12869;
			_t_12871 = -1.0f * ty3_9_1;
			_t_12872 = ty2_8_1 + _t_12871;
			_t_12873 = -1.0f * _t_12872;
			_t_12874 = _t_12873 < 0.0f;
			if(_t_12874)
				{
					float _t_12875;
					float _t_12876;
				
					_t_12875 = -1.0f * tx2_5_1;
					_t_12876 = tx3_6_1 + _t_12875;
					_t_12877 = _t_12876;
				
				}
		else
				{
					float _t_12878;
					float _t_12879;
					float _t_12880;
				
					_t_12878 = -1.0f * tx2_5_1;
					_t_12879 = tx3_6_1 + _t_12878;
					_t_12880 = -1.0f * _t_12879;
					_t_12877 = _t_12880;
				
				}
		
			_t_12881 = _t_12877 * _t_483;
			_t_12882 = -1.0f * ty3_9_1;
			_t_12883 = ty2_8_1 + _t_12882;
			_t_12884 = -1.0f * _t_12883;
			_t_12885 = _t_12884 < 0.0f;
			if(_t_12885)
				{
					float _t_12886;
					float _t_12887;
				
					_t_12886 = -1.0f * tx2_5_1;
					_t_12887 = tx3_6_1 + _t_12886;
					_t_12888 = _t_12887;
				
				}
		else
				{
					float _t_12889;
					float _t_12890;
					float _t_12891;
				
					_t_12889 = -1.0f * tx2_5_1;
					_t_12890 = tx3_6_1 + _t_12889;
					_t_12891 = -1.0f * _t_12890;
					_t_12888 = _t_12891;
				
				}
		
			_t_12892 = _t_12888 * _t_483;
			_t_12893 = _t_12881 * _t_12892;
			_t_12894 = -1.0f * ty3_9_1;
			_t_12895 = ty2_8_1 + _t_12894;
			_t_12896 = -1.0f * _t_12895;
			_t_12897 = _t_12896 < 0.0f;
			if(_t_12897)
				{
					float _t_12898;
					float _t_12899;
				
					_t_12898 = -1.0f * ty3_9_1;
					_t_12899 = ty2_8_1 + _t_12898;
					_t_12900 = _t_12899;
				
				}
		else
				{
					float _t_12901;
					float _t_12902;
					float _t_12903;
				
					_t_12901 = -1.0f * ty3_9_1;
					_t_12902 = ty2_8_1 + _t_12901;
					_t_12903 = -1.0f * _t_12902;
					_t_12900 = _t_12903;
				
				}
		
			_t_12904 = _t_12900 * _t_483;
			_t_12905 = 1.0f + _t_12904;
			_t_12906 = 1.0f / _t_12905;
			_t_12907 = _t_12893 * _t_12906;
			_t_12908 = _t_12907 * -1.0f;
			_t_12909 = 1.0f + _t_12908;
			_t_12910 = 0.0f < _t_12909;
			if(_t_12910)
				{
				
					_t_12911 = py1_13_1;
				
				}
		else
				{
				
					_t_12911 = py0_12_1;
				
				}
		
			_t_12912 = _t_12870 * _t_12911;
			_t_12913 = _t_12831 + _t_12912;
			_t_12914 = y__3535_1 < _t_12913;
			_t_12915 = _t_12804 && _t_12914;
			_t_12916 = -1.0f * ty3_9_1;
			_t_12917 = ty2_8_1 + _t_12916;
			_t_12918 = -1.0f * _t_12917;
			_t_12919 = _t_12918 < 0.0f;
			if(_t_12919)
				{
					float _t_12920;
					float _t_12921;
				
					_t_12920 = -1.0f * ty3_9_1;
					_t_12921 = ty2_8_1 + _t_12920;
					_t_12922 = _t_12921;
				
				}
		else
				{
					float _t_12923;
					float _t_12924;
					float _t_12925;
				
					_t_12923 = -1.0f * ty3_9_1;
					_t_12924 = ty2_8_1 + _t_12923;
					_t_12925 = -1.0f * _t_12924;
					_t_12922 = _t_12925;
				
				}
		
			_t_12926 = _t_12922 * _t_483;
			_t_12927 = -1.0f * ty3_9_1;
			_t_12928 = ty2_8_1 + _t_12927;
			_t_12929 = -1.0f * _t_12928;
			_t_12930 = _t_12929 < 0.0f;
			if(_t_12930)
				{
					float _t_12931;
					float _t_12932;
				
					_t_12931 = -1.0f * ty3_9_1;
					_t_12932 = ty2_8_1 + _t_12931;
					_t_12933 = _t_12932;
				
				}
		else
				{
					float _t_12934;
					float _t_12935;
					float _t_12936;
				
					_t_12934 = -1.0f * ty3_9_1;
					_t_12935 = ty2_8_1 + _t_12934;
					_t_12936 = -1.0f * _t_12935;
					_t_12933 = _t_12936;
				
				}
		
			_t_12937 = _t_12933 * _t_483;
			_t_12938 = 0.0f < _t_12937;
			if(_t_12938)
				{
				
					_t_12939 = px0_10_1;
				
				}
		else
				{
				
					_t_12939 = px1_11_1;
				
				}
		
			_t_12940 = _t_12926 * _t_12939;
			_t_12941 = -1.0f * ty3_9_1;
			_t_12942 = ty2_8_1 + _t_12941;
			_t_12943 = -1.0f * _t_12942;
			_t_12944 = _t_12943 < 0.0f;
			if(_t_12944)
				{
					float _t_12945;
					float _t_12946;
				
					_t_12945 = -1.0f * tx2_5_1;
					_t_12946 = tx3_6_1 + _t_12945;
					_t_12947 = _t_12946;
				
				}
		else
				{
					float _t_12948;
					float _t_12949;
					float _t_12950;
				
					_t_12948 = -1.0f * tx2_5_1;
					_t_12949 = tx3_6_1 + _t_12948;
					_t_12950 = -1.0f * _t_12949;
					_t_12947 = _t_12950;
				
				}
		
			_t_12951 = _t_12947 * _t_483;
			_t_12952 = -1.0f * ty3_9_1;
			_t_12953 = ty2_8_1 + _t_12952;
			_t_12954 = -1.0f * _t_12953;
			_t_12955 = _t_12954 < 0.0f;
			if(_t_12955)
				{
					float _t_12956;
					float _t_12957;
				
					_t_12956 = -1.0f * tx2_5_1;
					_t_12957 = tx3_6_1 + _t_12956;
					_t_12958 = _t_12957;
				
				}
		else
				{
					float _t_12959;
					float _t_12960;
					float _t_12961;
				
					_t_12959 = -1.0f * tx2_5_1;
					_t_12960 = tx3_6_1 + _t_12959;
					_t_12961 = -1.0f * _t_12960;
					_t_12958 = _t_12961;
				
				}
		
			_t_12962 = _t_12958 * _t_483;
			_t_12963 = 0.0f < _t_12962;
			if(_t_12963)
				{
				
					_t_12964 = py0_12_1;
				
				}
		else
				{
				
					_t_12964 = py1_13_1;
				
				}
		
			_t_12965 = _t_12951 * _t_12964;
			_t_12966 = _t_12940 + _t_12965;
			_t_12967 = _t_12966 < _t_12565;
			_t_12968 = -1.0f * ty3_9_1;
			_t_12969 = ty2_8_1 + _t_12968;
			_t_12970 = -1.0f * _t_12969;
			_t_12971 = _t_12970 < 0.0f;
			if(_t_12971)
				{
					float _t_12972;
					float _t_12973;
				
					_t_12972 = -1.0f * ty3_9_1;
					_t_12973 = ty2_8_1 + _t_12972;
					_t_12974 = _t_12973;
				
				}
		else
				{
					float _t_12975;
					float _t_12976;
					float _t_12977;
				
					_t_12975 = -1.0f * ty3_9_1;
					_t_12976 = ty2_8_1 + _t_12975;
					_t_12977 = -1.0f * _t_12976;
					_t_12974 = _t_12977;
				
				}
		
			_t_12978 = _t_12974 * _t_483;
			_t_12979 = -1.0f * ty3_9_1;
			_t_12980 = ty2_8_1 + _t_12979;
			_t_12981 = -1.0f * _t_12980;
			_t_12982 = _t_12981 < 0.0f;
			if(_t_12982)
				{
					float _t_12983;
					float _t_12984;
				
					_t_12983 = -1.0f * ty3_9_1;
					_t_12984 = ty2_8_1 + _t_12983;
					_t_12985 = _t_12984;
				
				}
		else
				{
					float _t_12986;
					float _t_12987;
					float _t_12988;
				
					_t_12986 = -1.0f * ty3_9_1;
					_t_12987 = ty2_8_1 + _t_12986;
					_t_12988 = -1.0f * _t_12987;
					_t_12985 = _t_12988;
				
				}
		
			_t_12989 = _t_12985 * _t_483;
			_t_12990 = 0.0f < _t_12989;
			if(_t_12990)
				{
				
					_t_12991 = px1_11_1;
				
				}
		else
				{
				
					_t_12991 = px0_10_1;
				
				}
		
			_t_12992 = _t_12978 * _t_12991;
			_t_12993 = -1.0f * ty3_9_1;
			_t_12994 = ty2_8_1 + _t_12993;
			_t_12995 = -1.0f * _t_12994;
			_t_12996 = _t_12995 < 0.0f;
			if(_t_12996)
				{
					float _t_12997;
					float _t_12998;
				
					_t_12997 = -1.0f * tx2_5_1;
					_t_12998 = tx3_6_1 + _t_12997;
					_t_12999 = _t_12998;
				
				}
		else
				{
					float _t_13000;
					float _t_13001;
					float _t_13002;
				
					_t_13000 = -1.0f * tx2_5_1;
					_t_13001 = tx3_6_1 + _t_13000;
					_t_13002 = -1.0f * _t_13001;
					_t_12999 = _t_13002;
				
				}
		
			_t_13003 = _t_12999 * _t_483;
			_t_13004 = -1.0f * ty3_9_1;
			_t_13005 = ty2_8_1 + _t_13004;
			_t_13006 = -1.0f * _t_13005;
			_t_13007 = _t_13006 < 0.0f;
			if(_t_13007)
				{
					float _t_13008;
					float _t_13009;
				
					_t_13008 = -1.0f * tx2_5_1;
					_t_13009 = tx3_6_1 + _t_13008;
					_t_13010 = _t_13009;
				
				}
		else
				{
					float _t_13011;
					float _t_13012;
					float _t_13013;
				
					_t_13011 = -1.0f * tx2_5_1;
					_t_13012 = tx3_6_1 + _t_13011;
					_t_13013 = -1.0f * _t_13012;
					_t_13010 = _t_13013;
				
				}
		
			_t_13014 = _t_13010 * _t_483;
			_t_13015 = 0.0f < _t_13014;
			if(_t_13015)
				{
				
					_t_13016 = py1_13_1;
				
				}
		else
				{
				
					_t_13016 = py0_12_1;
				
				}
		
			_t_13017 = _t_13003 * _t_13016;
			_t_13018 = _t_12992 + _t_13017;
			_t_13019 = _t_12565 < _t_13018;
			_t_13020 = _t_12967 && _t_13019;
			_t_13021 = _t_12915 && _t_13020;
			if(_t_13021)
				{
				
					_t_13022 = 1.0f;
				
				}
		else
				{
				
					_t_13022 = 0.0f;
				
				}
		
			_t_13023 = _t_13022 * _t_483;
			_t_13024 = _t_13023;
		
		}
else
		{
		
			_t_13024 = 0.0f;
		
		}

	_t_12566 = _t_12687 * _t_13024;

	return _t_12566;
}
__device__ float tegpixellet_block_45(float ty2_8_1,float ty3_9_1,float _t_483,float _t_12565,float tx3_6_1,float tx2_5_1,float y__3535_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1){
	float _t_12567;
	float _t_12568;
	float _t_12569;
	bool _t_12570;
	float _t_12573;
	float _t_12577;
	float _t_12578;
	float _t_12579;
	float _t_12580;
	float _t_12581;
	bool _t_12582;
	float _t_12585;
	float _t_12589;
	float _t_12590;
	float _t_12591;
	float _t_12592;
	float _t_12593;
	float _t_12594;
	float _t_12595;
	bool _t_12596;
	float _t_12599;
	float _t_12603;
	float _t_12604;
	float _t_12605;
	float _t_12606;
	bool _t_12607;
	float _t_12610;
	float _t_12614;
	float _t_12615;
	float _t_12616;
	float _t_12617;
	float _t_12618;
	bool _t_12619;
	float _t_12622;
	float _t_12626;
	float _t_12627;
	float _t_12628;
	float _t_12629;
	float _t_12630;
	float _t_12631;
	float _t_12632;
	float _t_12633;
	float _t_12634;
	float _t_12635;
	bool _t_12636;
	float _t_12639;
	float _t_12643;
	float _t_12644;
	float _t_12645;

	float _t_12566;

	_t_12567 = -1.0f * ty3_9_1;
	_t_12568 = ty2_8_1 + _t_12567;
	_t_12569 = -1.0f * _t_12568;
	_t_12570 = _t_12569 < 0.0f;
	if(_t_12570)
		{
			float _t_12571;
			float _t_12572;
		
			_t_12571 = -1.0f * ty3_9_1;
			_t_12572 = ty2_8_1 + _t_12571;
			_t_12573 = _t_12572;
		
		}
else
		{
			float _t_12574;
			float _t_12575;
			float _t_12576;
		
			_t_12574 = -1.0f * ty3_9_1;
			_t_12575 = ty2_8_1 + _t_12574;
			_t_12576 = -1.0f * _t_12575;
			_t_12573 = _t_12576;
		
		}

	_t_12577 = _t_12573 * _t_483;
	_t_12578 = _t_12577 * _t_12565;
	_t_12579 = -1.0f * ty3_9_1;
	_t_12580 = ty2_8_1 + _t_12579;
	_t_12581 = -1.0f * _t_12580;
	_t_12582 = _t_12581 < 0.0f;
	if(_t_12582)
		{
			float _t_12583;
			float _t_12584;
		
			_t_12583 = -1.0f * tx2_5_1;
			_t_12584 = tx3_6_1 + _t_12583;
			_t_12585 = _t_12584;
		
		}
else
		{
			float _t_12586;
			float _t_12587;
			float _t_12588;
		
			_t_12586 = -1.0f * tx2_5_1;
			_t_12587 = tx3_6_1 + _t_12586;
			_t_12588 = -1.0f * _t_12587;
			_t_12585 = _t_12588;
		
		}

	_t_12589 = _t_12585 * _t_483;
	_t_12590 = _t_12589 * -1.0f;
	_t_12591 = _t_12590 * y__3535_1;
	_t_12592 = _t_12578 + _t_12591;
	_t_12593 = -1.0f * ty3_9_1;
	_t_12594 = ty2_8_1 + _t_12593;
	_t_12595 = -1.0f * _t_12594;
	_t_12596 = _t_12595 < 0.0f;
	if(_t_12596)
		{
			float _t_12597;
			float _t_12598;
		
			_t_12597 = -1.0f * tx2_5_1;
			_t_12598 = tx3_6_1 + _t_12597;
			_t_12599 = _t_12598;
		
		}
else
		{
			float _t_12600;
			float _t_12601;
			float _t_12602;
		
			_t_12600 = -1.0f * tx2_5_1;
			_t_12601 = tx3_6_1 + _t_12600;
			_t_12602 = -1.0f * _t_12601;
			_t_12599 = _t_12602;
		
		}

	_t_12603 = _t_12599 * _t_483;
	_t_12604 = -1.0f * ty3_9_1;
	_t_12605 = ty2_8_1 + _t_12604;
	_t_12606 = -1.0f * _t_12605;
	_t_12607 = _t_12606 < 0.0f;
	if(_t_12607)
		{
			float _t_12608;
			float _t_12609;
		
			_t_12608 = -1.0f * tx2_5_1;
			_t_12609 = tx3_6_1 + _t_12608;
			_t_12610 = _t_12609;
		
		}
else
		{
			float _t_12611;
			float _t_12612;
			float _t_12613;
		
			_t_12611 = -1.0f * tx2_5_1;
			_t_12612 = tx3_6_1 + _t_12611;
			_t_12613 = -1.0f * _t_12612;
			_t_12610 = _t_12613;
		
		}

	_t_12614 = _t_12610 * _t_483;
	_t_12615 = _t_12603 * _t_12614;
	_t_12616 = -1.0f * ty3_9_1;
	_t_12617 = ty2_8_1 + _t_12616;
	_t_12618 = -1.0f * _t_12617;
	_t_12619 = _t_12618 < 0.0f;
	if(_t_12619)
		{
			float _t_12620;
			float _t_12621;
		
			_t_12620 = -1.0f * ty3_9_1;
			_t_12621 = ty2_8_1 + _t_12620;
			_t_12622 = _t_12621;
		
		}
else
		{
			float _t_12623;
			float _t_12624;
			float _t_12625;
		
			_t_12623 = -1.0f * ty3_9_1;
			_t_12624 = ty2_8_1 + _t_12623;
			_t_12625 = -1.0f * _t_12624;
			_t_12622 = _t_12625;
		
		}

	_t_12626 = _t_12622 * _t_483;
	_t_12627 = 1.0f + _t_12626;
	_t_12628 = 1.0f / _t_12627;
	_t_12629 = _t_12615 * _t_12628;
	_t_12630 = _t_12629 * -1.0f;
	_t_12631 = 1.0f + _t_12630;
	_t_12632 = _t_12631 * y__3535_1;
	_t_12633 = -1.0f * ty3_9_1;
	_t_12634 = ty2_8_1 + _t_12633;
	_t_12635 = -1.0f * _t_12634;
	_t_12636 = _t_12635 < 0.0f;
	if(_t_12636)
		{
			float _t_12637;
			float _t_12638;
		
			_t_12637 = -1.0f * tx2_5_1;
			_t_12638 = tx3_6_1 + _t_12637;
			_t_12639 = _t_12638;
		
		}
else
		{
			float _t_12640;
			float _t_12641;
			float _t_12642;
		
			_t_12640 = -1.0f * tx2_5_1;
			_t_12641 = tx3_6_1 + _t_12640;
			_t_12642 = -1.0f * _t_12641;
			_t_12639 = _t_12642;
		
		}

	_t_12643 = _t_12639 * _t_483;
	_t_12644 = _t_12643 * _t_12565;
	_t_12645 = _t_12632 + _t_12644;
	_t_12566 = tegpixellet_block_46(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty1_7_1,tx1_4_1,ty3_9_1,_t_12592,_t_12645,ty2_8_1,tx2_5_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,_t_483,y__3535_1,_t_12565);

	return _t_12566;
}
__device__ float tegpixelbody_block_30(float ty2_8_1,float ty3_9_1,float _t_483,float px0_10_1,float px1_11_1,float tx3_6_1,float tx2_5_1,float py0_12_1,float py1_13_1,float y__3535_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_12409;
	float _t_12410;
	float _t_12411;
	bool _t_12412;
	float _t_12415;
	float _t_12419;
	float _t_12420;
	float _t_12421;
	float _t_12422;
	bool _t_12423;
	float _t_12426;
	float _t_12430;
	bool _t_12431;
	float _t_12432;
	float _t_12433;
	float _t_12434;
	float _t_12435;
	float _t_12436;
	bool _t_12437;
	float _t_12440;
	float _t_12444;
	float _t_12445;
	float _t_12446;
	float _t_12447;
	bool _t_12448;
	float _t_12451;
	float _t_12455;
	bool _t_12456;
	float _t_12457;
	float _t_12458;
	float _t_12459;
	float _t_12460;
	float _t_12461;
	float _t_12462;
	bool _t_12463;
	float _t_12468;
	float _t_12474;
	float _t_12475;
	float _t_12476;
	float _t_12477;
	bool _t_12478;
	float _t_12479;
	float _t_12480;
	float _t_12481;
	bool _t_12482;
	float _t_12485;
	float _t_12489;
	float _t_12490;
	float _t_12491;
	float _t_12492;
	bool _t_12493;
	float _t_12496;
	float _t_12500;
	bool _t_12501;
	float _t_12502;
	float _t_12503;
	float _t_12504;
	float _t_12505;
	float _t_12506;
	bool _t_12507;
	float _t_12510;
	float _t_12514;
	float _t_12515;
	float _t_12516;
	float _t_12517;
	bool _t_12518;
	float _t_12521;
	float _t_12525;
	bool _t_12526;
	float _t_12527;
	float _t_12528;
	float _t_12529;
	float _t_12530;
	float _t_12531;
	float _t_12532;
	bool _t_12533;
	float _t_12538;
	float _t_12544;
	float _t_12545;
	float _t_12546;
	float _t_12547;
	bool _t_12548;
	bool _t_12549;

	float _t_12408;

	_t_12409 = -1.0f * ty3_9_1;
	_t_12410 = ty2_8_1 + _t_12409;
	_t_12411 = -1.0f * _t_12410;
	_t_12412 = _t_12411 < 0.0f;
	if(_t_12412)
		{
			float _t_12413;
			float _t_12414;
		
			_t_12413 = -1.0f * ty3_9_1;
			_t_12414 = ty2_8_1 + _t_12413;
			_t_12415 = _t_12414;
		
		}
else
		{
			float _t_12416;
			float _t_12417;
			float _t_12418;
		
			_t_12416 = -1.0f * ty3_9_1;
			_t_12417 = ty2_8_1 + _t_12416;
			_t_12418 = -1.0f * _t_12417;
			_t_12415 = _t_12418;
		
		}

	_t_12419 = _t_12415 * _t_483;
	_t_12420 = -1.0f * ty3_9_1;
	_t_12421 = ty2_8_1 + _t_12420;
	_t_12422 = -1.0f * _t_12421;
	_t_12423 = _t_12422 < 0.0f;
	if(_t_12423)
		{
			float _t_12424;
			float _t_12425;
		
			_t_12424 = -1.0f * ty3_9_1;
			_t_12425 = ty2_8_1 + _t_12424;
			_t_12426 = _t_12425;
		
		}
else
		{
			float _t_12427;
			float _t_12428;
			float _t_12429;
		
			_t_12427 = -1.0f * ty3_9_1;
			_t_12428 = ty2_8_1 + _t_12427;
			_t_12429 = -1.0f * _t_12428;
			_t_12426 = _t_12429;
		
		}

	_t_12430 = _t_12426 * _t_483;
	_t_12431 = 0.0f < _t_12430;
	if(_t_12431)
		{
		
			_t_12432 = px0_10_1;
		
		}
else
		{
		
			_t_12432 = px1_11_1;
		
		}

	_t_12433 = _t_12419 * _t_12432;
	_t_12434 = -1.0f * ty3_9_1;
	_t_12435 = ty2_8_1 + _t_12434;
	_t_12436 = -1.0f * _t_12435;
	_t_12437 = _t_12436 < 0.0f;
	if(_t_12437)
		{
			float _t_12438;
			float _t_12439;
		
			_t_12438 = -1.0f * tx2_5_1;
			_t_12439 = tx3_6_1 + _t_12438;
			_t_12440 = _t_12439;
		
		}
else
		{
			float _t_12441;
			float _t_12442;
			float _t_12443;
		
			_t_12441 = -1.0f * tx2_5_1;
			_t_12442 = tx3_6_1 + _t_12441;
			_t_12443 = -1.0f * _t_12442;
			_t_12440 = _t_12443;
		
		}

	_t_12444 = _t_12440 * _t_483;
	_t_12445 = -1.0f * ty3_9_1;
	_t_12446 = ty2_8_1 + _t_12445;
	_t_12447 = -1.0f * _t_12446;
	_t_12448 = _t_12447 < 0.0f;
	if(_t_12448)
		{
			float _t_12449;
			float _t_12450;
		
			_t_12449 = -1.0f * tx2_5_1;
			_t_12450 = tx3_6_1 + _t_12449;
			_t_12451 = _t_12450;
		
		}
else
		{
			float _t_12452;
			float _t_12453;
			float _t_12454;
		
			_t_12452 = -1.0f * tx2_5_1;
			_t_12453 = tx3_6_1 + _t_12452;
			_t_12454 = -1.0f * _t_12453;
			_t_12451 = _t_12454;
		
		}

	_t_12455 = _t_12451 * _t_483;
	_t_12456 = 0.0f < _t_12455;
	if(_t_12456)
		{
		
			_t_12457 = py0_12_1;
		
		}
else
		{
		
			_t_12457 = py1_13_1;
		
		}

	_t_12458 = _t_12444 * _t_12457;
	_t_12459 = _t_12433 + _t_12458;
	_t_12460 = -1.0f * ty3_9_1;
	_t_12461 = ty2_8_1 + _t_12460;
	_t_12462 = -1.0f * _t_12461;
	_t_12463 = _t_12462 < 0.0f;
	if(_t_12463)
		{
			float _t_12464;
			float _t_12465;
			float _t_12466;
			float _t_12467;
		
			_t_12464 = tx2_5_1 * ty3_9_1;
			_t_12465 = tx3_6_1 * ty2_8_1;
			_t_12466 = _t_12465 * -1.0f;
			_t_12467 = _t_12464 + _t_12466;
			_t_12468 = _t_12467;
		
		}
else
		{
			float _t_12469;
			float _t_12470;
			float _t_12471;
			float _t_12472;
			float _t_12473;
		
			_t_12469 = tx2_5_1 * ty3_9_1;
			_t_12470 = tx3_6_1 * ty2_8_1;
			_t_12471 = _t_12470 * -1.0f;
			_t_12472 = _t_12469 + _t_12471;
			_t_12473 = -1.0f * _t_12472;
			_t_12468 = _t_12473;
		
		}

	_t_12474 = -1.0f * _t_12468;
	_t_12475 = _t_12474 * _t_483;
	_t_12476 = _t_12475 * -1.0f;
	_t_12477 = _t_12459 + _t_12476;
	_t_12478 = _t_12477 < 0.0f;
	_t_12479 = -1.0f * ty3_9_1;
	_t_12480 = ty2_8_1 + _t_12479;
	_t_12481 = -1.0f * _t_12480;
	_t_12482 = _t_12481 < 0.0f;
	if(_t_12482)
		{
			float _t_12483;
			float _t_12484;
		
			_t_12483 = -1.0f * ty3_9_1;
			_t_12484 = ty2_8_1 + _t_12483;
			_t_12485 = _t_12484;
		
		}
else
		{
			float _t_12486;
			float _t_12487;
			float _t_12488;
		
			_t_12486 = -1.0f * ty3_9_1;
			_t_12487 = ty2_8_1 + _t_12486;
			_t_12488 = -1.0f * _t_12487;
			_t_12485 = _t_12488;
		
		}

	_t_12489 = _t_12485 * _t_483;
	_t_12490 = -1.0f * ty3_9_1;
	_t_12491 = ty2_8_1 + _t_12490;
	_t_12492 = -1.0f * _t_12491;
	_t_12493 = _t_12492 < 0.0f;
	if(_t_12493)
		{
			float _t_12494;
			float _t_12495;
		
			_t_12494 = -1.0f * ty3_9_1;
			_t_12495 = ty2_8_1 + _t_12494;
			_t_12496 = _t_12495;
		
		}
else
		{
			float _t_12497;
			float _t_12498;
			float _t_12499;
		
			_t_12497 = -1.0f * ty3_9_1;
			_t_12498 = ty2_8_1 + _t_12497;
			_t_12499 = -1.0f * _t_12498;
			_t_12496 = _t_12499;
		
		}

	_t_12500 = _t_12496 * _t_483;
	_t_12501 = 0.0f < _t_12500;
	if(_t_12501)
		{
		
			_t_12502 = px1_11_1;
		
		}
else
		{
		
			_t_12502 = px0_10_1;
		
		}

	_t_12503 = _t_12489 * _t_12502;
	_t_12504 = -1.0f * ty3_9_1;
	_t_12505 = ty2_8_1 + _t_12504;
	_t_12506 = -1.0f * _t_12505;
	_t_12507 = _t_12506 < 0.0f;
	if(_t_12507)
		{
			float _t_12508;
			float _t_12509;
		
			_t_12508 = -1.0f * tx2_5_1;
			_t_12509 = tx3_6_1 + _t_12508;
			_t_12510 = _t_12509;
		
		}
else
		{
			float _t_12511;
			float _t_12512;
			float _t_12513;
		
			_t_12511 = -1.0f * tx2_5_1;
			_t_12512 = tx3_6_1 + _t_12511;
			_t_12513 = -1.0f * _t_12512;
			_t_12510 = _t_12513;
		
		}

	_t_12514 = _t_12510 * _t_483;
	_t_12515 = -1.0f * ty3_9_1;
	_t_12516 = ty2_8_1 + _t_12515;
	_t_12517 = -1.0f * _t_12516;
	_t_12518 = _t_12517 < 0.0f;
	if(_t_12518)
		{
			float _t_12519;
			float _t_12520;
		
			_t_12519 = -1.0f * tx2_5_1;
			_t_12520 = tx3_6_1 + _t_12519;
			_t_12521 = _t_12520;
		
		}
else
		{
			float _t_12522;
			float _t_12523;
			float _t_12524;
		
			_t_12522 = -1.0f * tx2_5_1;
			_t_12523 = tx3_6_1 + _t_12522;
			_t_12524 = -1.0f * _t_12523;
			_t_12521 = _t_12524;
		
		}

	_t_12525 = _t_12521 * _t_483;
	_t_12526 = 0.0f < _t_12525;
	if(_t_12526)
		{
		
			_t_12527 = py1_13_1;
		
		}
else
		{
		
			_t_12527 = py0_12_1;
		
		}

	_t_12528 = _t_12514 * _t_12527;
	_t_12529 = _t_12503 + _t_12528;
	_t_12530 = -1.0f * ty3_9_1;
	_t_12531 = ty2_8_1 + _t_12530;
	_t_12532 = -1.0f * _t_12531;
	_t_12533 = _t_12532 < 0.0f;
	if(_t_12533)
		{
			float _t_12534;
			float _t_12535;
			float _t_12536;
			float _t_12537;
		
			_t_12534 = tx2_5_1 * ty3_9_1;
			_t_12535 = tx3_6_1 * ty2_8_1;
			_t_12536 = _t_12535 * -1.0f;
			_t_12537 = _t_12534 + _t_12536;
			_t_12538 = _t_12537;
		
		}
else
		{
			float _t_12539;
			float _t_12540;
			float _t_12541;
			float _t_12542;
			float _t_12543;
		
			_t_12539 = tx2_5_1 * ty3_9_1;
			_t_12540 = tx3_6_1 * ty2_8_1;
			_t_12541 = _t_12540 * -1.0f;
			_t_12542 = _t_12539 + _t_12541;
			_t_12543 = -1.0f * _t_12542;
			_t_12538 = _t_12543;
		
		}

	_t_12544 = -1.0f * _t_12538;
	_t_12545 = _t_12544 * _t_483;
	_t_12546 = _t_12545 * -1.0f;
	_t_12547 = _t_12529 + _t_12546;
	_t_12548 = 0.0f < _t_12547;
	_t_12549 = _t_12478 && _t_12548;
	if(_t_12549)
		{
			float _t_12550;
			float _t_12551;
			float _t_12552;
			bool _t_12553;
			float _t_12558;
			float _t_12564;
			float _t_12565;
			float _t_12566;
		
			_t_12550 = -1.0f * ty3_9_1;
			_t_12551 = ty2_8_1 + _t_12550;
			_t_12552 = -1.0f * _t_12551;
			_t_12553 = _t_12552 < 0.0f;
			if(_t_12553)
				{
					float _t_12554;
					float _t_12555;
					float _t_12556;
					float _t_12557;
				
					_t_12554 = tx2_5_1 * ty3_9_1;
					_t_12555 = tx3_6_1 * ty2_8_1;
					_t_12556 = _t_12555 * -1.0f;
					_t_12557 = _t_12554 + _t_12556;
					_t_12558 = _t_12557;
				
				}
		else
				{
					float _t_12559;
					float _t_12560;
					float _t_12561;
					float _t_12562;
					float _t_12563;
				
					_t_12559 = tx2_5_1 * ty3_9_1;
					_t_12560 = tx3_6_1 * ty2_8_1;
					_t_12561 = _t_12560 * -1.0f;
					_t_12562 = _t_12559 + _t_12561;
					_t_12563 = -1.0f * _t_12562;
					_t_12558 = _t_12563;
				
				}
		
			_t_12564 = -1.0f * _t_12558;
			_t_12565 = _t_12564 * _t_483;
			_t_12566 = tegpixellet_block_45(ty2_8_1,ty3_9_1,_t_483,_t_12565,tx3_6_1,tx2_5_1,y__3535_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1);
			_t_12408 = _t_12566;
		
		}
else
		{
		
			_t_12408 = 0.0f;
		
		}


	return _t_12408;
}
__device__ float tegpixelintegrator_30(float ty3_9_1,float pc1_15_1,float tc2_19_1,float ty2_8_1,float _t_12298,float ty1_7_1,float pc0_14_1,float _t_483,float tx3_6_1,float tx1_4_1,float tx2_5_1,float py1_13_1,float pc2_16_1,float px1_11_1,float tc0_17_1,float py0_12_1,float tc1_18_1,float _t_12407,float px0_10_1){
    float y__3535_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_12407 - _t_12298)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__3535_1 = _t_12298 + __step__ * (i + (float)(0.5));
        float _t_12408;
		_t_12408 = tegpixelbody_block_30(ty2_8_1,ty3_9_1,_t_483,px0_10_1,px1_11_1,tx3_6_1,tx2_5_1,py0_12_1,py1_13_1,y__3535_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);;
        __output__ = __output__ + _t_12408 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_14(float ty2_8_1,float ty3_9_1,float tx3_6_1,float tx2_5_1,float _t_483,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_12190;
	float _t_12191;
	float _t_12192;
	bool _t_12193;
	float _t_12196;
	float _t_12200;
	float _t_12201;
	float _t_12202;
	float _t_12203;
	float _t_12204;
	bool _t_12205;
	float _t_12208;
	float _t_12212;
	float _t_12213;
	bool _t_12214;
	float _t_12215;
	float _t_12216;
	float _t_12217;
	float _t_12218;
	float _t_12219;
	bool _t_12220;
	float _t_12223;
	float _t_12227;
	float _t_12228;
	float _t_12229;
	float _t_12230;
	bool _t_12231;
	float _t_12234;
	float _t_12238;
	float _t_12239;
	float _t_12240;
	float _t_12241;
	float _t_12242;
	bool _t_12243;
	float _t_12246;
	float _t_12250;
	float _t_12251;
	float _t_12252;
	float _t_12253;
	float _t_12254;
	float _t_12255;
	float _t_12256;
	float _t_12257;
	float _t_12258;
	bool _t_12259;
	float _t_12262;
	float _t_12266;
	float _t_12267;
	float _t_12268;
	float _t_12269;
	bool _t_12270;
	float _t_12273;
	float _t_12277;
	float _t_12278;
	float _t_12279;
	float _t_12280;
	float _t_12281;
	bool _t_12282;
	float _t_12285;
	float _t_12289;
	float _t_12290;
	float _t_12291;
	float _t_12292;
	float _t_12293;
	float _t_12294;
	bool _t_12295;
	float _t_12296;
	float _t_12297;
	float _t_12298;
	float _t_12299;
	float _t_12300;
	float _t_12301;
	bool _t_12302;
	float _t_12305;
	float _t_12309;
	float _t_12310;
	float _t_12311;
	float _t_12312;
	float _t_12313;
	bool _t_12314;
	float _t_12317;
	float _t_12321;
	float _t_12322;
	bool _t_12323;
	float _t_12324;
	float _t_12325;
	float _t_12326;
	float _t_12327;
	float _t_12328;
	bool _t_12329;
	float _t_12332;
	float _t_12336;
	float _t_12337;
	float _t_12338;
	float _t_12339;
	bool _t_12340;
	float _t_12343;
	float _t_12347;
	float _t_12348;
	float _t_12349;
	float _t_12350;
	float _t_12351;
	bool _t_12352;
	float _t_12355;
	float _t_12359;
	float _t_12360;
	float _t_12361;
	float _t_12362;
	float _t_12363;
	float _t_12364;
	float _t_12365;
	float _t_12366;
	float _t_12367;
	bool _t_12368;
	float _t_12371;
	float _t_12375;
	float _t_12376;
	float _t_12377;
	float _t_12378;
	bool _t_12379;
	float _t_12382;
	float _t_12386;
	float _t_12387;
	float _t_12388;
	float _t_12389;
	float _t_12390;
	bool _t_12391;
	float _t_12394;
	float _t_12398;
	float _t_12399;
	float _t_12400;
	float _t_12401;
	float _t_12402;
	float _t_12403;
	bool _t_12404;
	float _t_12405;
	float _t_12406;
	float _t_12407;

	float _t_484;

	_t_12190 = -1.0f * ty3_9_1;
	_t_12191 = ty2_8_1 + _t_12190;
	_t_12192 = -1.0f * _t_12191;
	_t_12193 = _t_12192 < 0.0f;
	if(_t_12193)
		{
			float _t_12194;
			float _t_12195;
		
			_t_12194 = -1.0f * tx2_5_1;
			_t_12195 = tx3_6_1 + _t_12194;
			_t_12196 = _t_12195;
		
		}
else
		{
			float _t_12197;
			float _t_12198;
			float _t_12199;
		
			_t_12197 = -1.0f * tx2_5_1;
			_t_12198 = tx3_6_1 + _t_12197;
			_t_12199 = -1.0f * _t_12198;
			_t_12196 = _t_12199;
		
		}

	_t_12200 = _t_12196 * _t_483;
	_t_12201 = _t_12200 * -1.0f;
	_t_12202 = -1.0f * ty3_9_1;
	_t_12203 = ty2_8_1 + _t_12202;
	_t_12204 = -1.0f * _t_12203;
	_t_12205 = _t_12204 < 0.0f;
	if(_t_12205)
		{
			float _t_12206;
			float _t_12207;
		
			_t_12206 = -1.0f * tx2_5_1;
			_t_12207 = tx3_6_1 + _t_12206;
			_t_12208 = _t_12207;
		
		}
else
		{
			float _t_12209;
			float _t_12210;
			float _t_12211;
		
			_t_12209 = -1.0f * tx2_5_1;
			_t_12210 = tx3_6_1 + _t_12209;
			_t_12211 = -1.0f * _t_12210;
			_t_12208 = _t_12211;
		
		}

	_t_12212 = _t_12208 * _t_483;
	_t_12213 = _t_12212 * -1.0f;
	_t_12214 = 0.0f < _t_12213;
	if(_t_12214)
		{
		
			_t_12215 = px0_10_1;
		
		}
else
		{
		
			_t_12215 = px1_11_1;
		
		}

	_t_12216 = _t_12201 * _t_12215;
	_t_12217 = -1.0f * ty3_9_1;
	_t_12218 = ty2_8_1 + _t_12217;
	_t_12219 = -1.0f * _t_12218;
	_t_12220 = _t_12219 < 0.0f;
	if(_t_12220)
		{
			float _t_12221;
			float _t_12222;
		
			_t_12221 = -1.0f * tx2_5_1;
			_t_12222 = tx3_6_1 + _t_12221;
			_t_12223 = _t_12222;
		
		}
else
		{
			float _t_12224;
			float _t_12225;
			float _t_12226;
		
			_t_12224 = -1.0f * tx2_5_1;
			_t_12225 = tx3_6_1 + _t_12224;
			_t_12226 = -1.0f * _t_12225;
			_t_12223 = _t_12226;
		
		}

	_t_12227 = _t_12223 * _t_483;
	_t_12228 = -1.0f * ty3_9_1;
	_t_12229 = ty2_8_1 + _t_12228;
	_t_12230 = -1.0f * _t_12229;
	_t_12231 = _t_12230 < 0.0f;
	if(_t_12231)
		{
			float _t_12232;
			float _t_12233;
		
			_t_12232 = -1.0f * tx2_5_1;
			_t_12233 = tx3_6_1 + _t_12232;
			_t_12234 = _t_12233;
		
		}
else
		{
			float _t_12235;
			float _t_12236;
			float _t_12237;
		
			_t_12235 = -1.0f * tx2_5_1;
			_t_12236 = tx3_6_1 + _t_12235;
			_t_12237 = -1.0f * _t_12236;
			_t_12234 = _t_12237;
		
		}

	_t_12238 = _t_12234 * _t_483;
	_t_12239 = _t_12227 * _t_12238;
	_t_12240 = -1.0f * ty3_9_1;
	_t_12241 = ty2_8_1 + _t_12240;
	_t_12242 = -1.0f * _t_12241;
	_t_12243 = _t_12242 < 0.0f;
	if(_t_12243)
		{
			float _t_12244;
			float _t_12245;
		
			_t_12244 = -1.0f * ty3_9_1;
			_t_12245 = ty2_8_1 + _t_12244;
			_t_12246 = _t_12245;
		
		}
else
		{
			float _t_12247;
			float _t_12248;
			float _t_12249;
		
			_t_12247 = -1.0f * ty3_9_1;
			_t_12248 = ty2_8_1 + _t_12247;
			_t_12249 = -1.0f * _t_12248;
			_t_12246 = _t_12249;
		
		}

	_t_12250 = _t_12246 * _t_483;
	_t_12251 = 1.0f + _t_12250;
	_t_12252 = 1.0f / _t_12251;
	_t_12253 = _t_12239 * _t_12252;
	_t_12254 = _t_12253 * -1.0f;
	_t_12255 = 1.0f + _t_12254;
	_t_12256 = -1.0f * ty3_9_1;
	_t_12257 = ty2_8_1 + _t_12256;
	_t_12258 = -1.0f * _t_12257;
	_t_12259 = _t_12258 < 0.0f;
	if(_t_12259)
		{
			float _t_12260;
			float _t_12261;
		
			_t_12260 = -1.0f * tx2_5_1;
			_t_12261 = tx3_6_1 + _t_12260;
			_t_12262 = _t_12261;
		
		}
else
		{
			float _t_12263;
			float _t_12264;
			float _t_12265;
		
			_t_12263 = -1.0f * tx2_5_1;
			_t_12264 = tx3_6_1 + _t_12263;
			_t_12265 = -1.0f * _t_12264;
			_t_12262 = _t_12265;
		
		}

	_t_12266 = _t_12262 * _t_483;
	_t_12267 = -1.0f * ty3_9_1;
	_t_12268 = ty2_8_1 + _t_12267;
	_t_12269 = -1.0f * _t_12268;
	_t_12270 = _t_12269 < 0.0f;
	if(_t_12270)
		{
			float _t_12271;
			float _t_12272;
		
			_t_12271 = -1.0f * tx2_5_1;
			_t_12272 = tx3_6_1 + _t_12271;
			_t_12273 = _t_12272;
		
		}
else
		{
			float _t_12274;
			float _t_12275;
			float _t_12276;
		
			_t_12274 = -1.0f * tx2_5_1;
			_t_12275 = tx3_6_1 + _t_12274;
			_t_12276 = -1.0f * _t_12275;
			_t_12273 = _t_12276;
		
		}

	_t_12277 = _t_12273 * _t_483;
	_t_12278 = _t_12266 * _t_12277;
	_t_12279 = -1.0f * ty3_9_1;
	_t_12280 = ty2_8_1 + _t_12279;
	_t_12281 = -1.0f * _t_12280;
	_t_12282 = _t_12281 < 0.0f;
	if(_t_12282)
		{
			float _t_12283;
			float _t_12284;
		
			_t_12283 = -1.0f * ty3_9_1;
			_t_12284 = ty2_8_1 + _t_12283;
			_t_12285 = _t_12284;
		
		}
else
		{
			float _t_12286;
			float _t_12287;
			float _t_12288;
		
			_t_12286 = -1.0f * ty3_9_1;
			_t_12287 = ty2_8_1 + _t_12286;
			_t_12288 = -1.0f * _t_12287;
			_t_12285 = _t_12288;
		
		}

	_t_12289 = _t_12285 * _t_483;
	_t_12290 = 1.0f + _t_12289;
	_t_12291 = 1.0f / _t_12290;
	_t_12292 = _t_12278 * _t_12291;
	_t_12293 = _t_12292 * -1.0f;
	_t_12294 = 1.0f + _t_12293;
	_t_12295 = 0.0f < _t_12294;
	if(_t_12295)
		{
		
			_t_12296 = py0_12_1;
		
		}
else
		{
		
			_t_12296 = py1_13_1;
		
		}

	_t_12297 = _t_12255 * _t_12296;
	_t_12298 = _t_12216 + _t_12297;
	_t_12299 = -1.0f * ty3_9_1;
	_t_12300 = ty2_8_1 + _t_12299;
	_t_12301 = -1.0f * _t_12300;
	_t_12302 = _t_12301 < 0.0f;
	if(_t_12302)
		{
			float _t_12303;
			float _t_12304;
		
			_t_12303 = -1.0f * tx2_5_1;
			_t_12304 = tx3_6_1 + _t_12303;
			_t_12305 = _t_12304;
		
		}
else
		{
			float _t_12306;
			float _t_12307;
			float _t_12308;
		
			_t_12306 = -1.0f * tx2_5_1;
			_t_12307 = tx3_6_1 + _t_12306;
			_t_12308 = -1.0f * _t_12307;
			_t_12305 = _t_12308;
		
		}

	_t_12309 = _t_12305 * _t_483;
	_t_12310 = _t_12309 * -1.0f;
	_t_12311 = -1.0f * ty3_9_1;
	_t_12312 = ty2_8_1 + _t_12311;
	_t_12313 = -1.0f * _t_12312;
	_t_12314 = _t_12313 < 0.0f;
	if(_t_12314)
		{
			float _t_12315;
			float _t_12316;
		
			_t_12315 = -1.0f * tx2_5_1;
			_t_12316 = tx3_6_1 + _t_12315;
			_t_12317 = _t_12316;
		
		}
else
		{
			float _t_12318;
			float _t_12319;
			float _t_12320;
		
			_t_12318 = -1.0f * tx2_5_1;
			_t_12319 = tx3_6_1 + _t_12318;
			_t_12320 = -1.0f * _t_12319;
			_t_12317 = _t_12320;
		
		}

	_t_12321 = _t_12317 * _t_483;
	_t_12322 = _t_12321 * -1.0f;
	_t_12323 = 0.0f < _t_12322;
	if(_t_12323)
		{
		
			_t_12324 = px1_11_1;
		
		}
else
		{
		
			_t_12324 = px0_10_1;
		
		}

	_t_12325 = _t_12310 * _t_12324;
	_t_12326 = -1.0f * ty3_9_1;
	_t_12327 = ty2_8_1 + _t_12326;
	_t_12328 = -1.0f * _t_12327;
	_t_12329 = _t_12328 < 0.0f;
	if(_t_12329)
		{
			float _t_12330;
			float _t_12331;
		
			_t_12330 = -1.0f * tx2_5_1;
			_t_12331 = tx3_6_1 + _t_12330;
			_t_12332 = _t_12331;
		
		}
else
		{
			float _t_12333;
			float _t_12334;
			float _t_12335;
		
			_t_12333 = -1.0f * tx2_5_1;
			_t_12334 = tx3_6_1 + _t_12333;
			_t_12335 = -1.0f * _t_12334;
			_t_12332 = _t_12335;
		
		}

	_t_12336 = _t_12332 * _t_483;
	_t_12337 = -1.0f * ty3_9_1;
	_t_12338 = ty2_8_1 + _t_12337;
	_t_12339 = -1.0f * _t_12338;
	_t_12340 = _t_12339 < 0.0f;
	if(_t_12340)
		{
			float _t_12341;
			float _t_12342;
		
			_t_12341 = -1.0f * tx2_5_1;
			_t_12342 = tx3_6_1 + _t_12341;
			_t_12343 = _t_12342;
		
		}
else
		{
			float _t_12344;
			float _t_12345;
			float _t_12346;
		
			_t_12344 = -1.0f * tx2_5_1;
			_t_12345 = tx3_6_1 + _t_12344;
			_t_12346 = -1.0f * _t_12345;
			_t_12343 = _t_12346;
		
		}

	_t_12347 = _t_12343 * _t_483;
	_t_12348 = _t_12336 * _t_12347;
	_t_12349 = -1.0f * ty3_9_1;
	_t_12350 = ty2_8_1 + _t_12349;
	_t_12351 = -1.0f * _t_12350;
	_t_12352 = _t_12351 < 0.0f;
	if(_t_12352)
		{
			float _t_12353;
			float _t_12354;
		
			_t_12353 = -1.0f * ty3_9_1;
			_t_12354 = ty2_8_1 + _t_12353;
			_t_12355 = _t_12354;
		
		}
else
		{
			float _t_12356;
			float _t_12357;
			float _t_12358;
		
			_t_12356 = -1.0f * ty3_9_1;
			_t_12357 = ty2_8_1 + _t_12356;
			_t_12358 = -1.0f * _t_12357;
			_t_12355 = _t_12358;
		
		}

	_t_12359 = _t_12355 * _t_483;
	_t_12360 = 1.0f + _t_12359;
	_t_12361 = 1.0f / _t_12360;
	_t_12362 = _t_12348 * _t_12361;
	_t_12363 = _t_12362 * -1.0f;
	_t_12364 = 1.0f + _t_12363;
	_t_12365 = -1.0f * ty3_9_1;
	_t_12366 = ty2_8_1 + _t_12365;
	_t_12367 = -1.0f * _t_12366;
	_t_12368 = _t_12367 < 0.0f;
	if(_t_12368)
		{
			float _t_12369;
			float _t_12370;
		
			_t_12369 = -1.0f * tx2_5_1;
			_t_12370 = tx3_6_1 + _t_12369;
			_t_12371 = _t_12370;
		
		}
else
		{
			float _t_12372;
			float _t_12373;
			float _t_12374;
		
			_t_12372 = -1.0f * tx2_5_1;
			_t_12373 = tx3_6_1 + _t_12372;
			_t_12374 = -1.0f * _t_12373;
			_t_12371 = _t_12374;
		
		}

	_t_12375 = _t_12371 * _t_483;
	_t_12376 = -1.0f * ty3_9_1;
	_t_12377 = ty2_8_1 + _t_12376;
	_t_12378 = -1.0f * _t_12377;
	_t_12379 = _t_12378 < 0.0f;
	if(_t_12379)
		{
			float _t_12380;
			float _t_12381;
		
			_t_12380 = -1.0f * tx2_5_1;
			_t_12381 = tx3_6_1 + _t_12380;
			_t_12382 = _t_12381;
		
		}
else
		{
			float _t_12383;
			float _t_12384;
			float _t_12385;
		
			_t_12383 = -1.0f * tx2_5_1;
			_t_12384 = tx3_6_1 + _t_12383;
			_t_12385 = -1.0f * _t_12384;
			_t_12382 = _t_12385;
		
		}

	_t_12386 = _t_12382 * _t_483;
	_t_12387 = _t_12375 * _t_12386;
	_t_12388 = -1.0f * ty3_9_1;
	_t_12389 = ty2_8_1 + _t_12388;
	_t_12390 = -1.0f * _t_12389;
	_t_12391 = _t_12390 < 0.0f;
	if(_t_12391)
		{
			float _t_12392;
			float _t_12393;
		
			_t_12392 = -1.0f * ty3_9_1;
			_t_12393 = ty2_8_1 + _t_12392;
			_t_12394 = _t_12393;
		
		}
else
		{
			float _t_12395;
			float _t_12396;
			float _t_12397;
		
			_t_12395 = -1.0f * ty3_9_1;
			_t_12396 = ty2_8_1 + _t_12395;
			_t_12397 = -1.0f * _t_12396;
			_t_12394 = _t_12397;
		
		}

	_t_12398 = _t_12394 * _t_483;
	_t_12399 = 1.0f + _t_12398;
	_t_12400 = 1.0f / _t_12399;
	_t_12401 = _t_12387 * _t_12400;
	_t_12402 = _t_12401 * -1.0f;
	_t_12403 = 1.0f + _t_12402;
	_t_12404 = 0.0f < _t_12403;
	if(_t_12404)
		{
		
			_t_12405 = py1_13_1;
		
		}
else
		{
		
			_t_12405 = py0_12_1;
		
		}

	_t_12406 = _t_12364 * _t_12405;
	_t_12407 = _t_12325 + _t_12406;
	_t_484 = tegpixelintegrator_30(ty3_9_1,pc1_15_1,tc2_19_1,ty2_8_1,_t_12298,ty1_7_1,pc0_14_1,_t_483,tx3_6_1,tx1_4_1,tx2_5_1,py1_13_1,pc2_16_1,px1_11_1,tc0_17_1,py0_12_1,tc1_18_1,_t_12407,px0_10_1);

	return _t_484;
}
__device__ float tegpixellet_block_48(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx3_6_1,float ty1_7_1,float tx1_4_1,float ty3_9_1,float _t_13427,float _t_13480,float ty2_8_1,float tx2_5_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float _t_511,float y__3609_1,float _t_13400){
	float _t_13481;
	float _t_13482;
	float _t_13483;
	float _t_13484;
	float _t_13485;
	float _t_13486;
	float _t_13487;
	float _t_13488;
	float _t_13489;
	float _t_13490;
	float _t_13491;
	float _t_13492;
	float _t_13493;
	float _t_13494;
	float _t_13495;
	float _t_13496;
	float _t_13497;
	float _t_13498;
	float _t_13499;
	float _t_13500;
	float _t_13501;
	float _t_13502;
	float _t_13503;
	bool _t_13504;
	float _t_13505;
	float _t_13506;
	float _t_13507;
	float _t_13508;
	float _t_13509;
	float _t_13510;
	float _t_13511;
	float _t_13512;
	float _t_13513;
	float _t_13514;
	float _t_13515;
	float _t_13516;
	float _t_13517;
	float _t_13518;
	bool _t_13519;
	float _t_13520;
	float _t_13521;
	float _t_13522;
	float _t_13523;
	bool _t_13524;
	bool _t_13525;
	bool _t_13526;
	bool _t_13527;
	bool _t_13528;
	bool _t_13529;
	bool _t_13530;
	float _t_13860;

	float _t_13401;

	_t_13481 = -1.0f * pc0_14_1;
	_t_13482 = tc0_17_1 + _t_13481;
	_t_13483 = _t_13482 * _t_13482;
	_t_13484 = -1.0f * pc1_15_1;
	_t_13485 = tc1_18_1 + _t_13484;
	_t_13486 = _t_13485 * _t_13485;
	_t_13487 = _t_13483 + _t_13486;
	_t_13488 = -1.0f * pc2_16_1;
	_t_13489 = tc2_19_1 + _t_13488;
	_t_13490 = _t_13489 * _t_13489;
	_t_13491 = _t_13487 + _t_13490;
	_t_13492 = tx3_6_1 * ty1_7_1;
	_t_13493 = tx1_4_1 * ty3_9_1;
	_t_13494 = _t_13493 * -1.0f;
	_t_13495 = _t_13492 + _t_13494;
	_t_13496 = -1.0f * ty1_7_1;
	_t_13497 = ty3_9_1 + _t_13496;
	_t_13498 = _t_13497 * _t_13427;
	_t_13499 = _t_13495 + _t_13498;
	_t_13500 = -1.0f * tx3_6_1;
	_t_13501 = tx1_4_1 + _t_13500;
	_t_13502 = _t_13501 * _t_13480;
	_t_13503 = _t_13499 + _t_13502;
	_t_13504 = _t_13503 < 0.0f;
	if(_t_13504)
		{
		
			_t_13505 = 1.0f;
		
		}
else
		{
		
			_t_13505 = 0.0f;
		
		}

	_t_13506 = _t_13491 * _t_13505;
	_t_13507 = tx1_4_1 * ty2_8_1;
	_t_13508 = tx2_5_1 * ty1_7_1;
	_t_13509 = _t_13508 * -1.0f;
	_t_13510 = _t_13507 + _t_13509;
	_t_13511 = -1.0f * ty2_8_1;
	_t_13512 = ty1_7_1 + _t_13511;
	_t_13513 = _t_13512 * _t_13427;
	_t_13514 = _t_13510 + _t_13513;
	_t_13515 = -1.0f * tx1_4_1;
	_t_13516 = tx2_5_1 + _t_13515;
	_t_13517 = _t_13516 * _t_13480;
	_t_13518 = _t_13514 + _t_13517;
	_t_13519 = _t_13518 < 0.0f;
	if(_t_13519)
		{
		
			_t_13520 = 1.0f;
		
		}
else
		{
		
			_t_13520 = 0.0f;
		
		}

	_t_13521 = _t_13506 * _t_13520;
	_t_13522 = _t_13521 * _t_13427;
	_t_13523 = _t_13522 * -1.0f;
	_t_13524 = py0_12_1 < _t_13480;
	_t_13525 = _t_13480 < py1_13_1;
	_t_13526 = _t_13524 && _t_13525;
	_t_13527 = px0_10_1 < _t_13427;
	_t_13528 = _t_13427 < px1_11_1;
	_t_13529 = _t_13527 && _t_13528;
	_t_13530 = _t_13526 && _t_13529;
	if(_t_13530)
		{
			float _t_13531;
			float _t_13532;
			float _t_13533;
			bool _t_13534;
			float _t_13537;
			float _t_13541;
			float _t_13542;
			float _t_13543;
			float _t_13544;
			float _t_13545;
			bool _t_13546;
			float _t_13549;
			float _t_13553;
			float _t_13554;
			bool _t_13555;
			float _t_13556;
			float _t_13557;
			float _t_13558;
			float _t_13559;
			float _t_13560;
			bool _t_13561;
			float _t_13564;
			float _t_13568;
			float _t_13569;
			float _t_13570;
			float _t_13571;
			bool _t_13572;
			float _t_13575;
			float _t_13579;
			float _t_13580;
			float _t_13581;
			float _t_13582;
			float _t_13583;
			bool _t_13584;
			float _t_13587;
			float _t_13591;
			float _t_13592;
			float _t_13593;
			float _t_13594;
			float _t_13595;
			float _t_13596;
			float _t_13597;
			float _t_13598;
			float _t_13599;
			bool _t_13600;
			float _t_13603;
			float _t_13607;
			float _t_13608;
			float _t_13609;
			float _t_13610;
			bool _t_13611;
			float _t_13614;
			float _t_13618;
			float _t_13619;
			float _t_13620;
			float _t_13621;
			float _t_13622;
			bool _t_13623;
			float _t_13626;
			float _t_13630;
			float _t_13631;
			float _t_13632;
			float _t_13633;
			float _t_13634;
			float _t_13635;
			bool _t_13636;
			float _t_13637;
			float _t_13638;
			float _t_13639;
			bool _t_13640;
			float _t_13641;
			float _t_13642;
			float _t_13643;
			bool _t_13644;
			float _t_13647;
			float _t_13651;
			float _t_13652;
			float _t_13653;
			float _t_13654;
			float _t_13655;
			bool _t_13656;
			float _t_13659;
			float _t_13663;
			float _t_13664;
			bool _t_13665;
			float _t_13666;
			float _t_13667;
			float _t_13668;
			float _t_13669;
			float _t_13670;
			bool _t_13671;
			float _t_13674;
			float _t_13678;
			float _t_13679;
			float _t_13680;
			float _t_13681;
			bool _t_13682;
			float _t_13685;
			float _t_13689;
			float _t_13690;
			float _t_13691;
			float _t_13692;
			float _t_13693;
			bool _t_13694;
			float _t_13697;
			float _t_13701;
			float _t_13702;
			float _t_13703;
			float _t_13704;
			float _t_13705;
			float _t_13706;
			float _t_13707;
			float _t_13708;
			float _t_13709;
			bool _t_13710;
			float _t_13713;
			float _t_13717;
			float _t_13718;
			float _t_13719;
			float _t_13720;
			bool _t_13721;
			float _t_13724;
			float _t_13728;
			float _t_13729;
			float _t_13730;
			float _t_13731;
			float _t_13732;
			bool _t_13733;
			float _t_13736;
			float _t_13740;
			float _t_13741;
			float _t_13742;
			float _t_13743;
			float _t_13744;
			float _t_13745;
			bool _t_13746;
			float _t_13747;
			float _t_13748;
			float _t_13749;
			bool _t_13750;
			bool _t_13751;
			float _t_13752;
			float _t_13753;
			float _t_13754;
			bool _t_13755;
			float _t_13758;
			float _t_13762;
			float _t_13763;
			float _t_13764;
			float _t_13765;
			bool _t_13766;
			float _t_13769;
			float _t_13773;
			bool _t_13774;
			float _t_13775;
			float _t_13776;
			float _t_13777;
			float _t_13778;
			float _t_13779;
			bool _t_13780;
			float _t_13783;
			float _t_13787;
			float _t_13788;
			float _t_13789;
			float _t_13790;
			bool _t_13791;
			float _t_13794;
			float _t_13798;
			bool _t_13799;
			float _t_13800;
			float _t_13801;
			float _t_13802;
			bool _t_13803;
			float _t_13804;
			float _t_13805;
			float _t_13806;
			bool _t_13807;
			float _t_13810;
			float _t_13814;
			float _t_13815;
			float _t_13816;
			float _t_13817;
			bool _t_13818;
			float _t_13821;
			float _t_13825;
			bool _t_13826;
			float _t_13827;
			float _t_13828;
			float _t_13829;
			float _t_13830;
			float _t_13831;
			bool _t_13832;
			float _t_13835;
			float _t_13839;
			float _t_13840;
			float _t_13841;
			float _t_13842;
			bool _t_13843;
			float _t_13846;
			float _t_13850;
			bool _t_13851;
			float _t_13852;
			float _t_13853;
			float _t_13854;
			bool _t_13855;
			bool _t_13856;
			bool _t_13857;
			float _t_13858;
			float _t_13859;
		
			_t_13531 = -1.0f * ty3_9_1;
			_t_13532 = ty2_8_1 + _t_13531;
			_t_13533 = -1.0f * _t_13532;
			_t_13534 = _t_13533 < 0.0f;
			if(_t_13534)
				{
					float _t_13535;
					float _t_13536;
				
					_t_13535 = -1.0f * tx2_5_1;
					_t_13536 = tx3_6_1 + _t_13535;
					_t_13537 = _t_13536;
				
				}
		else
				{
					float _t_13538;
					float _t_13539;
					float _t_13540;
				
					_t_13538 = -1.0f * tx2_5_1;
					_t_13539 = tx3_6_1 + _t_13538;
					_t_13540 = -1.0f * _t_13539;
					_t_13537 = _t_13540;
				
				}
		
			_t_13541 = _t_13537 * _t_511;
			_t_13542 = _t_13541 * -1.0f;
			_t_13543 = -1.0f * ty3_9_1;
			_t_13544 = ty2_8_1 + _t_13543;
			_t_13545 = -1.0f * _t_13544;
			_t_13546 = _t_13545 < 0.0f;
			if(_t_13546)
				{
					float _t_13547;
					float _t_13548;
				
					_t_13547 = -1.0f * tx2_5_1;
					_t_13548 = tx3_6_1 + _t_13547;
					_t_13549 = _t_13548;
				
				}
		else
				{
					float _t_13550;
					float _t_13551;
					float _t_13552;
				
					_t_13550 = -1.0f * tx2_5_1;
					_t_13551 = tx3_6_1 + _t_13550;
					_t_13552 = -1.0f * _t_13551;
					_t_13549 = _t_13552;
				
				}
		
			_t_13553 = _t_13549 * _t_511;
			_t_13554 = _t_13553 * -1.0f;
			_t_13555 = 0.0f < _t_13554;
			if(_t_13555)
				{
				
					_t_13556 = px0_10_1;
				
				}
		else
				{
				
					_t_13556 = px1_11_1;
				
				}
		
			_t_13557 = _t_13542 * _t_13556;
			_t_13558 = -1.0f * ty3_9_1;
			_t_13559 = ty2_8_1 + _t_13558;
			_t_13560 = -1.0f * _t_13559;
			_t_13561 = _t_13560 < 0.0f;
			if(_t_13561)
				{
					float _t_13562;
					float _t_13563;
				
					_t_13562 = -1.0f * tx2_5_1;
					_t_13563 = tx3_6_1 + _t_13562;
					_t_13564 = _t_13563;
				
				}
		else
				{
					float _t_13565;
					float _t_13566;
					float _t_13567;
				
					_t_13565 = -1.0f * tx2_5_1;
					_t_13566 = tx3_6_1 + _t_13565;
					_t_13567 = -1.0f * _t_13566;
					_t_13564 = _t_13567;
				
				}
		
			_t_13568 = _t_13564 * _t_511;
			_t_13569 = -1.0f * ty3_9_1;
			_t_13570 = ty2_8_1 + _t_13569;
			_t_13571 = -1.0f * _t_13570;
			_t_13572 = _t_13571 < 0.0f;
			if(_t_13572)
				{
					float _t_13573;
					float _t_13574;
				
					_t_13573 = -1.0f * tx2_5_1;
					_t_13574 = tx3_6_1 + _t_13573;
					_t_13575 = _t_13574;
				
				}
		else
				{
					float _t_13576;
					float _t_13577;
					float _t_13578;
				
					_t_13576 = -1.0f * tx2_5_1;
					_t_13577 = tx3_6_1 + _t_13576;
					_t_13578 = -1.0f * _t_13577;
					_t_13575 = _t_13578;
				
				}
		
			_t_13579 = _t_13575 * _t_511;
			_t_13580 = _t_13568 * _t_13579;
			_t_13581 = -1.0f * ty3_9_1;
			_t_13582 = ty2_8_1 + _t_13581;
			_t_13583 = -1.0f * _t_13582;
			_t_13584 = _t_13583 < 0.0f;
			if(_t_13584)
				{
					float _t_13585;
					float _t_13586;
				
					_t_13585 = -1.0f * ty3_9_1;
					_t_13586 = ty2_8_1 + _t_13585;
					_t_13587 = _t_13586;
				
				}
		else
				{
					float _t_13588;
					float _t_13589;
					float _t_13590;
				
					_t_13588 = -1.0f * ty3_9_1;
					_t_13589 = ty2_8_1 + _t_13588;
					_t_13590 = -1.0f * _t_13589;
					_t_13587 = _t_13590;
				
				}
		
			_t_13591 = _t_13587 * _t_511;
			_t_13592 = 1.0f + _t_13591;
			_t_13593 = 1.0f / _t_13592;
			_t_13594 = _t_13580 * _t_13593;
			_t_13595 = _t_13594 * -1.0f;
			_t_13596 = 1.0f + _t_13595;
			_t_13597 = -1.0f * ty3_9_1;
			_t_13598 = ty2_8_1 + _t_13597;
			_t_13599 = -1.0f * _t_13598;
			_t_13600 = _t_13599 < 0.0f;
			if(_t_13600)
				{
					float _t_13601;
					float _t_13602;
				
					_t_13601 = -1.0f * tx2_5_1;
					_t_13602 = tx3_6_1 + _t_13601;
					_t_13603 = _t_13602;
				
				}
		else
				{
					float _t_13604;
					float _t_13605;
					float _t_13606;
				
					_t_13604 = -1.0f * tx2_5_1;
					_t_13605 = tx3_6_1 + _t_13604;
					_t_13606 = -1.0f * _t_13605;
					_t_13603 = _t_13606;
				
				}
		
			_t_13607 = _t_13603 * _t_511;
			_t_13608 = -1.0f * ty3_9_1;
			_t_13609 = ty2_8_1 + _t_13608;
			_t_13610 = -1.0f * _t_13609;
			_t_13611 = _t_13610 < 0.0f;
			if(_t_13611)
				{
					float _t_13612;
					float _t_13613;
				
					_t_13612 = -1.0f * tx2_5_1;
					_t_13613 = tx3_6_1 + _t_13612;
					_t_13614 = _t_13613;
				
				}
		else
				{
					float _t_13615;
					float _t_13616;
					float _t_13617;
				
					_t_13615 = -1.0f * tx2_5_1;
					_t_13616 = tx3_6_1 + _t_13615;
					_t_13617 = -1.0f * _t_13616;
					_t_13614 = _t_13617;
				
				}
		
			_t_13618 = _t_13614 * _t_511;
			_t_13619 = _t_13607 * _t_13618;
			_t_13620 = -1.0f * ty3_9_1;
			_t_13621 = ty2_8_1 + _t_13620;
			_t_13622 = -1.0f * _t_13621;
			_t_13623 = _t_13622 < 0.0f;
			if(_t_13623)
				{
					float _t_13624;
					float _t_13625;
				
					_t_13624 = -1.0f * ty3_9_1;
					_t_13625 = ty2_8_1 + _t_13624;
					_t_13626 = _t_13625;
				
				}
		else
				{
					float _t_13627;
					float _t_13628;
					float _t_13629;
				
					_t_13627 = -1.0f * ty3_9_1;
					_t_13628 = ty2_8_1 + _t_13627;
					_t_13629 = -1.0f * _t_13628;
					_t_13626 = _t_13629;
				
				}
		
			_t_13630 = _t_13626 * _t_511;
			_t_13631 = 1.0f + _t_13630;
			_t_13632 = 1.0f / _t_13631;
			_t_13633 = _t_13619 * _t_13632;
			_t_13634 = _t_13633 * -1.0f;
			_t_13635 = 1.0f + _t_13634;
			_t_13636 = 0.0f < _t_13635;
			if(_t_13636)
				{
				
					_t_13637 = py0_12_1;
				
				}
		else
				{
				
					_t_13637 = py1_13_1;
				
				}
		
			_t_13638 = _t_13596 * _t_13637;
			_t_13639 = _t_13557 + _t_13638;
			_t_13640 = _t_13639 < y__3609_1;
			_t_13641 = -1.0f * ty3_9_1;
			_t_13642 = ty2_8_1 + _t_13641;
			_t_13643 = -1.0f * _t_13642;
			_t_13644 = _t_13643 < 0.0f;
			if(_t_13644)
				{
					float _t_13645;
					float _t_13646;
				
					_t_13645 = -1.0f * tx2_5_1;
					_t_13646 = tx3_6_1 + _t_13645;
					_t_13647 = _t_13646;
				
				}
		else
				{
					float _t_13648;
					float _t_13649;
					float _t_13650;
				
					_t_13648 = -1.0f * tx2_5_1;
					_t_13649 = tx3_6_1 + _t_13648;
					_t_13650 = -1.0f * _t_13649;
					_t_13647 = _t_13650;
				
				}
		
			_t_13651 = _t_13647 * _t_511;
			_t_13652 = _t_13651 * -1.0f;
			_t_13653 = -1.0f * ty3_9_1;
			_t_13654 = ty2_8_1 + _t_13653;
			_t_13655 = -1.0f * _t_13654;
			_t_13656 = _t_13655 < 0.0f;
			if(_t_13656)
				{
					float _t_13657;
					float _t_13658;
				
					_t_13657 = -1.0f * tx2_5_1;
					_t_13658 = tx3_6_1 + _t_13657;
					_t_13659 = _t_13658;
				
				}
		else
				{
					float _t_13660;
					float _t_13661;
					float _t_13662;
				
					_t_13660 = -1.0f * tx2_5_1;
					_t_13661 = tx3_6_1 + _t_13660;
					_t_13662 = -1.0f * _t_13661;
					_t_13659 = _t_13662;
				
				}
		
			_t_13663 = _t_13659 * _t_511;
			_t_13664 = _t_13663 * -1.0f;
			_t_13665 = 0.0f < _t_13664;
			if(_t_13665)
				{
				
					_t_13666 = px1_11_1;
				
				}
		else
				{
				
					_t_13666 = px0_10_1;
				
				}
		
			_t_13667 = _t_13652 * _t_13666;
			_t_13668 = -1.0f * ty3_9_1;
			_t_13669 = ty2_8_1 + _t_13668;
			_t_13670 = -1.0f * _t_13669;
			_t_13671 = _t_13670 < 0.0f;
			if(_t_13671)
				{
					float _t_13672;
					float _t_13673;
				
					_t_13672 = -1.0f * tx2_5_1;
					_t_13673 = tx3_6_1 + _t_13672;
					_t_13674 = _t_13673;
				
				}
		else
				{
					float _t_13675;
					float _t_13676;
					float _t_13677;
				
					_t_13675 = -1.0f * tx2_5_1;
					_t_13676 = tx3_6_1 + _t_13675;
					_t_13677 = -1.0f * _t_13676;
					_t_13674 = _t_13677;
				
				}
		
			_t_13678 = _t_13674 * _t_511;
			_t_13679 = -1.0f * ty3_9_1;
			_t_13680 = ty2_8_1 + _t_13679;
			_t_13681 = -1.0f * _t_13680;
			_t_13682 = _t_13681 < 0.0f;
			if(_t_13682)
				{
					float _t_13683;
					float _t_13684;
				
					_t_13683 = -1.0f * tx2_5_1;
					_t_13684 = tx3_6_1 + _t_13683;
					_t_13685 = _t_13684;
				
				}
		else
				{
					float _t_13686;
					float _t_13687;
					float _t_13688;
				
					_t_13686 = -1.0f * tx2_5_1;
					_t_13687 = tx3_6_1 + _t_13686;
					_t_13688 = -1.0f * _t_13687;
					_t_13685 = _t_13688;
				
				}
		
			_t_13689 = _t_13685 * _t_511;
			_t_13690 = _t_13678 * _t_13689;
			_t_13691 = -1.0f * ty3_9_1;
			_t_13692 = ty2_8_1 + _t_13691;
			_t_13693 = -1.0f * _t_13692;
			_t_13694 = _t_13693 < 0.0f;
			if(_t_13694)
				{
					float _t_13695;
					float _t_13696;
				
					_t_13695 = -1.0f * ty3_9_1;
					_t_13696 = ty2_8_1 + _t_13695;
					_t_13697 = _t_13696;
				
				}
		else
				{
					float _t_13698;
					float _t_13699;
					float _t_13700;
				
					_t_13698 = -1.0f * ty3_9_1;
					_t_13699 = ty2_8_1 + _t_13698;
					_t_13700 = -1.0f * _t_13699;
					_t_13697 = _t_13700;
				
				}
		
			_t_13701 = _t_13697 * _t_511;
			_t_13702 = 1.0f + _t_13701;
			_t_13703 = 1.0f / _t_13702;
			_t_13704 = _t_13690 * _t_13703;
			_t_13705 = _t_13704 * -1.0f;
			_t_13706 = 1.0f + _t_13705;
			_t_13707 = -1.0f * ty3_9_1;
			_t_13708 = ty2_8_1 + _t_13707;
			_t_13709 = -1.0f * _t_13708;
			_t_13710 = _t_13709 < 0.0f;
			if(_t_13710)
				{
					float _t_13711;
					float _t_13712;
				
					_t_13711 = -1.0f * tx2_5_1;
					_t_13712 = tx3_6_1 + _t_13711;
					_t_13713 = _t_13712;
				
				}
		else
				{
					float _t_13714;
					float _t_13715;
					float _t_13716;
				
					_t_13714 = -1.0f * tx2_5_1;
					_t_13715 = tx3_6_1 + _t_13714;
					_t_13716 = -1.0f * _t_13715;
					_t_13713 = _t_13716;
				
				}
		
			_t_13717 = _t_13713 * _t_511;
			_t_13718 = -1.0f * ty3_9_1;
			_t_13719 = ty2_8_1 + _t_13718;
			_t_13720 = -1.0f * _t_13719;
			_t_13721 = _t_13720 < 0.0f;
			if(_t_13721)
				{
					float _t_13722;
					float _t_13723;
				
					_t_13722 = -1.0f * tx2_5_1;
					_t_13723 = tx3_6_1 + _t_13722;
					_t_13724 = _t_13723;
				
				}
		else
				{
					float _t_13725;
					float _t_13726;
					float _t_13727;
				
					_t_13725 = -1.0f * tx2_5_1;
					_t_13726 = tx3_6_1 + _t_13725;
					_t_13727 = -1.0f * _t_13726;
					_t_13724 = _t_13727;
				
				}
		
			_t_13728 = _t_13724 * _t_511;
			_t_13729 = _t_13717 * _t_13728;
			_t_13730 = -1.0f * ty3_9_1;
			_t_13731 = ty2_8_1 + _t_13730;
			_t_13732 = -1.0f * _t_13731;
			_t_13733 = _t_13732 < 0.0f;
			if(_t_13733)
				{
					float _t_13734;
					float _t_13735;
				
					_t_13734 = -1.0f * ty3_9_1;
					_t_13735 = ty2_8_1 + _t_13734;
					_t_13736 = _t_13735;
				
				}
		else
				{
					float _t_13737;
					float _t_13738;
					float _t_13739;
				
					_t_13737 = -1.0f * ty3_9_1;
					_t_13738 = ty2_8_1 + _t_13737;
					_t_13739 = -1.0f * _t_13738;
					_t_13736 = _t_13739;
				
				}
		
			_t_13740 = _t_13736 * _t_511;
			_t_13741 = 1.0f + _t_13740;
			_t_13742 = 1.0f / _t_13741;
			_t_13743 = _t_13729 * _t_13742;
			_t_13744 = _t_13743 * -1.0f;
			_t_13745 = 1.0f + _t_13744;
			_t_13746 = 0.0f < _t_13745;
			if(_t_13746)
				{
				
					_t_13747 = py1_13_1;
				
				}
		else
				{
				
					_t_13747 = py0_12_1;
				
				}
		
			_t_13748 = _t_13706 * _t_13747;
			_t_13749 = _t_13667 + _t_13748;
			_t_13750 = y__3609_1 < _t_13749;
			_t_13751 = _t_13640 && _t_13750;
			_t_13752 = -1.0f * ty3_9_1;
			_t_13753 = ty2_8_1 + _t_13752;
			_t_13754 = -1.0f * _t_13753;
			_t_13755 = _t_13754 < 0.0f;
			if(_t_13755)
				{
					float _t_13756;
					float _t_13757;
				
					_t_13756 = -1.0f * ty3_9_1;
					_t_13757 = ty2_8_1 + _t_13756;
					_t_13758 = _t_13757;
				
				}
		else
				{
					float _t_13759;
					float _t_13760;
					float _t_13761;
				
					_t_13759 = -1.0f * ty3_9_1;
					_t_13760 = ty2_8_1 + _t_13759;
					_t_13761 = -1.0f * _t_13760;
					_t_13758 = _t_13761;
				
				}
		
			_t_13762 = _t_13758 * _t_511;
			_t_13763 = -1.0f * ty3_9_1;
			_t_13764 = ty2_8_1 + _t_13763;
			_t_13765 = -1.0f * _t_13764;
			_t_13766 = _t_13765 < 0.0f;
			if(_t_13766)
				{
					float _t_13767;
					float _t_13768;
				
					_t_13767 = -1.0f * ty3_9_1;
					_t_13768 = ty2_8_1 + _t_13767;
					_t_13769 = _t_13768;
				
				}
		else
				{
					float _t_13770;
					float _t_13771;
					float _t_13772;
				
					_t_13770 = -1.0f * ty3_9_1;
					_t_13771 = ty2_8_1 + _t_13770;
					_t_13772 = -1.0f * _t_13771;
					_t_13769 = _t_13772;
				
				}
		
			_t_13773 = _t_13769 * _t_511;
			_t_13774 = 0.0f < _t_13773;
			if(_t_13774)
				{
				
					_t_13775 = px0_10_1;
				
				}
		else
				{
				
					_t_13775 = px1_11_1;
				
				}
		
			_t_13776 = _t_13762 * _t_13775;
			_t_13777 = -1.0f * ty3_9_1;
			_t_13778 = ty2_8_1 + _t_13777;
			_t_13779 = -1.0f * _t_13778;
			_t_13780 = _t_13779 < 0.0f;
			if(_t_13780)
				{
					float _t_13781;
					float _t_13782;
				
					_t_13781 = -1.0f * tx2_5_1;
					_t_13782 = tx3_6_1 + _t_13781;
					_t_13783 = _t_13782;
				
				}
		else
				{
					float _t_13784;
					float _t_13785;
					float _t_13786;
				
					_t_13784 = -1.0f * tx2_5_1;
					_t_13785 = tx3_6_1 + _t_13784;
					_t_13786 = -1.0f * _t_13785;
					_t_13783 = _t_13786;
				
				}
		
			_t_13787 = _t_13783 * _t_511;
			_t_13788 = -1.0f * ty3_9_1;
			_t_13789 = ty2_8_1 + _t_13788;
			_t_13790 = -1.0f * _t_13789;
			_t_13791 = _t_13790 < 0.0f;
			if(_t_13791)
				{
					float _t_13792;
					float _t_13793;
				
					_t_13792 = -1.0f * tx2_5_1;
					_t_13793 = tx3_6_1 + _t_13792;
					_t_13794 = _t_13793;
				
				}
		else
				{
					float _t_13795;
					float _t_13796;
					float _t_13797;
				
					_t_13795 = -1.0f * tx2_5_1;
					_t_13796 = tx3_6_1 + _t_13795;
					_t_13797 = -1.0f * _t_13796;
					_t_13794 = _t_13797;
				
				}
		
			_t_13798 = _t_13794 * _t_511;
			_t_13799 = 0.0f < _t_13798;
			if(_t_13799)
				{
				
					_t_13800 = py0_12_1;
				
				}
		else
				{
				
					_t_13800 = py1_13_1;
				
				}
		
			_t_13801 = _t_13787 * _t_13800;
			_t_13802 = _t_13776 + _t_13801;
			_t_13803 = _t_13802 < _t_13400;
			_t_13804 = -1.0f * ty3_9_1;
			_t_13805 = ty2_8_1 + _t_13804;
			_t_13806 = -1.0f * _t_13805;
			_t_13807 = _t_13806 < 0.0f;
			if(_t_13807)
				{
					float _t_13808;
					float _t_13809;
				
					_t_13808 = -1.0f * ty3_9_1;
					_t_13809 = ty2_8_1 + _t_13808;
					_t_13810 = _t_13809;
				
				}
		else
				{
					float _t_13811;
					float _t_13812;
					float _t_13813;
				
					_t_13811 = -1.0f * ty3_9_1;
					_t_13812 = ty2_8_1 + _t_13811;
					_t_13813 = -1.0f * _t_13812;
					_t_13810 = _t_13813;
				
				}
		
			_t_13814 = _t_13810 * _t_511;
			_t_13815 = -1.0f * ty3_9_1;
			_t_13816 = ty2_8_1 + _t_13815;
			_t_13817 = -1.0f * _t_13816;
			_t_13818 = _t_13817 < 0.0f;
			if(_t_13818)
				{
					float _t_13819;
					float _t_13820;
				
					_t_13819 = -1.0f * ty3_9_1;
					_t_13820 = ty2_8_1 + _t_13819;
					_t_13821 = _t_13820;
				
				}
		else
				{
					float _t_13822;
					float _t_13823;
					float _t_13824;
				
					_t_13822 = -1.0f * ty3_9_1;
					_t_13823 = ty2_8_1 + _t_13822;
					_t_13824 = -1.0f * _t_13823;
					_t_13821 = _t_13824;
				
				}
		
			_t_13825 = _t_13821 * _t_511;
			_t_13826 = 0.0f < _t_13825;
			if(_t_13826)
				{
				
					_t_13827 = px1_11_1;
				
				}
		else
				{
				
					_t_13827 = px0_10_1;
				
				}
		
			_t_13828 = _t_13814 * _t_13827;
			_t_13829 = -1.0f * ty3_9_1;
			_t_13830 = ty2_8_1 + _t_13829;
			_t_13831 = -1.0f * _t_13830;
			_t_13832 = _t_13831 < 0.0f;
			if(_t_13832)
				{
					float _t_13833;
					float _t_13834;
				
					_t_13833 = -1.0f * tx2_5_1;
					_t_13834 = tx3_6_1 + _t_13833;
					_t_13835 = _t_13834;
				
				}
		else
				{
					float _t_13836;
					float _t_13837;
					float _t_13838;
				
					_t_13836 = -1.0f * tx2_5_1;
					_t_13837 = tx3_6_1 + _t_13836;
					_t_13838 = -1.0f * _t_13837;
					_t_13835 = _t_13838;
				
				}
		
			_t_13839 = _t_13835 * _t_511;
			_t_13840 = -1.0f * ty3_9_1;
			_t_13841 = ty2_8_1 + _t_13840;
			_t_13842 = -1.0f * _t_13841;
			_t_13843 = _t_13842 < 0.0f;
			if(_t_13843)
				{
					float _t_13844;
					float _t_13845;
				
					_t_13844 = -1.0f * tx2_5_1;
					_t_13845 = tx3_6_1 + _t_13844;
					_t_13846 = _t_13845;
				
				}
		else
				{
					float _t_13847;
					float _t_13848;
					float _t_13849;
				
					_t_13847 = -1.0f * tx2_5_1;
					_t_13848 = tx3_6_1 + _t_13847;
					_t_13849 = -1.0f * _t_13848;
					_t_13846 = _t_13849;
				
				}
		
			_t_13850 = _t_13846 * _t_511;
			_t_13851 = 0.0f < _t_13850;
			if(_t_13851)
				{
				
					_t_13852 = py1_13_1;
				
				}
		else
				{
				
					_t_13852 = py0_12_1;
				
				}
		
			_t_13853 = _t_13839 * _t_13852;
			_t_13854 = _t_13828 + _t_13853;
			_t_13855 = _t_13400 < _t_13854;
			_t_13856 = _t_13803 && _t_13855;
			_t_13857 = _t_13751 && _t_13856;
			if(_t_13857)
				{
				
					_t_13858 = 1.0f;
				
				}
		else
				{
				
					_t_13858 = 0.0f;
				
				}
		
			_t_13859 = _t_13858 * _t_511;
			_t_13860 = _t_13859;
		
		}
else
		{
		
			_t_13860 = 0.0f;
		
		}

	_t_13401 = _t_13523 * _t_13860;

	return _t_13401;
}
__device__ float tegpixellet_block_47(float ty2_8_1,float ty3_9_1,float _t_511,float _t_13400,float tx3_6_1,float tx2_5_1,float y__3609_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1){
	float _t_13402;
	float _t_13403;
	float _t_13404;
	bool _t_13405;
	float _t_13408;
	float _t_13412;
	float _t_13413;
	float _t_13414;
	float _t_13415;
	float _t_13416;
	bool _t_13417;
	float _t_13420;
	float _t_13424;
	float _t_13425;
	float _t_13426;
	float _t_13427;
	float _t_13428;
	float _t_13429;
	float _t_13430;
	bool _t_13431;
	float _t_13434;
	float _t_13438;
	float _t_13439;
	float _t_13440;
	float _t_13441;
	bool _t_13442;
	float _t_13445;
	float _t_13449;
	float _t_13450;
	float _t_13451;
	float _t_13452;
	float _t_13453;
	bool _t_13454;
	float _t_13457;
	float _t_13461;
	float _t_13462;
	float _t_13463;
	float _t_13464;
	float _t_13465;
	float _t_13466;
	float _t_13467;
	float _t_13468;
	float _t_13469;
	float _t_13470;
	bool _t_13471;
	float _t_13474;
	float _t_13478;
	float _t_13479;
	float _t_13480;

	float _t_13401;

	_t_13402 = -1.0f * ty3_9_1;
	_t_13403 = ty2_8_1 + _t_13402;
	_t_13404 = -1.0f * _t_13403;
	_t_13405 = _t_13404 < 0.0f;
	if(_t_13405)
		{
			float _t_13406;
			float _t_13407;
		
			_t_13406 = -1.0f * ty3_9_1;
			_t_13407 = ty2_8_1 + _t_13406;
			_t_13408 = _t_13407;
		
		}
else
		{
			float _t_13409;
			float _t_13410;
			float _t_13411;
		
			_t_13409 = -1.0f * ty3_9_1;
			_t_13410 = ty2_8_1 + _t_13409;
			_t_13411 = -1.0f * _t_13410;
			_t_13408 = _t_13411;
		
		}

	_t_13412 = _t_13408 * _t_511;
	_t_13413 = _t_13412 * _t_13400;
	_t_13414 = -1.0f * ty3_9_1;
	_t_13415 = ty2_8_1 + _t_13414;
	_t_13416 = -1.0f * _t_13415;
	_t_13417 = _t_13416 < 0.0f;
	if(_t_13417)
		{
			float _t_13418;
			float _t_13419;
		
			_t_13418 = -1.0f * tx2_5_1;
			_t_13419 = tx3_6_1 + _t_13418;
			_t_13420 = _t_13419;
		
		}
else
		{
			float _t_13421;
			float _t_13422;
			float _t_13423;
		
			_t_13421 = -1.0f * tx2_5_1;
			_t_13422 = tx3_6_1 + _t_13421;
			_t_13423 = -1.0f * _t_13422;
			_t_13420 = _t_13423;
		
		}

	_t_13424 = _t_13420 * _t_511;
	_t_13425 = _t_13424 * -1.0f;
	_t_13426 = _t_13425 * y__3609_1;
	_t_13427 = _t_13413 + _t_13426;
	_t_13428 = -1.0f * ty3_9_1;
	_t_13429 = ty2_8_1 + _t_13428;
	_t_13430 = -1.0f * _t_13429;
	_t_13431 = _t_13430 < 0.0f;
	if(_t_13431)
		{
			float _t_13432;
			float _t_13433;
		
			_t_13432 = -1.0f * tx2_5_1;
			_t_13433 = tx3_6_1 + _t_13432;
			_t_13434 = _t_13433;
		
		}
else
		{
			float _t_13435;
			float _t_13436;
			float _t_13437;
		
			_t_13435 = -1.0f * tx2_5_1;
			_t_13436 = tx3_6_1 + _t_13435;
			_t_13437 = -1.0f * _t_13436;
			_t_13434 = _t_13437;
		
		}

	_t_13438 = _t_13434 * _t_511;
	_t_13439 = -1.0f * ty3_9_1;
	_t_13440 = ty2_8_1 + _t_13439;
	_t_13441 = -1.0f * _t_13440;
	_t_13442 = _t_13441 < 0.0f;
	if(_t_13442)
		{
			float _t_13443;
			float _t_13444;
		
			_t_13443 = -1.0f * tx2_5_1;
			_t_13444 = tx3_6_1 + _t_13443;
			_t_13445 = _t_13444;
		
		}
else
		{
			float _t_13446;
			float _t_13447;
			float _t_13448;
		
			_t_13446 = -1.0f * tx2_5_1;
			_t_13447 = tx3_6_1 + _t_13446;
			_t_13448 = -1.0f * _t_13447;
			_t_13445 = _t_13448;
		
		}

	_t_13449 = _t_13445 * _t_511;
	_t_13450 = _t_13438 * _t_13449;
	_t_13451 = -1.0f * ty3_9_1;
	_t_13452 = ty2_8_1 + _t_13451;
	_t_13453 = -1.0f * _t_13452;
	_t_13454 = _t_13453 < 0.0f;
	if(_t_13454)
		{
			float _t_13455;
			float _t_13456;
		
			_t_13455 = -1.0f * ty3_9_1;
			_t_13456 = ty2_8_1 + _t_13455;
			_t_13457 = _t_13456;
		
		}
else
		{
			float _t_13458;
			float _t_13459;
			float _t_13460;
		
			_t_13458 = -1.0f * ty3_9_1;
			_t_13459 = ty2_8_1 + _t_13458;
			_t_13460 = -1.0f * _t_13459;
			_t_13457 = _t_13460;
		
		}

	_t_13461 = _t_13457 * _t_511;
	_t_13462 = 1.0f + _t_13461;
	_t_13463 = 1.0f / _t_13462;
	_t_13464 = _t_13450 * _t_13463;
	_t_13465 = _t_13464 * -1.0f;
	_t_13466 = 1.0f + _t_13465;
	_t_13467 = _t_13466 * y__3609_1;
	_t_13468 = -1.0f * ty3_9_1;
	_t_13469 = ty2_8_1 + _t_13468;
	_t_13470 = -1.0f * _t_13469;
	_t_13471 = _t_13470 < 0.0f;
	if(_t_13471)
		{
			float _t_13472;
			float _t_13473;
		
			_t_13472 = -1.0f * tx2_5_1;
			_t_13473 = tx3_6_1 + _t_13472;
			_t_13474 = _t_13473;
		
		}
else
		{
			float _t_13475;
			float _t_13476;
			float _t_13477;
		
			_t_13475 = -1.0f * tx2_5_1;
			_t_13476 = tx3_6_1 + _t_13475;
			_t_13477 = -1.0f * _t_13476;
			_t_13474 = _t_13477;
		
		}

	_t_13478 = _t_13474 * _t_511;
	_t_13479 = _t_13478 * _t_13400;
	_t_13480 = _t_13467 + _t_13479;
	_t_13401 = tegpixellet_block_48(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty1_7_1,tx1_4_1,ty3_9_1,_t_13427,_t_13480,ty2_8_1,tx2_5_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,_t_511,y__3609_1,_t_13400);

	return _t_13401;
}
__device__ float tegpixelbody_block_31(float ty2_8_1,float ty3_9_1,float _t_511,float px0_10_1,float px1_11_1,float tx3_6_1,float tx2_5_1,float py0_12_1,float py1_13_1,float y__3609_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_13244;
	float _t_13245;
	float _t_13246;
	bool _t_13247;
	float _t_13250;
	float _t_13254;
	float _t_13255;
	float _t_13256;
	float _t_13257;
	bool _t_13258;
	float _t_13261;
	float _t_13265;
	bool _t_13266;
	float _t_13267;
	float _t_13268;
	float _t_13269;
	float _t_13270;
	float _t_13271;
	bool _t_13272;
	float _t_13275;
	float _t_13279;
	float _t_13280;
	float _t_13281;
	float _t_13282;
	bool _t_13283;
	float _t_13286;
	float _t_13290;
	bool _t_13291;
	float _t_13292;
	float _t_13293;
	float _t_13294;
	float _t_13295;
	float _t_13296;
	float _t_13297;
	bool _t_13298;
	float _t_13303;
	float _t_13309;
	float _t_13310;
	float _t_13311;
	float _t_13312;
	bool _t_13313;
	float _t_13314;
	float _t_13315;
	float _t_13316;
	bool _t_13317;
	float _t_13320;
	float _t_13324;
	float _t_13325;
	float _t_13326;
	float _t_13327;
	bool _t_13328;
	float _t_13331;
	float _t_13335;
	bool _t_13336;
	float _t_13337;
	float _t_13338;
	float _t_13339;
	float _t_13340;
	float _t_13341;
	bool _t_13342;
	float _t_13345;
	float _t_13349;
	float _t_13350;
	float _t_13351;
	float _t_13352;
	bool _t_13353;
	float _t_13356;
	float _t_13360;
	bool _t_13361;
	float _t_13362;
	float _t_13363;
	float _t_13364;
	float _t_13365;
	float _t_13366;
	float _t_13367;
	bool _t_13368;
	float _t_13373;
	float _t_13379;
	float _t_13380;
	float _t_13381;
	float _t_13382;
	bool _t_13383;
	bool _t_13384;

	float _t_13243;

	_t_13244 = -1.0f * ty3_9_1;
	_t_13245 = ty2_8_1 + _t_13244;
	_t_13246 = -1.0f * _t_13245;
	_t_13247 = _t_13246 < 0.0f;
	if(_t_13247)
		{
			float _t_13248;
			float _t_13249;
		
			_t_13248 = -1.0f * ty3_9_1;
			_t_13249 = ty2_8_1 + _t_13248;
			_t_13250 = _t_13249;
		
		}
else
		{
			float _t_13251;
			float _t_13252;
			float _t_13253;
		
			_t_13251 = -1.0f * ty3_9_1;
			_t_13252 = ty2_8_1 + _t_13251;
			_t_13253 = -1.0f * _t_13252;
			_t_13250 = _t_13253;
		
		}

	_t_13254 = _t_13250 * _t_511;
	_t_13255 = -1.0f * ty3_9_1;
	_t_13256 = ty2_8_1 + _t_13255;
	_t_13257 = -1.0f * _t_13256;
	_t_13258 = _t_13257 < 0.0f;
	if(_t_13258)
		{
			float _t_13259;
			float _t_13260;
		
			_t_13259 = -1.0f * ty3_9_1;
			_t_13260 = ty2_8_1 + _t_13259;
			_t_13261 = _t_13260;
		
		}
else
		{
			float _t_13262;
			float _t_13263;
			float _t_13264;
		
			_t_13262 = -1.0f * ty3_9_1;
			_t_13263 = ty2_8_1 + _t_13262;
			_t_13264 = -1.0f * _t_13263;
			_t_13261 = _t_13264;
		
		}

	_t_13265 = _t_13261 * _t_511;
	_t_13266 = 0.0f < _t_13265;
	if(_t_13266)
		{
		
			_t_13267 = px0_10_1;
		
		}
else
		{
		
			_t_13267 = px1_11_1;
		
		}

	_t_13268 = _t_13254 * _t_13267;
	_t_13269 = -1.0f * ty3_9_1;
	_t_13270 = ty2_8_1 + _t_13269;
	_t_13271 = -1.0f * _t_13270;
	_t_13272 = _t_13271 < 0.0f;
	if(_t_13272)
		{
			float _t_13273;
			float _t_13274;
		
			_t_13273 = -1.0f * tx2_5_1;
			_t_13274 = tx3_6_1 + _t_13273;
			_t_13275 = _t_13274;
		
		}
else
		{
			float _t_13276;
			float _t_13277;
			float _t_13278;
		
			_t_13276 = -1.0f * tx2_5_1;
			_t_13277 = tx3_6_1 + _t_13276;
			_t_13278 = -1.0f * _t_13277;
			_t_13275 = _t_13278;
		
		}

	_t_13279 = _t_13275 * _t_511;
	_t_13280 = -1.0f * ty3_9_1;
	_t_13281 = ty2_8_1 + _t_13280;
	_t_13282 = -1.0f * _t_13281;
	_t_13283 = _t_13282 < 0.0f;
	if(_t_13283)
		{
			float _t_13284;
			float _t_13285;
		
			_t_13284 = -1.0f * tx2_5_1;
			_t_13285 = tx3_6_1 + _t_13284;
			_t_13286 = _t_13285;
		
		}
else
		{
			float _t_13287;
			float _t_13288;
			float _t_13289;
		
			_t_13287 = -1.0f * tx2_5_1;
			_t_13288 = tx3_6_1 + _t_13287;
			_t_13289 = -1.0f * _t_13288;
			_t_13286 = _t_13289;
		
		}

	_t_13290 = _t_13286 * _t_511;
	_t_13291 = 0.0f < _t_13290;
	if(_t_13291)
		{
		
			_t_13292 = py0_12_1;
		
		}
else
		{
		
			_t_13292 = py1_13_1;
		
		}

	_t_13293 = _t_13279 * _t_13292;
	_t_13294 = _t_13268 + _t_13293;
	_t_13295 = -1.0f * ty3_9_1;
	_t_13296 = ty2_8_1 + _t_13295;
	_t_13297 = -1.0f * _t_13296;
	_t_13298 = _t_13297 < 0.0f;
	if(_t_13298)
		{
			float _t_13299;
			float _t_13300;
			float _t_13301;
			float _t_13302;
		
			_t_13299 = tx2_5_1 * ty3_9_1;
			_t_13300 = tx3_6_1 * ty2_8_1;
			_t_13301 = _t_13300 * -1.0f;
			_t_13302 = _t_13299 + _t_13301;
			_t_13303 = _t_13302;
		
		}
else
		{
			float _t_13304;
			float _t_13305;
			float _t_13306;
			float _t_13307;
			float _t_13308;
		
			_t_13304 = tx2_5_1 * ty3_9_1;
			_t_13305 = tx3_6_1 * ty2_8_1;
			_t_13306 = _t_13305 * -1.0f;
			_t_13307 = _t_13304 + _t_13306;
			_t_13308 = -1.0f * _t_13307;
			_t_13303 = _t_13308;
		
		}

	_t_13309 = -1.0f * _t_13303;
	_t_13310 = _t_13309 * _t_511;
	_t_13311 = _t_13310 * -1.0f;
	_t_13312 = _t_13294 + _t_13311;
	_t_13313 = _t_13312 < 0.0f;
	_t_13314 = -1.0f * ty3_9_1;
	_t_13315 = ty2_8_1 + _t_13314;
	_t_13316 = -1.0f * _t_13315;
	_t_13317 = _t_13316 < 0.0f;
	if(_t_13317)
		{
			float _t_13318;
			float _t_13319;
		
			_t_13318 = -1.0f * ty3_9_1;
			_t_13319 = ty2_8_1 + _t_13318;
			_t_13320 = _t_13319;
		
		}
else
		{
			float _t_13321;
			float _t_13322;
			float _t_13323;
		
			_t_13321 = -1.0f * ty3_9_1;
			_t_13322 = ty2_8_1 + _t_13321;
			_t_13323 = -1.0f * _t_13322;
			_t_13320 = _t_13323;
		
		}

	_t_13324 = _t_13320 * _t_511;
	_t_13325 = -1.0f * ty3_9_1;
	_t_13326 = ty2_8_1 + _t_13325;
	_t_13327 = -1.0f * _t_13326;
	_t_13328 = _t_13327 < 0.0f;
	if(_t_13328)
		{
			float _t_13329;
			float _t_13330;
		
			_t_13329 = -1.0f * ty3_9_1;
			_t_13330 = ty2_8_1 + _t_13329;
			_t_13331 = _t_13330;
		
		}
else
		{
			float _t_13332;
			float _t_13333;
			float _t_13334;
		
			_t_13332 = -1.0f * ty3_9_1;
			_t_13333 = ty2_8_1 + _t_13332;
			_t_13334 = -1.0f * _t_13333;
			_t_13331 = _t_13334;
		
		}

	_t_13335 = _t_13331 * _t_511;
	_t_13336 = 0.0f < _t_13335;
	if(_t_13336)
		{
		
			_t_13337 = px1_11_1;
		
		}
else
		{
		
			_t_13337 = px0_10_1;
		
		}

	_t_13338 = _t_13324 * _t_13337;
	_t_13339 = -1.0f * ty3_9_1;
	_t_13340 = ty2_8_1 + _t_13339;
	_t_13341 = -1.0f * _t_13340;
	_t_13342 = _t_13341 < 0.0f;
	if(_t_13342)
		{
			float _t_13343;
			float _t_13344;
		
			_t_13343 = -1.0f * tx2_5_1;
			_t_13344 = tx3_6_1 + _t_13343;
			_t_13345 = _t_13344;
		
		}
else
		{
			float _t_13346;
			float _t_13347;
			float _t_13348;
		
			_t_13346 = -1.0f * tx2_5_1;
			_t_13347 = tx3_6_1 + _t_13346;
			_t_13348 = -1.0f * _t_13347;
			_t_13345 = _t_13348;
		
		}

	_t_13349 = _t_13345 * _t_511;
	_t_13350 = -1.0f * ty3_9_1;
	_t_13351 = ty2_8_1 + _t_13350;
	_t_13352 = -1.0f * _t_13351;
	_t_13353 = _t_13352 < 0.0f;
	if(_t_13353)
		{
			float _t_13354;
			float _t_13355;
		
			_t_13354 = -1.0f * tx2_5_1;
			_t_13355 = tx3_6_1 + _t_13354;
			_t_13356 = _t_13355;
		
		}
else
		{
			float _t_13357;
			float _t_13358;
			float _t_13359;
		
			_t_13357 = -1.0f * tx2_5_1;
			_t_13358 = tx3_6_1 + _t_13357;
			_t_13359 = -1.0f * _t_13358;
			_t_13356 = _t_13359;
		
		}

	_t_13360 = _t_13356 * _t_511;
	_t_13361 = 0.0f < _t_13360;
	if(_t_13361)
		{
		
			_t_13362 = py1_13_1;
		
		}
else
		{
		
			_t_13362 = py0_12_1;
		
		}

	_t_13363 = _t_13349 * _t_13362;
	_t_13364 = _t_13338 + _t_13363;
	_t_13365 = -1.0f * ty3_9_1;
	_t_13366 = ty2_8_1 + _t_13365;
	_t_13367 = -1.0f * _t_13366;
	_t_13368 = _t_13367 < 0.0f;
	if(_t_13368)
		{
			float _t_13369;
			float _t_13370;
			float _t_13371;
			float _t_13372;
		
			_t_13369 = tx2_5_1 * ty3_9_1;
			_t_13370 = tx3_6_1 * ty2_8_1;
			_t_13371 = _t_13370 * -1.0f;
			_t_13372 = _t_13369 + _t_13371;
			_t_13373 = _t_13372;
		
		}
else
		{
			float _t_13374;
			float _t_13375;
			float _t_13376;
			float _t_13377;
			float _t_13378;
		
			_t_13374 = tx2_5_1 * ty3_9_1;
			_t_13375 = tx3_6_1 * ty2_8_1;
			_t_13376 = _t_13375 * -1.0f;
			_t_13377 = _t_13374 + _t_13376;
			_t_13378 = -1.0f * _t_13377;
			_t_13373 = _t_13378;
		
		}

	_t_13379 = -1.0f * _t_13373;
	_t_13380 = _t_13379 * _t_511;
	_t_13381 = _t_13380 * -1.0f;
	_t_13382 = _t_13364 + _t_13381;
	_t_13383 = 0.0f < _t_13382;
	_t_13384 = _t_13313 && _t_13383;
	if(_t_13384)
		{
			float _t_13385;
			float _t_13386;
			float _t_13387;
			bool _t_13388;
			float _t_13393;
			float _t_13399;
			float _t_13400;
			float _t_13401;
		
			_t_13385 = -1.0f * ty3_9_1;
			_t_13386 = ty2_8_1 + _t_13385;
			_t_13387 = -1.0f * _t_13386;
			_t_13388 = _t_13387 < 0.0f;
			if(_t_13388)
				{
					float _t_13389;
					float _t_13390;
					float _t_13391;
					float _t_13392;
				
					_t_13389 = tx2_5_1 * ty3_9_1;
					_t_13390 = tx3_6_1 * ty2_8_1;
					_t_13391 = _t_13390 * -1.0f;
					_t_13392 = _t_13389 + _t_13391;
					_t_13393 = _t_13392;
				
				}
		else
				{
					float _t_13394;
					float _t_13395;
					float _t_13396;
					float _t_13397;
					float _t_13398;
				
					_t_13394 = tx2_5_1 * ty3_9_1;
					_t_13395 = tx3_6_1 * ty2_8_1;
					_t_13396 = _t_13395 * -1.0f;
					_t_13397 = _t_13394 + _t_13396;
					_t_13398 = -1.0f * _t_13397;
					_t_13393 = _t_13398;
				
				}
		
			_t_13399 = -1.0f * _t_13393;
			_t_13400 = _t_13399 * _t_511;
			_t_13401 = tegpixellet_block_47(ty2_8_1,ty3_9_1,_t_511,_t_13400,tx3_6_1,tx2_5_1,y__3609_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1);
			_t_13243 = _t_13401;
		
		}
else
		{
		
			_t_13243 = 0.0f;
		
		}


	return _t_13243;
}
__device__ float tegpixelintegrator_31(float _t_13133,float ty3_9_1,float pc1_15_1,float tc2_19_1,float ty2_8_1,float ty1_7_1,float pc0_14_1,float _t_13242,float tx3_6_1,float tx1_4_1,float tx2_5_1,float py1_13_1,float pc2_16_1,float px1_11_1,float tc0_17_1,float py0_12_1,float tc1_18_1,float px0_10_1,float _t_511){
    float y__3609_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_13242 - _t_13133)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__3609_1 = _t_13133 + __step__ * (i + (float)(0.5));
        float _t_13243;
		_t_13243 = tegpixelbody_block_31(ty2_8_1,ty3_9_1,_t_511,px0_10_1,px1_11_1,tx3_6_1,tx2_5_1,py0_12_1,py1_13_1,y__3609_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);;
        __output__ = __output__ + _t_13243 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_15(float ty2_8_1,float ty3_9_1,float tx3_6_1,float tx2_5_1,float _t_511,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_13025;
	float _t_13026;
	float _t_13027;
	bool _t_13028;
	float _t_13031;
	float _t_13035;
	float _t_13036;
	float _t_13037;
	float _t_13038;
	float _t_13039;
	bool _t_13040;
	float _t_13043;
	float _t_13047;
	float _t_13048;
	bool _t_13049;
	float _t_13050;
	float _t_13051;
	float _t_13052;
	float _t_13053;
	float _t_13054;
	bool _t_13055;
	float _t_13058;
	float _t_13062;
	float _t_13063;
	float _t_13064;
	float _t_13065;
	bool _t_13066;
	float _t_13069;
	float _t_13073;
	float _t_13074;
	float _t_13075;
	float _t_13076;
	float _t_13077;
	bool _t_13078;
	float _t_13081;
	float _t_13085;
	float _t_13086;
	float _t_13087;
	float _t_13088;
	float _t_13089;
	float _t_13090;
	float _t_13091;
	float _t_13092;
	float _t_13093;
	bool _t_13094;
	float _t_13097;
	float _t_13101;
	float _t_13102;
	float _t_13103;
	float _t_13104;
	bool _t_13105;
	float _t_13108;
	float _t_13112;
	float _t_13113;
	float _t_13114;
	float _t_13115;
	float _t_13116;
	bool _t_13117;
	float _t_13120;
	float _t_13124;
	float _t_13125;
	float _t_13126;
	float _t_13127;
	float _t_13128;
	float _t_13129;
	bool _t_13130;
	float _t_13131;
	float _t_13132;
	float _t_13133;
	float _t_13134;
	float _t_13135;
	float _t_13136;
	bool _t_13137;
	float _t_13140;
	float _t_13144;
	float _t_13145;
	float _t_13146;
	float _t_13147;
	float _t_13148;
	bool _t_13149;
	float _t_13152;
	float _t_13156;
	float _t_13157;
	bool _t_13158;
	float _t_13159;
	float _t_13160;
	float _t_13161;
	float _t_13162;
	float _t_13163;
	bool _t_13164;
	float _t_13167;
	float _t_13171;
	float _t_13172;
	float _t_13173;
	float _t_13174;
	bool _t_13175;
	float _t_13178;
	float _t_13182;
	float _t_13183;
	float _t_13184;
	float _t_13185;
	float _t_13186;
	bool _t_13187;
	float _t_13190;
	float _t_13194;
	float _t_13195;
	float _t_13196;
	float _t_13197;
	float _t_13198;
	float _t_13199;
	float _t_13200;
	float _t_13201;
	float _t_13202;
	bool _t_13203;
	float _t_13206;
	float _t_13210;
	float _t_13211;
	float _t_13212;
	float _t_13213;
	bool _t_13214;
	float _t_13217;
	float _t_13221;
	float _t_13222;
	float _t_13223;
	float _t_13224;
	float _t_13225;
	bool _t_13226;
	float _t_13229;
	float _t_13233;
	float _t_13234;
	float _t_13235;
	float _t_13236;
	float _t_13237;
	float _t_13238;
	bool _t_13239;
	float _t_13240;
	float _t_13241;
	float _t_13242;

	float _t_512;

	_t_13025 = -1.0f * ty3_9_1;
	_t_13026 = ty2_8_1 + _t_13025;
	_t_13027 = -1.0f * _t_13026;
	_t_13028 = _t_13027 < 0.0f;
	if(_t_13028)
		{
			float _t_13029;
			float _t_13030;
		
			_t_13029 = -1.0f * tx2_5_1;
			_t_13030 = tx3_6_1 + _t_13029;
			_t_13031 = _t_13030;
		
		}
else
		{
			float _t_13032;
			float _t_13033;
			float _t_13034;
		
			_t_13032 = -1.0f * tx2_5_1;
			_t_13033 = tx3_6_1 + _t_13032;
			_t_13034 = -1.0f * _t_13033;
			_t_13031 = _t_13034;
		
		}

	_t_13035 = _t_13031 * _t_511;
	_t_13036 = _t_13035 * -1.0f;
	_t_13037 = -1.0f * ty3_9_1;
	_t_13038 = ty2_8_1 + _t_13037;
	_t_13039 = -1.0f * _t_13038;
	_t_13040 = _t_13039 < 0.0f;
	if(_t_13040)
		{
			float _t_13041;
			float _t_13042;
		
			_t_13041 = -1.0f * tx2_5_1;
			_t_13042 = tx3_6_1 + _t_13041;
			_t_13043 = _t_13042;
		
		}
else
		{
			float _t_13044;
			float _t_13045;
			float _t_13046;
		
			_t_13044 = -1.0f * tx2_5_1;
			_t_13045 = tx3_6_1 + _t_13044;
			_t_13046 = -1.0f * _t_13045;
			_t_13043 = _t_13046;
		
		}

	_t_13047 = _t_13043 * _t_511;
	_t_13048 = _t_13047 * -1.0f;
	_t_13049 = 0.0f < _t_13048;
	if(_t_13049)
		{
		
			_t_13050 = px0_10_1;
		
		}
else
		{
		
			_t_13050 = px1_11_1;
		
		}

	_t_13051 = _t_13036 * _t_13050;
	_t_13052 = -1.0f * ty3_9_1;
	_t_13053 = ty2_8_1 + _t_13052;
	_t_13054 = -1.0f * _t_13053;
	_t_13055 = _t_13054 < 0.0f;
	if(_t_13055)
		{
			float _t_13056;
			float _t_13057;
		
			_t_13056 = -1.0f * tx2_5_1;
			_t_13057 = tx3_6_1 + _t_13056;
			_t_13058 = _t_13057;
		
		}
else
		{
			float _t_13059;
			float _t_13060;
			float _t_13061;
		
			_t_13059 = -1.0f * tx2_5_1;
			_t_13060 = tx3_6_1 + _t_13059;
			_t_13061 = -1.0f * _t_13060;
			_t_13058 = _t_13061;
		
		}

	_t_13062 = _t_13058 * _t_511;
	_t_13063 = -1.0f * ty3_9_1;
	_t_13064 = ty2_8_1 + _t_13063;
	_t_13065 = -1.0f * _t_13064;
	_t_13066 = _t_13065 < 0.0f;
	if(_t_13066)
		{
			float _t_13067;
			float _t_13068;
		
			_t_13067 = -1.0f * tx2_5_1;
			_t_13068 = tx3_6_1 + _t_13067;
			_t_13069 = _t_13068;
		
		}
else
		{
			float _t_13070;
			float _t_13071;
			float _t_13072;
		
			_t_13070 = -1.0f * tx2_5_1;
			_t_13071 = tx3_6_1 + _t_13070;
			_t_13072 = -1.0f * _t_13071;
			_t_13069 = _t_13072;
		
		}

	_t_13073 = _t_13069 * _t_511;
	_t_13074 = _t_13062 * _t_13073;
	_t_13075 = -1.0f * ty3_9_1;
	_t_13076 = ty2_8_1 + _t_13075;
	_t_13077 = -1.0f * _t_13076;
	_t_13078 = _t_13077 < 0.0f;
	if(_t_13078)
		{
			float _t_13079;
			float _t_13080;
		
			_t_13079 = -1.0f * ty3_9_1;
			_t_13080 = ty2_8_1 + _t_13079;
			_t_13081 = _t_13080;
		
		}
else
		{
			float _t_13082;
			float _t_13083;
			float _t_13084;
		
			_t_13082 = -1.0f * ty3_9_1;
			_t_13083 = ty2_8_1 + _t_13082;
			_t_13084 = -1.0f * _t_13083;
			_t_13081 = _t_13084;
		
		}

	_t_13085 = _t_13081 * _t_511;
	_t_13086 = 1.0f + _t_13085;
	_t_13087 = 1.0f / _t_13086;
	_t_13088 = _t_13074 * _t_13087;
	_t_13089 = _t_13088 * -1.0f;
	_t_13090 = 1.0f + _t_13089;
	_t_13091 = -1.0f * ty3_9_1;
	_t_13092 = ty2_8_1 + _t_13091;
	_t_13093 = -1.0f * _t_13092;
	_t_13094 = _t_13093 < 0.0f;
	if(_t_13094)
		{
			float _t_13095;
			float _t_13096;
		
			_t_13095 = -1.0f * tx2_5_1;
			_t_13096 = tx3_6_1 + _t_13095;
			_t_13097 = _t_13096;
		
		}
else
		{
			float _t_13098;
			float _t_13099;
			float _t_13100;
		
			_t_13098 = -1.0f * tx2_5_1;
			_t_13099 = tx3_6_1 + _t_13098;
			_t_13100 = -1.0f * _t_13099;
			_t_13097 = _t_13100;
		
		}

	_t_13101 = _t_13097 * _t_511;
	_t_13102 = -1.0f * ty3_9_1;
	_t_13103 = ty2_8_1 + _t_13102;
	_t_13104 = -1.0f * _t_13103;
	_t_13105 = _t_13104 < 0.0f;
	if(_t_13105)
		{
			float _t_13106;
			float _t_13107;
		
			_t_13106 = -1.0f * tx2_5_1;
			_t_13107 = tx3_6_1 + _t_13106;
			_t_13108 = _t_13107;
		
		}
else
		{
			float _t_13109;
			float _t_13110;
			float _t_13111;
		
			_t_13109 = -1.0f * tx2_5_1;
			_t_13110 = tx3_6_1 + _t_13109;
			_t_13111 = -1.0f * _t_13110;
			_t_13108 = _t_13111;
		
		}

	_t_13112 = _t_13108 * _t_511;
	_t_13113 = _t_13101 * _t_13112;
	_t_13114 = -1.0f * ty3_9_1;
	_t_13115 = ty2_8_1 + _t_13114;
	_t_13116 = -1.0f * _t_13115;
	_t_13117 = _t_13116 < 0.0f;
	if(_t_13117)
		{
			float _t_13118;
			float _t_13119;
		
			_t_13118 = -1.0f * ty3_9_1;
			_t_13119 = ty2_8_1 + _t_13118;
			_t_13120 = _t_13119;
		
		}
else
		{
			float _t_13121;
			float _t_13122;
			float _t_13123;
		
			_t_13121 = -1.0f * ty3_9_1;
			_t_13122 = ty2_8_1 + _t_13121;
			_t_13123 = -1.0f * _t_13122;
			_t_13120 = _t_13123;
		
		}

	_t_13124 = _t_13120 * _t_511;
	_t_13125 = 1.0f + _t_13124;
	_t_13126 = 1.0f / _t_13125;
	_t_13127 = _t_13113 * _t_13126;
	_t_13128 = _t_13127 * -1.0f;
	_t_13129 = 1.0f + _t_13128;
	_t_13130 = 0.0f < _t_13129;
	if(_t_13130)
		{
		
			_t_13131 = py0_12_1;
		
		}
else
		{
		
			_t_13131 = py1_13_1;
		
		}

	_t_13132 = _t_13090 * _t_13131;
	_t_13133 = _t_13051 + _t_13132;
	_t_13134 = -1.0f * ty3_9_1;
	_t_13135 = ty2_8_1 + _t_13134;
	_t_13136 = -1.0f * _t_13135;
	_t_13137 = _t_13136 < 0.0f;
	if(_t_13137)
		{
			float _t_13138;
			float _t_13139;
		
			_t_13138 = -1.0f * tx2_5_1;
			_t_13139 = tx3_6_1 + _t_13138;
			_t_13140 = _t_13139;
		
		}
else
		{
			float _t_13141;
			float _t_13142;
			float _t_13143;
		
			_t_13141 = -1.0f * tx2_5_1;
			_t_13142 = tx3_6_1 + _t_13141;
			_t_13143 = -1.0f * _t_13142;
			_t_13140 = _t_13143;
		
		}

	_t_13144 = _t_13140 * _t_511;
	_t_13145 = _t_13144 * -1.0f;
	_t_13146 = -1.0f * ty3_9_1;
	_t_13147 = ty2_8_1 + _t_13146;
	_t_13148 = -1.0f * _t_13147;
	_t_13149 = _t_13148 < 0.0f;
	if(_t_13149)
		{
			float _t_13150;
			float _t_13151;
		
			_t_13150 = -1.0f * tx2_5_1;
			_t_13151 = tx3_6_1 + _t_13150;
			_t_13152 = _t_13151;
		
		}
else
		{
			float _t_13153;
			float _t_13154;
			float _t_13155;
		
			_t_13153 = -1.0f * tx2_5_1;
			_t_13154 = tx3_6_1 + _t_13153;
			_t_13155 = -1.0f * _t_13154;
			_t_13152 = _t_13155;
		
		}

	_t_13156 = _t_13152 * _t_511;
	_t_13157 = _t_13156 * -1.0f;
	_t_13158 = 0.0f < _t_13157;
	if(_t_13158)
		{
		
			_t_13159 = px1_11_1;
		
		}
else
		{
		
			_t_13159 = px0_10_1;
		
		}

	_t_13160 = _t_13145 * _t_13159;
	_t_13161 = -1.0f * ty3_9_1;
	_t_13162 = ty2_8_1 + _t_13161;
	_t_13163 = -1.0f * _t_13162;
	_t_13164 = _t_13163 < 0.0f;
	if(_t_13164)
		{
			float _t_13165;
			float _t_13166;
		
			_t_13165 = -1.0f * tx2_5_1;
			_t_13166 = tx3_6_1 + _t_13165;
			_t_13167 = _t_13166;
		
		}
else
		{
			float _t_13168;
			float _t_13169;
			float _t_13170;
		
			_t_13168 = -1.0f * tx2_5_1;
			_t_13169 = tx3_6_1 + _t_13168;
			_t_13170 = -1.0f * _t_13169;
			_t_13167 = _t_13170;
		
		}

	_t_13171 = _t_13167 * _t_511;
	_t_13172 = -1.0f * ty3_9_1;
	_t_13173 = ty2_8_1 + _t_13172;
	_t_13174 = -1.0f * _t_13173;
	_t_13175 = _t_13174 < 0.0f;
	if(_t_13175)
		{
			float _t_13176;
			float _t_13177;
		
			_t_13176 = -1.0f * tx2_5_1;
			_t_13177 = tx3_6_1 + _t_13176;
			_t_13178 = _t_13177;
		
		}
else
		{
			float _t_13179;
			float _t_13180;
			float _t_13181;
		
			_t_13179 = -1.0f * tx2_5_1;
			_t_13180 = tx3_6_1 + _t_13179;
			_t_13181 = -1.0f * _t_13180;
			_t_13178 = _t_13181;
		
		}

	_t_13182 = _t_13178 * _t_511;
	_t_13183 = _t_13171 * _t_13182;
	_t_13184 = -1.0f * ty3_9_1;
	_t_13185 = ty2_8_1 + _t_13184;
	_t_13186 = -1.0f * _t_13185;
	_t_13187 = _t_13186 < 0.0f;
	if(_t_13187)
		{
			float _t_13188;
			float _t_13189;
		
			_t_13188 = -1.0f * ty3_9_1;
			_t_13189 = ty2_8_1 + _t_13188;
			_t_13190 = _t_13189;
		
		}
else
		{
			float _t_13191;
			float _t_13192;
			float _t_13193;
		
			_t_13191 = -1.0f * ty3_9_1;
			_t_13192 = ty2_8_1 + _t_13191;
			_t_13193 = -1.0f * _t_13192;
			_t_13190 = _t_13193;
		
		}

	_t_13194 = _t_13190 * _t_511;
	_t_13195 = 1.0f + _t_13194;
	_t_13196 = 1.0f / _t_13195;
	_t_13197 = _t_13183 * _t_13196;
	_t_13198 = _t_13197 * -1.0f;
	_t_13199 = 1.0f + _t_13198;
	_t_13200 = -1.0f * ty3_9_1;
	_t_13201 = ty2_8_1 + _t_13200;
	_t_13202 = -1.0f * _t_13201;
	_t_13203 = _t_13202 < 0.0f;
	if(_t_13203)
		{
			float _t_13204;
			float _t_13205;
		
			_t_13204 = -1.0f * tx2_5_1;
			_t_13205 = tx3_6_1 + _t_13204;
			_t_13206 = _t_13205;
		
		}
else
		{
			float _t_13207;
			float _t_13208;
			float _t_13209;
		
			_t_13207 = -1.0f * tx2_5_1;
			_t_13208 = tx3_6_1 + _t_13207;
			_t_13209 = -1.0f * _t_13208;
			_t_13206 = _t_13209;
		
		}

	_t_13210 = _t_13206 * _t_511;
	_t_13211 = -1.0f * ty3_9_1;
	_t_13212 = ty2_8_1 + _t_13211;
	_t_13213 = -1.0f * _t_13212;
	_t_13214 = _t_13213 < 0.0f;
	if(_t_13214)
		{
			float _t_13215;
			float _t_13216;
		
			_t_13215 = -1.0f * tx2_5_1;
			_t_13216 = tx3_6_1 + _t_13215;
			_t_13217 = _t_13216;
		
		}
else
		{
			float _t_13218;
			float _t_13219;
			float _t_13220;
		
			_t_13218 = -1.0f * tx2_5_1;
			_t_13219 = tx3_6_1 + _t_13218;
			_t_13220 = -1.0f * _t_13219;
			_t_13217 = _t_13220;
		
		}

	_t_13221 = _t_13217 * _t_511;
	_t_13222 = _t_13210 * _t_13221;
	_t_13223 = -1.0f * ty3_9_1;
	_t_13224 = ty2_8_1 + _t_13223;
	_t_13225 = -1.0f * _t_13224;
	_t_13226 = _t_13225 < 0.0f;
	if(_t_13226)
		{
			float _t_13227;
			float _t_13228;
		
			_t_13227 = -1.0f * ty3_9_1;
			_t_13228 = ty2_8_1 + _t_13227;
			_t_13229 = _t_13228;
		
		}
else
		{
			float _t_13230;
			float _t_13231;
			float _t_13232;
		
			_t_13230 = -1.0f * ty3_9_1;
			_t_13231 = ty2_8_1 + _t_13230;
			_t_13232 = -1.0f * _t_13231;
			_t_13229 = _t_13232;
		
		}

	_t_13233 = _t_13229 * _t_511;
	_t_13234 = 1.0f + _t_13233;
	_t_13235 = 1.0f / _t_13234;
	_t_13236 = _t_13222 * _t_13235;
	_t_13237 = _t_13236 * -1.0f;
	_t_13238 = 1.0f + _t_13237;
	_t_13239 = 0.0f < _t_13238;
	if(_t_13239)
		{
		
			_t_13240 = py1_13_1;
		
		}
else
		{
		
			_t_13240 = py0_12_1;
		
		}

	_t_13241 = _t_13199 * _t_13240;
	_t_13242 = _t_13160 + _t_13241;
	_t_512 = tegpixelintegrator_31(_t_13133,ty3_9_1,pc1_15_1,tc2_19_1,ty2_8_1,ty1_7_1,pc0_14_1,_t_13242,tx3_6_1,tx1_4_1,tx2_5_1,py1_13_1,pc2_16_1,px1_11_1,tc0_17_1,py0_12_1,tc1_18_1,px0_10_1,_t_511);

	return _t_512;
}
__device__ float tegpixellet_block_50(float py0_12_1,float _t_14316,float py1_13_1,float px0_10_1,float _t_14263,float px1_11_1,float ty2_8_1,float ty3_9_1,float tx3_6_1,float tx2_5_1,float _t_539,float y__3683_1,float _t_14236,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	bool _t_14317;
	bool _t_14318;
	bool _t_14319;
	bool _t_14320;
	bool _t_14321;
	bool _t_14322;
	bool _t_14323;
	float _t_14653;
	float _t_14654;
	float _t_14655;
	float _t_14656;
	float _t_14657;
	float _t_14658;
	float _t_14659;
	float _t_14660;
	float _t_14661;
	float _t_14662;
	float _t_14663;
	float _t_14664;
	float _t_14665;
	float _t_14666;
	float _t_14667;
	float _t_14668;
	float _t_14669;
	float _t_14670;
	float _t_14671;
	float _t_14672;
	float _t_14673;
	float _t_14674;
	float _t_14675;
	float _t_14676;
	bool _t_14677;
	float _t_14678;
	float _t_14679;
	float _t_14680;
	float _t_14681;
	float _t_14682;
	float _t_14683;
	float _t_14684;
	float _t_14685;
	float _t_14686;
	float _t_14687;
	float _t_14688;
	float _t_14689;
	float _t_14690;
	float _t_14691;
	bool _t_14692;
	float _t_14693;
	float _t_14694;
	float _t_14695;
	float _t_14696;
	float _t_14697;
	float _t_14698;
	float _t_14699;
	float _t_14700;
	float _t_14701;
	float _t_14702;
	float _t_14703;
	float _t_14704;
	float _t_14705;
	float _t_14706;
	float _t_14707;
	float _t_14708;
	float _t_14709;
	float _t_14710;
	float _t_14711;
	float _t_14712;
	float _t_14713;
	float _t_14714;
	float _t_14715;
	float _t_14716;
	float _t_14717;
	float _t_14718;
	float _t_14719;
	bool _t_14720;
	float _t_14721;
	float _t_14722;
	float _t_14723;
	float _t_14724;
	float _t_14725;
	float _t_14726;
	float _t_14727;
	float _t_14728;
	float _t_14729;
	float _t_14730;
	float _t_14731;
	float _t_14732;
	float _t_14733;
	float _t_14734;
	bool _t_14735;
	float _t_14736;
	float _t_14737;
	float _t_14738;
	float _t_14739;

	float _t_14237;

	_t_14317 = py0_12_1 < _t_14316;
	_t_14318 = _t_14316 < py1_13_1;
	_t_14319 = _t_14317 && _t_14318;
	_t_14320 = px0_10_1 < _t_14263;
	_t_14321 = _t_14263 < px1_11_1;
	_t_14322 = _t_14320 && _t_14321;
	_t_14323 = _t_14319 && _t_14322;
	if(_t_14323)
		{
			float _t_14324;
			float _t_14325;
			float _t_14326;
			bool _t_14327;
			float _t_14330;
			float _t_14334;
			float _t_14335;
			float _t_14336;
			float _t_14337;
			float _t_14338;
			bool _t_14339;
			float _t_14342;
			float _t_14346;
			float _t_14347;
			bool _t_14348;
			float _t_14349;
			float _t_14350;
			float _t_14351;
			float _t_14352;
			float _t_14353;
			bool _t_14354;
			float _t_14357;
			float _t_14361;
			float _t_14362;
			float _t_14363;
			float _t_14364;
			bool _t_14365;
			float _t_14368;
			float _t_14372;
			float _t_14373;
			float _t_14374;
			float _t_14375;
			float _t_14376;
			bool _t_14377;
			float _t_14380;
			float _t_14384;
			float _t_14385;
			float _t_14386;
			float _t_14387;
			float _t_14388;
			float _t_14389;
			float _t_14390;
			float _t_14391;
			float _t_14392;
			bool _t_14393;
			float _t_14396;
			float _t_14400;
			float _t_14401;
			float _t_14402;
			float _t_14403;
			bool _t_14404;
			float _t_14407;
			float _t_14411;
			float _t_14412;
			float _t_14413;
			float _t_14414;
			float _t_14415;
			bool _t_14416;
			float _t_14419;
			float _t_14423;
			float _t_14424;
			float _t_14425;
			float _t_14426;
			float _t_14427;
			float _t_14428;
			bool _t_14429;
			float _t_14430;
			float _t_14431;
			float _t_14432;
			bool _t_14433;
			float _t_14434;
			float _t_14435;
			float _t_14436;
			bool _t_14437;
			float _t_14440;
			float _t_14444;
			float _t_14445;
			float _t_14446;
			float _t_14447;
			float _t_14448;
			bool _t_14449;
			float _t_14452;
			float _t_14456;
			float _t_14457;
			bool _t_14458;
			float _t_14459;
			float _t_14460;
			float _t_14461;
			float _t_14462;
			float _t_14463;
			bool _t_14464;
			float _t_14467;
			float _t_14471;
			float _t_14472;
			float _t_14473;
			float _t_14474;
			bool _t_14475;
			float _t_14478;
			float _t_14482;
			float _t_14483;
			float _t_14484;
			float _t_14485;
			float _t_14486;
			bool _t_14487;
			float _t_14490;
			float _t_14494;
			float _t_14495;
			float _t_14496;
			float _t_14497;
			float _t_14498;
			float _t_14499;
			float _t_14500;
			float _t_14501;
			float _t_14502;
			bool _t_14503;
			float _t_14506;
			float _t_14510;
			float _t_14511;
			float _t_14512;
			float _t_14513;
			bool _t_14514;
			float _t_14517;
			float _t_14521;
			float _t_14522;
			float _t_14523;
			float _t_14524;
			float _t_14525;
			bool _t_14526;
			float _t_14529;
			float _t_14533;
			float _t_14534;
			float _t_14535;
			float _t_14536;
			float _t_14537;
			float _t_14538;
			bool _t_14539;
			float _t_14540;
			float _t_14541;
			float _t_14542;
			bool _t_14543;
			bool _t_14544;
			float _t_14545;
			float _t_14546;
			float _t_14547;
			bool _t_14548;
			float _t_14551;
			float _t_14555;
			float _t_14556;
			float _t_14557;
			float _t_14558;
			bool _t_14559;
			float _t_14562;
			float _t_14566;
			bool _t_14567;
			float _t_14568;
			float _t_14569;
			float _t_14570;
			float _t_14571;
			float _t_14572;
			bool _t_14573;
			float _t_14576;
			float _t_14580;
			float _t_14581;
			float _t_14582;
			float _t_14583;
			bool _t_14584;
			float _t_14587;
			float _t_14591;
			bool _t_14592;
			float _t_14593;
			float _t_14594;
			float _t_14595;
			bool _t_14596;
			float _t_14597;
			float _t_14598;
			float _t_14599;
			bool _t_14600;
			float _t_14603;
			float _t_14607;
			float _t_14608;
			float _t_14609;
			float _t_14610;
			bool _t_14611;
			float _t_14614;
			float _t_14618;
			bool _t_14619;
			float _t_14620;
			float _t_14621;
			float _t_14622;
			float _t_14623;
			float _t_14624;
			bool _t_14625;
			float _t_14628;
			float _t_14632;
			float _t_14633;
			float _t_14634;
			float _t_14635;
			bool _t_14636;
			float _t_14639;
			float _t_14643;
			bool _t_14644;
			float _t_14645;
			float _t_14646;
			float _t_14647;
			bool _t_14648;
			bool _t_14649;
			bool _t_14650;
			float _t_14651;
			float _t_14652;
		
			_t_14324 = -1.0f * ty3_9_1;
			_t_14325 = ty2_8_1 + _t_14324;
			_t_14326 = -1.0f * _t_14325;
			_t_14327 = _t_14326 < 0.0f;
			if(_t_14327)
				{
					float _t_14328;
					float _t_14329;
				
					_t_14328 = -1.0f * tx2_5_1;
					_t_14329 = tx3_6_1 + _t_14328;
					_t_14330 = _t_14329;
				
				}
		else
				{
					float _t_14331;
					float _t_14332;
					float _t_14333;
				
					_t_14331 = -1.0f * tx2_5_1;
					_t_14332 = tx3_6_1 + _t_14331;
					_t_14333 = -1.0f * _t_14332;
					_t_14330 = _t_14333;
				
				}
		
			_t_14334 = _t_14330 * _t_539;
			_t_14335 = _t_14334 * -1.0f;
			_t_14336 = -1.0f * ty3_9_1;
			_t_14337 = ty2_8_1 + _t_14336;
			_t_14338 = -1.0f * _t_14337;
			_t_14339 = _t_14338 < 0.0f;
			if(_t_14339)
				{
					float _t_14340;
					float _t_14341;
				
					_t_14340 = -1.0f * tx2_5_1;
					_t_14341 = tx3_6_1 + _t_14340;
					_t_14342 = _t_14341;
				
				}
		else
				{
					float _t_14343;
					float _t_14344;
					float _t_14345;
				
					_t_14343 = -1.0f * tx2_5_1;
					_t_14344 = tx3_6_1 + _t_14343;
					_t_14345 = -1.0f * _t_14344;
					_t_14342 = _t_14345;
				
				}
		
			_t_14346 = _t_14342 * _t_539;
			_t_14347 = _t_14346 * -1.0f;
			_t_14348 = 0.0f < _t_14347;
			if(_t_14348)
				{
				
					_t_14349 = px0_10_1;
				
				}
		else
				{
				
					_t_14349 = px1_11_1;
				
				}
		
			_t_14350 = _t_14335 * _t_14349;
			_t_14351 = -1.0f * ty3_9_1;
			_t_14352 = ty2_8_1 + _t_14351;
			_t_14353 = -1.0f * _t_14352;
			_t_14354 = _t_14353 < 0.0f;
			if(_t_14354)
				{
					float _t_14355;
					float _t_14356;
				
					_t_14355 = -1.0f * tx2_5_1;
					_t_14356 = tx3_6_1 + _t_14355;
					_t_14357 = _t_14356;
				
				}
		else
				{
					float _t_14358;
					float _t_14359;
					float _t_14360;
				
					_t_14358 = -1.0f * tx2_5_1;
					_t_14359 = tx3_6_1 + _t_14358;
					_t_14360 = -1.0f * _t_14359;
					_t_14357 = _t_14360;
				
				}
		
			_t_14361 = _t_14357 * _t_539;
			_t_14362 = -1.0f * ty3_9_1;
			_t_14363 = ty2_8_1 + _t_14362;
			_t_14364 = -1.0f * _t_14363;
			_t_14365 = _t_14364 < 0.0f;
			if(_t_14365)
				{
					float _t_14366;
					float _t_14367;
				
					_t_14366 = -1.0f * tx2_5_1;
					_t_14367 = tx3_6_1 + _t_14366;
					_t_14368 = _t_14367;
				
				}
		else
				{
					float _t_14369;
					float _t_14370;
					float _t_14371;
				
					_t_14369 = -1.0f * tx2_5_1;
					_t_14370 = tx3_6_1 + _t_14369;
					_t_14371 = -1.0f * _t_14370;
					_t_14368 = _t_14371;
				
				}
		
			_t_14372 = _t_14368 * _t_539;
			_t_14373 = _t_14361 * _t_14372;
			_t_14374 = -1.0f * ty3_9_1;
			_t_14375 = ty2_8_1 + _t_14374;
			_t_14376 = -1.0f * _t_14375;
			_t_14377 = _t_14376 < 0.0f;
			if(_t_14377)
				{
					float _t_14378;
					float _t_14379;
				
					_t_14378 = -1.0f * ty3_9_1;
					_t_14379 = ty2_8_1 + _t_14378;
					_t_14380 = _t_14379;
				
				}
		else
				{
					float _t_14381;
					float _t_14382;
					float _t_14383;
				
					_t_14381 = -1.0f * ty3_9_1;
					_t_14382 = ty2_8_1 + _t_14381;
					_t_14383 = -1.0f * _t_14382;
					_t_14380 = _t_14383;
				
				}
		
			_t_14384 = _t_14380 * _t_539;
			_t_14385 = 1.0f + _t_14384;
			_t_14386 = 1.0f / _t_14385;
			_t_14387 = _t_14373 * _t_14386;
			_t_14388 = _t_14387 * -1.0f;
			_t_14389 = 1.0f + _t_14388;
			_t_14390 = -1.0f * ty3_9_1;
			_t_14391 = ty2_8_1 + _t_14390;
			_t_14392 = -1.0f * _t_14391;
			_t_14393 = _t_14392 < 0.0f;
			if(_t_14393)
				{
					float _t_14394;
					float _t_14395;
				
					_t_14394 = -1.0f * tx2_5_1;
					_t_14395 = tx3_6_1 + _t_14394;
					_t_14396 = _t_14395;
				
				}
		else
				{
					float _t_14397;
					float _t_14398;
					float _t_14399;
				
					_t_14397 = -1.0f * tx2_5_1;
					_t_14398 = tx3_6_1 + _t_14397;
					_t_14399 = -1.0f * _t_14398;
					_t_14396 = _t_14399;
				
				}
		
			_t_14400 = _t_14396 * _t_539;
			_t_14401 = -1.0f * ty3_9_1;
			_t_14402 = ty2_8_1 + _t_14401;
			_t_14403 = -1.0f * _t_14402;
			_t_14404 = _t_14403 < 0.0f;
			if(_t_14404)
				{
					float _t_14405;
					float _t_14406;
				
					_t_14405 = -1.0f * tx2_5_1;
					_t_14406 = tx3_6_1 + _t_14405;
					_t_14407 = _t_14406;
				
				}
		else
				{
					float _t_14408;
					float _t_14409;
					float _t_14410;
				
					_t_14408 = -1.0f * tx2_5_1;
					_t_14409 = tx3_6_1 + _t_14408;
					_t_14410 = -1.0f * _t_14409;
					_t_14407 = _t_14410;
				
				}
		
			_t_14411 = _t_14407 * _t_539;
			_t_14412 = _t_14400 * _t_14411;
			_t_14413 = -1.0f * ty3_9_1;
			_t_14414 = ty2_8_1 + _t_14413;
			_t_14415 = -1.0f * _t_14414;
			_t_14416 = _t_14415 < 0.0f;
			if(_t_14416)
				{
					float _t_14417;
					float _t_14418;
				
					_t_14417 = -1.0f * ty3_9_1;
					_t_14418 = ty2_8_1 + _t_14417;
					_t_14419 = _t_14418;
				
				}
		else
				{
					float _t_14420;
					float _t_14421;
					float _t_14422;
				
					_t_14420 = -1.0f * ty3_9_1;
					_t_14421 = ty2_8_1 + _t_14420;
					_t_14422 = -1.0f * _t_14421;
					_t_14419 = _t_14422;
				
				}
		
			_t_14423 = _t_14419 * _t_539;
			_t_14424 = 1.0f + _t_14423;
			_t_14425 = 1.0f / _t_14424;
			_t_14426 = _t_14412 * _t_14425;
			_t_14427 = _t_14426 * -1.0f;
			_t_14428 = 1.0f + _t_14427;
			_t_14429 = 0.0f < _t_14428;
			if(_t_14429)
				{
				
					_t_14430 = py0_12_1;
				
				}
		else
				{
				
					_t_14430 = py1_13_1;
				
				}
		
			_t_14431 = _t_14389 * _t_14430;
			_t_14432 = _t_14350 + _t_14431;
			_t_14433 = _t_14432 < y__3683_1;
			_t_14434 = -1.0f * ty3_9_1;
			_t_14435 = ty2_8_1 + _t_14434;
			_t_14436 = -1.0f * _t_14435;
			_t_14437 = _t_14436 < 0.0f;
			if(_t_14437)
				{
					float _t_14438;
					float _t_14439;
				
					_t_14438 = -1.0f * tx2_5_1;
					_t_14439 = tx3_6_1 + _t_14438;
					_t_14440 = _t_14439;
				
				}
		else
				{
					float _t_14441;
					float _t_14442;
					float _t_14443;
				
					_t_14441 = -1.0f * tx2_5_1;
					_t_14442 = tx3_6_1 + _t_14441;
					_t_14443 = -1.0f * _t_14442;
					_t_14440 = _t_14443;
				
				}
		
			_t_14444 = _t_14440 * _t_539;
			_t_14445 = _t_14444 * -1.0f;
			_t_14446 = -1.0f * ty3_9_1;
			_t_14447 = ty2_8_1 + _t_14446;
			_t_14448 = -1.0f * _t_14447;
			_t_14449 = _t_14448 < 0.0f;
			if(_t_14449)
				{
					float _t_14450;
					float _t_14451;
				
					_t_14450 = -1.0f * tx2_5_1;
					_t_14451 = tx3_6_1 + _t_14450;
					_t_14452 = _t_14451;
				
				}
		else
				{
					float _t_14453;
					float _t_14454;
					float _t_14455;
				
					_t_14453 = -1.0f * tx2_5_1;
					_t_14454 = tx3_6_1 + _t_14453;
					_t_14455 = -1.0f * _t_14454;
					_t_14452 = _t_14455;
				
				}
		
			_t_14456 = _t_14452 * _t_539;
			_t_14457 = _t_14456 * -1.0f;
			_t_14458 = 0.0f < _t_14457;
			if(_t_14458)
				{
				
					_t_14459 = px1_11_1;
				
				}
		else
				{
				
					_t_14459 = px0_10_1;
				
				}
		
			_t_14460 = _t_14445 * _t_14459;
			_t_14461 = -1.0f * ty3_9_1;
			_t_14462 = ty2_8_1 + _t_14461;
			_t_14463 = -1.0f * _t_14462;
			_t_14464 = _t_14463 < 0.0f;
			if(_t_14464)
				{
					float _t_14465;
					float _t_14466;
				
					_t_14465 = -1.0f * tx2_5_1;
					_t_14466 = tx3_6_1 + _t_14465;
					_t_14467 = _t_14466;
				
				}
		else
				{
					float _t_14468;
					float _t_14469;
					float _t_14470;
				
					_t_14468 = -1.0f * tx2_5_1;
					_t_14469 = tx3_6_1 + _t_14468;
					_t_14470 = -1.0f * _t_14469;
					_t_14467 = _t_14470;
				
				}
		
			_t_14471 = _t_14467 * _t_539;
			_t_14472 = -1.0f * ty3_9_1;
			_t_14473 = ty2_8_1 + _t_14472;
			_t_14474 = -1.0f * _t_14473;
			_t_14475 = _t_14474 < 0.0f;
			if(_t_14475)
				{
					float _t_14476;
					float _t_14477;
				
					_t_14476 = -1.0f * tx2_5_1;
					_t_14477 = tx3_6_1 + _t_14476;
					_t_14478 = _t_14477;
				
				}
		else
				{
					float _t_14479;
					float _t_14480;
					float _t_14481;
				
					_t_14479 = -1.0f * tx2_5_1;
					_t_14480 = tx3_6_1 + _t_14479;
					_t_14481 = -1.0f * _t_14480;
					_t_14478 = _t_14481;
				
				}
		
			_t_14482 = _t_14478 * _t_539;
			_t_14483 = _t_14471 * _t_14482;
			_t_14484 = -1.0f * ty3_9_1;
			_t_14485 = ty2_8_1 + _t_14484;
			_t_14486 = -1.0f * _t_14485;
			_t_14487 = _t_14486 < 0.0f;
			if(_t_14487)
				{
					float _t_14488;
					float _t_14489;
				
					_t_14488 = -1.0f * ty3_9_1;
					_t_14489 = ty2_8_1 + _t_14488;
					_t_14490 = _t_14489;
				
				}
		else
				{
					float _t_14491;
					float _t_14492;
					float _t_14493;
				
					_t_14491 = -1.0f * ty3_9_1;
					_t_14492 = ty2_8_1 + _t_14491;
					_t_14493 = -1.0f * _t_14492;
					_t_14490 = _t_14493;
				
				}
		
			_t_14494 = _t_14490 * _t_539;
			_t_14495 = 1.0f + _t_14494;
			_t_14496 = 1.0f / _t_14495;
			_t_14497 = _t_14483 * _t_14496;
			_t_14498 = _t_14497 * -1.0f;
			_t_14499 = 1.0f + _t_14498;
			_t_14500 = -1.0f * ty3_9_1;
			_t_14501 = ty2_8_1 + _t_14500;
			_t_14502 = -1.0f * _t_14501;
			_t_14503 = _t_14502 < 0.0f;
			if(_t_14503)
				{
					float _t_14504;
					float _t_14505;
				
					_t_14504 = -1.0f * tx2_5_1;
					_t_14505 = tx3_6_1 + _t_14504;
					_t_14506 = _t_14505;
				
				}
		else
				{
					float _t_14507;
					float _t_14508;
					float _t_14509;
				
					_t_14507 = -1.0f * tx2_5_1;
					_t_14508 = tx3_6_1 + _t_14507;
					_t_14509 = -1.0f * _t_14508;
					_t_14506 = _t_14509;
				
				}
		
			_t_14510 = _t_14506 * _t_539;
			_t_14511 = -1.0f * ty3_9_1;
			_t_14512 = ty2_8_1 + _t_14511;
			_t_14513 = -1.0f * _t_14512;
			_t_14514 = _t_14513 < 0.0f;
			if(_t_14514)
				{
					float _t_14515;
					float _t_14516;
				
					_t_14515 = -1.0f * tx2_5_1;
					_t_14516 = tx3_6_1 + _t_14515;
					_t_14517 = _t_14516;
				
				}
		else
				{
					float _t_14518;
					float _t_14519;
					float _t_14520;
				
					_t_14518 = -1.0f * tx2_5_1;
					_t_14519 = tx3_6_1 + _t_14518;
					_t_14520 = -1.0f * _t_14519;
					_t_14517 = _t_14520;
				
				}
		
			_t_14521 = _t_14517 * _t_539;
			_t_14522 = _t_14510 * _t_14521;
			_t_14523 = -1.0f * ty3_9_1;
			_t_14524 = ty2_8_1 + _t_14523;
			_t_14525 = -1.0f * _t_14524;
			_t_14526 = _t_14525 < 0.0f;
			if(_t_14526)
				{
					float _t_14527;
					float _t_14528;
				
					_t_14527 = -1.0f * ty3_9_1;
					_t_14528 = ty2_8_1 + _t_14527;
					_t_14529 = _t_14528;
				
				}
		else
				{
					float _t_14530;
					float _t_14531;
					float _t_14532;
				
					_t_14530 = -1.0f * ty3_9_1;
					_t_14531 = ty2_8_1 + _t_14530;
					_t_14532 = -1.0f * _t_14531;
					_t_14529 = _t_14532;
				
				}
		
			_t_14533 = _t_14529 * _t_539;
			_t_14534 = 1.0f + _t_14533;
			_t_14535 = 1.0f / _t_14534;
			_t_14536 = _t_14522 * _t_14535;
			_t_14537 = _t_14536 * -1.0f;
			_t_14538 = 1.0f + _t_14537;
			_t_14539 = 0.0f < _t_14538;
			if(_t_14539)
				{
				
					_t_14540 = py1_13_1;
				
				}
		else
				{
				
					_t_14540 = py0_12_1;
				
				}
		
			_t_14541 = _t_14499 * _t_14540;
			_t_14542 = _t_14460 + _t_14541;
			_t_14543 = y__3683_1 < _t_14542;
			_t_14544 = _t_14433 && _t_14543;
			_t_14545 = -1.0f * ty3_9_1;
			_t_14546 = ty2_8_1 + _t_14545;
			_t_14547 = -1.0f * _t_14546;
			_t_14548 = _t_14547 < 0.0f;
			if(_t_14548)
				{
					float _t_14549;
					float _t_14550;
				
					_t_14549 = -1.0f * ty3_9_1;
					_t_14550 = ty2_8_1 + _t_14549;
					_t_14551 = _t_14550;
				
				}
		else
				{
					float _t_14552;
					float _t_14553;
					float _t_14554;
				
					_t_14552 = -1.0f * ty3_9_1;
					_t_14553 = ty2_8_1 + _t_14552;
					_t_14554 = -1.0f * _t_14553;
					_t_14551 = _t_14554;
				
				}
		
			_t_14555 = _t_14551 * _t_539;
			_t_14556 = -1.0f * ty3_9_1;
			_t_14557 = ty2_8_1 + _t_14556;
			_t_14558 = -1.0f * _t_14557;
			_t_14559 = _t_14558 < 0.0f;
			if(_t_14559)
				{
					float _t_14560;
					float _t_14561;
				
					_t_14560 = -1.0f * ty3_9_1;
					_t_14561 = ty2_8_1 + _t_14560;
					_t_14562 = _t_14561;
				
				}
		else
				{
					float _t_14563;
					float _t_14564;
					float _t_14565;
				
					_t_14563 = -1.0f * ty3_9_1;
					_t_14564 = ty2_8_1 + _t_14563;
					_t_14565 = -1.0f * _t_14564;
					_t_14562 = _t_14565;
				
				}
		
			_t_14566 = _t_14562 * _t_539;
			_t_14567 = 0.0f < _t_14566;
			if(_t_14567)
				{
				
					_t_14568 = px0_10_1;
				
				}
		else
				{
				
					_t_14568 = px1_11_1;
				
				}
		
			_t_14569 = _t_14555 * _t_14568;
			_t_14570 = -1.0f * ty3_9_1;
			_t_14571 = ty2_8_1 + _t_14570;
			_t_14572 = -1.0f * _t_14571;
			_t_14573 = _t_14572 < 0.0f;
			if(_t_14573)
				{
					float _t_14574;
					float _t_14575;
				
					_t_14574 = -1.0f * tx2_5_1;
					_t_14575 = tx3_6_1 + _t_14574;
					_t_14576 = _t_14575;
				
				}
		else
				{
					float _t_14577;
					float _t_14578;
					float _t_14579;
				
					_t_14577 = -1.0f * tx2_5_1;
					_t_14578 = tx3_6_1 + _t_14577;
					_t_14579 = -1.0f * _t_14578;
					_t_14576 = _t_14579;
				
				}
		
			_t_14580 = _t_14576 * _t_539;
			_t_14581 = -1.0f * ty3_9_1;
			_t_14582 = ty2_8_1 + _t_14581;
			_t_14583 = -1.0f * _t_14582;
			_t_14584 = _t_14583 < 0.0f;
			if(_t_14584)
				{
					float _t_14585;
					float _t_14586;
				
					_t_14585 = -1.0f * tx2_5_1;
					_t_14586 = tx3_6_1 + _t_14585;
					_t_14587 = _t_14586;
				
				}
		else
				{
					float _t_14588;
					float _t_14589;
					float _t_14590;
				
					_t_14588 = -1.0f * tx2_5_1;
					_t_14589 = tx3_6_1 + _t_14588;
					_t_14590 = -1.0f * _t_14589;
					_t_14587 = _t_14590;
				
				}
		
			_t_14591 = _t_14587 * _t_539;
			_t_14592 = 0.0f < _t_14591;
			if(_t_14592)
				{
				
					_t_14593 = py0_12_1;
				
				}
		else
				{
				
					_t_14593 = py1_13_1;
				
				}
		
			_t_14594 = _t_14580 * _t_14593;
			_t_14595 = _t_14569 + _t_14594;
			_t_14596 = _t_14595 < _t_14236;
			_t_14597 = -1.0f * ty3_9_1;
			_t_14598 = ty2_8_1 + _t_14597;
			_t_14599 = -1.0f * _t_14598;
			_t_14600 = _t_14599 < 0.0f;
			if(_t_14600)
				{
					float _t_14601;
					float _t_14602;
				
					_t_14601 = -1.0f * ty3_9_1;
					_t_14602 = ty2_8_1 + _t_14601;
					_t_14603 = _t_14602;
				
				}
		else
				{
					float _t_14604;
					float _t_14605;
					float _t_14606;
				
					_t_14604 = -1.0f * ty3_9_1;
					_t_14605 = ty2_8_1 + _t_14604;
					_t_14606 = -1.0f * _t_14605;
					_t_14603 = _t_14606;
				
				}
		
			_t_14607 = _t_14603 * _t_539;
			_t_14608 = -1.0f * ty3_9_1;
			_t_14609 = ty2_8_1 + _t_14608;
			_t_14610 = -1.0f * _t_14609;
			_t_14611 = _t_14610 < 0.0f;
			if(_t_14611)
				{
					float _t_14612;
					float _t_14613;
				
					_t_14612 = -1.0f * ty3_9_1;
					_t_14613 = ty2_8_1 + _t_14612;
					_t_14614 = _t_14613;
				
				}
		else
				{
					float _t_14615;
					float _t_14616;
					float _t_14617;
				
					_t_14615 = -1.0f * ty3_9_1;
					_t_14616 = ty2_8_1 + _t_14615;
					_t_14617 = -1.0f * _t_14616;
					_t_14614 = _t_14617;
				
				}
		
			_t_14618 = _t_14614 * _t_539;
			_t_14619 = 0.0f < _t_14618;
			if(_t_14619)
				{
				
					_t_14620 = px1_11_1;
				
				}
		else
				{
				
					_t_14620 = px0_10_1;
				
				}
		
			_t_14621 = _t_14607 * _t_14620;
			_t_14622 = -1.0f * ty3_9_1;
			_t_14623 = ty2_8_1 + _t_14622;
			_t_14624 = -1.0f * _t_14623;
			_t_14625 = _t_14624 < 0.0f;
			if(_t_14625)
				{
					float _t_14626;
					float _t_14627;
				
					_t_14626 = -1.0f * tx2_5_1;
					_t_14627 = tx3_6_1 + _t_14626;
					_t_14628 = _t_14627;
				
				}
		else
				{
					float _t_14629;
					float _t_14630;
					float _t_14631;
				
					_t_14629 = -1.0f * tx2_5_1;
					_t_14630 = tx3_6_1 + _t_14629;
					_t_14631 = -1.0f * _t_14630;
					_t_14628 = _t_14631;
				
				}
		
			_t_14632 = _t_14628 * _t_539;
			_t_14633 = -1.0f * ty3_9_1;
			_t_14634 = ty2_8_1 + _t_14633;
			_t_14635 = -1.0f * _t_14634;
			_t_14636 = _t_14635 < 0.0f;
			if(_t_14636)
				{
					float _t_14637;
					float _t_14638;
				
					_t_14637 = -1.0f * tx2_5_1;
					_t_14638 = tx3_6_1 + _t_14637;
					_t_14639 = _t_14638;
				
				}
		else
				{
					float _t_14640;
					float _t_14641;
					float _t_14642;
				
					_t_14640 = -1.0f * tx2_5_1;
					_t_14641 = tx3_6_1 + _t_14640;
					_t_14642 = -1.0f * _t_14641;
					_t_14639 = _t_14642;
				
				}
		
			_t_14643 = _t_14639 * _t_539;
			_t_14644 = 0.0f < _t_14643;
			if(_t_14644)
				{
				
					_t_14645 = py1_13_1;
				
				}
		else
				{
				
					_t_14645 = py0_12_1;
				
				}
		
			_t_14646 = _t_14632 * _t_14645;
			_t_14647 = _t_14621 + _t_14646;
			_t_14648 = _t_14236 < _t_14647;
			_t_14649 = _t_14596 && _t_14648;
			_t_14650 = _t_14544 && _t_14649;
			if(_t_14650)
				{
				
					_t_14651 = 1.0f;
				
				}
		else
				{
				
					_t_14651 = 0.0f;
				
				}
		
			_t_14652 = _t_14651 * _t_539;
			_t_14653 = _t_14652;
		
		}
else
		{
		
			_t_14653 = 0.0f;
		
		}

	_t_14654 = -1.0f * pc0_14_1;
	_t_14655 = tc0_17_1 + _t_14654;
	_t_14656 = _t_14655 * _t_14655;
	_t_14657 = -1.0f * pc1_15_1;
	_t_14658 = tc1_18_1 + _t_14657;
	_t_14659 = _t_14658 * _t_14658;
	_t_14660 = _t_14656 + _t_14659;
	_t_14661 = -1.0f * pc2_16_1;
	_t_14662 = tc2_19_1 + _t_14661;
	_t_14663 = _t_14662 * _t_14662;
	_t_14664 = _t_14660 + _t_14663;
	_t_14665 = tx3_6_1 * ty1_7_1;
	_t_14666 = tx1_4_1 * ty3_9_1;
	_t_14667 = _t_14666 * -1.0f;
	_t_14668 = _t_14665 + _t_14667;
	_t_14669 = -1.0f * ty1_7_1;
	_t_14670 = ty3_9_1 + _t_14669;
	_t_14671 = _t_14670 * _t_14263;
	_t_14672 = _t_14668 + _t_14671;
	_t_14673 = -1.0f * tx3_6_1;
	_t_14674 = tx1_4_1 + _t_14673;
	_t_14675 = _t_14674 * _t_14316;
	_t_14676 = _t_14672 + _t_14675;
	_t_14677 = _t_14676 < 0.0f;
	if(_t_14677)
		{
		
			_t_14678 = 1.0f;
		
		}
else
		{
		
			_t_14678 = 0.0f;
		
		}

	_t_14679 = _t_14664 * _t_14678;
	_t_14680 = tx1_4_1 * ty2_8_1;
	_t_14681 = tx2_5_1 * ty1_7_1;
	_t_14682 = _t_14681 * -1.0f;
	_t_14683 = _t_14680 + _t_14682;
	_t_14684 = -1.0f * ty2_8_1;
	_t_14685 = ty1_7_1 + _t_14684;
	_t_14686 = _t_14685 * _t_14263;
	_t_14687 = _t_14683 + _t_14686;
	_t_14688 = -1.0f * tx1_4_1;
	_t_14689 = tx2_5_1 + _t_14688;
	_t_14690 = _t_14689 * _t_14316;
	_t_14691 = _t_14687 + _t_14690;
	_t_14692 = _t_14691 < 0.0f;
	if(_t_14692)
		{
		
			_t_14693 = 1.0f;
		
		}
else
		{
		
			_t_14693 = 0.0f;
		
		}

	_t_14694 = _t_14679 * _t_14693;
	_t_14695 = _t_14694 * tx2_5_1;
	_t_14696 = _t_14695 * -1.0f;
	_t_14697 = -1.0f * pc0_14_1;
	_t_14698 = tc0_17_1 + _t_14697;
	_t_14699 = _t_14698 * _t_14698;
	_t_14700 = -1.0f * pc1_15_1;
	_t_14701 = tc1_18_1 + _t_14700;
	_t_14702 = _t_14701 * _t_14701;
	_t_14703 = _t_14699 + _t_14702;
	_t_14704 = -1.0f * pc2_16_1;
	_t_14705 = tc2_19_1 + _t_14704;
	_t_14706 = _t_14705 * _t_14705;
	_t_14707 = _t_14703 + _t_14706;
	_t_14708 = tx3_6_1 * ty1_7_1;
	_t_14709 = tx1_4_1 * ty3_9_1;
	_t_14710 = _t_14709 * -1.0f;
	_t_14711 = _t_14708 + _t_14710;
	_t_14712 = -1.0f * ty1_7_1;
	_t_14713 = ty3_9_1 + _t_14712;
	_t_14714 = _t_14713 * _t_14263;
	_t_14715 = _t_14711 + _t_14714;
	_t_14716 = -1.0f * tx3_6_1;
	_t_14717 = tx1_4_1 + _t_14716;
	_t_14718 = _t_14717 * _t_14316;
	_t_14719 = _t_14715 + _t_14718;
	_t_14720 = _t_14719 < 0.0f;
	if(_t_14720)
		{
		
			_t_14721 = 1.0f;
		
		}
else
		{
		
			_t_14721 = 0.0f;
		
		}

	_t_14722 = _t_14707 * _t_14721;
	_t_14723 = tx1_4_1 * ty2_8_1;
	_t_14724 = tx2_5_1 * ty1_7_1;
	_t_14725 = _t_14724 * -1.0f;
	_t_14726 = _t_14723 + _t_14725;
	_t_14727 = -1.0f * ty2_8_1;
	_t_14728 = ty1_7_1 + _t_14727;
	_t_14729 = _t_14728 * _t_14263;
	_t_14730 = _t_14726 + _t_14729;
	_t_14731 = -1.0f * tx1_4_1;
	_t_14732 = tx2_5_1 + _t_14731;
	_t_14733 = _t_14732 * _t_14316;
	_t_14734 = _t_14730 + _t_14733;
	_t_14735 = _t_14734 < 0.0f;
	if(_t_14735)
		{
		
			_t_14736 = 1.0f;
		
		}
else
		{
		
			_t_14736 = 0.0f;
		
		}

	_t_14737 = _t_14722 * _t_14736;
	_t_14738 = _t_14737 * _t_14263;
	_t_14739 = _t_14696 + _t_14738;
	_t_14237 = _t_14653 * _t_14739;

	return _t_14237;
}
__device__ float tegpixellet_block_49(float ty2_8_1,float ty3_9_1,float _t_539,float _t_14236,float tx3_6_1,float tx2_5_1,float y__3683_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_14238;
	float _t_14239;
	float _t_14240;
	bool _t_14241;
	float _t_14244;
	float _t_14248;
	float _t_14249;
	float _t_14250;
	float _t_14251;
	float _t_14252;
	bool _t_14253;
	float _t_14256;
	float _t_14260;
	float _t_14261;
	float _t_14262;
	float _t_14263;
	float _t_14264;
	float _t_14265;
	float _t_14266;
	bool _t_14267;
	float _t_14270;
	float _t_14274;
	float _t_14275;
	float _t_14276;
	float _t_14277;
	bool _t_14278;
	float _t_14281;
	float _t_14285;
	float _t_14286;
	float _t_14287;
	float _t_14288;
	float _t_14289;
	bool _t_14290;
	float _t_14293;
	float _t_14297;
	float _t_14298;
	float _t_14299;
	float _t_14300;
	float _t_14301;
	float _t_14302;
	float _t_14303;
	float _t_14304;
	float _t_14305;
	float _t_14306;
	bool _t_14307;
	float _t_14310;
	float _t_14314;
	float _t_14315;
	float _t_14316;

	float _t_14237;

	_t_14238 = -1.0f * ty3_9_1;
	_t_14239 = ty2_8_1 + _t_14238;
	_t_14240 = -1.0f * _t_14239;
	_t_14241 = _t_14240 < 0.0f;
	if(_t_14241)
		{
			float _t_14242;
			float _t_14243;
		
			_t_14242 = -1.0f * ty3_9_1;
			_t_14243 = ty2_8_1 + _t_14242;
			_t_14244 = _t_14243;
		
		}
else
		{
			float _t_14245;
			float _t_14246;
			float _t_14247;
		
			_t_14245 = -1.0f * ty3_9_1;
			_t_14246 = ty2_8_1 + _t_14245;
			_t_14247 = -1.0f * _t_14246;
			_t_14244 = _t_14247;
		
		}

	_t_14248 = _t_14244 * _t_539;
	_t_14249 = _t_14248 * _t_14236;
	_t_14250 = -1.0f * ty3_9_1;
	_t_14251 = ty2_8_1 + _t_14250;
	_t_14252 = -1.0f * _t_14251;
	_t_14253 = _t_14252 < 0.0f;
	if(_t_14253)
		{
			float _t_14254;
			float _t_14255;
		
			_t_14254 = -1.0f * tx2_5_1;
			_t_14255 = tx3_6_1 + _t_14254;
			_t_14256 = _t_14255;
		
		}
else
		{
			float _t_14257;
			float _t_14258;
			float _t_14259;
		
			_t_14257 = -1.0f * tx2_5_1;
			_t_14258 = tx3_6_1 + _t_14257;
			_t_14259 = -1.0f * _t_14258;
			_t_14256 = _t_14259;
		
		}

	_t_14260 = _t_14256 * _t_539;
	_t_14261 = _t_14260 * -1.0f;
	_t_14262 = _t_14261 * y__3683_1;
	_t_14263 = _t_14249 + _t_14262;
	_t_14264 = -1.0f * ty3_9_1;
	_t_14265 = ty2_8_1 + _t_14264;
	_t_14266 = -1.0f * _t_14265;
	_t_14267 = _t_14266 < 0.0f;
	if(_t_14267)
		{
			float _t_14268;
			float _t_14269;
		
			_t_14268 = -1.0f * tx2_5_1;
			_t_14269 = tx3_6_1 + _t_14268;
			_t_14270 = _t_14269;
		
		}
else
		{
			float _t_14271;
			float _t_14272;
			float _t_14273;
		
			_t_14271 = -1.0f * tx2_5_1;
			_t_14272 = tx3_6_1 + _t_14271;
			_t_14273 = -1.0f * _t_14272;
			_t_14270 = _t_14273;
		
		}

	_t_14274 = _t_14270 * _t_539;
	_t_14275 = -1.0f * ty3_9_1;
	_t_14276 = ty2_8_1 + _t_14275;
	_t_14277 = -1.0f * _t_14276;
	_t_14278 = _t_14277 < 0.0f;
	if(_t_14278)
		{
			float _t_14279;
			float _t_14280;
		
			_t_14279 = -1.0f * tx2_5_1;
			_t_14280 = tx3_6_1 + _t_14279;
			_t_14281 = _t_14280;
		
		}
else
		{
			float _t_14282;
			float _t_14283;
			float _t_14284;
		
			_t_14282 = -1.0f * tx2_5_1;
			_t_14283 = tx3_6_1 + _t_14282;
			_t_14284 = -1.0f * _t_14283;
			_t_14281 = _t_14284;
		
		}

	_t_14285 = _t_14281 * _t_539;
	_t_14286 = _t_14274 * _t_14285;
	_t_14287 = -1.0f * ty3_9_1;
	_t_14288 = ty2_8_1 + _t_14287;
	_t_14289 = -1.0f * _t_14288;
	_t_14290 = _t_14289 < 0.0f;
	if(_t_14290)
		{
			float _t_14291;
			float _t_14292;
		
			_t_14291 = -1.0f * ty3_9_1;
			_t_14292 = ty2_8_1 + _t_14291;
			_t_14293 = _t_14292;
		
		}
else
		{
			float _t_14294;
			float _t_14295;
			float _t_14296;
		
			_t_14294 = -1.0f * ty3_9_1;
			_t_14295 = ty2_8_1 + _t_14294;
			_t_14296 = -1.0f * _t_14295;
			_t_14293 = _t_14296;
		
		}

	_t_14297 = _t_14293 * _t_539;
	_t_14298 = 1.0f + _t_14297;
	_t_14299 = 1.0f / _t_14298;
	_t_14300 = _t_14286 * _t_14299;
	_t_14301 = _t_14300 * -1.0f;
	_t_14302 = 1.0f + _t_14301;
	_t_14303 = _t_14302 * y__3683_1;
	_t_14304 = -1.0f * ty3_9_1;
	_t_14305 = ty2_8_1 + _t_14304;
	_t_14306 = -1.0f * _t_14305;
	_t_14307 = _t_14306 < 0.0f;
	if(_t_14307)
		{
			float _t_14308;
			float _t_14309;
		
			_t_14308 = -1.0f * tx2_5_1;
			_t_14309 = tx3_6_1 + _t_14308;
			_t_14310 = _t_14309;
		
		}
else
		{
			float _t_14311;
			float _t_14312;
			float _t_14313;
		
			_t_14311 = -1.0f * tx2_5_1;
			_t_14312 = tx3_6_1 + _t_14311;
			_t_14313 = -1.0f * _t_14312;
			_t_14310 = _t_14313;
		
		}

	_t_14314 = _t_14310 * _t_539;
	_t_14315 = _t_14314 * _t_14236;
	_t_14316 = _t_14303 + _t_14315;
	_t_14237 = tegpixellet_block_50(py0_12_1,_t_14316,py1_13_1,px0_10_1,_t_14263,px1_11_1,ty2_8_1,ty3_9_1,tx3_6_1,tx2_5_1,_t_539,y__3683_1,_t_14236,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);

	return _t_14237;
}
__device__ float tegpixelbody_block_32(float ty2_8_1,float ty3_9_1,float _t_539,float px0_10_1,float px1_11_1,float tx3_6_1,float tx2_5_1,float py0_12_1,float py1_13_1,float y__3683_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_14080;
	float _t_14081;
	float _t_14082;
	bool _t_14083;
	float _t_14086;
	float _t_14090;
	float _t_14091;
	float _t_14092;
	float _t_14093;
	bool _t_14094;
	float _t_14097;
	float _t_14101;
	bool _t_14102;
	float _t_14103;
	float _t_14104;
	float _t_14105;
	float _t_14106;
	float _t_14107;
	bool _t_14108;
	float _t_14111;
	float _t_14115;
	float _t_14116;
	float _t_14117;
	float _t_14118;
	bool _t_14119;
	float _t_14122;
	float _t_14126;
	bool _t_14127;
	float _t_14128;
	float _t_14129;
	float _t_14130;
	float _t_14131;
	float _t_14132;
	float _t_14133;
	bool _t_14134;
	float _t_14139;
	float _t_14145;
	float _t_14146;
	float _t_14147;
	float _t_14148;
	bool _t_14149;
	float _t_14150;
	float _t_14151;
	float _t_14152;
	bool _t_14153;
	float _t_14156;
	float _t_14160;
	float _t_14161;
	float _t_14162;
	float _t_14163;
	bool _t_14164;
	float _t_14167;
	float _t_14171;
	bool _t_14172;
	float _t_14173;
	float _t_14174;
	float _t_14175;
	float _t_14176;
	float _t_14177;
	bool _t_14178;
	float _t_14181;
	float _t_14185;
	float _t_14186;
	float _t_14187;
	float _t_14188;
	bool _t_14189;
	float _t_14192;
	float _t_14196;
	bool _t_14197;
	float _t_14198;
	float _t_14199;
	float _t_14200;
	float _t_14201;
	float _t_14202;
	float _t_14203;
	bool _t_14204;
	float _t_14209;
	float _t_14215;
	float _t_14216;
	float _t_14217;
	float _t_14218;
	bool _t_14219;
	bool _t_14220;

	float _t_14079;

	_t_14080 = -1.0f * ty3_9_1;
	_t_14081 = ty2_8_1 + _t_14080;
	_t_14082 = -1.0f * _t_14081;
	_t_14083 = _t_14082 < 0.0f;
	if(_t_14083)
		{
			float _t_14084;
			float _t_14085;
		
			_t_14084 = -1.0f * ty3_9_1;
			_t_14085 = ty2_8_1 + _t_14084;
			_t_14086 = _t_14085;
		
		}
else
		{
			float _t_14087;
			float _t_14088;
			float _t_14089;
		
			_t_14087 = -1.0f * ty3_9_1;
			_t_14088 = ty2_8_1 + _t_14087;
			_t_14089 = -1.0f * _t_14088;
			_t_14086 = _t_14089;
		
		}

	_t_14090 = _t_14086 * _t_539;
	_t_14091 = -1.0f * ty3_9_1;
	_t_14092 = ty2_8_1 + _t_14091;
	_t_14093 = -1.0f * _t_14092;
	_t_14094 = _t_14093 < 0.0f;
	if(_t_14094)
		{
			float _t_14095;
			float _t_14096;
		
			_t_14095 = -1.0f * ty3_9_1;
			_t_14096 = ty2_8_1 + _t_14095;
			_t_14097 = _t_14096;
		
		}
else
		{
			float _t_14098;
			float _t_14099;
			float _t_14100;
		
			_t_14098 = -1.0f * ty3_9_1;
			_t_14099 = ty2_8_1 + _t_14098;
			_t_14100 = -1.0f * _t_14099;
			_t_14097 = _t_14100;
		
		}

	_t_14101 = _t_14097 * _t_539;
	_t_14102 = 0.0f < _t_14101;
	if(_t_14102)
		{
		
			_t_14103 = px0_10_1;
		
		}
else
		{
		
			_t_14103 = px1_11_1;
		
		}

	_t_14104 = _t_14090 * _t_14103;
	_t_14105 = -1.0f * ty3_9_1;
	_t_14106 = ty2_8_1 + _t_14105;
	_t_14107 = -1.0f * _t_14106;
	_t_14108 = _t_14107 < 0.0f;
	if(_t_14108)
		{
			float _t_14109;
			float _t_14110;
		
			_t_14109 = -1.0f * tx2_5_1;
			_t_14110 = tx3_6_1 + _t_14109;
			_t_14111 = _t_14110;
		
		}
else
		{
			float _t_14112;
			float _t_14113;
			float _t_14114;
		
			_t_14112 = -1.0f * tx2_5_1;
			_t_14113 = tx3_6_1 + _t_14112;
			_t_14114 = -1.0f * _t_14113;
			_t_14111 = _t_14114;
		
		}

	_t_14115 = _t_14111 * _t_539;
	_t_14116 = -1.0f * ty3_9_1;
	_t_14117 = ty2_8_1 + _t_14116;
	_t_14118 = -1.0f * _t_14117;
	_t_14119 = _t_14118 < 0.0f;
	if(_t_14119)
		{
			float _t_14120;
			float _t_14121;
		
			_t_14120 = -1.0f * tx2_5_1;
			_t_14121 = tx3_6_1 + _t_14120;
			_t_14122 = _t_14121;
		
		}
else
		{
			float _t_14123;
			float _t_14124;
			float _t_14125;
		
			_t_14123 = -1.0f * tx2_5_1;
			_t_14124 = tx3_6_1 + _t_14123;
			_t_14125 = -1.0f * _t_14124;
			_t_14122 = _t_14125;
		
		}

	_t_14126 = _t_14122 * _t_539;
	_t_14127 = 0.0f < _t_14126;
	if(_t_14127)
		{
		
			_t_14128 = py0_12_1;
		
		}
else
		{
		
			_t_14128 = py1_13_1;
		
		}

	_t_14129 = _t_14115 * _t_14128;
	_t_14130 = _t_14104 + _t_14129;
	_t_14131 = -1.0f * ty3_9_1;
	_t_14132 = ty2_8_1 + _t_14131;
	_t_14133 = -1.0f * _t_14132;
	_t_14134 = _t_14133 < 0.0f;
	if(_t_14134)
		{
			float _t_14135;
			float _t_14136;
			float _t_14137;
			float _t_14138;
		
			_t_14135 = tx2_5_1 * ty3_9_1;
			_t_14136 = tx3_6_1 * ty2_8_1;
			_t_14137 = _t_14136 * -1.0f;
			_t_14138 = _t_14135 + _t_14137;
			_t_14139 = _t_14138;
		
		}
else
		{
			float _t_14140;
			float _t_14141;
			float _t_14142;
			float _t_14143;
			float _t_14144;
		
			_t_14140 = tx2_5_1 * ty3_9_1;
			_t_14141 = tx3_6_1 * ty2_8_1;
			_t_14142 = _t_14141 * -1.0f;
			_t_14143 = _t_14140 + _t_14142;
			_t_14144 = -1.0f * _t_14143;
			_t_14139 = _t_14144;
		
		}

	_t_14145 = -1.0f * _t_14139;
	_t_14146 = _t_14145 * _t_539;
	_t_14147 = _t_14146 * -1.0f;
	_t_14148 = _t_14130 + _t_14147;
	_t_14149 = _t_14148 < 0.0f;
	_t_14150 = -1.0f * ty3_9_1;
	_t_14151 = ty2_8_1 + _t_14150;
	_t_14152 = -1.0f * _t_14151;
	_t_14153 = _t_14152 < 0.0f;
	if(_t_14153)
		{
			float _t_14154;
			float _t_14155;
		
			_t_14154 = -1.0f * ty3_9_1;
			_t_14155 = ty2_8_1 + _t_14154;
			_t_14156 = _t_14155;
		
		}
else
		{
			float _t_14157;
			float _t_14158;
			float _t_14159;
		
			_t_14157 = -1.0f * ty3_9_1;
			_t_14158 = ty2_8_1 + _t_14157;
			_t_14159 = -1.0f * _t_14158;
			_t_14156 = _t_14159;
		
		}

	_t_14160 = _t_14156 * _t_539;
	_t_14161 = -1.0f * ty3_9_1;
	_t_14162 = ty2_8_1 + _t_14161;
	_t_14163 = -1.0f * _t_14162;
	_t_14164 = _t_14163 < 0.0f;
	if(_t_14164)
		{
			float _t_14165;
			float _t_14166;
		
			_t_14165 = -1.0f * ty3_9_1;
			_t_14166 = ty2_8_1 + _t_14165;
			_t_14167 = _t_14166;
		
		}
else
		{
			float _t_14168;
			float _t_14169;
			float _t_14170;
		
			_t_14168 = -1.0f * ty3_9_1;
			_t_14169 = ty2_8_1 + _t_14168;
			_t_14170 = -1.0f * _t_14169;
			_t_14167 = _t_14170;
		
		}

	_t_14171 = _t_14167 * _t_539;
	_t_14172 = 0.0f < _t_14171;
	if(_t_14172)
		{
		
			_t_14173 = px1_11_1;
		
		}
else
		{
		
			_t_14173 = px0_10_1;
		
		}

	_t_14174 = _t_14160 * _t_14173;
	_t_14175 = -1.0f * ty3_9_1;
	_t_14176 = ty2_8_1 + _t_14175;
	_t_14177 = -1.0f * _t_14176;
	_t_14178 = _t_14177 < 0.0f;
	if(_t_14178)
		{
			float _t_14179;
			float _t_14180;
		
			_t_14179 = -1.0f * tx2_5_1;
			_t_14180 = tx3_6_1 + _t_14179;
			_t_14181 = _t_14180;
		
		}
else
		{
			float _t_14182;
			float _t_14183;
			float _t_14184;
		
			_t_14182 = -1.0f * tx2_5_1;
			_t_14183 = tx3_6_1 + _t_14182;
			_t_14184 = -1.0f * _t_14183;
			_t_14181 = _t_14184;
		
		}

	_t_14185 = _t_14181 * _t_539;
	_t_14186 = -1.0f * ty3_9_1;
	_t_14187 = ty2_8_1 + _t_14186;
	_t_14188 = -1.0f * _t_14187;
	_t_14189 = _t_14188 < 0.0f;
	if(_t_14189)
		{
			float _t_14190;
			float _t_14191;
		
			_t_14190 = -1.0f * tx2_5_1;
			_t_14191 = tx3_6_1 + _t_14190;
			_t_14192 = _t_14191;
		
		}
else
		{
			float _t_14193;
			float _t_14194;
			float _t_14195;
		
			_t_14193 = -1.0f * tx2_5_1;
			_t_14194 = tx3_6_1 + _t_14193;
			_t_14195 = -1.0f * _t_14194;
			_t_14192 = _t_14195;
		
		}

	_t_14196 = _t_14192 * _t_539;
	_t_14197 = 0.0f < _t_14196;
	if(_t_14197)
		{
		
			_t_14198 = py1_13_1;
		
		}
else
		{
		
			_t_14198 = py0_12_1;
		
		}

	_t_14199 = _t_14185 * _t_14198;
	_t_14200 = _t_14174 + _t_14199;
	_t_14201 = -1.0f * ty3_9_1;
	_t_14202 = ty2_8_1 + _t_14201;
	_t_14203 = -1.0f * _t_14202;
	_t_14204 = _t_14203 < 0.0f;
	if(_t_14204)
		{
			float _t_14205;
			float _t_14206;
			float _t_14207;
			float _t_14208;
		
			_t_14205 = tx2_5_1 * ty3_9_1;
			_t_14206 = tx3_6_1 * ty2_8_1;
			_t_14207 = _t_14206 * -1.0f;
			_t_14208 = _t_14205 + _t_14207;
			_t_14209 = _t_14208;
		
		}
else
		{
			float _t_14210;
			float _t_14211;
			float _t_14212;
			float _t_14213;
			float _t_14214;
		
			_t_14210 = tx2_5_1 * ty3_9_1;
			_t_14211 = tx3_6_1 * ty2_8_1;
			_t_14212 = _t_14211 * -1.0f;
			_t_14213 = _t_14210 + _t_14212;
			_t_14214 = -1.0f * _t_14213;
			_t_14209 = _t_14214;
		
		}

	_t_14215 = -1.0f * _t_14209;
	_t_14216 = _t_14215 * _t_539;
	_t_14217 = _t_14216 * -1.0f;
	_t_14218 = _t_14200 + _t_14217;
	_t_14219 = 0.0f < _t_14218;
	_t_14220 = _t_14149 && _t_14219;
	if(_t_14220)
		{
			float _t_14221;
			float _t_14222;
			float _t_14223;
			bool _t_14224;
			float _t_14229;
			float _t_14235;
			float _t_14236;
			float _t_14237;
		
			_t_14221 = -1.0f * ty3_9_1;
			_t_14222 = ty2_8_1 + _t_14221;
			_t_14223 = -1.0f * _t_14222;
			_t_14224 = _t_14223 < 0.0f;
			if(_t_14224)
				{
					float _t_14225;
					float _t_14226;
					float _t_14227;
					float _t_14228;
				
					_t_14225 = tx2_5_1 * ty3_9_1;
					_t_14226 = tx3_6_1 * ty2_8_1;
					_t_14227 = _t_14226 * -1.0f;
					_t_14228 = _t_14225 + _t_14227;
					_t_14229 = _t_14228;
				
				}
		else
				{
					float _t_14230;
					float _t_14231;
					float _t_14232;
					float _t_14233;
					float _t_14234;
				
					_t_14230 = tx2_5_1 * ty3_9_1;
					_t_14231 = tx3_6_1 * ty2_8_1;
					_t_14232 = _t_14231 * -1.0f;
					_t_14233 = _t_14230 + _t_14232;
					_t_14234 = -1.0f * _t_14233;
					_t_14229 = _t_14234;
				
				}
		
			_t_14235 = -1.0f * _t_14229;
			_t_14236 = _t_14235 * _t_539;
			_t_14237 = tegpixellet_block_49(ty2_8_1,ty3_9_1,_t_539,_t_14236,tx3_6_1,tx2_5_1,y__3683_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);
			_t_14079 = _t_14237;
		
		}
else
		{
		
			_t_14079 = 0.0f;
		
		}


	return _t_14079;
}
__device__ float tegpixelintegrator_32(float _t_539,float ty3_9_1,float pc1_15_1,float tc2_19_1,float ty2_8_1,float _t_14078,float ty1_7_1,float pc0_14_1,float tx3_6_1,float tx1_4_1,float tx2_5_1,float py1_13_1,float pc2_16_1,float px1_11_1,float tc0_17_1,float py0_12_1,float tc1_18_1,float px0_10_1,float _t_13969){
    float y__3683_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_14078 - _t_13969)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__3683_1 = _t_13969 + __step__ * (i + (float)(0.5));
        float _t_14079;
		_t_14079 = tegpixelbody_block_32(ty2_8_1,ty3_9_1,_t_539,px0_10_1,px1_11_1,tx3_6_1,tx2_5_1,py0_12_1,py1_13_1,y__3683_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);;
        __output__ = __output__ + _t_14079 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_16(float ty2_8_1,float ty3_9_1,float tx3_6_1,float tx2_5_1,float _t_539,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty1_7_1,float tx1_4_1){
	float _t_13861;
	float _t_13862;
	float _t_13863;
	bool _t_13864;
	float _t_13867;
	float _t_13871;
	float _t_13872;
	float _t_13873;
	float _t_13874;
	float _t_13875;
	bool _t_13876;
	float _t_13879;
	float _t_13883;
	float _t_13884;
	bool _t_13885;
	float _t_13886;
	float _t_13887;
	float _t_13888;
	float _t_13889;
	float _t_13890;
	bool _t_13891;
	float _t_13894;
	float _t_13898;
	float _t_13899;
	float _t_13900;
	float _t_13901;
	bool _t_13902;
	float _t_13905;
	float _t_13909;
	float _t_13910;
	float _t_13911;
	float _t_13912;
	float _t_13913;
	bool _t_13914;
	float _t_13917;
	float _t_13921;
	float _t_13922;
	float _t_13923;
	float _t_13924;
	float _t_13925;
	float _t_13926;
	float _t_13927;
	float _t_13928;
	float _t_13929;
	bool _t_13930;
	float _t_13933;
	float _t_13937;
	float _t_13938;
	float _t_13939;
	float _t_13940;
	bool _t_13941;
	float _t_13944;
	float _t_13948;
	float _t_13949;
	float _t_13950;
	float _t_13951;
	float _t_13952;
	bool _t_13953;
	float _t_13956;
	float _t_13960;
	float _t_13961;
	float _t_13962;
	float _t_13963;
	float _t_13964;
	float _t_13965;
	bool _t_13966;
	float _t_13967;
	float _t_13968;
	float _t_13969;
	float _t_13970;
	float _t_13971;
	float _t_13972;
	bool _t_13973;
	float _t_13976;
	float _t_13980;
	float _t_13981;
	float _t_13982;
	float _t_13983;
	float _t_13984;
	bool _t_13985;
	float _t_13988;
	float _t_13992;
	float _t_13993;
	bool _t_13994;
	float _t_13995;
	float _t_13996;
	float _t_13997;
	float _t_13998;
	float _t_13999;
	bool _t_14000;
	float _t_14003;
	float _t_14007;
	float _t_14008;
	float _t_14009;
	float _t_14010;
	bool _t_14011;
	float _t_14014;
	float _t_14018;
	float _t_14019;
	float _t_14020;
	float _t_14021;
	float _t_14022;
	bool _t_14023;
	float _t_14026;
	float _t_14030;
	float _t_14031;
	float _t_14032;
	float _t_14033;
	float _t_14034;
	float _t_14035;
	float _t_14036;
	float _t_14037;
	float _t_14038;
	bool _t_14039;
	float _t_14042;
	float _t_14046;
	float _t_14047;
	float _t_14048;
	float _t_14049;
	bool _t_14050;
	float _t_14053;
	float _t_14057;
	float _t_14058;
	float _t_14059;
	float _t_14060;
	float _t_14061;
	bool _t_14062;
	float _t_14065;
	float _t_14069;
	float _t_14070;
	float _t_14071;
	float _t_14072;
	float _t_14073;
	float _t_14074;
	bool _t_14075;
	float _t_14076;
	float _t_14077;
	float _t_14078;

	float _t_540;

	_t_13861 = -1.0f * ty3_9_1;
	_t_13862 = ty2_8_1 + _t_13861;
	_t_13863 = -1.0f * _t_13862;
	_t_13864 = _t_13863 < 0.0f;
	if(_t_13864)
		{
			float _t_13865;
			float _t_13866;
		
			_t_13865 = -1.0f * tx2_5_1;
			_t_13866 = tx3_6_1 + _t_13865;
			_t_13867 = _t_13866;
		
		}
else
		{
			float _t_13868;
			float _t_13869;
			float _t_13870;
		
			_t_13868 = -1.0f * tx2_5_1;
			_t_13869 = tx3_6_1 + _t_13868;
			_t_13870 = -1.0f * _t_13869;
			_t_13867 = _t_13870;
		
		}

	_t_13871 = _t_13867 * _t_539;
	_t_13872 = _t_13871 * -1.0f;
	_t_13873 = -1.0f * ty3_9_1;
	_t_13874 = ty2_8_1 + _t_13873;
	_t_13875 = -1.0f * _t_13874;
	_t_13876 = _t_13875 < 0.0f;
	if(_t_13876)
		{
			float _t_13877;
			float _t_13878;
		
			_t_13877 = -1.0f * tx2_5_1;
			_t_13878 = tx3_6_1 + _t_13877;
			_t_13879 = _t_13878;
		
		}
else
		{
			float _t_13880;
			float _t_13881;
			float _t_13882;
		
			_t_13880 = -1.0f * tx2_5_1;
			_t_13881 = tx3_6_1 + _t_13880;
			_t_13882 = -1.0f * _t_13881;
			_t_13879 = _t_13882;
		
		}

	_t_13883 = _t_13879 * _t_539;
	_t_13884 = _t_13883 * -1.0f;
	_t_13885 = 0.0f < _t_13884;
	if(_t_13885)
		{
		
			_t_13886 = px0_10_1;
		
		}
else
		{
		
			_t_13886 = px1_11_1;
		
		}

	_t_13887 = _t_13872 * _t_13886;
	_t_13888 = -1.0f * ty3_9_1;
	_t_13889 = ty2_8_1 + _t_13888;
	_t_13890 = -1.0f * _t_13889;
	_t_13891 = _t_13890 < 0.0f;
	if(_t_13891)
		{
			float _t_13892;
			float _t_13893;
		
			_t_13892 = -1.0f * tx2_5_1;
			_t_13893 = tx3_6_1 + _t_13892;
			_t_13894 = _t_13893;
		
		}
else
		{
			float _t_13895;
			float _t_13896;
			float _t_13897;
		
			_t_13895 = -1.0f * tx2_5_1;
			_t_13896 = tx3_6_1 + _t_13895;
			_t_13897 = -1.0f * _t_13896;
			_t_13894 = _t_13897;
		
		}

	_t_13898 = _t_13894 * _t_539;
	_t_13899 = -1.0f * ty3_9_1;
	_t_13900 = ty2_8_1 + _t_13899;
	_t_13901 = -1.0f * _t_13900;
	_t_13902 = _t_13901 < 0.0f;
	if(_t_13902)
		{
			float _t_13903;
			float _t_13904;
		
			_t_13903 = -1.0f * tx2_5_1;
			_t_13904 = tx3_6_1 + _t_13903;
			_t_13905 = _t_13904;
		
		}
else
		{
			float _t_13906;
			float _t_13907;
			float _t_13908;
		
			_t_13906 = -1.0f * tx2_5_1;
			_t_13907 = tx3_6_1 + _t_13906;
			_t_13908 = -1.0f * _t_13907;
			_t_13905 = _t_13908;
		
		}

	_t_13909 = _t_13905 * _t_539;
	_t_13910 = _t_13898 * _t_13909;
	_t_13911 = -1.0f * ty3_9_1;
	_t_13912 = ty2_8_1 + _t_13911;
	_t_13913 = -1.0f * _t_13912;
	_t_13914 = _t_13913 < 0.0f;
	if(_t_13914)
		{
			float _t_13915;
			float _t_13916;
		
			_t_13915 = -1.0f * ty3_9_1;
			_t_13916 = ty2_8_1 + _t_13915;
			_t_13917 = _t_13916;
		
		}
else
		{
			float _t_13918;
			float _t_13919;
			float _t_13920;
		
			_t_13918 = -1.0f * ty3_9_1;
			_t_13919 = ty2_8_1 + _t_13918;
			_t_13920 = -1.0f * _t_13919;
			_t_13917 = _t_13920;
		
		}

	_t_13921 = _t_13917 * _t_539;
	_t_13922 = 1.0f + _t_13921;
	_t_13923 = 1.0f / _t_13922;
	_t_13924 = _t_13910 * _t_13923;
	_t_13925 = _t_13924 * -1.0f;
	_t_13926 = 1.0f + _t_13925;
	_t_13927 = -1.0f * ty3_9_1;
	_t_13928 = ty2_8_1 + _t_13927;
	_t_13929 = -1.0f * _t_13928;
	_t_13930 = _t_13929 < 0.0f;
	if(_t_13930)
		{
			float _t_13931;
			float _t_13932;
		
			_t_13931 = -1.0f * tx2_5_1;
			_t_13932 = tx3_6_1 + _t_13931;
			_t_13933 = _t_13932;
		
		}
else
		{
			float _t_13934;
			float _t_13935;
			float _t_13936;
		
			_t_13934 = -1.0f * tx2_5_1;
			_t_13935 = tx3_6_1 + _t_13934;
			_t_13936 = -1.0f * _t_13935;
			_t_13933 = _t_13936;
		
		}

	_t_13937 = _t_13933 * _t_539;
	_t_13938 = -1.0f * ty3_9_1;
	_t_13939 = ty2_8_1 + _t_13938;
	_t_13940 = -1.0f * _t_13939;
	_t_13941 = _t_13940 < 0.0f;
	if(_t_13941)
		{
			float _t_13942;
			float _t_13943;
		
			_t_13942 = -1.0f * tx2_5_1;
			_t_13943 = tx3_6_1 + _t_13942;
			_t_13944 = _t_13943;
		
		}
else
		{
			float _t_13945;
			float _t_13946;
			float _t_13947;
		
			_t_13945 = -1.0f * tx2_5_1;
			_t_13946 = tx3_6_1 + _t_13945;
			_t_13947 = -1.0f * _t_13946;
			_t_13944 = _t_13947;
		
		}

	_t_13948 = _t_13944 * _t_539;
	_t_13949 = _t_13937 * _t_13948;
	_t_13950 = -1.0f * ty3_9_1;
	_t_13951 = ty2_8_1 + _t_13950;
	_t_13952 = -1.0f * _t_13951;
	_t_13953 = _t_13952 < 0.0f;
	if(_t_13953)
		{
			float _t_13954;
			float _t_13955;
		
			_t_13954 = -1.0f * ty3_9_1;
			_t_13955 = ty2_8_1 + _t_13954;
			_t_13956 = _t_13955;
		
		}
else
		{
			float _t_13957;
			float _t_13958;
			float _t_13959;
		
			_t_13957 = -1.0f * ty3_9_1;
			_t_13958 = ty2_8_1 + _t_13957;
			_t_13959 = -1.0f * _t_13958;
			_t_13956 = _t_13959;
		
		}

	_t_13960 = _t_13956 * _t_539;
	_t_13961 = 1.0f + _t_13960;
	_t_13962 = 1.0f / _t_13961;
	_t_13963 = _t_13949 * _t_13962;
	_t_13964 = _t_13963 * -1.0f;
	_t_13965 = 1.0f + _t_13964;
	_t_13966 = 0.0f < _t_13965;
	if(_t_13966)
		{
		
			_t_13967 = py0_12_1;
		
		}
else
		{
		
			_t_13967 = py1_13_1;
		
		}

	_t_13968 = _t_13926 * _t_13967;
	_t_13969 = _t_13887 + _t_13968;
	_t_13970 = -1.0f * ty3_9_1;
	_t_13971 = ty2_8_1 + _t_13970;
	_t_13972 = -1.0f * _t_13971;
	_t_13973 = _t_13972 < 0.0f;
	if(_t_13973)
		{
			float _t_13974;
			float _t_13975;
		
			_t_13974 = -1.0f * tx2_5_1;
			_t_13975 = tx3_6_1 + _t_13974;
			_t_13976 = _t_13975;
		
		}
else
		{
			float _t_13977;
			float _t_13978;
			float _t_13979;
		
			_t_13977 = -1.0f * tx2_5_1;
			_t_13978 = tx3_6_1 + _t_13977;
			_t_13979 = -1.0f * _t_13978;
			_t_13976 = _t_13979;
		
		}

	_t_13980 = _t_13976 * _t_539;
	_t_13981 = _t_13980 * -1.0f;
	_t_13982 = -1.0f * ty3_9_1;
	_t_13983 = ty2_8_1 + _t_13982;
	_t_13984 = -1.0f * _t_13983;
	_t_13985 = _t_13984 < 0.0f;
	if(_t_13985)
		{
			float _t_13986;
			float _t_13987;
		
			_t_13986 = -1.0f * tx2_5_1;
			_t_13987 = tx3_6_1 + _t_13986;
			_t_13988 = _t_13987;
		
		}
else
		{
			float _t_13989;
			float _t_13990;
			float _t_13991;
		
			_t_13989 = -1.0f * tx2_5_1;
			_t_13990 = tx3_6_1 + _t_13989;
			_t_13991 = -1.0f * _t_13990;
			_t_13988 = _t_13991;
		
		}

	_t_13992 = _t_13988 * _t_539;
	_t_13993 = _t_13992 * -1.0f;
	_t_13994 = 0.0f < _t_13993;
	if(_t_13994)
		{
		
			_t_13995 = px1_11_1;
		
		}
else
		{
		
			_t_13995 = px0_10_1;
		
		}

	_t_13996 = _t_13981 * _t_13995;
	_t_13997 = -1.0f * ty3_9_1;
	_t_13998 = ty2_8_1 + _t_13997;
	_t_13999 = -1.0f * _t_13998;
	_t_14000 = _t_13999 < 0.0f;
	if(_t_14000)
		{
			float _t_14001;
			float _t_14002;
		
			_t_14001 = -1.0f * tx2_5_1;
			_t_14002 = tx3_6_1 + _t_14001;
			_t_14003 = _t_14002;
		
		}
else
		{
			float _t_14004;
			float _t_14005;
			float _t_14006;
		
			_t_14004 = -1.0f * tx2_5_1;
			_t_14005 = tx3_6_1 + _t_14004;
			_t_14006 = -1.0f * _t_14005;
			_t_14003 = _t_14006;
		
		}

	_t_14007 = _t_14003 * _t_539;
	_t_14008 = -1.0f * ty3_9_1;
	_t_14009 = ty2_8_1 + _t_14008;
	_t_14010 = -1.0f * _t_14009;
	_t_14011 = _t_14010 < 0.0f;
	if(_t_14011)
		{
			float _t_14012;
			float _t_14013;
		
			_t_14012 = -1.0f * tx2_5_1;
			_t_14013 = tx3_6_1 + _t_14012;
			_t_14014 = _t_14013;
		
		}
else
		{
			float _t_14015;
			float _t_14016;
			float _t_14017;
		
			_t_14015 = -1.0f * tx2_5_1;
			_t_14016 = tx3_6_1 + _t_14015;
			_t_14017 = -1.0f * _t_14016;
			_t_14014 = _t_14017;
		
		}

	_t_14018 = _t_14014 * _t_539;
	_t_14019 = _t_14007 * _t_14018;
	_t_14020 = -1.0f * ty3_9_1;
	_t_14021 = ty2_8_1 + _t_14020;
	_t_14022 = -1.0f * _t_14021;
	_t_14023 = _t_14022 < 0.0f;
	if(_t_14023)
		{
			float _t_14024;
			float _t_14025;
		
			_t_14024 = -1.0f * ty3_9_1;
			_t_14025 = ty2_8_1 + _t_14024;
			_t_14026 = _t_14025;
		
		}
else
		{
			float _t_14027;
			float _t_14028;
			float _t_14029;
		
			_t_14027 = -1.0f * ty3_9_1;
			_t_14028 = ty2_8_1 + _t_14027;
			_t_14029 = -1.0f * _t_14028;
			_t_14026 = _t_14029;
		
		}

	_t_14030 = _t_14026 * _t_539;
	_t_14031 = 1.0f + _t_14030;
	_t_14032 = 1.0f / _t_14031;
	_t_14033 = _t_14019 * _t_14032;
	_t_14034 = _t_14033 * -1.0f;
	_t_14035 = 1.0f + _t_14034;
	_t_14036 = -1.0f * ty3_9_1;
	_t_14037 = ty2_8_1 + _t_14036;
	_t_14038 = -1.0f * _t_14037;
	_t_14039 = _t_14038 < 0.0f;
	if(_t_14039)
		{
			float _t_14040;
			float _t_14041;
		
			_t_14040 = -1.0f * tx2_5_1;
			_t_14041 = tx3_6_1 + _t_14040;
			_t_14042 = _t_14041;
		
		}
else
		{
			float _t_14043;
			float _t_14044;
			float _t_14045;
		
			_t_14043 = -1.0f * tx2_5_1;
			_t_14044 = tx3_6_1 + _t_14043;
			_t_14045 = -1.0f * _t_14044;
			_t_14042 = _t_14045;
		
		}

	_t_14046 = _t_14042 * _t_539;
	_t_14047 = -1.0f * ty3_9_1;
	_t_14048 = ty2_8_1 + _t_14047;
	_t_14049 = -1.0f * _t_14048;
	_t_14050 = _t_14049 < 0.0f;
	if(_t_14050)
		{
			float _t_14051;
			float _t_14052;
		
			_t_14051 = -1.0f * tx2_5_1;
			_t_14052 = tx3_6_1 + _t_14051;
			_t_14053 = _t_14052;
		
		}
else
		{
			float _t_14054;
			float _t_14055;
			float _t_14056;
		
			_t_14054 = -1.0f * tx2_5_1;
			_t_14055 = tx3_6_1 + _t_14054;
			_t_14056 = -1.0f * _t_14055;
			_t_14053 = _t_14056;
		
		}

	_t_14057 = _t_14053 * _t_539;
	_t_14058 = _t_14046 * _t_14057;
	_t_14059 = -1.0f * ty3_9_1;
	_t_14060 = ty2_8_1 + _t_14059;
	_t_14061 = -1.0f * _t_14060;
	_t_14062 = _t_14061 < 0.0f;
	if(_t_14062)
		{
			float _t_14063;
			float _t_14064;
		
			_t_14063 = -1.0f * ty3_9_1;
			_t_14064 = ty2_8_1 + _t_14063;
			_t_14065 = _t_14064;
		
		}
else
		{
			float _t_14066;
			float _t_14067;
			float _t_14068;
		
			_t_14066 = -1.0f * ty3_9_1;
			_t_14067 = ty2_8_1 + _t_14066;
			_t_14068 = -1.0f * _t_14067;
			_t_14065 = _t_14068;
		
		}

	_t_14069 = _t_14065 * _t_539;
	_t_14070 = 1.0f + _t_14069;
	_t_14071 = 1.0f / _t_14070;
	_t_14072 = _t_14058 * _t_14071;
	_t_14073 = _t_14072 * -1.0f;
	_t_14074 = 1.0f + _t_14073;
	_t_14075 = 0.0f < _t_14074;
	if(_t_14075)
		{
		
			_t_14076 = py1_13_1;
		
		}
else
		{
		
			_t_14076 = py0_12_1;
		
		}

	_t_14077 = _t_14035 * _t_14076;
	_t_14078 = _t_13996 + _t_14077;
	_t_540 = tegpixelintegrator_32(_t_539,ty3_9_1,pc1_15_1,tc2_19_1,ty2_8_1,_t_14078,ty1_7_1,pc0_14_1,tx3_6_1,tx1_4_1,tx2_5_1,py1_13_1,pc2_16_1,px1_11_1,tc0_17_1,py0_12_1,tc1_18_1,px0_10_1,_t_13969);

	return _t_540;
}
__device__ float tegpixellet_block_52(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float _t_15142,float _t_15195,float ty3_9_1,float tx3_6_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float _t_567,float y__3757_1,float _t_15115){
	float _t_15196;
	float _t_15197;
	float _t_15198;
	float _t_15199;
	float _t_15200;
	float _t_15201;
	float _t_15202;
	float _t_15203;
	float _t_15204;
	float _t_15205;
	float _t_15206;
	float _t_15207;
	float _t_15208;
	float _t_15209;
	float _t_15210;
	float _t_15211;
	float _t_15212;
	float _t_15213;
	float _t_15214;
	float _t_15215;
	float _t_15216;
	float _t_15217;
	float _t_15218;
	bool _t_15219;
	float _t_15220;
	float _t_15221;
	float _t_15222;
	float _t_15223;
	float _t_15224;
	float _t_15225;
	float _t_15226;
	float _t_15227;
	float _t_15228;
	float _t_15229;
	float _t_15230;
	float _t_15231;
	float _t_15232;
	bool _t_15233;
	float _t_15234;
	float _t_15235;
	float _t_15236;
	float _t_15237;
	bool _t_15238;
	bool _t_15239;
	bool _t_15240;
	bool _t_15241;
	bool _t_15242;
	bool _t_15243;
	bool _t_15244;
	float _t_15574;

	float _t_15116;

	_t_15196 = -1.0f * pc0_14_1;
	_t_15197 = tc0_17_1 + _t_15196;
	_t_15198 = _t_15197 * _t_15197;
	_t_15199 = -1.0f * pc1_15_1;
	_t_15200 = tc1_18_1 + _t_15199;
	_t_15201 = _t_15200 * _t_15200;
	_t_15202 = _t_15198 + _t_15201;
	_t_15203 = -1.0f * pc2_16_1;
	_t_15204 = tc2_19_1 + _t_15203;
	_t_15205 = _t_15204 * _t_15204;
	_t_15206 = _t_15202 + _t_15205;
	_t_15207 = tx1_4_1 * ty2_8_1;
	_t_15208 = tx2_5_1 * ty1_7_1;
	_t_15209 = _t_15208 * -1.0f;
	_t_15210 = _t_15207 + _t_15209;
	_t_15211 = -1.0f * ty2_8_1;
	_t_15212 = ty1_7_1 + _t_15211;
	_t_15213 = _t_15212 * _t_15142;
	_t_15214 = _t_15210 + _t_15213;
	_t_15215 = -1.0f * tx1_4_1;
	_t_15216 = tx2_5_1 + _t_15215;
	_t_15217 = _t_15216 * _t_15195;
	_t_15218 = _t_15214 + _t_15217;
	_t_15219 = _t_15218 < 0.0f;
	if(_t_15219)
		{
		
			_t_15220 = 1.0f;
		
		}
else
		{
		
			_t_15220 = 0.0f;
		
		}

	_t_15221 = tx2_5_1 * ty3_9_1;
	_t_15222 = tx3_6_1 * ty2_8_1;
	_t_15223 = _t_15222 * -1.0f;
	_t_15224 = _t_15221 + _t_15223;
	_t_15225 = -1.0f * ty3_9_1;
	_t_15226 = ty2_8_1 + _t_15225;
	_t_15227 = _t_15226 * _t_15142;
	_t_15228 = _t_15224 + _t_15227;
	_t_15229 = -1.0f * tx2_5_1;
	_t_15230 = tx3_6_1 + _t_15229;
	_t_15231 = _t_15230 * _t_15195;
	_t_15232 = _t_15228 + _t_15231;
	_t_15233 = _t_15232 < 0.0f;
	if(_t_15233)
		{
		
			_t_15234 = 1.0f;
		
		}
else
		{
		
			_t_15234 = 0.0f;
		
		}

	_t_15235 = _t_15220 * _t_15234;
	_t_15236 = _t_15206 * _t_15235;
	_t_15237 = _t_15236 * tx1_4_1;
	_t_15238 = py0_12_1 < _t_15195;
	_t_15239 = _t_15195 < py1_13_1;
	_t_15240 = _t_15238 && _t_15239;
	_t_15241 = px0_10_1 < _t_15142;
	_t_15242 = _t_15142 < px1_11_1;
	_t_15243 = _t_15241 && _t_15242;
	_t_15244 = _t_15240 && _t_15243;
	if(_t_15244)
		{
			float _t_15245;
			float _t_15246;
			float _t_15247;
			bool _t_15248;
			float _t_15251;
			float _t_15255;
			float _t_15256;
			float _t_15257;
			float _t_15258;
			float _t_15259;
			bool _t_15260;
			float _t_15263;
			float _t_15267;
			float _t_15268;
			bool _t_15269;
			float _t_15270;
			float _t_15271;
			float _t_15272;
			float _t_15273;
			float _t_15274;
			bool _t_15275;
			float _t_15278;
			float _t_15282;
			float _t_15283;
			float _t_15284;
			float _t_15285;
			bool _t_15286;
			float _t_15289;
			float _t_15293;
			float _t_15294;
			float _t_15295;
			float _t_15296;
			float _t_15297;
			bool _t_15298;
			float _t_15301;
			float _t_15305;
			float _t_15306;
			float _t_15307;
			float _t_15308;
			float _t_15309;
			float _t_15310;
			float _t_15311;
			float _t_15312;
			float _t_15313;
			bool _t_15314;
			float _t_15317;
			float _t_15321;
			float _t_15322;
			float _t_15323;
			float _t_15324;
			bool _t_15325;
			float _t_15328;
			float _t_15332;
			float _t_15333;
			float _t_15334;
			float _t_15335;
			float _t_15336;
			bool _t_15337;
			float _t_15340;
			float _t_15344;
			float _t_15345;
			float _t_15346;
			float _t_15347;
			float _t_15348;
			float _t_15349;
			bool _t_15350;
			float _t_15351;
			float _t_15352;
			float _t_15353;
			bool _t_15354;
			float _t_15355;
			float _t_15356;
			float _t_15357;
			bool _t_15358;
			float _t_15361;
			float _t_15365;
			float _t_15366;
			float _t_15367;
			float _t_15368;
			float _t_15369;
			bool _t_15370;
			float _t_15373;
			float _t_15377;
			float _t_15378;
			bool _t_15379;
			float _t_15380;
			float _t_15381;
			float _t_15382;
			float _t_15383;
			float _t_15384;
			bool _t_15385;
			float _t_15388;
			float _t_15392;
			float _t_15393;
			float _t_15394;
			float _t_15395;
			bool _t_15396;
			float _t_15399;
			float _t_15403;
			float _t_15404;
			float _t_15405;
			float _t_15406;
			float _t_15407;
			bool _t_15408;
			float _t_15411;
			float _t_15415;
			float _t_15416;
			float _t_15417;
			float _t_15418;
			float _t_15419;
			float _t_15420;
			float _t_15421;
			float _t_15422;
			float _t_15423;
			bool _t_15424;
			float _t_15427;
			float _t_15431;
			float _t_15432;
			float _t_15433;
			float _t_15434;
			bool _t_15435;
			float _t_15438;
			float _t_15442;
			float _t_15443;
			float _t_15444;
			float _t_15445;
			float _t_15446;
			bool _t_15447;
			float _t_15450;
			float _t_15454;
			float _t_15455;
			float _t_15456;
			float _t_15457;
			float _t_15458;
			float _t_15459;
			bool _t_15460;
			float _t_15461;
			float _t_15462;
			float _t_15463;
			bool _t_15464;
			bool _t_15465;
			float _t_15466;
			float _t_15467;
			float _t_15468;
			bool _t_15469;
			float _t_15472;
			float _t_15476;
			float _t_15477;
			float _t_15478;
			float _t_15479;
			bool _t_15480;
			float _t_15483;
			float _t_15487;
			bool _t_15488;
			float _t_15489;
			float _t_15490;
			float _t_15491;
			float _t_15492;
			float _t_15493;
			bool _t_15494;
			float _t_15497;
			float _t_15501;
			float _t_15502;
			float _t_15503;
			float _t_15504;
			bool _t_15505;
			float _t_15508;
			float _t_15512;
			bool _t_15513;
			float _t_15514;
			float _t_15515;
			float _t_15516;
			bool _t_15517;
			float _t_15518;
			float _t_15519;
			float _t_15520;
			bool _t_15521;
			float _t_15524;
			float _t_15528;
			float _t_15529;
			float _t_15530;
			float _t_15531;
			bool _t_15532;
			float _t_15535;
			float _t_15539;
			bool _t_15540;
			float _t_15541;
			float _t_15542;
			float _t_15543;
			float _t_15544;
			float _t_15545;
			bool _t_15546;
			float _t_15549;
			float _t_15553;
			float _t_15554;
			float _t_15555;
			float _t_15556;
			bool _t_15557;
			float _t_15560;
			float _t_15564;
			bool _t_15565;
			float _t_15566;
			float _t_15567;
			float _t_15568;
			bool _t_15569;
			bool _t_15570;
			bool _t_15571;
			float _t_15572;
			float _t_15573;
		
			_t_15245 = -1.0f * ty1_7_1;
			_t_15246 = ty3_9_1 + _t_15245;
			_t_15247 = -1.0f * _t_15246;
			_t_15248 = _t_15247 < 0.0f;
			if(_t_15248)
				{
					float _t_15249;
					float _t_15250;
				
					_t_15249 = -1.0f * tx3_6_1;
					_t_15250 = tx1_4_1 + _t_15249;
					_t_15251 = _t_15250;
				
				}
		else
				{
					float _t_15252;
					float _t_15253;
					float _t_15254;
				
					_t_15252 = -1.0f * tx3_6_1;
					_t_15253 = tx1_4_1 + _t_15252;
					_t_15254 = -1.0f * _t_15253;
					_t_15251 = _t_15254;
				
				}
		
			_t_15255 = _t_15251 * _t_567;
			_t_15256 = _t_15255 * -1.0f;
			_t_15257 = -1.0f * ty1_7_1;
			_t_15258 = ty3_9_1 + _t_15257;
			_t_15259 = -1.0f * _t_15258;
			_t_15260 = _t_15259 < 0.0f;
			if(_t_15260)
				{
					float _t_15261;
					float _t_15262;
				
					_t_15261 = -1.0f * tx3_6_1;
					_t_15262 = tx1_4_1 + _t_15261;
					_t_15263 = _t_15262;
				
				}
		else
				{
					float _t_15264;
					float _t_15265;
					float _t_15266;
				
					_t_15264 = -1.0f * tx3_6_1;
					_t_15265 = tx1_4_1 + _t_15264;
					_t_15266 = -1.0f * _t_15265;
					_t_15263 = _t_15266;
				
				}
		
			_t_15267 = _t_15263 * _t_567;
			_t_15268 = _t_15267 * -1.0f;
			_t_15269 = 0.0f < _t_15268;
			if(_t_15269)
				{
				
					_t_15270 = px0_10_1;
				
				}
		else
				{
				
					_t_15270 = px1_11_1;
				
				}
		
			_t_15271 = _t_15256 * _t_15270;
			_t_15272 = -1.0f * ty1_7_1;
			_t_15273 = ty3_9_1 + _t_15272;
			_t_15274 = -1.0f * _t_15273;
			_t_15275 = _t_15274 < 0.0f;
			if(_t_15275)
				{
					float _t_15276;
					float _t_15277;
				
					_t_15276 = -1.0f * tx3_6_1;
					_t_15277 = tx1_4_1 + _t_15276;
					_t_15278 = _t_15277;
				
				}
		else
				{
					float _t_15279;
					float _t_15280;
					float _t_15281;
				
					_t_15279 = -1.0f * tx3_6_1;
					_t_15280 = tx1_4_1 + _t_15279;
					_t_15281 = -1.0f * _t_15280;
					_t_15278 = _t_15281;
				
				}
		
			_t_15282 = _t_15278 * _t_567;
			_t_15283 = -1.0f * ty1_7_1;
			_t_15284 = ty3_9_1 + _t_15283;
			_t_15285 = -1.0f * _t_15284;
			_t_15286 = _t_15285 < 0.0f;
			if(_t_15286)
				{
					float _t_15287;
					float _t_15288;
				
					_t_15287 = -1.0f * tx3_6_1;
					_t_15288 = tx1_4_1 + _t_15287;
					_t_15289 = _t_15288;
				
				}
		else
				{
					float _t_15290;
					float _t_15291;
					float _t_15292;
				
					_t_15290 = -1.0f * tx3_6_1;
					_t_15291 = tx1_4_1 + _t_15290;
					_t_15292 = -1.0f * _t_15291;
					_t_15289 = _t_15292;
				
				}
		
			_t_15293 = _t_15289 * _t_567;
			_t_15294 = _t_15282 * _t_15293;
			_t_15295 = -1.0f * ty1_7_1;
			_t_15296 = ty3_9_1 + _t_15295;
			_t_15297 = -1.0f * _t_15296;
			_t_15298 = _t_15297 < 0.0f;
			if(_t_15298)
				{
					float _t_15299;
					float _t_15300;
				
					_t_15299 = -1.0f * ty1_7_1;
					_t_15300 = ty3_9_1 + _t_15299;
					_t_15301 = _t_15300;
				
				}
		else
				{
					float _t_15302;
					float _t_15303;
					float _t_15304;
				
					_t_15302 = -1.0f * ty1_7_1;
					_t_15303 = ty3_9_1 + _t_15302;
					_t_15304 = -1.0f * _t_15303;
					_t_15301 = _t_15304;
				
				}
		
			_t_15305 = _t_15301 * _t_567;
			_t_15306 = 1.0f + _t_15305;
			_t_15307 = 1.0f / _t_15306;
			_t_15308 = _t_15294 * _t_15307;
			_t_15309 = _t_15308 * -1.0f;
			_t_15310 = 1.0f + _t_15309;
			_t_15311 = -1.0f * ty1_7_1;
			_t_15312 = ty3_9_1 + _t_15311;
			_t_15313 = -1.0f * _t_15312;
			_t_15314 = _t_15313 < 0.0f;
			if(_t_15314)
				{
					float _t_15315;
					float _t_15316;
				
					_t_15315 = -1.0f * tx3_6_1;
					_t_15316 = tx1_4_1 + _t_15315;
					_t_15317 = _t_15316;
				
				}
		else
				{
					float _t_15318;
					float _t_15319;
					float _t_15320;
				
					_t_15318 = -1.0f * tx3_6_1;
					_t_15319 = tx1_4_1 + _t_15318;
					_t_15320 = -1.0f * _t_15319;
					_t_15317 = _t_15320;
				
				}
		
			_t_15321 = _t_15317 * _t_567;
			_t_15322 = -1.0f * ty1_7_1;
			_t_15323 = ty3_9_1 + _t_15322;
			_t_15324 = -1.0f * _t_15323;
			_t_15325 = _t_15324 < 0.0f;
			if(_t_15325)
				{
					float _t_15326;
					float _t_15327;
				
					_t_15326 = -1.0f * tx3_6_1;
					_t_15327 = tx1_4_1 + _t_15326;
					_t_15328 = _t_15327;
				
				}
		else
				{
					float _t_15329;
					float _t_15330;
					float _t_15331;
				
					_t_15329 = -1.0f * tx3_6_1;
					_t_15330 = tx1_4_1 + _t_15329;
					_t_15331 = -1.0f * _t_15330;
					_t_15328 = _t_15331;
				
				}
		
			_t_15332 = _t_15328 * _t_567;
			_t_15333 = _t_15321 * _t_15332;
			_t_15334 = -1.0f * ty1_7_1;
			_t_15335 = ty3_9_1 + _t_15334;
			_t_15336 = -1.0f * _t_15335;
			_t_15337 = _t_15336 < 0.0f;
			if(_t_15337)
				{
					float _t_15338;
					float _t_15339;
				
					_t_15338 = -1.0f * ty1_7_1;
					_t_15339 = ty3_9_1 + _t_15338;
					_t_15340 = _t_15339;
				
				}
		else
				{
					float _t_15341;
					float _t_15342;
					float _t_15343;
				
					_t_15341 = -1.0f * ty1_7_1;
					_t_15342 = ty3_9_1 + _t_15341;
					_t_15343 = -1.0f * _t_15342;
					_t_15340 = _t_15343;
				
				}
		
			_t_15344 = _t_15340 * _t_567;
			_t_15345 = 1.0f + _t_15344;
			_t_15346 = 1.0f / _t_15345;
			_t_15347 = _t_15333 * _t_15346;
			_t_15348 = _t_15347 * -1.0f;
			_t_15349 = 1.0f + _t_15348;
			_t_15350 = 0.0f < _t_15349;
			if(_t_15350)
				{
				
					_t_15351 = py0_12_1;
				
				}
		else
				{
				
					_t_15351 = py1_13_1;
				
				}
		
			_t_15352 = _t_15310 * _t_15351;
			_t_15353 = _t_15271 + _t_15352;
			_t_15354 = _t_15353 < y__3757_1;
			_t_15355 = -1.0f * ty1_7_1;
			_t_15356 = ty3_9_1 + _t_15355;
			_t_15357 = -1.0f * _t_15356;
			_t_15358 = _t_15357 < 0.0f;
			if(_t_15358)
				{
					float _t_15359;
					float _t_15360;
				
					_t_15359 = -1.0f * tx3_6_1;
					_t_15360 = tx1_4_1 + _t_15359;
					_t_15361 = _t_15360;
				
				}
		else
				{
					float _t_15362;
					float _t_15363;
					float _t_15364;
				
					_t_15362 = -1.0f * tx3_6_1;
					_t_15363 = tx1_4_1 + _t_15362;
					_t_15364 = -1.0f * _t_15363;
					_t_15361 = _t_15364;
				
				}
		
			_t_15365 = _t_15361 * _t_567;
			_t_15366 = _t_15365 * -1.0f;
			_t_15367 = -1.0f * ty1_7_1;
			_t_15368 = ty3_9_1 + _t_15367;
			_t_15369 = -1.0f * _t_15368;
			_t_15370 = _t_15369 < 0.0f;
			if(_t_15370)
				{
					float _t_15371;
					float _t_15372;
				
					_t_15371 = -1.0f * tx3_6_1;
					_t_15372 = tx1_4_1 + _t_15371;
					_t_15373 = _t_15372;
				
				}
		else
				{
					float _t_15374;
					float _t_15375;
					float _t_15376;
				
					_t_15374 = -1.0f * tx3_6_1;
					_t_15375 = tx1_4_1 + _t_15374;
					_t_15376 = -1.0f * _t_15375;
					_t_15373 = _t_15376;
				
				}
		
			_t_15377 = _t_15373 * _t_567;
			_t_15378 = _t_15377 * -1.0f;
			_t_15379 = 0.0f < _t_15378;
			if(_t_15379)
				{
				
					_t_15380 = px1_11_1;
				
				}
		else
				{
				
					_t_15380 = px0_10_1;
				
				}
		
			_t_15381 = _t_15366 * _t_15380;
			_t_15382 = -1.0f * ty1_7_1;
			_t_15383 = ty3_9_1 + _t_15382;
			_t_15384 = -1.0f * _t_15383;
			_t_15385 = _t_15384 < 0.0f;
			if(_t_15385)
				{
					float _t_15386;
					float _t_15387;
				
					_t_15386 = -1.0f * tx3_6_1;
					_t_15387 = tx1_4_1 + _t_15386;
					_t_15388 = _t_15387;
				
				}
		else
				{
					float _t_15389;
					float _t_15390;
					float _t_15391;
				
					_t_15389 = -1.0f * tx3_6_1;
					_t_15390 = tx1_4_1 + _t_15389;
					_t_15391 = -1.0f * _t_15390;
					_t_15388 = _t_15391;
				
				}
		
			_t_15392 = _t_15388 * _t_567;
			_t_15393 = -1.0f * ty1_7_1;
			_t_15394 = ty3_9_1 + _t_15393;
			_t_15395 = -1.0f * _t_15394;
			_t_15396 = _t_15395 < 0.0f;
			if(_t_15396)
				{
					float _t_15397;
					float _t_15398;
				
					_t_15397 = -1.0f * tx3_6_1;
					_t_15398 = tx1_4_1 + _t_15397;
					_t_15399 = _t_15398;
				
				}
		else
				{
					float _t_15400;
					float _t_15401;
					float _t_15402;
				
					_t_15400 = -1.0f * tx3_6_1;
					_t_15401 = tx1_4_1 + _t_15400;
					_t_15402 = -1.0f * _t_15401;
					_t_15399 = _t_15402;
				
				}
		
			_t_15403 = _t_15399 * _t_567;
			_t_15404 = _t_15392 * _t_15403;
			_t_15405 = -1.0f * ty1_7_1;
			_t_15406 = ty3_9_1 + _t_15405;
			_t_15407 = -1.0f * _t_15406;
			_t_15408 = _t_15407 < 0.0f;
			if(_t_15408)
				{
					float _t_15409;
					float _t_15410;
				
					_t_15409 = -1.0f * ty1_7_1;
					_t_15410 = ty3_9_1 + _t_15409;
					_t_15411 = _t_15410;
				
				}
		else
				{
					float _t_15412;
					float _t_15413;
					float _t_15414;
				
					_t_15412 = -1.0f * ty1_7_1;
					_t_15413 = ty3_9_1 + _t_15412;
					_t_15414 = -1.0f * _t_15413;
					_t_15411 = _t_15414;
				
				}
		
			_t_15415 = _t_15411 * _t_567;
			_t_15416 = 1.0f + _t_15415;
			_t_15417 = 1.0f / _t_15416;
			_t_15418 = _t_15404 * _t_15417;
			_t_15419 = _t_15418 * -1.0f;
			_t_15420 = 1.0f + _t_15419;
			_t_15421 = -1.0f * ty1_7_1;
			_t_15422 = ty3_9_1 + _t_15421;
			_t_15423 = -1.0f * _t_15422;
			_t_15424 = _t_15423 < 0.0f;
			if(_t_15424)
				{
					float _t_15425;
					float _t_15426;
				
					_t_15425 = -1.0f * tx3_6_1;
					_t_15426 = tx1_4_1 + _t_15425;
					_t_15427 = _t_15426;
				
				}
		else
				{
					float _t_15428;
					float _t_15429;
					float _t_15430;
				
					_t_15428 = -1.0f * tx3_6_1;
					_t_15429 = tx1_4_1 + _t_15428;
					_t_15430 = -1.0f * _t_15429;
					_t_15427 = _t_15430;
				
				}
		
			_t_15431 = _t_15427 * _t_567;
			_t_15432 = -1.0f * ty1_7_1;
			_t_15433 = ty3_9_1 + _t_15432;
			_t_15434 = -1.0f * _t_15433;
			_t_15435 = _t_15434 < 0.0f;
			if(_t_15435)
				{
					float _t_15436;
					float _t_15437;
				
					_t_15436 = -1.0f * tx3_6_1;
					_t_15437 = tx1_4_1 + _t_15436;
					_t_15438 = _t_15437;
				
				}
		else
				{
					float _t_15439;
					float _t_15440;
					float _t_15441;
				
					_t_15439 = -1.0f * tx3_6_1;
					_t_15440 = tx1_4_1 + _t_15439;
					_t_15441 = -1.0f * _t_15440;
					_t_15438 = _t_15441;
				
				}
		
			_t_15442 = _t_15438 * _t_567;
			_t_15443 = _t_15431 * _t_15442;
			_t_15444 = -1.0f * ty1_7_1;
			_t_15445 = ty3_9_1 + _t_15444;
			_t_15446 = -1.0f * _t_15445;
			_t_15447 = _t_15446 < 0.0f;
			if(_t_15447)
				{
					float _t_15448;
					float _t_15449;
				
					_t_15448 = -1.0f * ty1_7_1;
					_t_15449 = ty3_9_1 + _t_15448;
					_t_15450 = _t_15449;
				
				}
		else
				{
					float _t_15451;
					float _t_15452;
					float _t_15453;
				
					_t_15451 = -1.0f * ty1_7_1;
					_t_15452 = ty3_9_1 + _t_15451;
					_t_15453 = -1.0f * _t_15452;
					_t_15450 = _t_15453;
				
				}
		
			_t_15454 = _t_15450 * _t_567;
			_t_15455 = 1.0f + _t_15454;
			_t_15456 = 1.0f / _t_15455;
			_t_15457 = _t_15443 * _t_15456;
			_t_15458 = _t_15457 * -1.0f;
			_t_15459 = 1.0f + _t_15458;
			_t_15460 = 0.0f < _t_15459;
			if(_t_15460)
				{
				
					_t_15461 = py1_13_1;
				
				}
		else
				{
				
					_t_15461 = py0_12_1;
				
				}
		
			_t_15462 = _t_15420 * _t_15461;
			_t_15463 = _t_15381 + _t_15462;
			_t_15464 = y__3757_1 < _t_15463;
			_t_15465 = _t_15354 && _t_15464;
			_t_15466 = -1.0f * ty1_7_1;
			_t_15467 = ty3_9_1 + _t_15466;
			_t_15468 = -1.0f * _t_15467;
			_t_15469 = _t_15468 < 0.0f;
			if(_t_15469)
				{
					float _t_15470;
					float _t_15471;
				
					_t_15470 = -1.0f * ty1_7_1;
					_t_15471 = ty3_9_1 + _t_15470;
					_t_15472 = _t_15471;
				
				}
		else
				{
					float _t_15473;
					float _t_15474;
					float _t_15475;
				
					_t_15473 = -1.0f * ty1_7_1;
					_t_15474 = ty3_9_1 + _t_15473;
					_t_15475 = -1.0f * _t_15474;
					_t_15472 = _t_15475;
				
				}
		
			_t_15476 = _t_15472 * _t_567;
			_t_15477 = -1.0f * ty1_7_1;
			_t_15478 = ty3_9_1 + _t_15477;
			_t_15479 = -1.0f * _t_15478;
			_t_15480 = _t_15479 < 0.0f;
			if(_t_15480)
				{
					float _t_15481;
					float _t_15482;
				
					_t_15481 = -1.0f * ty1_7_1;
					_t_15482 = ty3_9_1 + _t_15481;
					_t_15483 = _t_15482;
				
				}
		else
				{
					float _t_15484;
					float _t_15485;
					float _t_15486;
				
					_t_15484 = -1.0f * ty1_7_1;
					_t_15485 = ty3_9_1 + _t_15484;
					_t_15486 = -1.0f * _t_15485;
					_t_15483 = _t_15486;
				
				}
		
			_t_15487 = _t_15483 * _t_567;
			_t_15488 = 0.0f < _t_15487;
			if(_t_15488)
				{
				
					_t_15489 = px0_10_1;
				
				}
		else
				{
				
					_t_15489 = px1_11_1;
				
				}
		
			_t_15490 = _t_15476 * _t_15489;
			_t_15491 = -1.0f * ty1_7_1;
			_t_15492 = ty3_9_1 + _t_15491;
			_t_15493 = -1.0f * _t_15492;
			_t_15494 = _t_15493 < 0.0f;
			if(_t_15494)
				{
					float _t_15495;
					float _t_15496;
				
					_t_15495 = -1.0f * tx3_6_1;
					_t_15496 = tx1_4_1 + _t_15495;
					_t_15497 = _t_15496;
				
				}
		else
				{
					float _t_15498;
					float _t_15499;
					float _t_15500;
				
					_t_15498 = -1.0f * tx3_6_1;
					_t_15499 = tx1_4_1 + _t_15498;
					_t_15500 = -1.0f * _t_15499;
					_t_15497 = _t_15500;
				
				}
		
			_t_15501 = _t_15497 * _t_567;
			_t_15502 = -1.0f * ty1_7_1;
			_t_15503 = ty3_9_1 + _t_15502;
			_t_15504 = -1.0f * _t_15503;
			_t_15505 = _t_15504 < 0.0f;
			if(_t_15505)
				{
					float _t_15506;
					float _t_15507;
				
					_t_15506 = -1.0f * tx3_6_1;
					_t_15507 = tx1_4_1 + _t_15506;
					_t_15508 = _t_15507;
				
				}
		else
				{
					float _t_15509;
					float _t_15510;
					float _t_15511;
				
					_t_15509 = -1.0f * tx3_6_1;
					_t_15510 = tx1_4_1 + _t_15509;
					_t_15511 = -1.0f * _t_15510;
					_t_15508 = _t_15511;
				
				}
		
			_t_15512 = _t_15508 * _t_567;
			_t_15513 = 0.0f < _t_15512;
			if(_t_15513)
				{
				
					_t_15514 = py0_12_1;
				
				}
		else
				{
				
					_t_15514 = py1_13_1;
				
				}
		
			_t_15515 = _t_15501 * _t_15514;
			_t_15516 = _t_15490 + _t_15515;
			_t_15517 = _t_15516 < _t_15115;
			_t_15518 = -1.0f * ty1_7_1;
			_t_15519 = ty3_9_1 + _t_15518;
			_t_15520 = -1.0f * _t_15519;
			_t_15521 = _t_15520 < 0.0f;
			if(_t_15521)
				{
					float _t_15522;
					float _t_15523;
				
					_t_15522 = -1.0f * ty1_7_1;
					_t_15523 = ty3_9_1 + _t_15522;
					_t_15524 = _t_15523;
				
				}
		else
				{
					float _t_15525;
					float _t_15526;
					float _t_15527;
				
					_t_15525 = -1.0f * ty1_7_1;
					_t_15526 = ty3_9_1 + _t_15525;
					_t_15527 = -1.0f * _t_15526;
					_t_15524 = _t_15527;
				
				}
		
			_t_15528 = _t_15524 * _t_567;
			_t_15529 = -1.0f * ty1_7_1;
			_t_15530 = ty3_9_1 + _t_15529;
			_t_15531 = -1.0f * _t_15530;
			_t_15532 = _t_15531 < 0.0f;
			if(_t_15532)
				{
					float _t_15533;
					float _t_15534;
				
					_t_15533 = -1.0f * ty1_7_1;
					_t_15534 = ty3_9_1 + _t_15533;
					_t_15535 = _t_15534;
				
				}
		else
				{
					float _t_15536;
					float _t_15537;
					float _t_15538;
				
					_t_15536 = -1.0f * ty1_7_1;
					_t_15537 = ty3_9_1 + _t_15536;
					_t_15538 = -1.0f * _t_15537;
					_t_15535 = _t_15538;
				
				}
		
			_t_15539 = _t_15535 * _t_567;
			_t_15540 = 0.0f < _t_15539;
			if(_t_15540)
				{
				
					_t_15541 = px1_11_1;
				
				}
		else
				{
				
					_t_15541 = px0_10_1;
				
				}
		
			_t_15542 = _t_15528 * _t_15541;
			_t_15543 = -1.0f * ty1_7_1;
			_t_15544 = ty3_9_1 + _t_15543;
			_t_15545 = -1.0f * _t_15544;
			_t_15546 = _t_15545 < 0.0f;
			if(_t_15546)
				{
					float _t_15547;
					float _t_15548;
				
					_t_15547 = -1.0f * tx3_6_1;
					_t_15548 = tx1_4_1 + _t_15547;
					_t_15549 = _t_15548;
				
				}
		else
				{
					float _t_15550;
					float _t_15551;
					float _t_15552;
				
					_t_15550 = -1.0f * tx3_6_1;
					_t_15551 = tx1_4_1 + _t_15550;
					_t_15552 = -1.0f * _t_15551;
					_t_15549 = _t_15552;
				
				}
		
			_t_15553 = _t_15549 * _t_567;
			_t_15554 = -1.0f * ty1_7_1;
			_t_15555 = ty3_9_1 + _t_15554;
			_t_15556 = -1.0f * _t_15555;
			_t_15557 = _t_15556 < 0.0f;
			if(_t_15557)
				{
					float _t_15558;
					float _t_15559;
				
					_t_15558 = -1.0f * tx3_6_1;
					_t_15559 = tx1_4_1 + _t_15558;
					_t_15560 = _t_15559;
				
				}
		else
				{
					float _t_15561;
					float _t_15562;
					float _t_15563;
				
					_t_15561 = -1.0f * tx3_6_1;
					_t_15562 = tx1_4_1 + _t_15561;
					_t_15563 = -1.0f * _t_15562;
					_t_15560 = _t_15563;
				
				}
		
			_t_15564 = _t_15560 * _t_567;
			_t_15565 = 0.0f < _t_15564;
			if(_t_15565)
				{
				
					_t_15566 = py1_13_1;
				
				}
		else
				{
				
					_t_15566 = py0_12_1;
				
				}
		
			_t_15567 = _t_15553 * _t_15566;
			_t_15568 = _t_15542 + _t_15567;
			_t_15569 = _t_15115 < _t_15568;
			_t_15570 = _t_15517 && _t_15569;
			_t_15571 = _t_15465 && _t_15570;
			if(_t_15571)
				{
				
					_t_15572 = 1.0f;
				
				}
		else
				{
				
					_t_15572 = 0.0f;
				
				}
		
			_t_15573 = _t_15572 * _t_567;
			_t_15574 = _t_15573;
		
		}
else
		{
		
			_t_15574 = 0.0f;
		
		}

	_t_15116 = _t_15237 * _t_15574;

	return _t_15116;
}
__device__ float tegpixellet_block_51(float ty3_9_1,float ty1_7_1,float _t_567,float _t_15115,float tx1_4_1,float tx3_6_1,float y__3757_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1){
	float _t_15117;
	float _t_15118;
	float _t_15119;
	bool _t_15120;
	float _t_15123;
	float _t_15127;
	float _t_15128;
	float _t_15129;
	float _t_15130;
	float _t_15131;
	bool _t_15132;
	float _t_15135;
	float _t_15139;
	float _t_15140;
	float _t_15141;
	float _t_15142;
	float _t_15143;
	float _t_15144;
	float _t_15145;
	bool _t_15146;
	float _t_15149;
	float _t_15153;
	float _t_15154;
	float _t_15155;
	float _t_15156;
	bool _t_15157;
	float _t_15160;
	float _t_15164;
	float _t_15165;
	float _t_15166;
	float _t_15167;
	float _t_15168;
	bool _t_15169;
	float _t_15172;
	float _t_15176;
	float _t_15177;
	float _t_15178;
	float _t_15179;
	float _t_15180;
	float _t_15181;
	float _t_15182;
	float _t_15183;
	float _t_15184;
	float _t_15185;
	bool _t_15186;
	float _t_15189;
	float _t_15193;
	float _t_15194;
	float _t_15195;

	float _t_15116;

	_t_15117 = -1.0f * ty1_7_1;
	_t_15118 = ty3_9_1 + _t_15117;
	_t_15119 = -1.0f * _t_15118;
	_t_15120 = _t_15119 < 0.0f;
	if(_t_15120)
		{
			float _t_15121;
			float _t_15122;
		
			_t_15121 = -1.0f * ty1_7_1;
			_t_15122 = ty3_9_1 + _t_15121;
			_t_15123 = _t_15122;
		
		}
else
		{
			float _t_15124;
			float _t_15125;
			float _t_15126;
		
			_t_15124 = -1.0f * ty1_7_1;
			_t_15125 = ty3_9_1 + _t_15124;
			_t_15126 = -1.0f * _t_15125;
			_t_15123 = _t_15126;
		
		}

	_t_15127 = _t_15123 * _t_567;
	_t_15128 = _t_15127 * _t_15115;
	_t_15129 = -1.0f * ty1_7_1;
	_t_15130 = ty3_9_1 + _t_15129;
	_t_15131 = -1.0f * _t_15130;
	_t_15132 = _t_15131 < 0.0f;
	if(_t_15132)
		{
			float _t_15133;
			float _t_15134;
		
			_t_15133 = -1.0f * tx3_6_1;
			_t_15134 = tx1_4_1 + _t_15133;
			_t_15135 = _t_15134;
		
		}
else
		{
			float _t_15136;
			float _t_15137;
			float _t_15138;
		
			_t_15136 = -1.0f * tx3_6_1;
			_t_15137 = tx1_4_1 + _t_15136;
			_t_15138 = -1.0f * _t_15137;
			_t_15135 = _t_15138;
		
		}

	_t_15139 = _t_15135 * _t_567;
	_t_15140 = _t_15139 * -1.0f;
	_t_15141 = _t_15140 * y__3757_1;
	_t_15142 = _t_15128 + _t_15141;
	_t_15143 = -1.0f * ty1_7_1;
	_t_15144 = ty3_9_1 + _t_15143;
	_t_15145 = -1.0f * _t_15144;
	_t_15146 = _t_15145 < 0.0f;
	if(_t_15146)
		{
			float _t_15147;
			float _t_15148;
		
			_t_15147 = -1.0f * tx3_6_1;
			_t_15148 = tx1_4_1 + _t_15147;
			_t_15149 = _t_15148;
		
		}
else
		{
			float _t_15150;
			float _t_15151;
			float _t_15152;
		
			_t_15150 = -1.0f * tx3_6_1;
			_t_15151 = tx1_4_1 + _t_15150;
			_t_15152 = -1.0f * _t_15151;
			_t_15149 = _t_15152;
		
		}

	_t_15153 = _t_15149 * _t_567;
	_t_15154 = -1.0f * ty1_7_1;
	_t_15155 = ty3_9_1 + _t_15154;
	_t_15156 = -1.0f * _t_15155;
	_t_15157 = _t_15156 < 0.0f;
	if(_t_15157)
		{
			float _t_15158;
			float _t_15159;
		
			_t_15158 = -1.0f * tx3_6_1;
			_t_15159 = tx1_4_1 + _t_15158;
			_t_15160 = _t_15159;
		
		}
else
		{
			float _t_15161;
			float _t_15162;
			float _t_15163;
		
			_t_15161 = -1.0f * tx3_6_1;
			_t_15162 = tx1_4_1 + _t_15161;
			_t_15163 = -1.0f * _t_15162;
			_t_15160 = _t_15163;
		
		}

	_t_15164 = _t_15160 * _t_567;
	_t_15165 = _t_15153 * _t_15164;
	_t_15166 = -1.0f * ty1_7_1;
	_t_15167 = ty3_9_1 + _t_15166;
	_t_15168 = -1.0f * _t_15167;
	_t_15169 = _t_15168 < 0.0f;
	if(_t_15169)
		{
			float _t_15170;
			float _t_15171;
		
			_t_15170 = -1.0f * ty1_7_1;
			_t_15171 = ty3_9_1 + _t_15170;
			_t_15172 = _t_15171;
		
		}
else
		{
			float _t_15173;
			float _t_15174;
			float _t_15175;
		
			_t_15173 = -1.0f * ty1_7_1;
			_t_15174 = ty3_9_1 + _t_15173;
			_t_15175 = -1.0f * _t_15174;
			_t_15172 = _t_15175;
		
		}

	_t_15176 = _t_15172 * _t_567;
	_t_15177 = 1.0f + _t_15176;
	_t_15178 = 1.0f / _t_15177;
	_t_15179 = _t_15165 * _t_15178;
	_t_15180 = _t_15179 * -1.0f;
	_t_15181 = 1.0f + _t_15180;
	_t_15182 = _t_15181 * y__3757_1;
	_t_15183 = -1.0f * ty1_7_1;
	_t_15184 = ty3_9_1 + _t_15183;
	_t_15185 = -1.0f * _t_15184;
	_t_15186 = _t_15185 < 0.0f;
	if(_t_15186)
		{
			float _t_15187;
			float _t_15188;
		
			_t_15187 = -1.0f * tx3_6_1;
			_t_15188 = tx1_4_1 + _t_15187;
			_t_15189 = _t_15188;
		
		}
else
		{
			float _t_15190;
			float _t_15191;
			float _t_15192;
		
			_t_15190 = -1.0f * tx3_6_1;
			_t_15191 = tx1_4_1 + _t_15190;
			_t_15192 = -1.0f * _t_15191;
			_t_15189 = _t_15192;
		
		}

	_t_15193 = _t_15189 * _t_567;
	_t_15194 = _t_15193 * _t_15115;
	_t_15195 = _t_15182 + _t_15194;
	_t_15116 = tegpixellet_block_52(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,_t_15142,_t_15195,ty3_9_1,tx3_6_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,_t_567,y__3757_1,_t_15115);

	return _t_15116;
}
__device__ float tegpixelbody_block_33(float ty3_9_1,float ty1_7_1,float _t_567,float px0_10_1,float px1_11_1,float tx1_4_1,float tx3_6_1,float py0_12_1,float py1_13_1,float y__3757_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_14959;
	float _t_14960;
	float _t_14961;
	bool _t_14962;
	float _t_14965;
	float _t_14969;
	float _t_14970;
	float _t_14971;
	float _t_14972;
	bool _t_14973;
	float _t_14976;
	float _t_14980;
	bool _t_14981;
	float _t_14982;
	float _t_14983;
	float _t_14984;
	float _t_14985;
	float _t_14986;
	bool _t_14987;
	float _t_14990;
	float _t_14994;
	float _t_14995;
	float _t_14996;
	float _t_14997;
	bool _t_14998;
	float _t_15001;
	float _t_15005;
	bool _t_15006;
	float _t_15007;
	float _t_15008;
	float _t_15009;
	float _t_15010;
	float _t_15011;
	float _t_15012;
	bool _t_15013;
	float _t_15018;
	float _t_15024;
	float _t_15025;
	float _t_15026;
	float _t_15027;
	bool _t_15028;
	float _t_15029;
	float _t_15030;
	float _t_15031;
	bool _t_15032;
	float _t_15035;
	float _t_15039;
	float _t_15040;
	float _t_15041;
	float _t_15042;
	bool _t_15043;
	float _t_15046;
	float _t_15050;
	bool _t_15051;
	float _t_15052;
	float _t_15053;
	float _t_15054;
	float _t_15055;
	float _t_15056;
	bool _t_15057;
	float _t_15060;
	float _t_15064;
	float _t_15065;
	float _t_15066;
	float _t_15067;
	bool _t_15068;
	float _t_15071;
	float _t_15075;
	bool _t_15076;
	float _t_15077;
	float _t_15078;
	float _t_15079;
	float _t_15080;
	float _t_15081;
	float _t_15082;
	bool _t_15083;
	float _t_15088;
	float _t_15094;
	float _t_15095;
	float _t_15096;
	float _t_15097;
	bool _t_15098;
	bool _t_15099;

	float _t_14958;

	_t_14959 = -1.0f * ty1_7_1;
	_t_14960 = ty3_9_1 + _t_14959;
	_t_14961 = -1.0f * _t_14960;
	_t_14962 = _t_14961 < 0.0f;
	if(_t_14962)
		{
			float _t_14963;
			float _t_14964;
		
			_t_14963 = -1.0f * ty1_7_1;
			_t_14964 = ty3_9_1 + _t_14963;
			_t_14965 = _t_14964;
		
		}
else
		{
			float _t_14966;
			float _t_14967;
			float _t_14968;
		
			_t_14966 = -1.0f * ty1_7_1;
			_t_14967 = ty3_9_1 + _t_14966;
			_t_14968 = -1.0f * _t_14967;
			_t_14965 = _t_14968;
		
		}

	_t_14969 = _t_14965 * _t_567;
	_t_14970 = -1.0f * ty1_7_1;
	_t_14971 = ty3_9_1 + _t_14970;
	_t_14972 = -1.0f * _t_14971;
	_t_14973 = _t_14972 < 0.0f;
	if(_t_14973)
		{
			float _t_14974;
			float _t_14975;
		
			_t_14974 = -1.0f * ty1_7_1;
			_t_14975 = ty3_9_1 + _t_14974;
			_t_14976 = _t_14975;
		
		}
else
		{
			float _t_14977;
			float _t_14978;
			float _t_14979;
		
			_t_14977 = -1.0f * ty1_7_1;
			_t_14978 = ty3_9_1 + _t_14977;
			_t_14979 = -1.0f * _t_14978;
			_t_14976 = _t_14979;
		
		}

	_t_14980 = _t_14976 * _t_567;
	_t_14981 = 0.0f < _t_14980;
	if(_t_14981)
		{
		
			_t_14982 = px0_10_1;
		
		}
else
		{
		
			_t_14982 = px1_11_1;
		
		}

	_t_14983 = _t_14969 * _t_14982;
	_t_14984 = -1.0f * ty1_7_1;
	_t_14985 = ty3_9_1 + _t_14984;
	_t_14986 = -1.0f * _t_14985;
	_t_14987 = _t_14986 < 0.0f;
	if(_t_14987)
		{
			float _t_14988;
			float _t_14989;
		
			_t_14988 = -1.0f * tx3_6_1;
			_t_14989 = tx1_4_1 + _t_14988;
			_t_14990 = _t_14989;
		
		}
else
		{
			float _t_14991;
			float _t_14992;
			float _t_14993;
		
			_t_14991 = -1.0f * tx3_6_1;
			_t_14992 = tx1_4_1 + _t_14991;
			_t_14993 = -1.0f * _t_14992;
			_t_14990 = _t_14993;
		
		}

	_t_14994 = _t_14990 * _t_567;
	_t_14995 = -1.0f * ty1_7_1;
	_t_14996 = ty3_9_1 + _t_14995;
	_t_14997 = -1.0f * _t_14996;
	_t_14998 = _t_14997 < 0.0f;
	if(_t_14998)
		{
			float _t_14999;
			float _t_15000;
		
			_t_14999 = -1.0f * tx3_6_1;
			_t_15000 = tx1_4_1 + _t_14999;
			_t_15001 = _t_15000;
		
		}
else
		{
			float _t_15002;
			float _t_15003;
			float _t_15004;
		
			_t_15002 = -1.0f * tx3_6_1;
			_t_15003 = tx1_4_1 + _t_15002;
			_t_15004 = -1.0f * _t_15003;
			_t_15001 = _t_15004;
		
		}

	_t_15005 = _t_15001 * _t_567;
	_t_15006 = 0.0f < _t_15005;
	if(_t_15006)
		{
		
			_t_15007 = py0_12_1;
		
		}
else
		{
		
			_t_15007 = py1_13_1;
		
		}

	_t_15008 = _t_14994 * _t_15007;
	_t_15009 = _t_14983 + _t_15008;
	_t_15010 = -1.0f * ty1_7_1;
	_t_15011 = ty3_9_1 + _t_15010;
	_t_15012 = -1.0f * _t_15011;
	_t_15013 = _t_15012 < 0.0f;
	if(_t_15013)
		{
			float _t_15014;
			float _t_15015;
			float _t_15016;
			float _t_15017;
		
			_t_15014 = tx3_6_1 * ty1_7_1;
			_t_15015 = tx1_4_1 * ty3_9_1;
			_t_15016 = _t_15015 * -1.0f;
			_t_15017 = _t_15014 + _t_15016;
			_t_15018 = _t_15017;
		
		}
else
		{
			float _t_15019;
			float _t_15020;
			float _t_15021;
			float _t_15022;
			float _t_15023;
		
			_t_15019 = tx3_6_1 * ty1_7_1;
			_t_15020 = tx1_4_1 * ty3_9_1;
			_t_15021 = _t_15020 * -1.0f;
			_t_15022 = _t_15019 + _t_15021;
			_t_15023 = -1.0f * _t_15022;
			_t_15018 = _t_15023;
		
		}

	_t_15024 = -1.0f * _t_15018;
	_t_15025 = _t_15024 * _t_567;
	_t_15026 = _t_15025 * -1.0f;
	_t_15027 = _t_15009 + _t_15026;
	_t_15028 = _t_15027 < 0.0f;
	_t_15029 = -1.0f * ty1_7_1;
	_t_15030 = ty3_9_1 + _t_15029;
	_t_15031 = -1.0f * _t_15030;
	_t_15032 = _t_15031 < 0.0f;
	if(_t_15032)
		{
			float _t_15033;
			float _t_15034;
		
			_t_15033 = -1.0f * ty1_7_1;
			_t_15034 = ty3_9_1 + _t_15033;
			_t_15035 = _t_15034;
		
		}
else
		{
			float _t_15036;
			float _t_15037;
			float _t_15038;
		
			_t_15036 = -1.0f * ty1_7_1;
			_t_15037 = ty3_9_1 + _t_15036;
			_t_15038 = -1.0f * _t_15037;
			_t_15035 = _t_15038;
		
		}

	_t_15039 = _t_15035 * _t_567;
	_t_15040 = -1.0f * ty1_7_1;
	_t_15041 = ty3_9_1 + _t_15040;
	_t_15042 = -1.0f * _t_15041;
	_t_15043 = _t_15042 < 0.0f;
	if(_t_15043)
		{
			float _t_15044;
			float _t_15045;
		
			_t_15044 = -1.0f * ty1_7_1;
			_t_15045 = ty3_9_1 + _t_15044;
			_t_15046 = _t_15045;
		
		}
else
		{
			float _t_15047;
			float _t_15048;
			float _t_15049;
		
			_t_15047 = -1.0f * ty1_7_1;
			_t_15048 = ty3_9_1 + _t_15047;
			_t_15049 = -1.0f * _t_15048;
			_t_15046 = _t_15049;
		
		}

	_t_15050 = _t_15046 * _t_567;
	_t_15051 = 0.0f < _t_15050;
	if(_t_15051)
		{
		
			_t_15052 = px1_11_1;
		
		}
else
		{
		
			_t_15052 = px0_10_1;
		
		}

	_t_15053 = _t_15039 * _t_15052;
	_t_15054 = -1.0f * ty1_7_1;
	_t_15055 = ty3_9_1 + _t_15054;
	_t_15056 = -1.0f * _t_15055;
	_t_15057 = _t_15056 < 0.0f;
	if(_t_15057)
		{
			float _t_15058;
			float _t_15059;
		
			_t_15058 = -1.0f * tx3_6_1;
			_t_15059 = tx1_4_1 + _t_15058;
			_t_15060 = _t_15059;
		
		}
else
		{
			float _t_15061;
			float _t_15062;
			float _t_15063;
		
			_t_15061 = -1.0f * tx3_6_1;
			_t_15062 = tx1_4_1 + _t_15061;
			_t_15063 = -1.0f * _t_15062;
			_t_15060 = _t_15063;
		
		}

	_t_15064 = _t_15060 * _t_567;
	_t_15065 = -1.0f * ty1_7_1;
	_t_15066 = ty3_9_1 + _t_15065;
	_t_15067 = -1.0f * _t_15066;
	_t_15068 = _t_15067 < 0.0f;
	if(_t_15068)
		{
			float _t_15069;
			float _t_15070;
		
			_t_15069 = -1.0f * tx3_6_1;
			_t_15070 = tx1_4_1 + _t_15069;
			_t_15071 = _t_15070;
		
		}
else
		{
			float _t_15072;
			float _t_15073;
			float _t_15074;
		
			_t_15072 = -1.0f * tx3_6_1;
			_t_15073 = tx1_4_1 + _t_15072;
			_t_15074 = -1.0f * _t_15073;
			_t_15071 = _t_15074;
		
		}

	_t_15075 = _t_15071 * _t_567;
	_t_15076 = 0.0f < _t_15075;
	if(_t_15076)
		{
		
			_t_15077 = py1_13_1;
		
		}
else
		{
		
			_t_15077 = py0_12_1;
		
		}

	_t_15078 = _t_15064 * _t_15077;
	_t_15079 = _t_15053 + _t_15078;
	_t_15080 = -1.0f * ty1_7_1;
	_t_15081 = ty3_9_1 + _t_15080;
	_t_15082 = -1.0f * _t_15081;
	_t_15083 = _t_15082 < 0.0f;
	if(_t_15083)
		{
			float _t_15084;
			float _t_15085;
			float _t_15086;
			float _t_15087;
		
			_t_15084 = tx3_6_1 * ty1_7_1;
			_t_15085 = tx1_4_1 * ty3_9_1;
			_t_15086 = _t_15085 * -1.0f;
			_t_15087 = _t_15084 + _t_15086;
			_t_15088 = _t_15087;
		
		}
else
		{
			float _t_15089;
			float _t_15090;
			float _t_15091;
			float _t_15092;
			float _t_15093;
		
			_t_15089 = tx3_6_1 * ty1_7_1;
			_t_15090 = tx1_4_1 * ty3_9_1;
			_t_15091 = _t_15090 * -1.0f;
			_t_15092 = _t_15089 + _t_15091;
			_t_15093 = -1.0f * _t_15092;
			_t_15088 = _t_15093;
		
		}

	_t_15094 = -1.0f * _t_15088;
	_t_15095 = _t_15094 * _t_567;
	_t_15096 = _t_15095 * -1.0f;
	_t_15097 = _t_15079 + _t_15096;
	_t_15098 = 0.0f < _t_15097;
	_t_15099 = _t_15028 && _t_15098;
	if(_t_15099)
		{
			float _t_15100;
			float _t_15101;
			float _t_15102;
			bool _t_15103;
			float _t_15108;
			float _t_15114;
			float _t_15115;
			float _t_15116;
		
			_t_15100 = -1.0f * ty1_7_1;
			_t_15101 = ty3_9_1 + _t_15100;
			_t_15102 = -1.0f * _t_15101;
			_t_15103 = _t_15102 < 0.0f;
			if(_t_15103)
				{
					float _t_15104;
					float _t_15105;
					float _t_15106;
					float _t_15107;
				
					_t_15104 = tx3_6_1 * ty1_7_1;
					_t_15105 = tx1_4_1 * ty3_9_1;
					_t_15106 = _t_15105 * -1.0f;
					_t_15107 = _t_15104 + _t_15106;
					_t_15108 = _t_15107;
				
				}
		else
				{
					float _t_15109;
					float _t_15110;
					float _t_15111;
					float _t_15112;
					float _t_15113;
				
					_t_15109 = tx3_6_1 * ty1_7_1;
					_t_15110 = tx1_4_1 * ty3_9_1;
					_t_15111 = _t_15110 * -1.0f;
					_t_15112 = _t_15109 + _t_15111;
					_t_15113 = -1.0f * _t_15112;
					_t_15108 = _t_15113;
				
				}
		
			_t_15114 = -1.0f * _t_15108;
			_t_15115 = _t_15114 * _t_567;
			_t_15116 = tegpixellet_block_51(ty3_9_1,ty1_7_1,_t_567,_t_15115,tx1_4_1,tx3_6_1,y__3757_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1);
			_t_14958 = _t_15116;
		
		}
else
		{
		
			_t_14958 = 0.0f;
		
		}


	return _t_14958;
}
__device__ float tegpixelintegrator_33(float ty3_9_1,float pc1_15_1,float _t_14957,float tc2_19_1,float ty2_8_1,float pc0_14_1,float _t_567,float ty1_7_1,float tx1_4_1,float tx3_6_1,float py1_13_1,float pc2_16_1,float tx2_5_1,float px1_11_1,float tc0_17_1,float _t_14848,float py0_12_1,float tc1_18_1,float px0_10_1){
    float y__3757_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_14957 - _t_14848)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__3757_1 = _t_14848 + __step__ * (i + (float)(0.5));
        float _t_14958;
		_t_14958 = tegpixelbody_block_33(ty3_9_1,ty1_7_1,_t_567,px0_10_1,px1_11_1,tx1_4_1,tx3_6_1,py0_12_1,py1_13_1,y__3757_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);;
        __output__ = __output__ + _t_14958 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_17(float ty3_9_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float _t_567,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_14740;
	float _t_14741;
	float _t_14742;
	bool _t_14743;
	float _t_14746;
	float _t_14750;
	float _t_14751;
	float _t_14752;
	float _t_14753;
	float _t_14754;
	bool _t_14755;
	float _t_14758;
	float _t_14762;
	float _t_14763;
	bool _t_14764;
	float _t_14765;
	float _t_14766;
	float _t_14767;
	float _t_14768;
	float _t_14769;
	bool _t_14770;
	float _t_14773;
	float _t_14777;
	float _t_14778;
	float _t_14779;
	float _t_14780;
	bool _t_14781;
	float _t_14784;
	float _t_14788;
	float _t_14789;
	float _t_14790;
	float _t_14791;
	float _t_14792;
	bool _t_14793;
	float _t_14796;
	float _t_14800;
	float _t_14801;
	float _t_14802;
	float _t_14803;
	float _t_14804;
	float _t_14805;
	float _t_14806;
	float _t_14807;
	float _t_14808;
	bool _t_14809;
	float _t_14812;
	float _t_14816;
	float _t_14817;
	float _t_14818;
	float _t_14819;
	bool _t_14820;
	float _t_14823;
	float _t_14827;
	float _t_14828;
	float _t_14829;
	float _t_14830;
	float _t_14831;
	bool _t_14832;
	float _t_14835;
	float _t_14839;
	float _t_14840;
	float _t_14841;
	float _t_14842;
	float _t_14843;
	float _t_14844;
	bool _t_14845;
	float _t_14846;
	float _t_14847;
	float _t_14848;
	float _t_14849;
	float _t_14850;
	float _t_14851;
	bool _t_14852;
	float _t_14855;
	float _t_14859;
	float _t_14860;
	float _t_14861;
	float _t_14862;
	float _t_14863;
	bool _t_14864;
	float _t_14867;
	float _t_14871;
	float _t_14872;
	bool _t_14873;
	float _t_14874;
	float _t_14875;
	float _t_14876;
	float _t_14877;
	float _t_14878;
	bool _t_14879;
	float _t_14882;
	float _t_14886;
	float _t_14887;
	float _t_14888;
	float _t_14889;
	bool _t_14890;
	float _t_14893;
	float _t_14897;
	float _t_14898;
	float _t_14899;
	float _t_14900;
	float _t_14901;
	bool _t_14902;
	float _t_14905;
	float _t_14909;
	float _t_14910;
	float _t_14911;
	float _t_14912;
	float _t_14913;
	float _t_14914;
	float _t_14915;
	float _t_14916;
	float _t_14917;
	bool _t_14918;
	float _t_14921;
	float _t_14925;
	float _t_14926;
	float _t_14927;
	float _t_14928;
	bool _t_14929;
	float _t_14932;
	float _t_14936;
	float _t_14937;
	float _t_14938;
	float _t_14939;
	float _t_14940;
	bool _t_14941;
	float _t_14944;
	float _t_14948;
	float _t_14949;
	float _t_14950;
	float _t_14951;
	float _t_14952;
	float _t_14953;
	bool _t_14954;
	float _t_14955;
	float _t_14956;
	float _t_14957;

	float _t_568;

	_t_14740 = -1.0f * ty1_7_1;
	_t_14741 = ty3_9_1 + _t_14740;
	_t_14742 = -1.0f * _t_14741;
	_t_14743 = _t_14742 < 0.0f;
	if(_t_14743)
		{
			float _t_14744;
			float _t_14745;
		
			_t_14744 = -1.0f * tx3_6_1;
			_t_14745 = tx1_4_1 + _t_14744;
			_t_14746 = _t_14745;
		
		}
else
		{
			float _t_14747;
			float _t_14748;
			float _t_14749;
		
			_t_14747 = -1.0f * tx3_6_1;
			_t_14748 = tx1_4_1 + _t_14747;
			_t_14749 = -1.0f * _t_14748;
			_t_14746 = _t_14749;
		
		}

	_t_14750 = _t_14746 * _t_567;
	_t_14751 = _t_14750 * -1.0f;
	_t_14752 = -1.0f * ty1_7_1;
	_t_14753 = ty3_9_1 + _t_14752;
	_t_14754 = -1.0f * _t_14753;
	_t_14755 = _t_14754 < 0.0f;
	if(_t_14755)
		{
			float _t_14756;
			float _t_14757;
		
			_t_14756 = -1.0f * tx3_6_1;
			_t_14757 = tx1_4_1 + _t_14756;
			_t_14758 = _t_14757;
		
		}
else
		{
			float _t_14759;
			float _t_14760;
			float _t_14761;
		
			_t_14759 = -1.0f * tx3_6_1;
			_t_14760 = tx1_4_1 + _t_14759;
			_t_14761 = -1.0f * _t_14760;
			_t_14758 = _t_14761;
		
		}

	_t_14762 = _t_14758 * _t_567;
	_t_14763 = _t_14762 * -1.0f;
	_t_14764 = 0.0f < _t_14763;
	if(_t_14764)
		{
		
			_t_14765 = px0_10_1;
		
		}
else
		{
		
			_t_14765 = px1_11_1;
		
		}

	_t_14766 = _t_14751 * _t_14765;
	_t_14767 = -1.0f * ty1_7_1;
	_t_14768 = ty3_9_1 + _t_14767;
	_t_14769 = -1.0f * _t_14768;
	_t_14770 = _t_14769 < 0.0f;
	if(_t_14770)
		{
			float _t_14771;
			float _t_14772;
		
			_t_14771 = -1.0f * tx3_6_1;
			_t_14772 = tx1_4_1 + _t_14771;
			_t_14773 = _t_14772;
		
		}
else
		{
			float _t_14774;
			float _t_14775;
			float _t_14776;
		
			_t_14774 = -1.0f * tx3_6_1;
			_t_14775 = tx1_4_1 + _t_14774;
			_t_14776 = -1.0f * _t_14775;
			_t_14773 = _t_14776;
		
		}

	_t_14777 = _t_14773 * _t_567;
	_t_14778 = -1.0f * ty1_7_1;
	_t_14779 = ty3_9_1 + _t_14778;
	_t_14780 = -1.0f * _t_14779;
	_t_14781 = _t_14780 < 0.0f;
	if(_t_14781)
		{
			float _t_14782;
			float _t_14783;
		
			_t_14782 = -1.0f * tx3_6_1;
			_t_14783 = tx1_4_1 + _t_14782;
			_t_14784 = _t_14783;
		
		}
else
		{
			float _t_14785;
			float _t_14786;
			float _t_14787;
		
			_t_14785 = -1.0f * tx3_6_1;
			_t_14786 = tx1_4_1 + _t_14785;
			_t_14787 = -1.0f * _t_14786;
			_t_14784 = _t_14787;
		
		}

	_t_14788 = _t_14784 * _t_567;
	_t_14789 = _t_14777 * _t_14788;
	_t_14790 = -1.0f * ty1_7_1;
	_t_14791 = ty3_9_1 + _t_14790;
	_t_14792 = -1.0f * _t_14791;
	_t_14793 = _t_14792 < 0.0f;
	if(_t_14793)
		{
			float _t_14794;
			float _t_14795;
		
			_t_14794 = -1.0f * ty1_7_1;
			_t_14795 = ty3_9_1 + _t_14794;
			_t_14796 = _t_14795;
		
		}
else
		{
			float _t_14797;
			float _t_14798;
			float _t_14799;
		
			_t_14797 = -1.0f * ty1_7_1;
			_t_14798 = ty3_9_1 + _t_14797;
			_t_14799 = -1.0f * _t_14798;
			_t_14796 = _t_14799;
		
		}

	_t_14800 = _t_14796 * _t_567;
	_t_14801 = 1.0f + _t_14800;
	_t_14802 = 1.0f / _t_14801;
	_t_14803 = _t_14789 * _t_14802;
	_t_14804 = _t_14803 * -1.0f;
	_t_14805 = 1.0f + _t_14804;
	_t_14806 = -1.0f * ty1_7_1;
	_t_14807 = ty3_9_1 + _t_14806;
	_t_14808 = -1.0f * _t_14807;
	_t_14809 = _t_14808 < 0.0f;
	if(_t_14809)
		{
			float _t_14810;
			float _t_14811;
		
			_t_14810 = -1.0f * tx3_6_1;
			_t_14811 = tx1_4_1 + _t_14810;
			_t_14812 = _t_14811;
		
		}
else
		{
			float _t_14813;
			float _t_14814;
			float _t_14815;
		
			_t_14813 = -1.0f * tx3_6_1;
			_t_14814 = tx1_4_1 + _t_14813;
			_t_14815 = -1.0f * _t_14814;
			_t_14812 = _t_14815;
		
		}

	_t_14816 = _t_14812 * _t_567;
	_t_14817 = -1.0f * ty1_7_1;
	_t_14818 = ty3_9_1 + _t_14817;
	_t_14819 = -1.0f * _t_14818;
	_t_14820 = _t_14819 < 0.0f;
	if(_t_14820)
		{
			float _t_14821;
			float _t_14822;
		
			_t_14821 = -1.0f * tx3_6_1;
			_t_14822 = tx1_4_1 + _t_14821;
			_t_14823 = _t_14822;
		
		}
else
		{
			float _t_14824;
			float _t_14825;
			float _t_14826;
		
			_t_14824 = -1.0f * tx3_6_1;
			_t_14825 = tx1_4_1 + _t_14824;
			_t_14826 = -1.0f * _t_14825;
			_t_14823 = _t_14826;
		
		}

	_t_14827 = _t_14823 * _t_567;
	_t_14828 = _t_14816 * _t_14827;
	_t_14829 = -1.0f * ty1_7_1;
	_t_14830 = ty3_9_1 + _t_14829;
	_t_14831 = -1.0f * _t_14830;
	_t_14832 = _t_14831 < 0.0f;
	if(_t_14832)
		{
			float _t_14833;
			float _t_14834;
		
			_t_14833 = -1.0f * ty1_7_1;
			_t_14834 = ty3_9_1 + _t_14833;
			_t_14835 = _t_14834;
		
		}
else
		{
			float _t_14836;
			float _t_14837;
			float _t_14838;
		
			_t_14836 = -1.0f * ty1_7_1;
			_t_14837 = ty3_9_1 + _t_14836;
			_t_14838 = -1.0f * _t_14837;
			_t_14835 = _t_14838;
		
		}

	_t_14839 = _t_14835 * _t_567;
	_t_14840 = 1.0f + _t_14839;
	_t_14841 = 1.0f / _t_14840;
	_t_14842 = _t_14828 * _t_14841;
	_t_14843 = _t_14842 * -1.0f;
	_t_14844 = 1.0f + _t_14843;
	_t_14845 = 0.0f < _t_14844;
	if(_t_14845)
		{
		
			_t_14846 = py0_12_1;
		
		}
else
		{
		
			_t_14846 = py1_13_1;
		
		}

	_t_14847 = _t_14805 * _t_14846;
	_t_14848 = _t_14766 + _t_14847;
	_t_14849 = -1.0f * ty1_7_1;
	_t_14850 = ty3_9_1 + _t_14849;
	_t_14851 = -1.0f * _t_14850;
	_t_14852 = _t_14851 < 0.0f;
	if(_t_14852)
		{
			float _t_14853;
			float _t_14854;
		
			_t_14853 = -1.0f * tx3_6_1;
			_t_14854 = tx1_4_1 + _t_14853;
			_t_14855 = _t_14854;
		
		}
else
		{
			float _t_14856;
			float _t_14857;
			float _t_14858;
		
			_t_14856 = -1.0f * tx3_6_1;
			_t_14857 = tx1_4_1 + _t_14856;
			_t_14858 = -1.0f * _t_14857;
			_t_14855 = _t_14858;
		
		}

	_t_14859 = _t_14855 * _t_567;
	_t_14860 = _t_14859 * -1.0f;
	_t_14861 = -1.0f * ty1_7_1;
	_t_14862 = ty3_9_1 + _t_14861;
	_t_14863 = -1.0f * _t_14862;
	_t_14864 = _t_14863 < 0.0f;
	if(_t_14864)
		{
			float _t_14865;
			float _t_14866;
		
			_t_14865 = -1.0f * tx3_6_1;
			_t_14866 = tx1_4_1 + _t_14865;
			_t_14867 = _t_14866;
		
		}
else
		{
			float _t_14868;
			float _t_14869;
			float _t_14870;
		
			_t_14868 = -1.0f * tx3_6_1;
			_t_14869 = tx1_4_1 + _t_14868;
			_t_14870 = -1.0f * _t_14869;
			_t_14867 = _t_14870;
		
		}

	_t_14871 = _t_14867 * _t_567;
	_t_14872 = _t_14871 * -1.0f;
	_t_14873 = 0.0f < _t_14872;
	if(_t_14873)
		{
		
			_t_14874 = px1_11_1;
		
		}
else
		{
		
			_t_14874 = px0_10_1;
		
		}

	_t_14875 = _t_14860 * _t_14874;
	_t_14876 = -1.0f * ty1_7_1;
	_t_14877 = ty3_9_1 + _t_14876;
	_t_14878 = -1.0f * _t_14877;
	_t_14879 = _t_14878 < 0.0f;
	if(_t_14879)
		{
			float _t_14880;
			float _t_14881;
		
			_t_14880 = -1.0f * tx3_6_1;
			_t_14881 = tx1_4_1 + _t_14880;
			_t_14882 = _t_14881;
		
		}
else
		{
			float _t_14883;
			float _t_14884;
			float _t_14885;
		
			_t_14883 = -1.0f * tx3_6_1;
			_t_14884 = tx1_4_1 + _t_14883;
			_t_14885 = -1.0f * _t_14884;
			_t_14882 = _t_14885;
		
		}

	_t_14886 = _t_14882 * _t_567;
	_t_14887 = -1.0f * ty1_7_1;
	_t_14888 = ty3_9_1 + _t_14887;
	_t_14889 = -1.0f * _t_14888;
	_t_14890 = _t_14889 < 0.0f;
	if(_t_14890)
		{
			float _t_14891;
			float _t_14892;
		
			_t_14891 = -1.0f * tx3_6_1;
			_t_14892 = tx1_4_1 + _t_14891;
			_t_14893 = _t_14892;
		
		}
else
		{
			float _t_14894;
			float _t_14895;
			float _t_14896;
		
			_t_14894 = -1.0f * tx3_6_1;
			_t_14895 = tx1_4_1 + _t_14894;
			_t_14896 = -1.0f * _t_14895;
			_t_14893 = _t_14896;
		
		}

	_t_14897 = _t_14893 * _t_567;
	_t_14898 = _t_14886 * _t_14897;
	_t_14899 = -1.0f * ty1_7_1;
	_t_14900 = ty3_9_1 + _t_14899;
	_t_14901 = -1.0f * _t_14900;
	_t_14902 = _t_14901 < 0.0f;
	if(_t_14902)
		{
			float _t_14903;
			float _t_14904;
		
			_t_14903 = -1.0f * ty1_7_1;
			_t_14904 = ty3_9_1 + _t_14903;
			_t_14905 = _t_14904;
		
		}
else
		{
			float _t_14906;
			float _t_14907;
			float _t_14908;
		
			_t_14906 = -1.0f * ty1_7_1;
			_t_14907 = ty3_9_1 + _t_14906;
			_t_14908 = -1.0f * _t_14907;
			_t_14905 = _t_14908;
		
		}

	_t_14909 = _t_14905 * _t_567;
	_t_14910 = 1.0f + _t_14909;
	_t_14911 = 1.0f / _t_14910;
	_t_14912 = _t_14898 * _t_14911;
	_t_14913 = _t_14912 * -1.0f;
	_t_14914 = 1.0f + _t_14913;
	_t_14915 = -1.0f * ty1_7_1;
	_t_14916 = ty3_9_1 + _t_14915;
	_t_14917 = -1.0f * _t_14916;
	_t_14918 = _t_14917 < 0.0f;
	if(_t_14918)
		{
			float _t_14919;
			float _t_14920;
		
			_t_14919 = -1.0f * tx3_6_1;
			_t_14920 = tx1_4_1 + _t_14919;
			_t_14921 = _t_14920;
		
		}
else
		{
			float _t_14922;
			float _t_14923;
			float _t_14924;
		
			_t_14922 = -1.0f * tx3_6_1;
			_t_14923 = tx1_4_1 + _t_14922;
			_t_14924 = -1.0f * _t_14923;
			_t_14921 = _t_14924;
		
		}

	_t_14925 = _t_14921 * _t_567;
	_t_14926 = -1.0f * ty1_7_1;
	_t_14927 = ty3_9_1 + _t_14926;
	_t_14928 = -1.0f * _t_14927;
	_t_14929 = _t_14928 < 0.0f;
	if(_t_14929)
		{
			float _t_14930;
			float _t_14931;
		
			_t_14930 = -1.0f * tx3_6_1;
			_t_14931 = tx1_4_1 + _t_14930;
			_t_14932 = _t_14931;
		
		}
else
		{
			float _t_14933;
			float _t_14934;
			float _t_14935;
		
			_t_14933 = -1.0f * tx3_6_1;
			_t_14934 = tx1_4_1 + _t_14933;
			_t_14935 = -1.0f * _t_14934;
			_t_14932 = _t_14935;
		
		}

	_t_14936 = _t_14932 * _t_567;
	_t_14937 = _t_14925 * _t_14936;
	_t_14938 = -1.0f * ty1_7_1;
	_t_14939 = ty3_9_1 + _t_14938;
	_t_14940 = -1.0f * _t_14939;
	_t_14941 = _t_14940 < 0.0f;
	if(_t_14941)
		{
			float _t_14942;
			float _t_14943;
		
			_t_14942 = -1.0f * ty1_7_1;
			_t_14943 = ty3_9_1 + _t_14942;
			_t_14944 = _t_14943;
		
		}
else
		{
			float _t_14945;
			float _t_14946;
			float _t_14947;
		
			_t_14945 = -1.0f * ty1_7_1;
			_t_14946 = ty3_9_1 + _t_14945;
			_t_14947 = -1.0f * _t_14946;
			_t_14944 = _t_14947;
		
		}

	_t_14948 = _t_14944 * _t_567;
	_t_14949 = 1.0f + _t_14948;
	_t_14950 = 1.0f / _t_14949;
	_t_14951 = _t_14937 * _t_14950;
	_t_14952 = _t_14951 * -1.0f;
	_t_14953 = 1.0f + _t_14952;
	_t_14954 = 0.0f < _t_14953;
	if(_t_14954)
		{
		
			_t_14955 = py1_13_1;
		
		}
else
		{
		
			_t_14955 = py0_12_1;
		
		}

	_t_14956 = _t_14914 * _t_14955;
	_t_14957 = _t_14875 + _t_14956;
	_t_568 = tegpixelintegrator_33(ty3_9_1,pc1_15_1,_t_14957,tc2_19_1,ty2_8_1,pc0_14_1,_t_567,ty1_7_1,tx1_4_1,tx3_6_1,py1_13_1,pc2_16_1,tx2_5_1,px1_11_1,tc0_17_1,_t_14848,py0_12_1,tc1_18_1,px0_10_1);

	return _t_568;
}
__device__ float tegpixellet_block_54(float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float tx1_4_1,float ty2_8_1,float tx2_5_1,float ty1_7_1,float _t_15977,float _t_16030,float ty3_9_1,float tx3_6_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1,float _t_595,float y__3831_1,float _t_15950){
	float _t_16031;
	float _t_16032;
	float _t_16033;
	float _t_16034;
	float _t_16035;
	float _t_16036;
	float _t_16037;
	float _t_16038;
	float _t_16039;
	float _t_16040;
	float _t_16041;
	float _t_16042;
	float _t_16043;
	float _t_16044;
	float _t_16045;
	float _t_16046;
	float _t_16047;
	float _t_16048;
	float _t_16049;
	float _t_16050;
	float _t_16051;
	float _t_16052;
	float _t_16053;
	bool _t_16054;
	float _t_16055;
	float _t_16056;
	float _t_16057;
	float _t_16058;
	float _t_16059;
	float _t_16060;
	float _t_16061;
	float _t_16062;
	float _t_16063;
	float _t_16064;
	float _t_16065;
	float _t_16066;
	float _t_16067;
	bool _t_16068;
	float _t_16069;
	float _t_16070;
	float _t_16071;
	float _t_16072;
	float _t_16073;
	bool _t_16074;
	bool _t_16075;
	bool _t_16076;
	bool _t_16077;
	bool _t_16078;
	bool _t_16079;
	bool _t_16080;
	float _t_16410;

	float _t_15951;

	_t_16031 = -1.0f * pc0_14_1;
	_t_16032 = tc0_17_1 + _t_16031;
	_t_16033 = _t_16032 * _t_16032;
	_t_16034 = -1.0f * pc1_15_1;
	_t_16035 = tc1_18_1 + _t_16034;
	_t_16036 = _t_16035 * _t_16035;
	_t_16037 = _t_16033 + _t_16036;
	_t_16038 = -1.0f * pc2_16_1;
	_t_16039 = tc2_19_1 + _t_16038;
	_t_16040 = _t_16039 * _t_16039;
	_t_16041 = _t_16037 + _t_16040;
	_t_16042 = tx1_4_1 * ty2_8_1;
	_t_16043 = tx2_5_1 * ty1_7_1;
	_t_16044 = _t_16043 * -1.0f;
	_t_16045 = _t_16042 + _t_16044;
	_t_16046 = -1.0f * ty2_8_1;
	_t_16047 = ty1_7_1 + _t_16046;
	_t_16048 = _t_16047 * _t_15977;
	_t_16049 = _t_16045 + _t_16048;
	_t_16050 = -1.0f * tx1_4_1;
	_t_16051 = tx2_5_1 + _t_16050;
	_t_16052 = _t_16051 * _t_16030;
	_t_16053 = _t_16049 + _t_16052;
	_t_16054 = _t_16053 < 0.0f;
	if(_t_16054)
		{
		
			_t_16055 = 1.0f;
		
		}
else
		{
		
			_t_16055 = 0.0f;
		
		}

	_t_16056 = tx2_5_1 * ty3_9_1;
	_t_16057 = tx3_6_1 * ty2_8_1;
	_t_16058 = _t_16057 * -1.0f;
	_t_16059 = _t_16056 + _t_16058;
	_t_16060 = -1.0f * ty3_9_1;
	_t_16061 = ty2_8_1 + _t_16060;
	_t_16062 = _t_16061 * _t_15977;
	_t_16063 = _t_16059 + _t_16062;
	_t_16064 = -1.0f * tx2_5_1;
	_t_16065 = tx3_6_1 + _t_16064;
	_t_16066 = _t_16065 * _t_16030;
	_t_16067 = _t_16063 + _t_16066;
	_t_16068 = _t_16067 < 0.0f;
	if(_t_16068)
		{
		
			_t_16069 = 1.0f;
		
		}
else
		{
		
			_t_16069 = 0.0f;
		
		}

	_t_16070 = _t_16055 * _t_16069;
	_t_16071 = _t_16041 * _t_16070;
	_t_16072 = _t_16071 * _t_15977;
	_t_16073 = _t_16072 * -1.0f;
	_t_16074 = py0_12_1 < _t_16030;
	_t_16075 = _t_16030 < py1_13_1;
	_t_16076 = _t_16074 && _t_16075;
	_t_16077 = px0_10_1 < _t_15977;
	_t_16078 = _t_15977 < px1_11_1;
	_t_16079 = _t_16077 && _t_16078;
	_t_16080 = _t_16076 && _t_16079;
	if(_t_16080)
		{
			float _t_16081;
			float _t_16082;
			float _t_16083;
			bool _t_16084;
			float _t_16087;
			float _t_16091;
			float _t_16092;
			float _t_16093;
			float _t_16094;
			float _t_16095;
			bool _t_16096;
			float _t_16099;
			float _t_16103;
			float _t_16104;
			bool _t_16105;
			float _t_16106;
			float _t_16107;
			float _t_16108;
			float _t_16109;
			float _t_16110;
			bool _t_16111;
			float _t_16114;
			float _t_16118;
			float _t_16119;
			float _t_16120;
			float _t_16121;
			bool _t_16122;
			float _t_16125;
			float _t_16129;
			float _t_16130;
			float _t_16131;
			float _t_16132;
			float _t_16133;
			bool _t_16134;
			float _t_16137;
			float _t_16141;
			float _t_16142;
			float _t_16143;
			float _t_16144;
			float _t_16145;
			float _t_16146;
			float _t_16147;
			float _t_16148;
			float _t_16149;
			bool _t_16150;
			float _t_16153;
			float _t_16157;
			float _t_16158;
			float _t_16159;
			float _t_16160;
			bool _t_16161;
			float _t_16164;
			float _t_16168;
			float _t_16169;
			float _t_16170;
			float _t_16171;
			float _t_16172;
			bool _t_16173;
			float _t_16176;
			float _t_16180;
			float _t_16181;
			float _t_16182;
			float _t_16183;
			float _t_16184;
			float _t_16185;
			bool _t_16186;
			float _t_16187;
			float _t_16188;
			float _t_16189;
			bool _t_16190;
			float _t_16191;
			float _t_16192;
			float _t_16193;
			bool _t_16194;
			float _t_16197;
			float _t_16201;
			float _t_16202;
			float _t_16203;
			float _t_16204;
			float _t_16205;
			bool _t_16206;
			float _t_16209;
			float _t_16213;
			float _t_16214;
			bool _t_16215;
			float _t_16216;
			float _t_16217;
			float _t_16218;
			float _t_16219;
			float _t_16220;
			bool _t_16221;
			float _t_16224;
			float _t_16228;
			float _t_16229;
			float _t_16230;
			float _t_16231;
			bool _t_16232;
			float _t_16235;
			float _t_16239;
			float _t_16240;
			float _t_16241;
			float _t_16242;
			float _t_16243;
			bool _t_16244;
			float _t_16247;
			float _t_16251;
			float _t_16252;
			float _t_16253;
			float _t_16254;
			float _t_16255;
			float _t_16256;
			float _t_16257;
			float _t_16258;
			float _t_16259;
			bool _t_16260;
			float _t_16263;
			float _t_16267;
			float _t_16268;
			float _t_16269;
			float _t_16270;
			bool _t_16271;
			float _t_16274;
			float _t_16278;
			float _t_16279;
			float _t_16280;
			float _t_16281;
			float _t_16282;
			bool _t_16283;
			float _t_16286;
			float _t_16290;
			float _t_16291;
			float _t_16292;
			float _t_16293;
			float _t_16294;
			float _t_16295;
			bool _t_16296;
			float _t_16297;
			float _t_16298;
			float _t_16299;
			bool _t_16300;
			bool _t_16301;
			float _t_16302;
			float _t_16303;
			float _t_16304;
			bool _t_16305;
			float _t_16308;
			float _t_16312;
			float _t_16313;
			float _t_16314;
			float _t_16315;
			bool _t_16316;
			float _t_16319;
			float _t_16323;
			bool _t_16324;
			float _t_16325;
			float _t_16326;
			float _t_16327;
			float _t_16328;
			float _t_16329;
			bool _t_16330;
			float _t_16333;
			float _t_16337;
			float _t_16338;
			float _t_16339;
			float _t_16340;
			bool _t_16341;
			float _t_16344;
			float _t_16348;
			bool _t_16349;
			float _t_16350;
			float _t_16351;
			float _t_16352;
			bool _t_16353;
			float _t_16354;
			float _t_16355;
			float _t_16356;
			bool _t_16357;
			float _t_16360;
			float _t_16364;
			float _t_16365;
			float _t_16366;
			float _t_16367;
			bool _t_16368;
			float _t_16371;
			float _t_16375;
			bool _t_16376;
			float _t_16377;
			float _t_16378;
			float _t_16379;
			float _t_16380;
			float _t_16381;
			bool _t_16382;
			float _t_16385;
			float _t_16389;
			float _t_16390;
			float _t_16391;
			float _t_16392;
			bool _t_16393;
			float _t_16396;
			float _t_16400;
			bool _t_16401;
			float _t_16402;
			float _t_16403;
			float _t_16404;
			bool _t_16405;
			bool _t_16406;
			bool _t_16407;
			float _t_16408;
			float _t_16409;
		
			_t_16081 = -1.0f * ty1_7_1;
			_t_16082 = ty3_9_1 + _t_16081;
			_t_16083 = -1.0f * _t_16082;
			_t_16084 = _t_16083 < 0.0f;
			if(_t_16084)
				{
					float _t_16085;
					float _t_16086;
				
					_t_16085 = -1.0f * tx3_6_1;
					_t_16086 = tx1_4_1 + _t_16085;
					_t_16087 = _t_16086;
				
				}
		else
				{
					float _t_16088;
					float _t_16089;
					float _t_16090;
				
					_t_16088 = -1.0f * tx3_6_1;
					_t_16089 = tx1_4_1 + _t_16088;
					_t_16090 = -1.0f * _t_16089;
					_t_16087 = _t_16090;
				
				}
		
			_t_16091 = _t_16087 * _t_595;
			_t_16092 = _t_16091 * -1.0f;
			_t_16093 = -1.0f * ty1_7_1;
			_t_16094 = ty3_9_1 + _t_16093;
			_t_16095 = -1.0f * _t_16094;
			_t_16096 = _t_16095 < 0.0f;
			if(_t_16096)
				{
					float _t_16097;
					float _t_16098;
				
					_t_16097 = -1.0f * tx3_6_1;
					_t_16098 = tx1_4_1 + _t_16097;
					_t_16099 = _t_16098;
				
				}
		else
				{
					float _t_16100;
					float _t_16101;
					float _t_16102;
				
					_t_16100 = -1.0f * tx3_6_1;
					_t_16101 = tx1_4_1 + _t_16100;
					_t_16102 = -1.0f * _t_16101;
					_t_16099 = _t_16102;
				
				}
		
			_t_16103 = _t_16099 * _t_595;
			_t_16104 = _t_16103 * -1.0f;
			_t_16105 = 0.0f < _t_16104;
			if(_t_16105)
				{
				
					_t_16106 = px0_10_1;
				
				}
		else
				{
				
					_t_16106 = px1_11_1;
				
				}
		
			_t_16107 = _t_16092 * _t_16106;
			_t_16108 = -1.0f * ty1_7_1;
			_t_16109 = ty3_9_1 + _t_16108;
			_t_16110 = -1.0f * _t_16109;
			_t_16111 = _t_16110 < 0.0f;
			if(_t_16111)
				{
					float _t_16112;
					float _t_16113;
				
					_t_16112 = -1.0f * tx3_6_1;
					_t_16113 = tx1_4_1 + _t_16112;
					_t_16114 = _t_16113;
				
				}
		else
				{
					float _t_16115;
					float _t_16116;
					float _t_16117;
				
					_t_16115 = -1.0f * tx3_6_1;
					_t_16116 = tx1_4_1 + _t_16115;
					_t_16117 = -1.0f * _t_16116;
					_t_16114 = _t_16117;
				
				}
		
			_t_16118 = _t_16114 * _t_595;
			_t_16119 = -1.0f * ty1_7_1;
			_t_16120 = ty3_9_1 + _t_16119;
			_t_16121 = -1.0f * _t_16120;
			_t_16122 = _t_16121 < 0.0f;
			if(_t_16122)
				{
					float _t_16123;
					float _t_16124;
				
					_t_16123 = -1.0f * tx3_6_1;
					_t_16124 = tx1_4_1 + _t_16123;
					_t_16125 = _t_16124;
				
				}
		else
				{
					float _t_16126;
					float _t_16127;
					float _t_16128;
				
					_t_16126 = -1.0f * tx3_6_1;
					_t_16127 = tx1_4_1 + _t_16126;
					_t_16128 = -1.0f * _t_16127;
					_t_16125 = _t_16128;
				
				}
		
			_t_16129 = _t_16125 * _t_595;
			_t_16130 = _t_16118 * _t_16129;
			_t_16131 = -1.0f * ty1_7_1;
			_t_16132 = ty3_9_1 + _t_16131;
			_t_16133 = -1.0f * _t_16132;
			_t_16134 = _t_16133 < 0.0f;
			if(_t_16134)
				{
					float _t_16135;
					float _t_16136;
				
					_t_16135 = -1.0f * ty1_7_1;
					_t_16136 = ty3_9_1 + _t_16135;
					_t_16137 = _t_16136;
				
				}
		else
				{
					float _t_16138;
					float _t_16139;
					float _t_16140;
				
					_t_16138 = -1.0f * ty1_7_1;
					_t_16139 = ty3_9_1 + _t_16138;
					_t_16140 = -1.0f * _t_16139;
					_t_16137 = _t_16140;
				
				}
		
			_t_16141 = _t_16137 * _t_595;
			_t_16142 = 1.0f + _t_16141;
			_t_16143 = 1.0f / _t_16142;
			_t_16144 = _t_16130 * _t_16143;
			_t_16145 = _t_16144 * -1.0f;
			_t_16146 = 1.0f + _t_16145;
			_t_16147 = -1.0f * ty1_7_1;
			_t_16148 = ty3_9_1 + _t_16147;
			_t_16149 = -1.0f * _t_16148;
			_t_16150 = _t_16149 < 0.0f;
			if(_t_16150)
				{
					float _t_16151;
					float _t_16152;
				
					_t_16151 = -1.0f * tx3_6_1;
					_t_16152 = tx1_4_1 + _t_16151;
					_t_16153 = _t_16152;
				
				}
		else
				{
					float _t_16154;
					float _t_16155;
					float _t_16156;
				
					_t_16154 = -1.0f * tx3_6_1;
					_t_16155 = tx1_4_1 + _t_16154;
					_t_16156 = -1.0f * _t_16155;
					_t_16153 = _t_16156;
				
				}
		
			_t_16157 = _t_16153 * _t_595;
			_t_16158 = -1.0f * ty1_7_1;
			_t_16159 = ty3_9_1 + _t_16158;
			_t_16160 = -1.0f * _t_16159;
			_t_16161 = _t_16160 < 0.0f;
			if(_t_16161)
				{
					float _t_16162;
					float _t_16163;
				
					_t_16162 = -1.0f * tx3_6_1;
					_t_16163 = tx1_4_1 + _t_16162;
					_t_16164 = _t_16163;
				
				}
		else
				{
					float _t_16165;
					float _t_16166;
					float _t_16167;
				
					_t_16165 = -1.0f * tx3_6_1;
					_t_16166 = tx1_4_1 + _t_16165;
					_t_16167 = -1.0f * _t_16166;
					_t_16164 = _t_16167;
				
				}
		
			_t_16168 = _t_16164 * _t_595;
			_t_16169 = _t_16157 * _t_16168;
			_t_16170 = -1.0f * ty1_7_1;
			_t_16171 = ty3_9_1 + _t_16170;
			_t_16172 = -1.0f * _t_16171;
			_t_16173 = _t_16172 < 0.0f;
			if(_t_16173)
				{
					float _t_16174;
					float _t_16175;
				
					_t_16174 = -1.0f * ty1_7_1;
					_t_16175 = ty3_9_1 + _t_16174;
					_t_16176 = _t_16175;
				
				}
		else
				{
					float _t_16177;
					float _t_16178;
					float _t_16179;
				
					_t_16177 = -1.0f * ty1_7_1;
					_t_16178 = ty3_9_1 + _t_16177;
					_t_16179 = -1.0f * _t_16178;
					_t_16176 = _t_16179;
				
				}
		
			_t_16180 = _t_16176 * _t_595;
			_t_16181 = 1.0f + _t_16180;
			_t_16182 = 1.0f / _t_16181;
			_t_16183 = _t_16169 * _t_16182;
			_t_16184 = _t_16183 * -1.0f;
			_t_16185 = 1.0f + _t_16184;
			_t_16186 = 0.0f < _t_16185;
			if(_t_16186)
				{
				
					_t_16187 = py0_12_1;
				
				}
		else
				{
				
					_t_16187 = py1_13_1;
				
				}
		
			_t_16188 = _t_16146 * _t_16187;
			_t_16189 = _t_16107 + _t_16188;
			_t_16190 = _t_16189 < y__3831_1;
			_t_16191 = -1.0f * ty1_7_1;
			_t_16192 = ty3_9_1 + _t_16191;
			_t_16193 = -1.0f * _t_16192;
			_t_16194 = _t_16193 < 0.0f;
			if(_t_16194)
				{
					float _t_16195;
					float _t_16196;
				
					_t_16195 = -1.0f * tx3_6_1;
					_t_16196 = tx1_4_1 + _t_16195;
					_t_16197 = _t_16196;
				
				}
		else
				{
					float _t_16198;
					float _t_16199;
					float _t_16200;
				
					_t_16198 = -1.0f * tx3_6_1;
					_t_16199 = tx1_4_1 + _t_16198;
					_t_16200 = -1.0f * _t_16199;
					_t_16197 = _t_16200;
				
				}
		
			_t_16201 = _t_16197 * _t_595;
			_t_16202 = _t_16201 * -1.0f;
			_t_16203 = -1.0f * ty1_7_1;
			_t_16204 = ty3_9_1 + _t_16203;
			_t_16205 = -1.0f * _t_16204;
			_t_16206 = _t_16205 < 0.0f;
			if(_t_16206)
				{
					float _t_16207;
					float _t_16208;
				
					_t_16207 = -1.0f * tx3_6_1;
					_t_16208 = tx1_4_1 + _t_16207;
					_t_16209 = _t_16208;
				
				}
		else
				{
					float _t_16210;
					float _t_16211;
					float _t_16212;
				
					_t_16210 = -1.0f * tx3_6_1;
					_t_16211 = tx1_4_1 + _t_16210;
					_t_16212 = -1.0f * _t_16211;
					_t_16209 = _t_16212;
				
				}
		
			_t_16213 = _t_16209 * _t_595;
			_t_16214 = _t_16213 * -1.0f;
			_t_16215 = 0.0f < _t_16214;
			if(_t_16215)
				{
				
					_t_16216 = px1_11_1;
				
				}
		else
				{
				
					_t_16216 = px0_10_1;
				
				}
		
			_t_16217 = _t_16202 * _t_16216;
			_t_16218 = -1.0f * ty1_7_1;
			_t_16219 = ty3_9_1 + _t_16218;
			_t_16220 = -1.0f * _t_16219;
			_t_16221 = _t_16220 < 0.0f;
			if(_t_16221)
				{
					float _t_16222;
					float _t_16223;
				
					_t_16222 = -1.0f * tx3_6_1;
					_t_16223 = tx1_4_1 + _t_16222;
					_t_16224 = _t_16223;
				
				}
		else
				{
					float _t_16225;
					float _t_16226;
					float _t_16227;
				
					_t_16225 = -1.0f * tx3_6_1;
					_t_16226 = tx1_4_1 + _t_16225;
					_t_16227 = -1.0f * _t_16226;
					_t_16224 = _t_16227;
				
				}
		
			_t_16228 = _t_16224 * _t_595;
			_t_16229 = -1.0f * ty1_7_1;
			_t_16230 = ty3_9_1 + _t_16229;
			_t_16231 = -1.0f * _t_16230;
			_t_16232 = _t_16231 < 0.0f;
			if(_t_16232)
				{
					float _t_16233;
					float _t_16234;
				
					_t_16233 = -1.0f * tx3_6_1;
					_t_16234 = tx1_4_1 + _t_16233;
					_t_16235 = _t_16234;
				
				}
		else
				{
					float _t_16236;
					float _t_16237;
					float _t_16238;
				
					_t_16236 = -1.0f * tx3_6_1;
					_t_16237 = tx1_4_1 + _t_16236;
					_t_16238 = -1.0f * _t_16237;
					_t_16235 = _t_16238;
				
				}
		
			_t_16239 = _t_16235 * _t_595;
			_t_16240 = _t_16228 * _t_16239;
			_t_16241 = -1.0f * ty1_7_1;
			_t_16242 = ty3_9_1 + _t_16241;
			_t_16243 = -1.0f * _t_16242;
			_t_16244 = _t_16243 < 0.0f;
			if(_t_16244)
				{
					float _t_16245;
					float _t_16246;
				
					_t_16245 = -1.0f * ty1_7_1;
					_t_16246 = ty3_9_1 + _t_16245;
					_t_16247 = _t_16246;
				
				}
		else
				{
					float _t_16248;
					float _t_16249;
					float _t_16250;
				
					_t_16248 = -1.0f * ty1_7_1;
					_t_16249 = ty3_9_1 + _t_16248;
					_t_16250 = -1.0f * _t_16249;
					_t_16247 = _t_16250;
				
				}
		
			_t_16251 = _t_16247 * _t_595;
			_t_16252 = 1.0f + _t_16251;
			_t_16253 = 1.0f / _t_16252;
			_t_16254 = _t_16240 * _t_16253;
			_t_16255 = _t_16254 * -1.0f;
			_t_16256 = 1.0f + _t_16255;
			_t_16257 = -1.0f * ty1_7_1;
			_t_16258 = ty3_9_1 + _t_16257;
			_t_16259 = -1.0f * _t_16258;
			_t_16260 = _t_16259 < 0.0f;
			if(_t_16260)
				{
					float _t_16261;
					float _t_16262;
				
					_t_16261 = -1.0f * tx3_6_1;
					_t_16262 = tx1_4_1 + _t_16261;
					_t_16263 = _t_16262;
				
				}
		else
				{
					float _t_16264;
					float _t_16265;
					float _t_16266;
				
					_t_16264 = -1.0f * tx3_6_1;
					_t_16265 = tx1_4_1 + _t_16264;
					_t_16266 = -1.0f * _t_16265;
					_t_16263 = _t_16266;
				
				}
		
			_t_16267 = _t_16263 * _t_595;
			_t_16268 = -1.0f * ty1_7_1;
			_t_16269 = ty3_9_1 + _t_16268;
			_t_16270 = -1.0f * _t_16269;
			_t_16271 = _t_16270 < 0.0f;
			if(_t_16271)
				{
					float _t_16272;
					float _t_16273;
				
					_t_16272 = -1.0f * tx3_6_1;
					_t_16273 = tx1_4_1 + _t_16272;
					_t_16274 = _t_16273;
				
				}
		else
				{
					float _t_16275;
					float _t_16276;
					float _t_16277;
				
					_t_16275 = -1.0f * tx3_6_1;
					_t_16276 = tx1_4_1 + _t_16275;
					_t_16277 = -1.0f * _t_16276;
					_t_16274 = _t_16277;
				
				}
		
			_t_16278 = _t_16274 * _t_595;
			_t_16279 = _t_16267 * _t_16278;
			_t_16280 = -1.0f * ty1_7_1;
			_t_16281 = ty3_9_1 + _t_16280;
			_t_16282 = -1.0f * _t_16281;
			_t_16283 = _t_16282 < 0.0f;
			if(_t_16283)
				{
					float _t_16284;
					float _t_16285;
				
					_t_16284 = -1.0f * ty1_7_1;
					_t_16285 = ty3_9_1 + _t_16284;
					_t_16286 = _t_16285;
				
				}
		else
				{
					float _t_16287;
					float _t_16288;
					float _t_16289;
				
					_t_16287 = -1.0f * ty1_7_1;
					_t_16288 = ty3_9_1 + _t_16287;
					_t_16289 = -1.0f * _t_16288;
					_t_16286 = _t_16289;
				
				}
		
			_t_16290 = _t_16286 * _t_595;
			_t_16291 = 1.0f + _t_16290;
			_t_16292 = 1.0f / _t_16291;
			_t_16293 = _t_16279 * _t_16292;
			_t_16294 = _t_16293 * -1.0f;
			_t_16295 = 1.0f + _t_16294;
			_t_16296 = 0.0f < _t_16295;
			if(_t_16296)
				{
				
					_t_16297 = py1_13_1;
				
				}
		else
				{
				
					_t_16297 = py0_12_1;
				
				}
		
			_t_16298 = _t_16256 * _t_16297;
			_t_16299 = _t_16217 + _t_16298;
			_t_16300 = y__3831_1 < _t_16299;
			_t_16301 = _t_16190 && _t_16300;
			_t_16302 = -1.0f * ty1_7_1;
			_t_16303 = ty3_9_1 + _t_16302;
			_t_16304 = -1.0f * _t_16303;
			_t_16305 = _t_16304 < 0.0f;
			if(_t_16305)
				{
					float _t_16306;
					float _t_16307;
				
					_t_16306 = -1.0f * ty1_7_1;
					_t_16307 = ty3_9_1 + _t_16306;
					_t_16308 = _t_16307;
				
				}
		else
				{
					float _t_16309;
					float _t_16310;
					float _t_16311;
				
					_t_16309 = -1.0f * ty1_7_1;
					_t_16310 = ty3_9_1 + _t_16309;
					_t_16311 = -1.0f * _t_16310;
					_t_16308 = _t_16311;
				
				}
		
			_t_16312 = _t_16308 * _t_595;
			_t_16313 = -1.0f * ty1_7_1;
			_t_16314 = ty3_9_1 + _t_16313;
			_t_16315 = -1.0f * _t_16314;
			_t_16316 = _t_16315 < 0.0f;
			if(_t_16316)
				{
					float _t_16317;
					float _t_16318;
				
					_t_16317 = -1.0f * ty1_7_1;
					_t_16318 = ty3_9_1 + _t_16317;
					_t_16319 = _t_16318;
				
				}
		else
				{
					float _t_16320;
					float _t_16321;
					float _t_16322;
				
					_t_16320 = -1.0f * ty1_7_1;
					_t_16321 = ty3_9_1 + _t_16320;
					_t_16322 = -1.0f * _t_16321;
					_t_16319 = _t_16322;
				
				}
		
			_t_16323 = _t_16319 * _t_595;
			_t_16324 = 0.0f < _t_16323;
			if(_t_16324)
				{
				
					_t_16325 = px0_10_1;
				
				}
		else
				{
				
					_t_16325 = px1_11_1;
				
				}
		
			_t_16326 = _t_16312 * _t_16325;
			_t_16327 = -1.0f * ty1_7_1;
			_t_16328 = ty3_9_1 + _t_16327;
			_t_16329 = -1.0f * _t_16328;
			_t_16330 = _t_16329 < 0.0f;
			if(_t_16330)
				{
					float _t_16331;
					float _t_16332;
				
					_t_16331 = -1.0f * tx3_6_1;
					_t_16332 = tx1_4_1 + _t_16331;
					_t_16333 = _t_16332;
				
				}
		else
				{
					float _t_16334;
					float _t_16335;
					float _t_16336;
				
					_t_16334 = -1.0f * tx3_6_1;
					_t_16335 = tx1_4_1 + _t_16334;
					_t_16336 = -1.0f * _t_16335;
					_t_16333 = _t_16336;
				
				}
		
			_t_16337 = _t_16333 * _t_595;
			_t_16338 = -1.0f * ty1_7_1;
			_t_16339 = ty3_9_1 + _t_16338;
			_t_16340 = -1.0f * _t_16339;
			_t_16341 = _t_16340 < 0.0f;
			if(_t_16341)
				{
					float _t_16342;
					float _t_16343;
				
					_t_16342 = -1.0f * tx3_6_1;
					_t_16343 = tx1_4_1 + _t_16342;
					_t_16344 = _t_16343;
				
				}
		else
				{
					float _t_16345;
					float _t_16346;
					float _t_16347;
				
					_t_16345 = -1.0f * tx3_6_1;
					_t_16346 = tx1_4_1 + _t_16345;
					_t_16347 = -1.0f * _t_16346;
					_t_16344 = _t_16347;
				
				}
		
			_t_16348 = _t_16344 * _t_595;
			_t_16349 = 0.0f < _t_16348;
			if(_t_16349)
				{
				
					_t_16350 = py0_12_1;
				
				}
		else
				{
				
					_t_16350 = py1_13_1;
				
				}
		
			_t_16351 = _t_16337 * _t_16350;
			_t_16352 = _t_16326 + _t_16351;
			_t_16353 = _t_16352 < _t_15950;
			_t_16354 = -1.0f * ty1_7_1;
			_t_16355 = ty3_9_1 + _t_16354;
			_t_16356 = -1.0f * _t_16355;
			_t_16357 = _t_16356 < 0.0f;
			if(_t_16357)
				{
					float _t_16358;
					float _t_16359;
				
					_t_16358 = -1.0f * ty1_7_1;
					_t_16359 = ty3_9_1 + _t_16358;
					_t_16360 = _t_16359;
				
				}
		else
				{
					float _t_16361;
					float _t_16362;
					float _t_16363;
				
					_t_16361 = -1.0f * ty1_7_1;
					_t_16362 = ty3_9_1 + _t_16361;
					_t_16363 = -1.0f * _t_16362;
					_t_16360 = _t_16363;
				
				}
		
			_t_16364 = _t_16360 * _t_595;
			_t_16365 = -1.0f * ty1_7_1;
			_t_16366 = ty3_9_1 + _t_16365;
			_t_16367 = -1.0f * _t_16366;
			_t_16368 = _t_16367 < 0.0f;
			if(_t_16368)
				{
					float _t_16369;
					float _t_16370;
				
					_t_16369 = -1.0f * ty1_7_1;
					_t_16370 = ty3_9_1 + _t_16369;
					_t_16371 = _t_16370;
				
				}
		else
				{
					float _t_16372;
					float _t_16373;
					float _t_16374;
				
					_t_16372 = -1.0f * ty1_7_1;
					_t_16373 = ty3_9_1 + _t_16372;
					_t_16374 = -1.0f * _t_16373;
					_t_16371 = _t_16374;
				
				}
		
			_t_16375 = _t_16371 * _t_595;
			_t_16376 = 0.0f < _t_16375;
			if(_t_16376)
				{
				
					_t_16377 = px1_11_1;
				
				}
		else
				{
				
					_t_16377 = px0_10_1;
				
				}
		
			_t_16378 = _t_16364 * _t_16377;
			_t_16379 = -1.0f * ty1_7_1;
			_t_16380 = ty3_9_1 + _t_16379;
			_t_16381 = -1.0f * _t_16380;
			_t_16382 = _t_16381 < 0.0f;
			if(_t_16382)
				{
					float _t_16383;
					float _t_16384;
				
					_t_16383 = -1.0f * tx3_6_1;
					_t_16384 = tx1_4_1 + _t_16383;
					_t_16385 = _t_16384;
				
				}
		else
				{
					float _t_16386;
					float _t_16387;
					float _t_16388;
				
					_t_16386 = -1.0f * tx3_6_1;
					_t_16387 = tx1_4_1 + _t_16386;
					_t_16388 = -1.0f * _t_16387;
					_t_16385 = _t_16388;
				
				}
		
			_t_16389 = _t_16385 * _t_595;
			_t_16390 = -1.0f * ty1_7_1;
			_t_16391 = ty3_9_1 + _t_16390;
			_t_16392 = -1.0f * _t_16391;
			_t_16393 = _t_16392 < 0.0f;
			if(_t_16393)
				{
					float _t_16394;
					float _t_16395;
				
					_t_16394 = -1.0f * tx3_6_1;
					_t_16395 = tx1_4_1 + _t_16394;
					_t_16396 = _t_16395;
				
				}
		else
				{
					float _t_16397;
					float _t_16398;
					float _t_16399;
				
					_t_16397 = -1.0f * tx3_6_1;
					_t_16398 = tx1_4_1 + _t_16397;
					_t_16399 = -1.0f * _t_16398;
					_t_16396 = _t_16399;
				
				}
		
			_t_16400 = _t_16396 * _t_595;
			_t_16401 = 0.0f < _t_16400;
			if(_t_16401)
				{
				
					_t_16402 = py1_13_1;
				
				}
		else
				{
				
					_t_16402 = py0_12_1;
				
				}
		
			_t_16403 = _t_16389 * _t_16402;
			_t_16404 = _t_16378 + _t_16403;
			_t_16405 = _t_15950 < _t_16404;
			_t_16406 = _t_16353 && _t_16405;
			_t_16407 = _t_16301 && _t_16406;
			if(_t_16407)
				{
				
					_t_16408 = 1.0f;
				
				}
		else
				{
				
					_t_16408 = 0.0f;
				
				}
		
			_t_16409 = _t_16408 * _t_595;
			_t_16410 = _t_16409;
		
		}
else
		{
		
			_t_16410 = 0.0f;
		
		}

	_t_15951 = _t_16073 * _t_16410;

	return _t_15951;
}
__device__ float tegpixellet_block_53(float ty3_9_1,float ty1_7_1,float _t_595,float _t_15950,float tx1_4_1,float tx3_6_1,float y__3831_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1,float py0_12_1,float py1_13_1,float px0_10_1,float px1_11_1){
	float _t_15952;
	float _t_15953;
	float _t_15954;
	bool _t_15955;
	float _t_15958;
	float _t_15962;
	float _t_15963;
	float _t_15964;
	float _t_15965;
	float _t_15966;
	bool _t_15967;
	float _t_15970;
	float _t_15974;
	float _t_15975;
	float _t_15976;
	float _t_15977;
	float _t_15978;
	float _t_15979;
	float _t_15980;
	bool _t_15981;
	float _t_15984;
	float _t_15988;
	float _t_15989;
	float _t_15990;
	float _t_15991;
	bool _t_15992;
	float _t_15995;
	float _t_15999;
	float _t_16000;
	float _t_16001;
	float _t_16002;
	float _t_16003;
	bool _t_16004;
	float _t_16007;
	float _t_16011;
	float _t_16012;
	float _t_16013;
	float _t_16014;
	float _t_16015;
	float _t_16016;
	float _t_16017;
	float _t_16018;
	float _t_16019;
	float _t_16020;
	bool _t_16021;
	float _t_16024;
	float _t_16028;
	float _t_16029;
	float _t_16030;

	float _t_15951;

	_t_15952 = -1.0f * ty1_7_1;
	_t_15953 = ty3_9_1 + _t_15952;
	_t_15954 = -1.0f * _t_15953;
	_t_15955 = _t_15954 < 0.0f;
	if(_t_15955)
		{
			float _t_15956;
			float _t_15957;
		
			_t_15956 = -1.0f * ty1_7_1;
			_t_15957 = ty3_9_1 + _t_15956;
			_t_15958 = _t_15957;
		
		}
else
		{
			float _t_15959;
			float _t_15960;
			float _t_15961;
		
			_t_15959 = -1.0f * ty1_7_1;
			_t_15960 = ty3_9_1 + _t_15959;
			_t_15961 = -1.0f * _t_15960;
			_t_15958 = _t_15961;
		
		}

	_t_15962 = _t_15958 * _t_595;
	_t_15963 = _t_15962 * _t_15950;
	_t_15964 = -1.0f * ty1_7_1;
	_t_15965 = ty3_9_1 + _t_15964;
	_t_15966 = -1.0f * _t_15965;
	_t_15967 = _t_15966 < 0.0f;
	if(_t_15967)
		{
			float _t_15968;
			float _t_15969;
		
			_t_15968 = -1.0f * tx3_6_1;
			_t_15969 = tx1_4_1 + _t_15968;
			_t_15970 = _t_15969;
		
		}
else
		{
			float _t_15971;
			float _t_15972;
			float _t_15973;
		
			_t_15971 = -1.0f * tx3_6_1;
			_t_15972 = tx1_4_1 + _t_15971;
			_t_15973 = -1.0f * _t_15972;
			_t_15970 = _t_15973;
		
		}

	_t_15974 = _t_15970 * _t_595;
	_t_15975 = _t_15974 * -1.0f;
	_t_15976 = _t_15975 * y__3831_1;
	_t_15977 = _t_15963 + _t_15976;
	_t_15978 = -1.0f * ty1_7_1;
	_t_15979 = ty3_9_1 + _t_15978;
	_t_15980 = -1.0f * _t_15979;
	_t_15981 = _t_15980 < 0.0f;
	if(_t_15981)
		{
			float _t_15982;
			float _t_15983;
		
			_t_15982 = -1.0f * tx3_6_1;
			_t_15983 = tx1_4_1 + _t_15982;
			_t_15984 = _t_15983;
		
		}
else
		{
			float _t_15985;
			float _t_15986;
			float _t_15987;
		
			_t_15985 = -1.0f * tx3_6_1;
			_t_15986 = tx1_4_1 + _t_15985;
			_t_15987 = -1.0f * _t_15986;
			_t_15984 = _t_15987;
		
		}

	_t_15988 = _t_15984 * _t_595;
	_t_15989 = -1.0f * ty1_7_1;
	_t_15990 = ty3_9_1 + _t_15989;
	_t_15991 = -1.0f * _t_15990;
	_t_15992 = _t_15991 < 0.0f;
	if(_t_15992)
		{
			float _t_15993;
			float _t_15994;
		
			_t_15993 = -1.0f * tx3_6_1;
			_t_15994 = tx1_4_1 + _t_15993;
			_t_15995 = _t_15994;
		
		}
else
		{
			float _t_15996;
			float _t_15997;
			float _t_15998;
		
			_t_15996 = -1.0f * tx3_6_1;
			_t_15997 = tx1_4_1 + _t_15996;
			_t_15998 = -1.0f * _t_15997;
			_t_15995 = _t_15998;
		
		}

	_t_15999 = _t_15995 * _t_595;
	_t_16000 = _t_15988 * _t_15999;
	_t_16001 = -1.0f * ty1_7_1;
	_t_16002 = ty3_9_1 + _t_16001;
	_t_16003 = -1.0f * _t_16002;
	_t_16004 = _t_16003 < 0.0f;
	if(_t_16004)
		{
			float _t_16005;
			float _t_16006;
		
			_t_16005 = -1.0f * ty1_7_1;
			_t_16006 = ty3_9_1 + _t_16005;
			_t_16007 = _t_16006;
		
		}
else
		{
			float _t_16008;
			float _t_16009;
			float _t_16010;
		
			_t_16008 = -1.0f * ty1_7_1;
			_t_16009 = ty3_9_1 + _t_16008;
			_t_16010 = -1.0f * _t_16009;
			_t_16007 = _t_16010;
		
		}

	_t_16011 = _t_16007 * _t_595;
	_t_16012 = 1.0f + _t_16011;
	_t_16013 = 1.0f / _t_16012;
	_t_16014 = _t_16000 * _t_16013;
	_t_16015 = _t_16014 * -1.0f;
	_t_16016 = 1.0f + _t_16015;
	_t_16017 = _t_16016 * y__3831_1;
	_t_16018 = -1.0f * ty1_7_1;
	_t_16019 = ty3_9_1 + _t_16018;
	_t_16020 = -1.0f * _t_16019;
	_t_16021 = _t_16020 < 0.0f;
	if(_t_16021)
		{
			float _t_16022;
			float _t_16023;
		
			_t_16022 = -1.0f * tx3_6_1;
			_t_16023 = tx1_4_1 + _t_16022;
			_t_16024 = _t_16023;
		
		}
else
		{
			float _t_16025;
			float _t_16026;
			float _t_16027;
		
			_t_16025 = -1.0f * tx3_6_1;
			_t_16026 = tx1_4_1 + _t_16025;
			_t_16027 = -1.0f * _t_16026;
			_t_16024 = _t_16027;
		
		}

	_t_16028 = _t_16024 * _t_595;
	_t_16029 = _t_16028 * _t_15950;
	_t_16030 = _t_16017 + _t_16029;
	_t_15951 = tegpixellet_block_54(tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx1_4_1,ty2_8_1,tx2_5_1,ty1_7_1,_t_15977,_t_16030,ty3_9_1,tx3_6_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1,_t_595,y__3831_1,_t_15950);

	return _t_15951;
}
__device__ float tegpixelbody_block_34(float ty3_9_1,float ty1_7_1,float _t_595,float px0_10_1,float px1_11_1,float tx1_4_1,float tx3_6_1,float py0_12_1,float py1_13_1,float y__3831_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_15794;
	float _t_15795;
	float _t_15796;
	bool _t_15797;
	float _t_15800;
	float _t_15804;
	float _t_15805;
	float _t_15806;
	float _t_15807;
	bool _t_15808;
	float _t_15811;
	float _t_15815;
	bool _t_15816;
	float _t_15817;
	float _t_15818;
	float _t_15819;
	float _t_15820;
	float _t_15821;
	bool _t_15822;
	float _t_15825;
	float _t_15829;
	float _t_15830;
	float _t_15831;
	float _t_15832;
	bool _t_15833;
	float _t_15836;
	float _t_15840;
	bool _t_15841;
	float _t_15842;
	float _t_15843;
	float _t_15844;
	float _t_15845;
	float _t_15846;
	float _t_15847;
	bool _t_15848;
	float _t_15853;
	float _t_15859;
	float _t_15860;
	float _t_15861;
	float _t_15862;
	bool _t_15863;
	float _t_15864;
	float _t_15865;
	float _t_15866;
	bool _t_15867;
	float _t_15870;
	float _t_15874;
	float _t_15875;
	float _t_15876;
	float _t_15877;
	bool _t_15878;
	float _t_15881;
	float _t_15885;
	bool _t_15886;
	float _t_15887;
	float _t_15888;
	float _t_15889;
	float _t_15890;
	float _t_15891;
	bool _t_15892;
	float _t_15895;
	float _t_15899;
	float _t_15900;
	float _t_15901;
	float _t_15902;
	bool _t_15903;
	float _t_15906;
	float _t_15910;
	bool _t_15911;
	float _t_15912;
	float _t_15913;
	float _t_15914;
	float _t_15915;
	float _t_15916;
	float _t_15917;
	bool _t_15918;
	float _t_15923;
	float _t_15929;
	float _t_15930;
	float _t_15931;
	float _t_15932;
	bool _t_15933;
	bool _t_15934;

	float _t_15793;

	_t_15794 = -1.0f * ty1_7_1;
	_t_15795 = ty3_9_1 + _t_15794;
	_t_15796 = -1.0f * _t_15795;
	_t_15797 = _t_15796 < 0.0f;
	if(_t_15797)
		{
			float _t_15798;
			float _t_15799;
		
			_t_15798 = -1.0f * ty1_7_1;
			_t_15799 = ty3_9_1 + _t_15798;
			_t_15800 = _t_15799;
		
		}
else
		{
			float _t_15801;
			float _t_15802;
			float _t_15803;
		
			_t_15801 = -1.0f * ty1_7_1;
			_t_15802 = ty3_9_1 + _t_15801;
			_t_15803 = -1.0f * _t_15802;
			_t_15800 = _t_15803;
		
		}

	_t_15804 = _t_15800 * _t_595;
	_t_15805 = -1.0f * ty1_7_1;
	_t_15806 = ty3_9_1 + _t_15805;
	_t_15807 = -1.0f * _t_15806;
	_t_15808 = _t_15807 < 0.0f;
	if(_t_15808)
		{
			float _t_15809;
			float _t_15810;
		
			_t_15809 = -1.0f * ty1_7_1;
			_t_15810 = ty3_9_1 + _t_15809;
			_t_15811 = _t_15810;
		
		}
else
		{
			float _t_15812;
			float _t_15813;
			float _t_15814;
		
			_t_15812 = -1.0f * ty1_7_1;
			_t_15813 = ty3_9_1 + _t_15812;
			_t_15814 = -1.0f * _t_15813;
			_t_15811 = _t_15814;
		
		}

	_t_15815 = _t_15811 * _t_595;
	_t_15816 = 0.0f < _t_15815;
	if(_t_15816)
		{
		
			_t_15817 = px0_10_1;
		
		}
else
		{
		
			_t_15817 = px1_11_1;
		
		}

	_t_15818 = _t_15804 * _t_15817;
	_t_15819 = -1.0f * ty1_7_1;
	_t_15820 = ty3_9_1 + _t_15819;
	_t_15821 = -1.0f * _t_15820;
	_t_15822 = _t_15821 < 0.0f;
	if(_t_15822)
		{
			float _t_15823;
			float _t_15824;
		
			_t_15823 = -1.0f * tx3_6_1;
			_t_15824 = tx1_4_1 + _t_15823;
			_t_15825 = _t_15824;
		
		}
else
		{
			float _t_15826;
			float _t_15827;
			float _t_15828;
		
			_t_15826 = -1.0f * tx3_6_1;
			_t_15827 = tx1_4_1 + _t_15826;
			_t_15828 = -1.0f * _t_15827;
			_t_15825 = _t_15828;
		
		}

	_t_15829 = _t_15825 * _t_595;
	_t_15830 = -1.0f * ty1_7_1;
	_t_15831 = ty3_9_1 + _t_15830;
	_t_15832 = -1.0f * _t_15831;
	_t_15833 = _t_15832 < 0.0f;
	if(_t_15833)
		{
			float _t_15834;
			float _t_15835;
		
			_t_15834 = -1.0f * tx3_6_1;
			_t_15835 = tx1_4_1 + _t_15834;
			_t_15836 = _t_15835;
		
		}
else
		{
			float _t_15837;
			float _t_15838;
			float _t_15839;
		
			_t_15837 = -1.0f * tx3_6_1;
			_t_15838 = tx1_4_1 + _t_15837;
			_t_15839 = -1.0f * _t_15838;
			_t_15836 = _t_15839;
		
		}

	_t_15840 = _t_15836 * _t_595;
	_t_15841 = 0.0f < _t_15840;
	if(_t_15841)
		{
		
			_t_15842 = py0_12_1;
		
		}
else
		{
		
			_t_15842 = py1_13_1;
		
		}

	_t_15843 = _t_15829 * _t_15842;
	_t_15844 = _t_15818 + _t_15843;
	_t_15845 = -1.0f * ty1_7_1;
	_t_15846 = ty3_9_1 + _t_15845;
	_t_15847 = -1.0f * _t_15846;
	_t_15848 = _t_15847 < 0.0f;
	if(_t_15848)
		{
			float _t_15849;
			float _t_15850;
			float _t_15851;
			float _t_15852;
		
			_t_15849 = tx3_6_1 * ty1_7_1;
			_t_15850 = tx1_4_1 * ty3_9_1;
			_t_15851 = _t_15850 * -1.0f;
			_t_15852 = _t_15849 + _t_15851;
			_t_15853 = _t_15852;
		
		}
else
		{
			float _t_15854;
			float _t_15855;
			float _t_15856;
			float _t_15857;
			float _t_15858;
		
			_t_15854 = tx3_6_1 * ty1_7_1;
			_t_15855 = tx1_4_1 * ty3_9_1;
			_t_15856 = _t_15855 * -1.0f;
			_t_15857 = _t_15854 + _t_15856;
			_t_15858 = -1.0f * _t_15857;
			_t_15853 = _t_15858;
		
		}

	_t_15859 = -1.0f * _t_15853;
	_t_15860 = _t_15859 * _t_595;
	_t_15861 = _t_15860 * -1.0f;
	_t_15862 = _t_15844 + _t_15861;
	_t_15863 = _t_15862 < 0.0f;
	_t_15864 = -1.0f * ty1_7_1;
	_t_15865 = ty3_9_1 + _t_15864;
	_t_15866 = -1.0f * _t_15865;
	_t_15867 = _t_15866 < 0.0f;
	if(_t_15867)
		{
			float _t_15868;
			float _t_15869;
		
			_t_15868 = -1.0f * ty1_7_1;
			_t_15869 = ty3_9_1 + _t_15868;
			_t_15870 = _t_15869;
		
		}
else
		{
			float _t_15871;
			float _t_15872;
			float _t_15873;
		
			_t_15871 = -1.0f * ty1_7_1;
			_t_15872 = ty3_9_1 + _t_15871;
			_t_15873 = -1.0f * _t_15872;
			_t_15870 = _t_15873;
		
		}

	_t_15874 = _t_15870 * _t_595;
	_t_15875 = -1.0f * ty1_7_1;
	_t_15876 = ty3_9_1 + _t_15875;
	_t_15877 = -1.0f * _t_15876;
	_t_15878 = _t_15877 < 0.0f;
	if(_t_15878)
		{
			float _t_15879;
			float _t_15880;
		
			_t_15879 = -1.0f * ty1_7_1;
			_t_15880 = ty3_9_1 + _t_15879;
			_t_15881 = _t_15880;
		
		}
else
		{
			float _t_15882;
			float _t_15883;
			float _t_15884;
		
			_t_15882 = -1.0f * ty1_7_1;
			_t_15883 = ty3_9_1 + _t_15882;
			_t_15884 = -1.0f * _t_15883;
			_t_15881 = _t_15884;
		
		}

	_t_15885 = _t_15881 * _t_595;
	_t_15886 = 0.0f < _t_15885;
	if(_t_15886)
		{
		
			_t_15887 = px1_11_1;
		
		}
else
		{
		
			_t_15887 = px0_10_1;
		
		}

	_t_15888 = _t_15874 * _t_15887;
	_t_15889 = -1.0f * ty1_7_1;
	_t_15890 = ty3_9_1 + _t_15889;
	_t_15891 = -1.0f * _t_15890;
	_t_15892 = _t_15891 < 0.0f;
	if(_t_15892)
		{
			float _t_15893;
			float _t_15894;
		
			_t_15893 = -1.0f * tx3_6_1;
			_t_15894 = tx1_4_1 + _t_15893;
			_t_15895 = _t_15894;
		
		}
else
		{
			float _t_15896;
			float _t_15897;
			float _t_15898;
		
			_t_15896 = -1.0f * tx3_6_1;
			_t_15897 = tx1_4_1 + _t_15896;
			_t_15898 = -1.0f * _t_15897;
			_t_15895 = _t_15898;
		
		}

	_t_15899 = _t_15895 * _t_595;
	_t_15900 = -1.0f * ty1_7_1;
	_t_15901 = ty3_9_1 + _t_15900;
	_t_15902 = -1.0f * _t_15901;
	_t_15903 = _t_15902 < 0.0f;
	if(_t_15903)
		{
			float _t_15904;
			float _t_15905;
		
			_t_15904 = -1.0f * tx3_6_1;
			_t_15905 = tx1_4_1 + _t_15904;
			_t_15906 = _t_15905;
		
		}
else
		{
			float _t_15907;
			float _t_15908;
			float _t_15909;
		
			_t_15907 = -1.0f * tx3_6_1;
			_t_15908 = tx1_4_1 + _t_15907;
			_t_15909 = -1.0f * _t_15908;
			_t_15906 = _t_15909;
		
		}

	_t_15910 = _t_15906 * _t_595;
	_t_15911 = 0.0f < _t_15910;
	if(_t_15911)
		{
		
			_t_15912 = py1_13_1;
		
		}
else
		{
		
			_t_15912 = py0_12_1;
		
		}

	_t_15913 = _t_15899 * _t_15912;
	_t_15914 = _t_15888 + _t_15913;
	_t_15915 = -1.0f * ty1_7_1;
	_t_15916 = ty3_9_1 + _t_15915;
	_t_15917 = -1.0f * _t_15916;
	_t_15918 = _t_15917 < 0.0f;
	if(_t_15918)
		{
			float _t_15919;
			float _t_15920;
			float _t_15921;
			float _t_15922;
		
			_t_15919 = tx3_6_1 * ty1_7_1;
			_t_15920 = tx1_4_1 * ty3_9_1;
			_t_15921 = _t_15920 * -1.0f;
			_t_15922 = _t_15919 + _t_15921;
			_t_15923 = _t_15922;
		
		}
else
		{
			float _t_15924;
			float _t_15925;
			float _t_15926;
			float _t_15927;
			float _t_15928;
		
			_t_15924 = tx3_6_1 * ty1_7_1;
			_t_15925 = tx1_4_1 * ty3_9_1;
			_t_15926 = _t_15925 * -1.0f;
			_t_15927 = _t_15924 + _t_15926;
			_t_15928 = -1.0f * _t_15927;
			_t_15923 = _t_15928;
		
		}

	_t_15929 = -1.0f * _t_15923;
	_t_15930 = _t_15929 * _t_595;
	_t_15931 = _t_15930 * -1.0f;
	_t_15932 = _t_15914 + _t_15931;
	_t_15933 = 0.0f < _t_15932;
	_t_15934 = _t_15863 && _t_15933;
	if(_t_15934)
		{
			float _t_15935;
			float _t_15936;
			float _t_15937;
			bool _t_15938;
			float _t_15943;
			float _t_15949;
			float _t_15950;
			float _t_15951;
		
			_t_15935 = -1.0f * ty1_7_1;
			_t_15936 = ty3_9_1 + _t_15935;
			_t_15937 = -1.0f * _t_15936;
			_t_15938 = _t_15937 < 0.0f;
			if(_t_15938)
				{
					float _t_15939;
					float _t_15940;
					float _t_15941;
					float _t_15942;
				
					_t_15939 = tx3_6_1 * ty1_7_1;
					_t_15940 = tx1_4_1 * ty3_9_1;
					_t_15941 = _t_15940 * -1.0f;
					_t_15942 = _t_15939 + _t_15941;
					_t_15943 = _t_15942;
				
				}
		else
				{
					float _t_15944;
					float _t_15945;
					float _t_15946;
					float _t_15947;
					float _t_15948;
				
					_t_15944 = tx3_6_1 * ty1_7_1;
					_t_15945 = tx1_4_1 * ty3_9_1;
					_t_15946 = _t_15945 * -1.0f;
					_t_15947 = _t_15944 + _t_15946;
					_t_15948 = -1.0f * _t_15947;
					_t_15943 = _t_15948;
				
				}
		
			_t_15949 = -1.0f * _t_15943;
			_t_15950 = _t_15949 * _t_595;
			_t_15951 = tegpixellet_block_53(ty3_9_1,ty1_7_1,_t_595,_t_15950,tx1_4_1,tx3_6_1,y__3831_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1,py0_12_1,py1_13_1,px0_10_1,px1_11_1);
			_t_15793 = _t_15951;
		
		}
else
		{
		
			_t_15793 = 0.0f;
		
		}


	return _t_15793;
}
__device__ float tegpixelintegrator_34(float ty3_9_1,float pc1_15_1,float _t_595,float tc2_19_1,float _t_15683,float ty2_8_1,float pc0_14_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float py1_13_1,float pc2_16_1,float tx2_5_1,float px1_11_1,float tc0_17_1,float py0_12_1,float _t_15792,float tc1_18_1,float px0_10_1){
    float y__3831_1;
    float __output__ = 0;
    float __step__ = ((float)(_t_15792 - _t_15683)) / 50;
    for (unsigned int i = 0; i < 50; i++) {
        y__3831_1 = _t_15683 + __step__ * (i + (float)(0.5));
        float _t_15793;
		_t_15793 = tegpixelbody_block_34(ty3_9_1,ty1_7_1,_t_595,px0_10_1,px1_11_1,tx1_4_1,tx3_6_1,py0_12_1,py1_13_1,y__3831_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);;
        __output__ = __output__ + _t_15793 * __step__;
    }
    return __output__;
}
__device__ float tegpixellet_block_18(float ty3_9_1,float ty1_7_1,float tx1_4_1,float tx3_6_1,float _t_595,float px1_11_1,float px0_10_1,float py1_13_1,float py0_12_1,float tc0_17_1,float pc0_14_1,float tc1_18_1,float pc1_15_1,float tc2_19_1,float pc2_16_1,float ty2_8_1,float tx2_5_1){
	float _t_15575;
	float _t_15576;
	float _t_15577;
	bool _t_15578;
	float _t_15581;
	float _t_15585;
	float _t_15586;
	float _t_15587;
	float _t_15588;
	float _t_15589;
	bool _t_15590;
	float _t_15593;
	float _t_15597;
	float _t_15598;
	bool _t_15599;
	float _t_15600;
	float _t_15601;
	float _t_15602;
	float _t_15603;
	float _t_15604;
	bool _t_15605;
	float _t_15608;
	float _t_15612;
	float _t_15613;
	float _t_15614;
	float _t_15615;
	bool _t_15616;
	float _t_15619;
	float _t_15623;
	float _t_15624;
	float _t_15625;
	float _t_15626;
	float _t_15627;
	bool _t_15628;
	float _t_15631;
	float _t_15635;
	float _t_15636;
	float _t_15637;
	float _t_15638;
	float _t_15639;
	float _t_15640;
	float _t_15641;
	float _t_15642;
	float _t_15643;
	bool _t_15644;
	float _t_15647;
	float _t_15651;
	float _t_15652;
	float _t_15653;
	float _t_15654;
	bool _t_15655;
	float _t_15658;
	float _t_15662;
	float _t_15663;
	float _t_15664;
	float _t_15665;
	float _t_15666;
	bool _t_15667;
	float _t_15670;
	float _t_15674;
	float _t_15675;
	float _t_15676;
	float _t_15677;
	float _t_15678;
	float _t_15679;
	bool _t_15680;
	float _t_15681;
	float _t_15682;
	float _t_15683;
	float _t_15684;
	float _t_15685;
	float _t_15686;
	bool _t_15687;
	float _t_15690;
	float _t_15694;
	float _t_15695;
	float _t_15696;
	float _t_15697;
	float _t_15698;
	bool _t_15699;
	float _t_15702;
	float _t_15706;
	float _t_15707;
	bool _t_15708;
	float _t_15709;
	float _t_15710;
	float _t_15711;
	float _t_15712;
	float _t_15713;
	bool _t_15714;
	float _t_15717;
	float _t_15721;
	float _t_15722;
	float _t_15723;
	float _t_15724;
	bool _t_15725;
	float _t_15728;
	float _t_15732;
	float _t_15733;
	float _t_15734;
	float _t_15735;
	float _t_15736;
	bool _t_15737;
	float _t_15740;
	float _t_15744;
	float _t_15745;
	float _t_15746;
	float _t_15747;
	float _t_15748;
	float _t_15749;
	float _t_15750;
	float _t_15751;
	float _t_15752;
	bool _t_15753;
	float _t_15756;
	float _t_15760;
	float _t_15761;
	float _t_15762;
	float _t_15763;
	bool _t_15764;
	float _t_15767;
	float _t_15771;
	float _t_15772;
	float _t_15773;
	float _t_15774;
	float _t_15775;
	bool _t_15776;
	float _t_15779;
	float _t_15783;
	float _t_15784;
	float _t_15785;
	float _t_15786;
	float _t_15787;
	float _t_15788;
	bool _t_15789;
	float _t_15790;
	float _t_15791;
	float _t_15792;

	float _t_596;

	_t_15575 = -1.0f * ty1_7_1;
	_t_15576 = ty3_9_1 + _t_15575;
	_t_15577 = -1.0f * _t_15576;
	_t_15578 = _t_15577 < 0.0f;
	if(_t_15578)
		{
			float _t_15579;
			float _t_15580;
		
			_t_15579 = -1.0f * tx3_6_1;
			_t_15580 = tx1_4_1 + _t_15579;
			_t_15581 = _t_15580;
		
		}
else
		{
			float _t_15582;
			float _t_15583;
			float _t_15584;
		
			_t_15582 = -1.0f * tx3_6_1;
			_t_15583 = tx1_4_1 + _t_15582;
			_t_15584 = -1.0f * _t_15583;
			_t_15581 = _t_15584;
		
		}

	_t_15585 = _t_15581 * _t_595;
	_t_15586 = _t_15585 * -1.0f;
	_t_15587 = -1.0f * ty1_7_1;
	_t_15588 = ty3_9_1 + _t_15587;
	_t_15589 = -1.0f * _t_15588;
	_t_15590 = _t_15589 < 0.0f;
	if(_t_15590)
		{
			float _t_15591;
			float _t_15592;
		
			_t_15591 = -1.0f * tx3_6_1;
			_t_15592 = tx1_4_1 + _t_15591;
			_t_15593 = _t_15592;
		
		}
else
		{
			float _t_15594;
			float _t_15595;
			float _t_15596;
		
			_t_15594 = -1.0f * tx3_6_1;
			_t_15595 = tx1_4_1 + _t_15594;
			_t_15596 = -1.0f * _t_15595;
			_t_15593 = _t_15596;
		
		}

	_t_15597 = _t_15593 * _t_595;
	_t_15598 = _t_15597 * -1.0f;
	_t_15599 = 0.0f < _t_15598;
	if(_t_15599)
		{
		
			_t_15600 = px0_10_1;
		
		}
else
		{
		
			_t_15600 = px1_11_1;
		
		}

	_t_15601 = _t_15586 * _t_15600;
	_t_15602 = -1.0f * ty1_7_1;
	_t_15603 = ty3_9_1 + _t_15602;
	_t_15604 = -1.0f * _t_15603;
	_t_15605 = _t_15604 < 0.0f;
	if(_t_15605)
		{
			float _t_15606;
			float _t_15607;
		
			_t_15606 = -1.0f * tx3_6_1;
			_t_15607 = tx1_4_1 + _t_15606;
			_t_15608 = _t_15607;
		
		}
else
		{
			float _t_15609;
			float _t_15610;
			float _t_15611;
		
			_t_15609 = -1.0f * tx3_6_1;
			_t_15610 = tx1_4_1 + _t_15609;
			_t_15611 = -1.0f * _t_15610;
			_t_15608 = _t_15611;
		
		}

	_t_15612 = _t_15608 * _t_595;
	_t_15613 = -1.0f * ty1_7_1;
	_t_15614 = ty3_9_1 + _t_15613;
	_t_15615 = -1.0f * _t_15614;
	_t_15616 = _t_15615 < 0.0f;
	if(_t_15616)
		{
			float _t_15617;
			float _t_15618;
		
			_t_15617 = -1.0f * tx3_6_1;
			_t_15618 = tx1_4_1 + _t_15617;
			_t_15619 = _t_15618;
		
		}
else
		{
			float _t_15620;
			float _t_15621;
			float _t_15622;
		
			_t_15620 = -1.0f * tx3_6_1;
			_t_15621 = tx1_4_1 + _t_15620;
			_t_15622 = -1.0f * _t_15621;
			_t_15619 = _t_15622;
		
		}

	_t_15623 = _t_15619 * _t_595;
	_t_15624 = _t_15612 * _t_15623;
	_t_15625 = -1.0f * ty1_7_1;
	_t_15626 = ty3_9_1 + _t_15625;
	_t_15627 = -1.0f * _t_15626;
	_t_15628 = _t_15627 < 0.0f;
	if(_t_15628)
		{
			float _t_15629;
			float _t_15630;
		
			_t_15629 = -1.0f * ty1_7_1;
			_t_15630 = ty3_9_1 + _t_15629;
			_t_15631 = _t_15630;
		
		}
else
		{
			float _t_15632;
			float _t_15633;
			float _t_15634;
		
			_t_15632 = -1.0f * ty1_7_1;
			_t_15633 = ty3_9_1 + _t_15632;
			_t_15634 = -1.0f * _t_15633;
			_t_15631 = _t_15634;
		
		}

	_t_15635 = _t_15631 * _t_595;
	_t_15636 = 1.0f + _t_15635;
	_t_15637 = 1.0f / _t_15636;
	_t_15638 = _t_15624 * _t_15637;
	_t_15639 = _t_15638 * -1.0f;
	_t_15640 = 1.0f + _t_15639;
	_t_15641 = -1.0f * ty1_7_1;
	_t_15642 = ty3_9_1 + _t_15641;
	_t_15643 = -1.0f * _t_15642;
	_t_15644 = _t_15643 < 0.0f;
	if(_t_15644)
		{
			float _t_15645;
			float _t_15646;
		
			_t_15645 = -1.0f * tx3_6_1;
			_t_15646 = tx1_4_1 + _t_15645;
			_t_15647 = _t_15646;
		
		}
else
		{
			float _t_15648;
			float _t_15649;
			float _t_15650;
		
			_t_15648 = -1.0f * tx3_6_1;
			_t_15649 = tx1_4_1 + _t_15648;
			_t_15650 = -1.0f * _t_15649;
			_t_15647 = _t_15650;
		
		}

	_t_15651 = _t_15647 * _t_595;
	_t_15652 = -1.0f * ty1_7_1;
	_t_15653 = ty3_9_1 + _t_15652;
	_t_15654 = -1.0f * _t_15653;
	_t_15655 = _t_15654 < 0.0f;
	if(_t_15655)
		{
			float _t_15656;
			float _t_15657;
		
			_t_15656 = -1.0f * tx3_6_1;
			_t_15657 = tx1_4_1 + _t_15656;
			_t_15658 = _t_15657;
		
		}
else
		{
			float _t_15659;
			float _t_15660;
			float _t_15661;
		
			_t_15659 = -1.0f * tx3_6_1;
			_t_15660 = tx1_4_1 + _t_15659;
			_t_15661 = -1.0f * _t_15660;
			_t_15658 = _t_15661;
		
		}

	_t_15662 = _t_15658 * _t_595;
	_t_15663 = _t_15651 * _t_15662;
	_t_15664 = -1.0f * ty1_7_1;
	_t_15665 = ty3_9_1 + _t_15664;
	_t_15666 = -1.0f * _t_15665;
	_t_15667 = _t_15666 < 0.0f;
	if(_t_15667)
		{
			float _t_15668;
			float _t_15669;
		
			_t_15668 = -1.0f * ty1_7_1;
			_t_15669 = ty3_9_1 + _t_15668;
			_t_15670 = _t_15669;
		
		}
else
		{
			float _t_15671;
			float _t_15672;
			float _t_15673;
		
			_t_15671 = -1.0f * ty1_7_1;
			_t_15672 = ty3_9_1 + _t_15671;
			_t_15673 = -1.0f * _t_15672;
			_t_15670 = _t_15673;
		
		}

	_t_15674 = _t_15670 * _t_595;
	_t_15675 = 1.0f + _t_15674;
	_t_15676 = 1.0f / _t_15675;
	_t_15677 = _t_15663 * _t_15676;
	_t_15678 = _t_15677 * -1.0f;
	_t_15679 = 1.0f + _t_15678;
	_t_15680 = 0.0f < _t_15679;
	if(_t_15680)
		{
		
			_t_15681 = py0_12_1;
		
		}
else
		{
		
			_t_15681 = py1_13_1;
		
		}

	_t_15682 = _t_15640 * _t_15681;
	_t_15683 = _t_15601 + _t_15682;
	_t_15684 = -1.0f * ty1_7_1;
	_t_15685 = ty3_9_1 + _t_15684;
	_t_15686 = -1.0f * _t_15685;
	_t_15687 = _t_15686 < 0.0f;
	if(_t_15687)
		{
			float _t_15688;
			float _t_15689;
		
			_t_15688 = -1.0f * tx3_6_1;
			_t_15689 = tx1_4_1 + _t_15688;
			_t_15690 = _t_15689;
		
		}
else
		{
			float _t_15691;
			float _t_15692;
			float _t_15693;
		
			_t_15691 = -1.0f * tx3_6_1;
			_t_15692 = tx1_4_1 + _t_15691;
			_t_15693 = -1.0f * _t_15692;
			_t_15690 = _t_15693;
		
		}

	_t_15694 = _t_15690 * _t_595;
	_t_15695 = _t_15694 * -1.0f;
	_t_15696 = -1.0f * ty1_7_1;
	_t_15697 = ty3_9_1 + _t_15696;
	_t_15698 = -1.0f * _t_15697;
	_t_15699 = _t_15698 < 0.0f;
	if(_t_15699)
		{
			float _t_15700;
			float _t_15701;
		
			_t_15700 = -1.0f * tx3_6_1;
			_t_15701 = tx1_4_1 + _t_15700;
			_t_15702 = _t_15701;
		
		}
else
		{
			float _t_15703;
			float _t_15704;
			float _t_15705;
		
			_t_15703 = -1.0f * tx3_6_1;
			_t_15704 = tx1_4_1 + _t_15703;
			_t_15705 = -1.0f * _t_15704;
			_t_15702 = _t_15705;
		
		}

	_t_15706 = _t_15702 * _t_595;
	_t_15707 = _t_15706 * -1.0f;
	_t_15708 = 0.0f < _t_15707;
	if(_t_15708)
		{
		
			_t_15709 = px1_11_1;
		
		}
else
		{
		
			_t_15709 = px0_10_1;
		
		}

	_t_15710 = _t_15695 * _t_15709;
	_t_15711 = -1.0f * ty1_7_1;
	_t_15712 = ty3_9_1 + _t_15711;
	_t_15713 = -1.0f * _t_15712;
	_t_15714 = _t_15713 < 0.0f;
	if(_t_15714)
		{
			float _t_15715;
			float _t_15716;
		
			_t_15715 = -1.0f * tx3_6_1;
			_t_15716 = tx1_4_1 + _t_15715;
			_t_15717 = _t_15716;
		
		}
else
		{
			float _t_15718;
			float _t_15719;
			float _t_15720;
		
			_t_15718 = -1.0f * tx3_6_1;
			_t_15719 = tx1_4_1 + _t_15718;
			_t_15720 = -1.0f * _t_15719;
			_t_15717 = _t_15720;
		
		}

	_t_15721 = _t_15717 * _t_595;
	_t_15722 = -1.0f * ty1_7_1;
	_t_15723 = ty3_9_1 + _t_15722;
	_t_15724 = -1.0f * _t_15723;
	_t_15725 = _t_15724 < 0.0f;
	if(_t_15725)
		{
			float _t_15726;
			float _t_15727;
		
			_t_15726 = -1.0f * tx3_6_1;
			_t_15727 = tx1_4_1 + _t_15726;
			_t_15728 = _t_15727;
		
		}
else
		{
			float _t_15729;
			float _t_15730;
			float _t_15731;
		
			_t_15729 = -1.0f * tx3_6_1;
			_t_15730 = tx1_4_1 + _t_15729;
			_t_15731 = -1.0f * _t_15730;
			_t_15728 = _t_15731;
		
		}

	_t_15732 = _t_15728 * _t_595;
	_t_15733 = _t_15721 * _t_15732;
	_t_15734 = -1.0f * ty1_7_1;
	_t_15735 = ty3_9_1 + _t_15734;
	_t_15736 = -1.0f * _t_15735;
	_t_15737 = _t_15736 < 0.0f;
	if(_t_15737)
		{
			float _t_15738;
			float _t_15739;
		
			_t_15738 = -1.0f * ty1_7_1;
			_t_15739 = ty3_9_1 + _t_15738;
			_t_15740 = _t_15739;
		
		}
else
		{
			float _t_15741;
			float _t_15742;
			float _t_15743;
		
			_t_15741 = -1.0f * ty1_7_1;
			_t_15742 = ty3_9_1 + _t_15741;
			_t_15743 = -1.0f * _t_15742;
			_t_15740 = _t_15743;
		
		}

	_t_15744 = _t_15740 * _t_595;
	_t_15745 = 1.0f + _t_15744;
	_t_15746 = 1.0f / _t_15745;
	_t_15747 = _t_15733 * _t_15746;
	_t_15748 = _t_15747 * -1.0f;
	_t_15749 = 1.0f + _t_15748;
	_t_15750 = -1.0f * ty1_7_1;
	_t_15751 = ty3_9_1 + _t_15750;
	_t_15752 = -1.0f * _t_15751;
	_t_15753 = _t_15752 < 0.0f;
	if(_t_15753)
		{
			float _t_15754;
			float _t_15755;
		
			_t_15754 = -1.0f * tx3_6_1;
			_t_15755 = tx1_4_1 + _t_15754;
			_t_15756 = _t_15755;
		
		}
else
		{
			float _t_15757;
			float _t_15758;
			float _t_15759;
		
			_t_15757 = -1.0f * tx3_6_1;
			_t_15758 = tx1_4_1 + _t_15757;
			_t_15759 = -1.0f * _t_15758;
			_t_15756 = _t_15759;
		
		}

	_t_15760 = _t_15756 * _t_595;
	_t_15761 = -1.0f * ty1_7_1;
	_t_15762 = ty3_9_1 + _t_15761;
	_t_15763 = -1.0f * _t_15762;
	_t_15764 = _t_15763 < 0.0f;
	if(_t_15764)
		{
			float _t_15765;
			float _t_15766;
		
			_t_15765 = -1.0f * tx3_6_1;
			_t_15766 = tx1_4_1 + _t_15765;
			_t_15767 = _t_15766;
		
		}
else
		{
			float _t_15768;
			float _t_15769;
			float _t_15770;
		
			_t_15768 = -1.0f * tx3_6_1;
			_t_15769 = tx1_4_1 + _t_15768;
			_t_15770 = -1.0f * _t_15769;
			_t_15767 = _t_15770;
		
		}

	_t_15771 = _t_15767 * _t_595;
	_t_15772 = _t_15760 * _t_15771;
	_t_15773 = -1.0f * ty1_7_1;
	_t_15774 = ty3_9_1 + _t_15773;
	_t_15775 = -1.0f * _t_15774;
	_t_15776 = _t_15775 < 0.0f;
	if(_t_15776)
		{
			float _t_15777;
			float _t_15778;
		
			_t_15777 = -1.0f * ty1_7_1;
			_t_15778 = ty3_9_1 + _t_15777;
			_t_15779 = _t_15778;
		
		}
else
		{
			float _t_15780;
			float _t_15781;
			float _t_15782;
		
			_t_15780 = -1.0f * ty1_7_1;
			_t_15781 = ty3_9_1 + _t_15780;
			_t_15782 = -1.0f * _t_15781;
			_t_15779 = _t_15782;
		
		}

	_t_15783 = _t_15779 * _t_595;
	_t_15784 = 1.0f + _t_15783;
	_t_15785 = 1.0f / _t_15784;
	_t_15786 = _t_15772 * _t_15785;
	_t_15787 = _t_15786 * -1.0f;
	_t_15788 = 1.0f + _t_15787;
	_t_15789 = 0.0f < _t_15788;
	if(_t_15789)
		{
		
			_t_15790 = py1_13_1;
		
		}
else
		{
		
			_t_15790 = py0_12_1;
		
		}

	_t_15791 = _t_15749 * _t_15790;
	_t_15792 = _t_15710 + _t_15791;
	_t_596 = tegpixelintegrator_34(ty3_9_1,pc1_15_1,_t_595,tc2_19_1,_t_15683,ty2_8_1,pc0_14_1,ty1_7_1,tx1_4_1,tx3_6_1,py1_13_1,pc2_16_1,tx2_5_1,px1_11_1,tc0_17_1,py0_12_1,_t_15792,tc1_18_1,px0_10_1);

	return _t_596;
}
__device__ generic_array<float,16> tegpixel(float tx1_4_1,float ty1_7_1,float tx2_5_1,float ty2_8_1,float tx3_6_1,float ty3_9_1,float px0_10_1,float px1_11_1,float py0_12_1,float py1_13_1,float pc0_14_1,float pc1_15_1,float pc2_16_1,float tc0_17_1,float tc1_18_1,float tc2_19_1){
	float _t_2;
	float _t_4;
	float _t_6;
	float _t_7;
	float _t_9;
	float _t_11;
	float _t_13;
	float _t_15;
	float _t_17;
	float _t_19;
	float _t_21;
	generic_array<float,16> _t_22;
	generic_array<float,16> _t_23;
	generic_array<float,16> _t_24;
	generic_array<float,16> _t_25;
	generic_array<float,16> _t_26;
	generic_array<float,16> _t_27;
	generic_array<float,16> _t_28;
	generic_array<float,16> _t_29;
	generic_array<float,16> _t_30;
	generic_array<float,16> _t_31;
	generic_array<float,16> _t_32;
	generic_array<float,16> _t_33;
	generic_array<float,16> _t_34;
	generic_array<float,16> _t_35;
	generic_array<float,16> _t_36;
	generic_array<float,16> _t_37;
	generic_array<float,16> _t_38;
	generic_array<float,16> _t_39;
	generic_array<float,16> _t_40;
	generic_array<float,16> _t_41;
	generic_array<float,16> _t_42;
	generic_array<float,16> _t_43;
	generic_array<float,16> _t_44;
	generic_array<float,16> _t_45;
	generic_array<float,16> _t_46;
	generic_array<float,16> _t_47;
	generic_array<float,16> _t_48;
	generic_array<float,16> _t_49;
	generic_array<float,16> _t_50;
	generic_array<float,16> _t_51;
	generic_array<float,16> _t_52;
	generic_array<float,16> _t_53;
	generic_array<float,16> _t_54;
	generic_array<float,16> _t_55;
	generic_array<float,16> _t_56;
	generic_array<float,16> _t_57;
	generic_array<float,16> _t_58;
	generic_array<float,16> _t_59;
	generic_array<float,16> _t_60;
	generic_array<float,16> _t_61;
	generic_array<float,16> _t_62;
	generic_array<float,16> _t_63;
	generic_array<float,16> _t_64;
	generic_array<float,16> _t_65;
	generic_array<float,16> _t_66;
	generic_array<float,16> _t_67;
	generic_array<float,16> _t_68;
	generic_array<float,16> _t_69;
	generic_array<float,16> _t_70;
	generic_array<float,16> _t_71;
	generic_array<float,16> _t_72;
	generic_array<float,16> _t_73;
	generic_array<float,16> _t_74;
	generic_array<float,16> _t_75;
	generic_array<float,16> _t_76;
	generic_array<float,16> _t_77;
	generic_array<float,16> _t_78;
	generic_array<float,16> _t_79;
	generic_array<float,16> _t_80;
	generic_array<float,16> _t_81;
	generic_array<float,16> _t_82;
	generic_array<float,16> _t_83;
	generic_array<float,16> _t_84;
	generic_array<float,16> _t_85;
	generic_array<float,16> _t_86;
	generic_array<float,16> _t_87;
	generic_array<float,16> _t_88;
	generic_array<float,16> _t_89;
	generic_array<float,16> _t_90;
	generic_array<float,16> _t_91;
	generic_array<float,16> _t_92;
	generic_array<float,16> _t_93;
	generic_array<float,16> _t_94;
	float _t_95;
	float _t_96;
	float _t_97;
	bool _t_98;
	float _t_101;
	float _t_105;
	float _t_106;
	float _t_107;
	float _t_108;
	bool _t_109;
	float _t_112;
	float _t_116;
	float _t_117;
	float _t_118;
	float _t_119;
	float _t_120;
	generic_array<float,16> _t_121;
	generic_array<float,16> _t_122;
	float _t_123;
	float _t_124;
	float _t_125;
	bool _t_126;
	float _t_129;
	float _t_133;
	float _t_134;
	float _t_135;
	float _t_136;
	bool _t_137;
	float _t_140;
	float _t_144;
	float _t_145;
	float _t_146;
	float _t_147;
	float _t_148;
	generic_array<float,16> _t_149;
	generic_array<float,16> _t_150;
	float _t_151;
	float _t_152;
	float _t_153;
	bool _t_154;
	float _t_157;
	float _t_161;
	float _t_162;
	float _t_163;
	float _t_164;
	bool _t_165;
	float _t_168;
	float _t_172;
	float _t_173;
	float _t_174;
	float _t_175;
	float _t_176;
	generic_array<float,16> _t_177;
	generic_array<float,16> _t_178;
	float _t_179;
	float _t_180;
	float _t_181;
	bool _t_182;
	float _t_185;
	float _t_189;
	float _t_190;
	float _t_191;
	float _t_192;
	bool _t_193;
	float _t_196;
	float _t_200;
	float _t_201;
	float _t_202;
	float _t_203;
	float _t_204;
	generic_array<float,16> _t_205;
	generic_array<float,16> _t_206;
	float _t_207;
	float _t_208;
	float _t_209;
	bool _t_210;
	float _t_213;
	float _t_217;
	float _t_218;
	float _t_219;
	float _t_220;
	bool _t_221;
	float _t_224;
	float _t_228;
	float _t_229;
	float _t_230;
	float _t_231;
	float _t_232;
	generic_array<float,16> _t_233;
	generic_array<float,16> _t_234;
	float _t_235;
	float _t_236;
	float _t_237;
	bool _t_238;
	float _t_241;
	float _t_245;
	float _t_246;
	float _t_247;
	float _t_248;
	bool _t_249;
	float _t_252;
	float _t_256;
	float _t_257;
	float _t_258;
	float _t_259;
	float _t_260;
	generic_array<float,16> _t_261;
	generic_array<float,16> _t_262;
	float _t_263;
	float _t_264;
	float _t_265;
	bool _t_266;
	float _t_269;
	float _t_273;
	float _t_274;
	float _t_275;
	float _t_276;
	bool _t_277;
	float _t_280;
	float _t_284;
	float _t_285;
	float _t_286;
	float _t_287;
	float _t_288;
	generic_array<float,16> _t_289;
	generic_array<float,16> _t_290;
	float _t_291;
	float _t_292;
	float _t_293;
	bool _t_294;
	float _t_297;
	float _t_301;
	float _t_302;
	float _t_303;
	float _t_304;
	bool _t_305;
	float _t_308;
	float _t_312;
	float _t_313;
	float _t_314;
	float _t_315;
	float _t_316;
	generic_array<float,16> _t_317;
	generic_array<float,16> _t_318;
	float _t_319;
	float _t_320;
	float _t_321;
	bool _t_322;
	float _t_325;
	float _t_329;
	float _t_330;
	float _t_331;
	float _t_332;
	bool _t_333;
	float _t_336;
	float _t_340;
	float _t_341;
	float _t_342;
	float _t_343;
	float _t_344;
	generic_array<float,16> _t_345;
	generic_array<float,16> _t_346;
	float _t_347;
	float _t_348;
	float _t_349;
	bool _t_350;
	float _t_353;
	float _t_357;
	float _t_358;
	float _t_359;
	float _t_360;
	bool _t_361;
	float _t_364;
	float _t_368;
	float _t_369;
	float _t_370;
	float _t_371;
	float _t_372;
	generic_array<float,16> _t_373;
	generic_array<float,16> _t_374;
	float _t_375;
	float _t_376;
	float _t_377;
	bool _t_378;
	float _t_381;
	float _t_385;
	float _t_386;
	float _t_387;
	float _t_388;
	bool _t_389;
	float _t_392;
	float _t_396;
	float _t_397;
	float _t_398;
	float _t_399;
	float _t_400;
	generic_array<float,16> _t_401;
	generic_array<float,16> _t_402;
	float _t_403;
	float _t_404;
	float _t_405;
	bool _t_406;
	float _t_409;
	float _t_413;
	float _t_414;
	float _t_415;
	float _t_416;
	bool _t_417;
	float _t_420;
	float _t_424;
	float _t_425;
	float _t_426;
	float _t_427;
	float _t_428;
	generic_array<float,16> _t_429;
	generic_array<float,16> _t_430;
	float _t_431;
	float _t_432;
	float _t_433;
	bool _t_434;
	float _t_437;
	float _t_441;
	float _t_442;
	float _t_443;
	float _t_444;
	bool _t_445;
	float _t_448;
	float _t_452;
	float _t_453;
	float _t_454;
	float _t_455;
	float _t_456;
	generic_array<float,16> _t_457;
	generic_array<float,16> _t_458;
	float _t_459;
	float _t_460;
	float _t_461;
	bool _t_462;
	float _t_465;
	float _t_469;
	float _t_470;
	float _t_471;
	float _t_472;
	bool _t_473;
	float _t_476;
	float _t_480;
	float _t_481;
	float _t_482;
	float _t_483;
	float _t_484;
	generic_array<float,16> _t_485;
	generic_array<float,16> _t_486;
	float _t_487;
	float _t_488;
	float _t_489;
	bool _t_490;
	float _t_493;
	float _t_497;
	float _t_498;
	float _t_499;
	float _t_500;
	bool _t_501;
	float _t_504;
	float _t_508;
	float _t_509;
	float _t_510;
	float _t_511;
	float _t_512;
	generic_array<float,16> _t_513;
	generic_array<float,16> _t_514;
	float _t_515;
	float _t_516;
	float _t_517;
	bool _t_518;
	float _t_521;
	float _t_525;
	float _t_526;
	float _t_527;
	float _t_528;
	bool _t_529;
	float _t_532;
	float _t_536;
	float _t_537;
	float _t_538;
	float _t_539;
	float _t_540;
	generic_array<float,16> _t_541;
	generic_array<float,16> _t_542;
	float _t_543;
	float _t_544;
	float _t_545;
	bool _t_546;
	float _t_549;
	float _t_553;
	float _t_554;
	float _t_555;
	float _t_556;
	bool _t_557;
	float _t_560;
	float _t_564;
	float _t_565;
	float _t_566;
	float _t_567;
	float _t_568;
	generic_array<float,16> _t_569;
	generic_array<float,16> _t_570;
	float _t_571;
	float _t_572;
	float _t_573;
	bool _t_574;
	float _t_577;
	float _t_581;
	float _t_582;
	float _t_583;
	float _t_584;
	bool _t_585;
	float _t_588;
	float _t_592;
	float _t_593;
	float _t_594;
	float _t_595;
	float _t_596;
	generic_array<float,16> _t_597;

	generic_array<float,16> _t_598;

	_t_2 = tegpixelintegrator_1(tx1_4_1,tx3_6_1,pc1_15_1,pc2_16_1,tx2_5_1,ty3_9_1,tc0_17_1,py1_13_1,py0_12_1,ty1_7_1,tc2_19_1,tc1_18_1,px0_10_1,ty2_8_1,pc0_14_1);
	_t_4 = tegpixelintegrator_2(tx1_4_1,tx3_6_1,pc1_15_1,pc2_16_1,tx2_5_1,ty3_9_1,tc0_17_1,px1_11_1,py1_13_1,py0_12_1,ty1_7_1,tc2_19_1,tc1_18_1,ty2_8_1,pc0_14_1);
	_t_6 = tegpixelintegrator_3(tx1_4_1,tx3_6_1,pc1_15_1,pc2_16_1,tx2_5_1,ty3_9_1,tc0_17_1,px1_11_1,py0_12_1,ty1_7_1,tc2_19_1,tc1_18_1,px0_10_1,ty2_8_1,pc0_14_1);
	_t_7 = -1.0f * _t_6;
	_t_9 = tegpixelintegrator_4(tx1_4_1,tx3_6_1,pc1_15_1,pc2_16_1,tx2_5_1,py1_13_1,tc0_17_1,ty3_9_1,px1_11_1,ty1_7_1,tc2_19_1,tc1_18_1,px0_10_1,ty2_8_1,pc0_14_1);
	_t_11 = tegpixelintegrator_5(tx1_4_1,tx3_6_1,tx2_5_1,ty3_9_1,py1_13_1,px1_11_1,tc0_17_1,py0_12_1,pc0_14_1,px0_10_1,ty2_8_1,ty1_7_1);
	_t_13 = tegpixelintegrator_6(tx1_4_1,tx3_6_1,tx2_5_1,ty3_9_1,pc1_15_1,py1_13_1,px1_11_1,py0_12_1,tc1_18_1,px0_10_1,ty2_8_1,ty1_7_1);
	_t_15 = tegpixelintegrator_7(tx1_4_1,tx3_6_1,tx2_5_1,ty3_9_1,pc2_16_1,py1_13_1,px1_11_1,py0_12_1,tc2_19_1,px0_10_1,ty2_8_1,ty1_7_1);
	_t_17 = tegpixelintegrator_8(tx1_4_1,tx3_6_1,tx2_5_1,ty3_9_1,py1_13_1,px1_11_1,tc0_17_1,py0_12_1,pc0_14_1,px0_10_1,ty2_8_1,ty1_7_1);
	_t_19 = tegpixelintegrator_9(tx1_4_1,tx3_6_1,tx2_5_1,ty3_9_1,pc1_15_1,py1_13_1,px1_11_1,py0_12_1,tc1_18_1,px0_10_1,ty2_8_1,ty1_7_1);
	_t_21 = tegpixelintegrator_10(tx1_4_1,tx3_6_1,tx2_5_1,ty3_9_1,pc2_16_1,py1_13_1,px1_11_1,py0_12_1,tc2_19_1,px0_10_1,ty2_8_1,ty1_7_1);
	_t_22[0] =0.0f;
_t_22[1] =0.0f;
_t_22[2] =0.0f;
_t_22[3] =0.0f;
_t_22[4] =0.0f;
_t_22[5] =0.0f;
_t_22[6] =_t_2;
_t_22[7] =_t_4;
_t_22[8] =_t_7;
_t_22[9] =_t_9;
_t_22[10] =_t_11;
_t_22[11] =_t_13;
_t_22[12] =_t_15;
_t_22[13] =_t_17;
_t_22[14] =_t_19;
_t_22[15] =_t_21;
	_t_23[0] =0.0f;
_t_23[1] =0.0f;
_t_23[2] =0.0f;
_t_23[3] =0.0f;
_t_23[4] =0.0f;
_t_23[5] =0.0f;
_t_23[6] =0.0f;
_t_23[7] =0.0f;
_t_23[8] =0.0f;
_t_23[9] =0.0f;
_t_23[10] =0.0f;
_t_23[11] =0.0f;
_t_23[12] =0.0f;
_t_23[13] =0.0f;
_t_23[14] =0.0f;
_t_23[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_24[__iter__] = _t_22[__iter__] + _t_23[__iter__];
	_t_25[0] =0.0f;
_t_25[1] =0.0f;
_t_25[2] =0.0f;
_t_25[3] =0.0f;
_t_25[4] =0.0f;
_t_25[5] =0.0f;
_t_25[6] =0.0f;
_t_25[7] =0.0f;
_t_25[8] =0.0f;
_t_25[9] =0.0f;
_t_25[10] =0.0f;
_t_25[11] =0.0f;
_t_25[12] =0.0f;
_t_25[13] =0.0f;
_t_25[14] =0.0f;
_t_25[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_26[__iter__] = _t_24[__iter__] + _t_25[__iter__];
	_t_27[0] =0.0f;
_t_27[1] =0.0f;
_t_27[2] =0.0f;
_t_27[3] =0.0f;
_t_27[4] =0.0f;
_t_27[5] =0.0f;
_t_27[6] =0.0f;
_t_27[7] =0.0f;
_t_27[8] =0.0f;
_t_27[9] =0.0f;
_t_27[10] =0.0f;
_t_27[11] =0.0f;
_t_27[12] =0.0f;
_t_27[13] =0.0f;
_t_27[14] =0.0f;
_t_27[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_28[__iter__] = _t_26[__iter__] + _t_27[__iter__];
	_t_29[0] =0.0f;
_t_29[1] =0.0f;
_t_29[2] =0.0f;
_t_29[3] =0.0f;
_t_29[4] =0.0f;
_t_29[5] =0.0f;
_t_29[6] =0.0f;
_t_29[7] =0.0f;
_t_29[8] =0.0f;
_t_29[9] =0.0f;
_t_29[10] =0.0f;
_t_29[11] =0.0f;
_t_29[12] =0.0f;
_t_29[13] =0.0f;
_t_29[14] =0.0f;
_t_29[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_30[__iter__] = _t_28[__iter__] + _t_29[__iter__];
	_t_31[0] =0.0f;
_t_31[1] =0.0f;
_t_31[2] =0.0f;
_t_31[3] =0.0f;
_t_31[4] =0.0f;
_t_31[5] =0.0f;
_t_31[6] =0.0f;
_t_31[7] =0.0f;
_t_31[8] =0.0f;
_t_31[9] =0.0f;
_t_31[10] =0.0f;
_t_31[11] =0.0f;
_t_31[12] =0.0f;
_t_31[13] =0.0f;
_t_31[14] =0.0f;
_t_31[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_32[__iter__] = _t_30[__iter__] + _t_31[__iter__];
	_t_33[0] =0.0f;
_t_33[1] =0.0f;
_t_33[2] =0.0f;
_t_33[3] =0.0f;
_t_33[4] =0.0f;
_t_33[5] =0.0f;
_t_33[6] =0.0f;
_t_33[7] =0.0f;
_t_33[8] =0.0f;
_t_33[9] =0.0f;
_t_33[10] =0.0f;
_t_33[11] =0.0f;
_t_33[12] =0.0f;
_t_33[13] =0.0f;
_t_33[14] =0.0f;
_t_33[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_34[__iter__] = _t_32[__iter__] + _t_33[__iter__];
	_t_35[0] =0.0f;
_t_35[1] =0.0f;
_t_35[2] =0.0f;
_t_35[3] =0.0f;
_t_35[4] =0.0f;
_t_35[5] =0.0f;
_t_35[6] =0.0f;
_t_35[7] =0.0f;
_t_35[8] =0.0f;
_t_35[9] =0.0f;
_t_35[10] =0.0f;
_t_35[11] =0.0f;
_t_35[12] =0.0f;
_t_35[13] =0.0f;
_t_35[14] =0.0f;
_t_35[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_36[__iter__] = _t_34[__iter__] + _t_35[__iter__];
	_t_37[0] =0.0f;
_t_37[1] =0.0f;
_t_37[2] =0.0f;
_t_37[3] =0.0f;
_t_37[4] =0.0f;
_t_37[5] =0.0f;
_t_37[6] =0.0f;
_t_37[7] =0.0f;
_t_37[8] =0.0f;
_t_37[9] =0.0f;
_t_37[10] =0.0f;
_t_37[11] =0.0f;
_t_37[12] =0.0f;
_t_37[13] =0.0f;
_t_37[14] =0.0f;
_t_37[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_38[__iter__] = _t_36[__iter__] + _t_37[__iter__];
	_t_39[0] =0.0f;
_t_39[1] =0.0f;
_t_39[2] =0.0f;
_t_39[3] =0.0f;
_t_39[4] =0.0f;
_t_39[5] =0.0f;
_t_39[6] =0.0f;
_t_39[7] =0.0f;
_t_39[8] =0.0f;
_t_39[9] =0.0f;
_t_39[10] =0.0f;
_t_39[11] =0.0f;
_t_39[12] =0.0f;
_t_39[13] =0.0f;
_t_39[14] =0.0f;
_t_39[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_40[__iter__] = _t_38[__iter__] + _t_39[__iter__];
	_t_41[0] =0.0f;
_t_41[1] =0.0f;
_t_41[2] =0.0f;
_t_41[3] =0.0f;
_t_41[4] =0.0f;
_t_41[5] =0.0f;
_t_41[6] =0.0f;
_t_41[7] =0.0f;
_t_41[8] =0.0f;
_t_41[9] =0.0f;
_t_41[10] =0.0f;
_t_41[11] =0.0f;
_t_41[12] =0.0f;
_t_41[13] =0.0f;
_t_41[14] =0.0f;
_t_41[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_42[__iter__] = _t_40[__iter__] + _t_41[__iter__];
	_t_43[0] =0.0f;
_t_43[1] =0.0f;
_t_43[2] =0.0f;
_t_43[3] =0.0f;
_t_43[4] =0.0f;
_t_43[5] =0.0f;
_t_43[6] =0.0f;
_t_43[7] =0.0f;
_t_43[8] =0.0f;
_t_43[9] =0.0f;
_t_43[10] =0.0f;
_t_43[11] =0.0f;
_t_43[12] =0.0f;
_t_43[13] =0.0f;
_t_43[14] =0.0f;
_t_43[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_44[__iter__] = _t_42[__iter__] + _t_43[__iter__];
	_t_45[0] =0.0f;
_t_45[1] =0.0f;
_t_45[2] =0.0f;
_t_45[3] =0.0f;
_t_45[4] =0.0f;
_t_45[5] =0.0f;
_t_45[6] =0.0f;
_t_45[7] =0.0f;
_t_45[8] =0.0f;
_t_45[9] =0.0f;
_t_45[10] =0.0f;
_t_45[11] =0.0f;
_t_45[12] =0.0f;
_t_45[13] =0.0f;
_t_45[14] =0.0f;
_t_45[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_46[__iter__] = _t_44[__iter__] + _t_45[__iter__];
	_t_47[0] =0.0f;
_t_47[1] =0.0f;
_t_47[2] =0.0f;
_t_47[3] =0.0f;
_t_47[4] =0.0f;
_t_47[5] =0.0f;
_t_47[6] =0.0f;
_t_47[7] =0.0f;
_t_47[8] =0.0f;
_t_47[9] =0.0f;
_t_47[10] =0.0f;
_t_47[11] =0.0f;
_t_47[12] =0.0f;
_t_47[13] =0.0f;
_t_47[14] =0.0f;
_t_47[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_48[__iter__] = _t_46[__iter__] + _t_47[__iter__];
	_t_49[0] =0.0f;
_t_49[1] =0.0f;
_t_49[2] =0.0f;
_t_49[3] =0.0f;
_t_49[4] =0.0f;
_t_49[5] =0.0f;
_t_49[6] =0.0f;
_t_49[7] =0.0f;
_t_49[8] =0.0f;
_t_49[9] =0.0f;
_t_49[10] =0.0f;
_t_49[11] =0.0f;
_t_49[12] =0.0f;
_t_49[13] =0.0f;
_t_49[14] =0.0f;
_t_49[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_50[__iter__] = _t_48[__iter__] + _t_49[__iter__];
	_t_51[0] =0.0f;
_t_51[1] =0.0f;
_t_51[2] =0.0f;
_t_51[3] =0.0f;
_t_51[4] =0.0f;
_t_51[5] =0.0f;
_t_51[6] =0.0f;
_t_51[7] =0.0f;
_t_51[8] =0.0f;
_t_51[9] =0.0f;
_t_51[10] =0.0f;
_t_51[11] =0.0f;
_t_51[12] =0.0f;
_t_51[13] =0.0f;
_t_51[14] =0.0f;
_t_51[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_52[__iter__] = _t_50[__iter__] + _t_51[__iter__];
	_t_53[0] =0.0f;
_t_53[1] =0.0f;
_t_53[2] =0.0f;
_t_53[3] =0.0f;
_t_53[4] =0.0f;
_t_53[5] =0.0f;
_t_53[6] =0.0f;
_t_53[7] =0.0f;
_t_53[8] =0.0f;
_t_53[9] =0.0f;
_t_53[10] =0.0f;
_t_53[11] =0.0f;
_t_53[12] =0.0f;
_t_53[13] =0.0f;
_t_53[14] =0.0f;
_t_53[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_54[__iter__] = _t_52[__iter__] + _t_53[__iter__];
	_t_55[0] =0.0f;
_t_55[1] =0.0f;
_t_55[2] =0.0f;
_t_55[3] =0.0f;
_t_55[4] =0.0f;
_t_55[5] =0.0f;
_t_55[6] =0.0f;
_t_55[7] =0.0f;
_t_55[8] =0.0f;
_t_55[9] =0.0f;
_t_55[10] =0.0f;
_t_55[11] =0.0f;
_t_55[12] =0.0f;
_t_55[13] =0.0f;
_t_55[14] =0.0f;
_t_55[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_56[__iter__] = _t_54[__iter__] + _t_55[__iter__];
	_t_57[0] =0.0f;
_t_57[1] =0.0f;
_t_57[2] =0.0f;
_t_57[3] =0.0f;
_t_57[4] =0.0f;
_t_57[5] =0.0f;
_t_57[6] =0.0f;
_t_57[7] =0.0f;
_t_57[8] =0.0f;
_t_57[9] =0.0f;
_t_57[10] =0.0f;
_t_57[11] =0.0f;
_t_57[12] =0.0f;
_t_57[13] =0.0f;
_t_57[14] =0.0f;
_t_57[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_58[__iter__] = _t_56[__iter__] + _t_57[__iter__];
	_t_59[0] =0.0f;
_t_59[1] =0.0f;
_t_59[2] =0.0f;
_t_59[3] =0.0f;
_t_59[4] =0.0f;
_t_59[5] =0.0f;
_t_59[6] =0.0f;
_t_59[7] =0.0f;
_t_59[8] =0.0f;
_t_59[9] =0.0f;
_t_59[10] =0.0f;
_t_59[11] =0.0f;
_t_59[12] =0.0f;
_t_59[13] =0.0f;
_t_59[14] =0.0f;
_t_59[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_60[__iter__] = _t_58[__iter__] + _t_59[__iter__];
	_t_61[0] =0.0f;
_t_61[1] =0.0f;
_t_61[2] =0.0f;
_t_61[3] =0.0f;
_t_61[4] =0.0f;
_t_61[5] =0.0f;
_t_61[6] =0.0f;
_t_61[7] =0.0f;
_t_61[8] =0.0f;
_t_61[9] =0.0f;
_t_61[10] =0.0f;
_t_61[11] =0.0f;
_t_61[12] =0.0f;
_t_61[13] =0.0f;
_t_61[14] =0.0f;
_t_61[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_62[__iter__] = _t_60[__iter__] + _t_61[__iter__];
	_t_63[0] =0.0f;
_t_63[1] =0.0f;
_t_63[2] =0.0f;
_t_63[3] =0.0f;
_t_63[4] =0.0f;
_t_63[5] =0.0f;
_t_63[6] =0.0f;
_t_63[7] =0.0f;
_t_63[8] =0.0f;
_t_63[9] =0.0f;
_t_63[10] =0.0f;
_t_63[11] =0.0f;
_t_63[12] =0.0f;
_t_63[13] =0.0f;
_t_63[14] =0.0f;
_t_63[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_64[__iter__] = _t_62[__iter__] + _t_63[__iter__];
	_t_65[0] =0.0f;
_t_65[1] =0.0f;
_t_65[2] =0.0f;
_t_65[3] =0.0f;
_t_65[4] =0.0f;
_t_65[5] =0.0f;
_t_65[6] =0.0f;
_t_65[7] =0.0f;
_t_65[8] =0.0f;
_t_65[9] =0.0f;
_t_65[10] =0.0f;
_t_65[11] =0.0f;
_t_65[12] =0.0f;
_t_65[13] =0.0f;
_t_65[14] =0.0f;
_t_65[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_66[__iter__] = _t_64[__iter__] + _t_65[__iter__];
	_t_67[0] =0.0f;
_t_67[1] =0.0f;
_t_67[2] =0.0f;
_t_67[3] =0.0f;
_t_67[4] =0.0f;
_t_67[5] =0.0f;
_t_67[6] =0.0f;
_t_67[7] =0.0f;
_t_67[8] =0.0f;
_t_67[9] =0.0f;
_t_67[10] =0.0f;
_t_67[11] =0.0f;
_t_67[12] =0.0f;
_t_67[13] =0.0f;
_t_67[14] =0.0f;
_t_67[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_68[__iter__] = _t_66[__iter__] + _t_67[__iter__];
	_t_69[0] =0.0f;
_t_69[1] =0.0f;
_t_69[2] =0.0f;
_t_69[3] =0.0f;
_t_69[4] =0.0f;
_t_69[5] =0.0f;
_t_69[6] =0.0f;
_t_69[7] =0.0f;
_t_69[8] =0.0f;
_t_69[9] =0.0f;
_t_69[10] =0.0f;
_t_69[11] =0.0f;
_t_69[12] =0.0f;
_t_69[13] =0.0f;
_t_69[14] =0.0f;
_t_69[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_70[__iter__] = _t_68[__iter__] + _t_69[__iter__];
	_t_71[0] =0.0f;
_t_71[1] =0.0f;
_t_71[2] =0.0f;
_t_71[3] =0.0f;
_t_71[4] =0.0f;
_t_71[5] =0.0f;
_t_71[6] =0.0f;
_t_71[7] =0.0f;
_t_71[8] =0.0f;
_t_71[9] =0.0f;
_t_71[10] =0.0f;
_t_71[11] =0.0f;
_t_71[12] =0.0f;
_t_71[13] =0.0f;
_t_71[14] =0.0f;
_t_71[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_72[__iter__] = _t_70[__iter__] + _t_71[__iter__];
	_t_73[0] =0.0f;
_t_73[1] =0.0f;
_t_73[2] =0.0f;
_t_73[3] =0.0f;
_t_73[4] =0.0f;
_t_73[5] =0.0f;
_t_73[6] =0.0f;
_t_73[7] =0.0f;
_t_73[8] =0.0f;
_t_73[9] =0.0f;
_t_73[10] =0.0f;
_t_73[11] =0.0f;
_t_73[12] =0.0f;
_t_73[13] =0.0f;
_t_73[14] =0.0f;
_t_73[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_74[__iter__] = _t_72[__iter__] + _t_73[__iter__];
	_t_75[0] =0.0f;
_t_75[1] =0.0f;
_t_75[2] =0.0f;
_t_75[3] =0.0f;
_t_75[4] =0.0f;
_t_75[5] =0.0f;
_t_75[6] =0.0f;
_t_75[7] =0.0f;
_t_75[8] =0.0f;
_t_75[9] =0.0f;
_t_75[10] =0.0f;
_t_75[11] =0.0f;
_t_75[12] =0.0f;
_t_75[13] =0.0f;
_t_75[14] =0.0f;
_t_75[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_76[__iter__] = _t_74[__iter__] + _t_75[__iter__];
	_t_77[0] =0.0f;
_t_77[1] =0.0f;
_t_77[2] =0.0f;
_t_77[3] =0.0f;
_t_77[4] =0.0f;
_t_77[5] =0.0f;
_t_77[6] =0.0f;
_t_77[7] =0.0f;
_t_77[8] =0.0f;
_t_77[9] =0.0f;
_t_77[10] =0.0f;
_t_77[11] =0.0f;
_t_77[12] =0.0f;
_t_77[13] =0.0f;
_t_77[14] =0.0f;
_t_77[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_78[__iter__] = _t_76[__iter__] + _t_77[__iter__];
	_t_79[0] =0.0f;
_t_79[1] =0.0f;
_t_79[2] =0.0f;
_t_79[3] =0.0f;
_t_79[4] =0.0f;
_t_79[5] =0.0f;
_t_79[6] =0.0f;
_t_79[7] =0.0f;
_t_79[8] =0.0f;
_t_79[9] =0.0f;
_t_79[10] =0.0f;
_t_79[11] =0.0f;
_t_79[12] =0.0f;
_t_79[13] =0.0f;
_t_79[14] =0.0f;
_t_79[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_80[__iter__] = _t_78[__iter__] + _t_79[__iter__];
	_t_81[0] =0.0f;
_t_81[1] =0.0f;
_t_81[2] =0.0f;
_t_81[3] =0.0f;
_t_81[4] =0.0f;
_t_81[5] =0.0f;
_t_81[6] =0.0f;
_t_81[7] =0.0f;
_t_81[8] =0.0f;
_t_81[9] =0.0f;
_t_81[10] =0.0f;
_t_81[11] =0.0f;
_t_81[12] =0.0f;
_t_81[13] =0.0f;
_t_81[14] =0.0f;
_t_81[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_82[__iter__] = _t_80[__iter__] + _t_81[__iter__];
	_t_83[0] =0.0f;
_t_83[1] =0.0f;
_t_83[2] =0.0f;
_t_83[3] =0.0f;
_t_83[4] =0.0f;
_t_83[5] =0.0f;
_t_83[6] =0.0f;
_t_83[7] =0.0f;
_t_83[8] =0.0f;
_t_83[9] =0.0f;
_t_83[10] =0.0f;
_t_83[11] =0.0f;
_t_83[12] =0.0f;
_t_83[13] =0.0f;
_t_83[14] =0.0f;
_t_83[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_84[__iter__] = _t_82[__iter__] + _t_83[__iter__];
	_t_85[0] =0.0f;
_t_85[1] =0.0f;
_t_85[2] =0.0f;
_t_85[3] =0.0f;
_t_85[4] =0.0f;
_t_85[5] =0.0f;
_t_85[6] =0.0f;
_t_85[7] =0.0f;
_t_85[8] =0.0f;
_t_85[9] =0.0f;
_t_85[10] =0.0f;
_t_85[11] =0.0f;
_t_85[12] =0.0f;
_t_85[13] =0.0f;
_t_85[14] =0.0f;
_t_85[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_86[__iter__] = _t_84[__iter__] + _t_85[__iter__];
	_t_87[0] =0.0f;
_t_87[1] =0.0f;
_t_87[2] =0.0f;
_t_87[3] =0.0f;
_t_87[4] =0.0f;
_t_87[5] =0.0f;
_t_87[6] =0.0f;
_t_87[7] =0.0f;
_t_87[8] =0.0f;
_t_87[9] =0.0f;
_t_87[10] =0.0f;
_t_87[11] =0.0f;
_t_87[12] =0.0f;
_t_87[13] =0.0f;
_t_87[14] =0.0f;
_t_87[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_88[__iter__] = _t_86[__iter__] + _t_87[__iter__];
	_t_89[0] =0.0f;
_t_89[1] =0.0f;
_t_89[2] =0.0f;
_t_89[3] =0.0f;
_t_89[4] =0.0f;
_t_89[5] =0.0f;
_t_89[6] =0.0f;
_t_89[7] =0.0f;
_t_89[8] =0.0f;
_t_89[9] =0.0f;
_t_89[10] =0.0f;
_t_89[11] =0.0f;
_t_89[12] =0.0f;
_t_89[13] =0.0f;
_t_89[14] =0.0f;
_t_89[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_90[__iter__] = _t_88[__iter__] + _t_89[__iter__];
	_t_91[0] =0.0f;
_t_91[1] =0.0f;
_t_91[2] =0.0f;
_t_91[3] =0.0f;
_t_91[4] =0.0f;
_t_91[5] =0.0f;
_t_91[6] =0.0f;
_t_91[7] =0.0f;
_t_91[8] =0.0f;
_t_91[9] =0.0f;
_t_91[10] =0.0f;
_t_91[11] =0.0f;
_t_91[12] =0.0f;
_t_91[13] =0.0f;
_t_91[14] =0.0f;
_t_91[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_92[__iter__] = _t_90[__iter__] + _t_91[__iter__];
	_t_93[0] =0.0f;
_t_93[1] =0.0f;
_t_93[2] =0.0f;
_t_93[3] =0.0f;
_t_93[4] =0.0f;
_t_93[5] =0.0f;
_t_93[6] =0.0f;
_t_93[7] =0.0f;
_t_93[8] =0.0f;
_t_93[9] =0.0f;
_t_93[10] =0.0f;
_t_93[11] =0.0f;
_t_93[12] =0.0f;
_t_93[13] =0.0f;
_t_93[14] =0.0f;
_t_93[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_94[__iter__] = _t_92[__iter__] + _t_93[__iter__];
	_t_95 = -1.0f * ty2_8_1;
	_t_96 = ty1_7_1 + _t_95;
	_t_97 = -1.0f * _t_96;
	_t_98 = _t_97 < 0.0f;
	if(_t_98)
		{
			float _t_99;
			float _t_100;
		
			_t_99 = -1.0f * ty2_8_1;
			_t_100 = ty1_7_1 + _t_99;
			_t_101 = _t_100;
		
		}
else
		{
			float _t_102;
			float _t_103;
			float _t_104;
		
			_t_102 = -1.0f * ty2_8_1;
			_t_103 = ty1_7_1 + _t_102;
			_t_104 = -1.0f * _t_103;
			_t_101 = _t_104;
		
		}

	_t_105 = _t_101 * _t_101;
	_t_106 = -1.0f * ty2_8_1;
	_t_107 = ty1_7_1 + _t_106;
	_t_108 = -1.0f * _t_107;
	_t_109 = _t_108 < 0.0f;
	if(_t_109)
		{
			float _t_110;
			float _t_111;
		
			_t_110 = -1.0f * tx1_4_1;
			_t_111 = tx2_5_1 + _t_110;
			_t_112 = _t_111;
		
		}
else
		{
			float _t_113;
			float _t_114;
			float _t_115;
		
			_t_113 = -1.0f * tx1_4_1;
			_t_114 = tx2_5_1 + _t_113;
			_t_115 = -1.0f * _t_114;
			_t_112 = _t_115;
		
		}

	_t_116 = _t_112 * _t_112;
	_t_117 = _t_105 + _t_116;
	_t_118 = sqrt(_t_117);
	_t_119 = 1.0f / _t_118;
	_t_120 = tegpixellet_block_1(ty1_7_1,ty2_8_1,tx2_5_1,tx1_4_1,_t_119,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);
	_t_121[0] =_t_120;
_t_121[1] =0.0f;
_t_121[2] =0.0f;
_t_121[3] =0.0f;
_t_121[4] =0.0f;
_t_121[5] =0.0f;
_t_121[6] =0.0f;
_t_121[7] =0.0f;
_t_121[8] =0.0f;
_t_121[9] =0.0f;
_t_121[10] =0.0f;
_t_121[11] =0.0f;
_t_121[12] =0.0f;
_t_121[13] =0.0f;
_t_121[14] =0.0f;
_t_121[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_122[__iter__] = _t_94[__iter__] + _t_121[__iter__];
	_t_123 = -1.0f * ty1_7_1;
	_t_124 = ty3_9_1 + _t_123;
	_t_125 = -1.0f * _t_124;
	_t_126 = _t_125 < 0.0f;
	if(_t_126)
		{
			float _t_127;
			float _t_128;
		
			_t_127 = -1.0f * ty1_7_1;
			_t_128 = ty3_9_1 + _t_127;
			_t_129 = _t_128;
		
		}
else
		{
			float _t_130;
			float _t_131;
			float _t_132;
		
			_t_130 = -1.0f * ty1_7_1;
			_t_131 = ty3_9_1 + _t_130;
			_t_132 = -1.0f * _t_131;
			_t_129 = _t_132;
		
		}

	_t_133 = _t_129 * _t_129;
	_t_134 = -1.0f * ty1_7_1;
	_t_135 = ty3_9_1 + _t_134;
	_t_136 = -1.0f * _t_135;
	_t_137 = _t_136 < 0.0f;
	if(_t_137)
		{
			float _t_138;
			float _t_139;
		
			_t_138 = -1.0f * tx3_6_1;
			_t_139 = tx1_4_1 + _t_138;
			_t_140 = _t_139;
		
		}
else
		{
			float _t_141;
			float _t_142;
			float _t_143;
		
			_t_141 = -1.0f * tx3_6_1;
			_t_142 = tx1_4_1 + _t_141;
			_t_143 = -1.0f * _t_142;
			_t_140 = _t_143;
		
		}

	_t_144 = _t_140 * _t_140;
	_t_145 = _t_133 + _t_144;
	_t_146 = sqrt(_t_145);
	_t_147 = 1.0f / _t_146;
	_t_148 = tegpixellet_block_2(ty3_9_1,ty1_7_1,tx1_4_1,tx3_6_1,_t_147,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);
	_t_149[0] =_t_148;
_t_149[1] =0.0f;
_t_149[2] =0.0f;
_t_149[3] =0.0f;
_t_149[4] =0.0f;
_t_149[5] =0.0f;
_t_149[6] =0.0f;
_t_149[7] =0.0f;
_t_149[8] =0.0f;
_t_149[9] =0.0f;
_t_149[10] =0.0f;
_t_149[11] =0.0f;
_t_149[12] =0.0f;
_t_149[13] =0.0f;
_t_149[14] =0.0f;
_t_149[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_150[__iter__] = _t_122[__iter__] + _t_149[__iter__];
	_t_151 = -1.0f * ty1_7_1;
	_t_152 = ty3_9_1 + _t_151;
	_t_153 = -1.0f * _t_152;
	_t_154 = _t_153 < 0.0f;
	if(_t_154)
		{
			float _t_155;
			float _t_156;
		
			_t_155 = -1.0f * ty1_7_1;
			_t_156 = ty3_9_1 + _t_155;
			_t_157 = _t_156;
		
		}
else
		{
			float _t_158;
			float _t_159;
			float _t_160;
		
			_t_158 = -1.0f * ty1_7_1;
			_t_159 = ty3_9_1 + _t_158;
			_t_160 = -1.0f * _t_159;
			_t_157 = _t_160;
		
		}

	_t_161 = _t_157 * _t_157;
	_t_162 = -1.0f * ty1_7_1;
	_t_163 = ty3_9_1 + _t_162;
	_t_164 = -1.0f * _t_163;
	_t_165 = _t_164 < 0.0f;
	if(_t_165)
		{
			float _t_166;
			float _t_167;
		
			_t_166 = -1.0f * tx3_6_1;
			_t_167 = tx1_4_1 + _t_166;
			_t_168 = _t_167;
		
		}
else
		{
			float _t_169;
			float _t_170;
			float _t_171;
		
			_t_169 = -1.0f * tx3_6_1;
			_t_170 = tx1_4_1 + _t_169;
			_t_171 = -1.0f * _t_170;
			_t_168 = _t_171;
		
		}

	_t_172 = _t_168 * _t_168;
	_t_173 = _t_161 + _t_172;
	_t_174 = sqrt(_t_173);
	_t_175 = 1.0f / _t_174;
	_t_176 = tegpixellet_block_3(ty3_9_1,ty1_7_1,tx1_4_1,tx3_6_1,_t_175,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);
	_t_177[0] =_t_176;
_t_177[1] =0.0f;
_t_177[2] =0.0f;
_t_177[3] =0.0f;
_t_177[4] =0.0f;
_t_177[5] =0.0f;
_t_177[6] =0.0f;
_t_177[7] =0.0f;
_t_177[8] =0.0f;
_t_177[9] =0.0f;
_t_177[10] =0.0f;
_t_177[11] =0.0f;
_t_177[12] =0.0f;
_t_177[13] =0.0f;
_t_177[14] =0.0f;
_t_177[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_178[__iter__] = _t_150[__iter__] + _t_177[__iter__];
	_t_179 = -1.0f * ty2_8_1;
	_t_180 = ty1_7_1 + _t_179;
	_t_181 = -1.0f * _t_180;
	_t_182 = _t_181 < 0.0f;
	if(_t_182)
		{
			float _t_183;
			float _t_184;
		
			_t_183 = -1.0f * ty2_8_1;
			_t_184 = ty1_7_1 + _t_183;
			_t_185 = _t_184;
		
		}
else
		{
			float _t_186;
			float _t_187;
			float _t_188;
		
			_t_186 = -1.0f * ty2_8_1;
			_t_187 = ty1_7_1 + _t_186;
			_t_188 = -1.0f * _t_187;
			_t_185 = _t_188;
		
		}

	_t_189 = _t_185 * _t_185;
	_t_190 = -1.0f * ty2_8_1;
	_t_191 = ty1_7_1 + _t_190;
	_t_192 = -1.0f * _t_191;
	_t_193 = _t_192 < 0.0f;
	if(_t_193)
		{
			float _t_194;
			float _t_195;
		
			_t_194 = -1.0f * tx1_4_1;
			_t_195 = tx2_5_1 + _t_194;
			_t_196 = _t_195;
		
		}
else
		{
			float _t_197;
			float _t_198;
			float _t_199;
		
			_t_197 = -1.0f * tx1_4_1;
			_t_198 = tx2_5_1 + _t_197;
			_t_199 = -1.0f * _t_198;
			_t_196 = _t_199;
		
		}

	_t_200 = _t_196 * _t_196;
	_t_201 = _t_189 + _t_200;
	_t_202 = sqrt(_t_201);
	_t_203 = 1.0f / _t_202;
	_t_204 = tegpixellet_block_4(ty1_7_1,ty2_8_1,tx2_5_1,tx1_4_1,_t_203,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);
	_t_205[0] =0.0f;
_t_205[1] =_t_204;
_t_205[2] =0.0f;
_t_205[3] =0.0f;
_t_205[4] =0.0f;
_t_205[5] =0.0f;
_t_205[6] =0.0f;
_t_205[7] =0.0f;
_t_205[8] =0.0f;
_t_205[9] =0.0f;
_t_205[10] =0.0f;
_t_205[11] =0.0f;
_t_205[12] =0.0f;
_t_205[13] =0.0f;
_t_205[14] =0.0f;
_t_205[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_206[__iter__] = _t_178[__iter__] + _t_205[__iter__];
	_t_207 = -1.0f * ty3_9_1;
	_t_208 = ty2_8_1 + _t_207;
	_t_209 = -1.0f * _t_208;
	_t_210 = _t_209 < 0.0f;
	if(_t_210)
		{
			float _t_211;
			float _t_212;
		
			_t_211 = -1.0f * ty3_9_1;
			_t_212 = ty2_8_1 + _t_211;
			_t_213 = _t_212;
		
		}
else
		{
			float _t_214;
			float _t_215;
			float _t_216;
		
			_t_214 = -1.0f * ty3_9_1;
			_t_215 = ty2_8_1 + _t_214;
			_t_216 = -1.0f * _t_215;
			_t_213 = _t_216;
		
		}

	_t_217 = _t_213 * _t_213;
	_t_218 = -1.0f * ty3_9_1;
	_t_219 = ty2_8_1 + _t_218;
	_t_220 = -1.0f * _t_219;
	_t_221 = _t_220 < 0.0f;
	if(_t_221)
		{
			float _t_222;
			float _t_223;
		
			_t_222 = -1.0f * tx2_5_1;
			_t_223 = tx3_6_1 + _t_222;
			_t_224 = _t_223;
		
		}
else
		{
			float _t_225;
			float _t_226;
			float _t_227;
		
			_t_225 = -1.0f * tx2_5_1;
			_t_226 = tx3_6_1 + _t_225;
			_t_227 = -1.0f * _t_226;
			_t_224 = _t_227;
		
		}

	_t_228 = _t_224 * _t_224;
	_t_229 = _t_217 + _t_228;
	_t_230 = sqrt(_t_229);
	_t_231 = 1.0f / _t_230;
	_t_232 = tegpixellet_block_5(ty2_8_1,ty3_9_1,tx3_6_1,tx2_5_1,_t_231,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);
	_t_233[0] =0.0f;
_t_233[1] =_t_232;
_t_233[2] =0.0f;
_t_233[3] =0.0f;
_t_233[4] =0.0f;
_t_233[5] =0.0f;
_t_233[6] =0.0f;
_t_233[7] =0.0f;
_t_233[8] =0.0f;
_t_233[9] =0.0f;
_t_233[10] =0.0f;
_t_233[11] =0.0f;
_t_233[12] =0.0f;
_t_233[13] =0.0f;
_t_233[14] =0.0f;
_t_233[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_234[__iter__] = _t_206[__iter__] + _t_233[__iter__];
	_t_235 = -1.0f * ty3_9_1;
	_t_236 = ty2_8_1 + _t_235;
	_t_237 = -1.0f * _t_236;
	_t_238 = _t_237 < 0.0f;
	if(_t_238)
		{
			float _t_239;
			float _t_240;
		
			_t_239 = -1.0f * ty3_9_1;
			_t_240 = ty2_8_1 + _t_239;
			_t_241 = _t_240;
		
		}
else
		{
			float _t_242;
			float _t_243;
			float _t_244;
		
			_t_242 = -1.0f * ty3_9_1;
			_t_243 = ty2_8_1 + _t_242;
			_t_244 = -1.0f * _t_243;
			_t_241 = _t_244;
		
		}

	_t_245 = _t_241 * _t_241;
	_t_246 = -1.0f * ty3_9_1;
	_t_247 = ty2_8_1 + _t_246;
	_t_248 = -1.0f * _t_247;
	_t_249 = _t_248 < 0.0f;
	if(_t_249)
		{
			float _t_250;
			float _t_251;
		
			_t_250 = -1.0f * tx2_5_1;
			_t_251 = tx3_6_1 + _t_250;
			_t_252 = _t_251;
		
		}
else
		{
			float _t_253;
			float _t_254;
			float _t_255;
		
			_t_253 = -1.0f * tx2_5_1;
			_t_254 = tx3_6_1 + _t_253;
			_t_255 = -1.0f * _t_254;
			_t_252 = _t_255;
		
		}

	_t_256 = _t_252 * _t_252;
	_t_257 = _t_245 + _t_256;
	_t_258 = sqrt(_t_257);
	_t_259 = 1.0f / _t_258;
	_t_260 = tegpixellet_block_6(ty2_8_1,ty3_9_1,tx3_6_1,tx2_5_1,_t_259,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);
	_t_261[0] =0.0f;
_t_261[1] =_t_260;
_t_261[2] =0.0f;
_t_261[3] =0.0f;
_t_261[4] =0.0f;
_t_261[5] =0.0f;
_t_261[6] =0.0f;
_t_261[7] =0.0f;
_t_261[8] =0.0f;
_t_261[9] =0.0f;
_t_261[10] =0.0f;
_t_261[11] =0.0f;
_t_261[12] =0.0f;
_t_261[13] =0.0f;
_t_261[14] =0.0f;
_t_261[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_262[__iter__] = _t_234[__iter__] + _t_261[__iter__];
	_t_263 = -1.0f * ty3_9_1;
	_t_264 = ty2_8_1 + _t_263;
	_t_265 = -1.0f * _t_264;
	_t_266 = _t_265 < 0.0f;
	if(_t_266)
		{
			float _t_267;
			float _t_268;
		
			_t_267 = -1.0f * ty3_9_1;
			_t_268 = ty2_8_1 + _t_267;
			_t_269 = _t_268;
		
		}
else
		{
			float _t_270;
			float _t_271;
			float _t_272;
		
			_t_270 = -1.0f * ty3_9_1;
			_t_271 = ty2_8_1 + _t_270;
			_t_272 = -1.0f * _t_271;
			_t_269 = _t_272;
		
		}

	_t_273 = _t_269 * _t_269;
	_t_274 = -1.0f * ty3_9_1;
	_t_275 = ty2_8_1 + _t_274;
	_t_276 = -1.0f * _t_275;
	_t_277 = _t_276 < 0.0f;
	if(_t_277)
		{
			float _t_278;
			float _t_279;
		
			_t_278 = -1.0f * tx2_5_1;
			_t_279 = tx3_6_1 + _t_278;
			_t_280 = _t_279;
		
		}
else
		{
			float _t_281;
			float _t_282;
			float _t_283;
		
			_t_281 = -1.0f * tx2_5_1;
			_t_282 = tx3_6_1 + _t_281;
			_t_283 = -1.0f * _t_282;
			_t_280 = _t_283;
		
		}

	_t_284 = _t_280 * _t_280;
	_t_285 = _t_273 + _t_284;
	_t_286 = sqrt(_t_285);
	_t_287 = 1.0f / _t_286;
	_t_288 = tegpixellet_block_7(ty2_8_1,ty3_9_1,tx3_6_1,tx2_5_1,_t_287,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);
	_t_289[0] =0.0f;
_t_289[1] =0.0f;
_t_289[2] =_t_288;
_t_289[3] =0.0f;
_t_289[4] =0.0f;
_t_289[5] =0.0f;
_t_289[6] =0.0f;
_t_289[7] =0.0f;
_t_289[8] =0.0f;
_t_289[9] =0.0f;
_t_289[10] =0.0f;
_t_289[11] =0.0f;
_t_289[12] =0.0f;
_t_289[13] =0.0f;
_t_289[14] =0.0f;
_t_289[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_290[__iter__] = _t_262[__iter__] + _t_289[__iter__];
	_t_291 = -1.0f * ty1_7_1;
	_t_292 = ty3_9_1 + _t_291;
	_t_293 = -1.0f * _t_292;
	_t_294 = _t_293 < 0.0f;
	if(_t_294)
		{
			float _t_295;
			float _t_296;
		
			_t_295 = -1.0f * ty1_7_1;
			_t_296 = ty3_9_1 + _t_295;
			_t_297 = _t_296;
		
		}
else
		{
			float _t_298;
			float _t_299;
			float _t_300;
		
			_t_298 = -1.0f * ty1_7_1;
			_t_299 = ty3_9_1 + _t_298;
			_t_300 = -1.0f * _t_299;
			_t_297 = _t_300;
		
		}

	_t_301 = _t_297 * _t_297;
	_t_302 = -1.0f * ty1_7_1;
	_t_303 = ty3_9_1 + _t_302;
	_t_304 = -1.0f * _t_303;
	_t_305 = _t_304 < 0.0f;
	if(_t_305)
		{
			float _t_306;
			float _t_307;
		
			_t_306 = -1.0f * tx3_6_1;
			_t_307 = tx1_4_1 + _t_306;
			_t_308 = _t_307;
		
		}
else
		{
			float _t_309;
			float _t_310;
			float _t_311;
		
			_t_309 = -1.0f * tx3_6_1;
			_t_310 = tx1_4_1 + _t_309;
			_t_311 = -1.0f * _t_310;
			_t_308 = _t_311;
		
		}

	_t_312 = _t_308 * _t_308;
	_t_313 = _t_301 + _t_312;
	_t_314 = sqrt(_t_313);
	_t_315 = 1.0f / _t_314;
	_t_316 = tegpixellet_block_8(ty3_9_1,ty1_7_1,tx1_4_1,tx3_6_1,_t_315,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);
	_t_317[0] =0.0f;
_t_317[1] =0.0f;
_t_317[2] =_t_316;
_t_317[3] =0.0f;
_t_317[4] =0.0f;
_t_317[5] =0.0f;
_t_317[6] =0.0f;
_t_317[7] =0.0f;
_t_317[8] =0.0f;
_t_317[9] =0.0f;
_t_317[10] =0.0f;
_t_317[11] =0.0f;
_t_317[12] =0.0f;
_t_317[13] =0.0f;
_t_317[14] =0.0f;
_t_317[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_318[__iter__] = _t_290[__iter__] + _t_317[__iter__];
	_t_319 = -1.0f * ty1_7_1;
	_t_320 = ty3_9_1 + _t_319;
	_t_321 = -1.0f * _t_320;
	_t_322 = _t_321 < 0.0f;
	if(_t_322)
		{
			float _t_323;
			float _t_324;
		
			_t_323 = -1.0f * ty1_7_1;
			_t_324 = ty3_9_1 + _t_323;
			_t_325 = _t_324;
		
		}
else
		{
			float _t_326;
			float _t_327;
			float _t_328;
		
			_t_326 = -1.0f * ty1_7_1;
			_t_327 = ty3_9_1 + _t_326;
			_t_328 = -1.0f * _t_327;
			_t_325 = _t_328;
		
		}

	_t_329 = _t_325 * _t_325;
	_t_330 = -1.0f * ty1_7_1;
	_t_331 = ty3_9_1 + _t_330;
	_t_332 = -1.0f * _t_331;
	_t_333 = _t_332 < 0.0f;
	if(_t_333)
		{
			float _t_334;
			float _t_335;
		
			_t_334 = -1.0f * tx3_6_1;
			_t_335 = tx1_4_1 + _t_334;
			_t_336 = _t_335;
		
		}
else
		{
			float _t_337;
			float _t_338;
			float _t_339;
		
			_t_337 = -1.0f * tx3_6_1;
			_t_338 = tx1_4_1 + _t_337;
			_t_339 = -1.0f * _t_338;
			_t_336 = _t_339;
		
		}

	_t_340 = _t_336 * _t_336;
	_t_341 = _t_329 + _t_340;
	_t_342 = sqrt(_t_341);
	_t_343 = 1.0f / _t_342;
	_t_344 = tegpixellet_block_9(ty3_9_1,ty1_7_1,tx1_4_1,tx3_6_1,_t_343,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);
	_t_345[0] =0.0f;
_t_345[1] =0.0f;
_t_345[2] =_t_344;
_t_345[3] =0.0f;
_t_345[4] =0.0f;
_t_345[5] =0.0f;
_t_345[6] =0.0f;
_t_345[7] =0.0f;
_t_345[8] =0.0f;
_t_345[9] =0.0f;
_t_345[10] =0.0f;
_t_345[11] =0.0f;
_t_345[12] =0.0f;
_t_345[13] =0.0f;
_t_345[14] =0.0f;
_t_345[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_346[__iter__] = _t_318[__iter__] + _t_345[__iter__];
	_t_347 = -1.0f * ty2_8_1;
	_t_348 = ty1_7_1 + _t_347;
	_t_349 = -1.0f * _t_348;
	_t_350 = _t_349 < 0.0f;
	if(_t_350)
		{
			float _t_351;
			float _t_352;
		
			_t_351 = -1.0f * ty2_8_1;
			_t_352 = ty1_7_1 + _t_351;
			_t_353 = _t_352;
		
		}
else
		{
			float _t_354;
			float _t_355;
			float _t_356;
		
			_t_354 = -1.0f * ty2_8_1;
			_t_355 = ty1_7_1 + _t_354;
			_t_356 = -1.0f * _t_355;
			_t_353 = _t_356;
		
		}

	_t_357 = _t_353 * _t_353;
	_t_358 = -1.0f * ty2_8_1;
	_t_359 = ty1_7_1 + _t_358;
	_t_360 = -1.0f * _t_359;
	_t_361 = _t_360 < 0.0f;
	if(_t_361)
		{
			float _t_362;
			float _t_363;
		
			_t_362 = -1.0f * tx1_4_1;
			_t_363 = tx2_5_1 + _t_362;
			_t_364 = _t_363;
		
		}
else
		{
			float _t_365;
			float _t_366;
			float _t_367;
		
			_t_365 = -1.0f * tx1_4_1;
			_t_366 = tx2_5_1 + _t_365;
			_t_367 = -1.0f * _t_366;
			_t_364 = _t_367;
		
		}

	_t_368 = _t_364 * _t_364;
	_t_369 = _t_357 + _t_368;
	_t_370 = sqrt(_t_369);
	_t_371 = 1.0f / _t_370;
	_t_372 = tegpixellet_block_10(ty1_7_1,ty2_8_1,tx2_5_1,tx1_4_1,_t_371,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);
	_t_373[0] =0.0f;
_t_373[1] =0.0f;
_t_373[2] =0.0f;
_t_373[3] =_t_372;
_t_373[4] =0.0f;
_t_373[5] =0.0f;
_t_373[6] =0.0f;
_t_373[7] =0.0f;
_t_373[8] =0.0f;
_t_373[9] =0.0f;
_t_373[10] =0.0f;
_t_373[11] =0.0f;
_t_373[12] =0.0f;
_t_373[13] =0.0f;
_t_373[14] =0.0f;
_t_373[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_374[__iter__] = _t_346[__iter__] + _t_373[__iter__];
	_t_375 = -1.0f * ty1_7_1;
	_t_376 = ty3_9_1 + _t_375;
	_t_377 = -1.0f * _t_376;
	_t_378 = _t_377 < 0.0f;
	if(_t_378)
		{
			float _t_379;
			float _t_380;
		
			_t_379 = -1.0f * ty1_7_1;
			_t_380 = ty3_9_1 + _t_379;
			_t_381 = _t_380;
		
		}
else
		{
			float _t_382;
			float _t_383;
			float _t_384;
		
			_t_382 = -1.0f * ty1_7_1;
			_t_383 = ty3_9_1 + _t_382;
			_t_384 = -1.0f * _t_383;
			_t_381 = _t_384;
		
		}

	_t_385 = _t_381 * _t_381;
	_t_386 = -1.0f * ty1_7_1;
	_t_387 = ty3_9_1 + _t_386;
	_t_388 = -1.0f * _t_387;
	_t_389 = _t_388 < 0.0f;
	if(_t_389)
		{
			float _t_390;
			float _t_391;
		
			_t_390 = -1.0f * tx3_6_1;
			_t_391 = tx1_4_1 + _t_390;
			_t_392 = _t_391;
		
		}
else
		{
			float _t_393;
			float _t_394;
			float _t_395;
		
			_t_393 = -1.0f * tx3_6_1;
			_t_394 = tx1_4_1 + _t_393;
			_t_395 = -1.0f * _t_394;
			_t_392 = _t_395;
		
		}

	_t_396 = _t_392 * _t_392;
	_t_397 = _t_385 + _t_396;
	_t_398 = sqrt(_t_397);
	_t_399 = 1.0f / _t_398;
	_t_400 = tegpixellet_block_11(ty3_9_1,ty1_7_1,tx1_4_1,tx3_6_1,_t_399,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);
	_t_401[0] =0.0f;
_t_401[1] =0.0f;
_t_401[2] =0.0f;
_t_401[3] =_t_400;
_t_401[4] =0.0f;
_t_401[5] =0.0f;
_t_401[6] =0.0f;
_t_401[7] =0.0f;
_t_401[8] =0.0f;
_t_401[9] =0.0f;
_t_401[10] =0.0f;
_t_401[11] =0.0f;
_t_401[12] =0.0f;
_t_401[13] =0.0f;
_t_401[14] =0.0f;
_t_401[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_402[__iter__] = _t_374[__iter__] + _t_401[__iter__];
	_t_403 = -1.0f * ty1_7_1;
	_t_404 = ty3_9_1 + _t_403;
	_t_405 = -1.0f * _t_404;
	_t_406 = _t_405 < 0.0f;
	if(_t_406)
		{
			float _t_407;
			float _t_408;
		
			_t_407 = -1.0f * ty1_7_1;
			_t_408 = ty3_9_1 + _t_407;
			_t_409 = _t_408;
		
		}
else
		{
			float _t_410;
			float _t_411;
			float _t_412;
		
			_t_410 = -1.0f * ty1_7_1;
			_t_411 = ty3_9_1 + _t_410;
			_t_412 = -1.0f * _t_411;
			_t_409 = _t_412;
		
		}

	_t_413 = _t_409 * _t_409;
	_t_414 = -1.0f * ty1_7_1;
	_t_415 = ty3_9_1 + _t_414;
	_t_416 = -1.0f * _t_415;
	_t_417 = _t_416 < 0.0f;
	if(_t_417)
		{
			float _t_418;
			float _t_419;
		
			_t_418 = -1.0f * tx3_6_1;
			_t_419 = tx1_4_1 + _t_418;
			_t_420 = _t_419;
		
		}
else
		{
			float _t_421;
			float _t_422;
			float _t_423;
		
			_t_421 = -1.0f * tx3_6_1;
			_t_422 = tx1_4_1 + _t_421;
			_t_423 = -1.0f * _t_422;
			_t_420 = _t_423;
		
		}

	_t_424 = _t_420 * _t_420;
	_t_425 = _t_413 + _t_424;
	_t_426 = sqrt(_t_425);
	_t_427 = 1.0f / _t_426;
	_t_428 = tegpixellet_block_12(ty3_9_1,ty1_7_1,tx1_4_1,tx3_6_1,_t_427,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);
	_t_429[0] =0.0f;
_t_429[1] =0.0f;
_t_429[2] =0.0f;
_t_429[3] =_t_428;
_t_429[4] =0.0f;
_t_429[5] =0.0f;
_t_429[6] =0.0f;
_t_429[7] =0.0f;
_t_429[8] =0.0f;
_t_429[9] =0.0f;
_t_429[10] =0.0f;
_t_429[11] =0.0f;
_t_429[12] =0.0f;
_t_429[13] =0.0f;
_t_429[14] =0.0f;
_t_429[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_430[__iter__] = _t_402[__iter__] + _t_429[__iter__];
	_t_431 = -1.0f * ty2_8_1;
	_t_432 = ty1_7_1 + _t_431;
	_t_433 = -1.0f * _t_432;
	_t_434 = _t_433 < 0.0f;
	if(_t_434)
		{
			float _t_435;
			float _t_436;
		
			_t_435 = -1.0f * ty2_8_1;
			_t_436 = ty1_7_1 + _t_435;
			_t_437 = _t_436;
		
		}
else
		{
			float _t_438;
			float _t_439;
			float _t_440;
		
			_t_438 = -1.0f * ty2_8_1;
			_t_439 = ty1_7_1 + _t_438;
			_t_440 = -1.0f * _t_439;
			_t_437 = _t_440;
		
		}

	_t_441 = _t_437 * _t_437;
	_t_442 = -1.0f * ty2_8_1;
	_t_443 = ty1_7_1 + _t_442;
	_t_444 = -1.0f * _t_443;
	_t_445 = _t_444 < 0.0f;
	if(_t_445)
		{
			float _t_446;
			float _t_447;
		
			_t_446 = -1.0f * tx1_4_1;
			_t_447 = tx2_5_1 + _t_446;
			_t_448 = _t_447;
		
		}
else
		{
			float _t_449;
			float _t_450;
			float _t_451;
		
			_t_449 = -1.0f * tx1_4_1;
			_t_450 = tx2_5_1 + _t_449;
			_t_451 = -1.0f * _t_450;
			_t_448 = _t_451;
		
		}

	_t_452 = _t_448 * _t_448;
	_t_453 = _t_441 + _t_452;
	_t_454 = sqrt(_t_453);
	_t_455 = 1.0f / _t_454;
	_t_456 = tegpixellet_block_13(ty1_7_1,ty2_8_1,tx2_5_1,tx1_4_1,_t_455,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,tx3_6_1,ty3_9_1);
	_t_457[0] =0.0f;
_t_457[1] =0.0f;
_t_457[2] =0.0f;
_t_457[3] =0.0f;
_t_457[4] =_t_456;
_t_457[5] =0.0f;
_t_457[6] =0.0f;
_t_457[7] =0.0f;
_t_457[8] =0.0f;
_t_457[9] =0.0f;
_t_457[10] =0.0f;
_t_457[11] =0.0f;
_t_457[12] =0.0f;
_t_457[13] =0.0f;
_t_457[14] =0.0f;
_t_457[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_458[__iter__] = _t_430[__iter__] + _t_457[__iter__];
	_t_459 = -1.0f * ty3_9_1;
	_t_460 = ty2_8_1 + _t_459;
	_t_461 = -1.0f * _t_460;
	_t_462 = _t_461 < 0.0f;
	if(_t_462)
		{
			float _t_463;
			float _t_464;
		
			_t_463 = -1.0f * ty3_9_1;
			_t_464 = ty2_8_1 + _t_463;
			_t_465 = _t_464;
		
		}
else
		{
			float _t_466;
			float _t_467;
			float _t_468;
		
			_t_466 = -1.0f * ty3_9_1;
			_t_467 = ty2_8_1 + _t_466;
			_t_468 = -1.0f * _t_467;
			_t_465 = _t_468;
		
		}

	_t_469 = _t_465 * _t_465;
	_t_470 = -1.0f * ty3_9_1;
	_t_471 = ty2_8_1 + _t_470;
	_t_472 = -1.0f * _t_471;
	_t_473 = _t_472 < 0.0f;
	if(_t_473)
		{
			float _t_474;
			float _t_475;
		
			_t_474 = -1.0f * tx2_5_1;
			_t_475 = tx3_6_1 + _t_474;
			_t_476 = _t_475;
		
		}
else
		{
			float _t_477;
			float _t_478;
			float _t_479;
		
			_t_477 = -1.0f * tx2_5_1;
			_t_478 = tx3_6_1 + _t_477;
			_t_479 = -1.0f * _t_478;
			_t_476 = _t_479;
		
		}

	_t_480 = _t_476 * _t_476;
	_t_481 = _t_469 + _t_480;
	_t_482 = sqrt(_t_481);
	_t_483 = 1.0f / _t_482;
	_t_484 = tegpixellet_block_14(ty2_8_1,ty3_9_1,tx3_6_1,tx2_5_1,_t_483,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);
	_t_485[0] =0.0f;
_t_485[1] =0.0f;
_t_485[2] =0.0f;
_t_485[3] =0.0f;
_t_485[4] =_t_484;
_t_485[5] =0.0f;
_t_485[6] =0.0f;
_t_485[7] =0.0f;
_t_485[8] =0.0f;
_t_485[9] =0.0f;
_t_485[10] =0.0f;
_t_485[11] =0.0f;
_t_485[12] =0.0f;
_t_485[13] =0.0f;
_t_485[14] =0.0f;
_t_485[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_486[__iter__] = _t_458[__iter__] + _t_485[__iter__];
	_t_487 = -1.0f * ty3_9_1;
	_t_488 = ty2_8_1 + _t_487;
	_t_489 = -1.0f * _t_488;
	_t_490 = _t_489 < 0.0f;
	if(_t_490)
		{
			float _t_491;
			float _t_492;
		
			_t_491 = -1.0f * ty3_9_1;
			_t_492 = ty2_8_1 + _t_491;
			_t_493 = _t_492;
		
		}
else
		{
			float _t_494;
			float _t_495;
			float _t_496;
		
			_t_494 = -1.0f * ty3_9_1;
			_t_495 = ty2_8_1 + _t_494;
			_t_496 = -1.0f * _t_495;
			_t_493 = _t_496;
		
		}

	_t_497 = _t_493 * _t_493;
	_t_498 = -1.0f * ty3_9_1;
	_t_499 = ty2_8_1 + _t_498;
	_t_500 = -1.0f * _t_499;
	_t_501 = _t_500 < 0.0f;
	if(_t_501)
		{
			float _t_502;
			float _t_503;
		
			_t_502 = -1.0f * tx2_5_1;
			_t_503 = tx3_6_1 + _t_502;
			_t_504 = _t_503;
		
		}
else
		{
			float _t_505;
			float _t_506;
			float _t_507;
		
			_t_505 = -1.0f * tx2_5_1;
			_t_506 = tx3_6_1 + _t_505;
			_t_507 = -1.0f * _t_506;
			_t_504 = _t_507;
		
		}

	_t_508 = _t_504 * _t_504;
	_t_509 = _t_497 + _t_508;
	_t_510 = sqrt(_t_509);
	_t_511 = 1.0f / _t_510;
	_t_512 = tegpixellet_block_15(ty2_8_1,ty3_9_1,tx3_6_1,tx2_5_1,_t_511,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);
	_t_513[0] =0.0f;
_t_513[1] =0.0f;
_t_513[2] =0.0f;
_t_513[3] =0.0f;
_t_513[4] =_t_512;
_t_513[5] =0.0f;
_t_513[6] =0.0f;
_t_513[7] =0.0f;
_t_513[8] =0.0f;
_t_513[9] =0.0f;
_t_513[10] =0.0f;
_t_513[11] =0.0f;
_t_513[12] =0.0f;
_t_513[13] =0.0f;
_t_513[14] =0.0f;
_t_513[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_514[__iter__] = _t_486[__iter__] + _t_513[__iter__];
	_t_515 = -1.0f * ty3_9_1;
	_t_516 = ty2_8_1 + _t_515;
	_t_517 = -1.0f * _t_516;
	_t_518 = _t_517 < 0.0f;
	if(_t_518)
		{
			float _t_519;
			float _t_520;
		
			_t_519 = -1.0f * ty3_9_1;
			_t_520 = ty2_8_1 + _t_519;
			_t_521 = _t_520;
		
		}
else
		{
			float _t_522;
			float _t_523;
			float _t_524;
		
			_t_522 = -1.0f * ty3_9_1;
			_t_523 = ty2_8_1 + _t_522;
			_t_524 = -1.0f * _t_523;
			_t_521 = _t_524;
		
		}

	_t_525 = _t_521 * _t_521;
	_t_526 = -1.0f * ty3_9_1;
	_t_527 = ty2_8_1 + _t_526;
	_t_528 = -1.0f * _t_527;
	_t_529 = _t_528 < 0.0f;
	if(_t_529)
		{
			float _t_530;
			float _t_531;
		
			_t_530 = -1.0f * tx2_5_1;
			_t_531 = tx3_6_1 + _t_530;
			_t_532 = _t_531;
		
		}
else
		{
			float _t_533;
			float _t_534;
			float _t_535;
		
			_t_533 = -1.0f * tx2_5_1;
			_t_534 = tx3_6_1 + _t_533;
			_t_535 = -1.0f * _t_534;
			_t_532 = _t_535;
		
		}

	_t_536 = _t_532 * _t_532;
	_t_537 = _t_525 + _t_536;
	_t_538 = sqrt(_t_537);
	_t_539 = 1.0f / _t_538;
	_t_540 = tegpixellet_block_16(ty2_8_1,ty3_9_1,tx3_6_1,tx2_5_1,_t_539,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty1_7_1,tx1_4_1);
	_t_541[0] =0.0f;
_t_541[1] =0.0f;
_t_541[2] =0.0f;
_t_541[3] =0.0f;
_t_541[4] =0.0f;
_t_541[5] =_t_540;
_t_541[6] =0.0f;
_t_541[7] =0.0f;
_t_541[8] =0.0f;
_t_541[9] =0.0f;
_t_541[10] =0.0f;
_t_541[11] =0.0f;
_t_541[12] =0.0f;
_t_541[13] =0.0f;
_t_541[14] =0.0f;
_t_541[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_542[__iter__] = _t_514[__iter__] + _t_541[__iter__];
	_t_543 = -1.0f * ty1_7_1;
	_t_544 = ty3_9_1 + _t_543;
	_t_545 = -1.0f * _t_544;
	_t_546 = _t_545 < 0.0f;
	if(_t_546)
		{
			float _t_547;
			float _t_548;
		
			_t_547 = -1.0f * ty1_7_1;
			_t_548 = ty3_9_1 + _t_547;
			_t_549 = _t_548;
		
		}
else
		{
			float _t_550;
			float _t_551;
			float _t_552;
		
			_t_550 = -1.0f * ty1_7_1;
			_t_551 = ty3_9_1 + _t_550;
			_t_552 = -1.0f * _t_551;
			_t_549 = _t_552;
		
		}

	_t_553 = _t_549 * _t_549;
	_t_554 = -1.0f * ty1_7_1;
	_t_555 = ty3_9_1 + _t_554;
	_t_556 = -1.0f * _t_555;
	_t_557 = _t_556 < 0.0f;
	if(_t_557)
		{
			float _t_558;
			float _t_559;
		
			_t_558 = -1.0f * tx3_6_1;
			_t_559 = tx1_4_1 + _t_558;
			_t_560 = _t_559;
		
		}
else
		{
			float _t_561;
			float _t_562;
			float _t_563;
		
			_t_561 = -1.0f * tx3_6_1;
			_t_562 = tx1_4_1 + _t_561;
			_t_563 = -1.0f * _t_562;
			_t_560 = _t_563;
		
		}

	_t_564 = _t_560 * _t_560;
	_t_565 = _t_553 + _t_564;
	_t_566 = sqrt(_t_565);
	_t_567 = 1.0f / _t_566;
	_t_568 = tegpixellet_block_17(ty3_9_1,ty1_7_1,tx1_4_1,tx3_6_1,_t_567,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);
	_t_569[0] =0.0f;
_t_569[1] =0.0f;
_t_569[2] =0.0f;
_t_569[3] =0.0f;
_t_569[4] =0.0f;
_t_569[5] =_t_568;
_t_569[6] =0.0f;
_t_569[7] =0.0f;
_t_569[8] =0.0f;
_t_569[9] =0.0f;
_t_569[10] =0.0f;
_t_569[11] =0.0f;
_t_569[12] =0.0f;
_t_569[13] =0.0f;
_t_569[14] =0.0f;
_t_569[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_570[__iter__] = _t_542[__iter__] + _t_569[__iter__];
	_t_571 = -1.0f * ty1_7_1;
	_t_572 = ty3_9_1 + _t_571;
	_t_573 = -1.0f * _t_572;
	_t_574 = _t_573 < 0.0f;
	if(_t_574)
		{
			float _t_575;
			float _t_576;
		
			_t_575 = -1.0f * ty1_7_1;
			_t_576 = ty3_9_1 + _t_575;
			_t_577 = _t_576;
		
		}
else
		{
			float _t_578;
			float _t_579;
			float _t_580;
		
			_t_578 = -1.0f * ty1_7_1;
			_t_579 = ty3_9_1 + _t_578;
			_t_580 = -1.0f * _t_579;
			_t_577 = _t_580;
		
		}

	_t_581 = _t_577 * _t_577;
	_t_582 = -1.0f * ty1_7_1;
	_t_583 = ty3_9_1 + _t_582;
	_t_584 = -1.0f * _t_583;
	_t_585 = _t_584 < 0.0f;
	if(_t_585)
		{
			float _t_586;
			float _t_587;
		
			_t_586 = -1.0f * tx3_6_1;
			_t_587 = tx1_4_1 + _t_586;
			_t_588 = _t_587;
		
		}
else
		{
			float _t_589;
			float _t_590;
			float _t_591;
		
			_t_589 = -1.0f * tx3_6_1;
			_t_590 = tx1_4_1 + _t_589;
			_t_591 = -1.0f * _t_590;
			_t_588 = _t_591;
		
		}

	_t_592 = _t_588 * _t_588;
	_t_593 = _t_581 + _t_592;
	_t_594 = sqrt(_t_593);
	_t_595 = 1.0f / _t_594;
	_t_596 = tegpixellet_block_18(ty3_9_1,ty1_7_1,tx1_4_1,tx3_6_1,_t_595,px1_11_1,px0_10_1,py1_13_1,py0_12_1,tc0_17_1,pc0_14_1,tc1_18_1,pc1_15_1,tc2_19_1,pc2_16_1,ty2_8_1,tx2_5_1);
	_t_597[0] =0.0f;
_t_597[1] =0.0f;
_t_597[2] =0.0f;
_t_597[3] =0.0f;
_t_597[4] =0.0f;
_t_597[5] =_t_596;
_t_597[6] =0.0f;
_t_597[7] =0.0f;
_t_597[8] =0.0f;
_t_597[9] =0.0f;
_t_597[10] =0.0f;
_t_597[11] =0.0f;
_t_597[12] =0.0f;
_t_597[13] =0.0f;
_t_597[14] =0.0f;
_t_597[15] =0.0f;
	for(int __iter__ = 0; __iter__ < 16; __iter__++)   _t_598[__iter__] = _t_570[__iter__] + _t_597[__iter__];

	return _t_598;
}
